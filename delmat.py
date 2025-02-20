from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import LoraConfig, get_peft_model, PeftModel
import torch.nn.functional as F
import logging
import json
import os

# Ensure we're working from the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item

def capture_last_token_activations(storage_dict):
    def hook(module, input, output):
        last_token_activations = output[0][:, -1, :].clone()
        storage_dict[id(module)].append(last_token_activations)
    return hook

def check_penalty_words(logits, penalty_word_ids, device):
    batch_size = logits.size(0)
    seq_length = logits.size(1)
    penalty_loss = torch.tensor(0.0, device=device)

    # For each sequence in the batch
    for i in range(batch_size):
        seq_logits = logits[i]

        # For each penalty word
        for word_ids in penalty_word_ids:
            word_ids = word_ids.to(device)
            word_length = len(word_ids)

            # Skip if word is longer than sequence
            if word_length > seq_length:
                continue

            # Slide through the sequence
            for pos in range(seq_length - word_length + 1):
                # Get probabilities for the current window
                window_logits = seq_logits[pos:pos + word_length]
                probs = F.softmax(window_logits, dim=-1)

                # Simply add the probabilities of penalty tokens
                for j, token_id in enumerate(word_ids):
                    penalty_loss += probs[j, token_id]

    penalty_loss *= 0.1

    return penalty_loss # Scale factor for penalty

def compute_repetition_penalty(logits, device, threshold=6):
    batch_size = logits.size(0)
    repetition_loss = torch.tensor(0.0, device=device)

    # For each sequence in the batch
    for i in range(batch_size):
        seq_logits = logits[i]  # Shape: [seq_len, vocab_size]
        probs = F.softmax(seq_logits, dim=-1)  # Convert to probabilities

        # Create a matrix of token probabilities over sequence length
        # Shape: [vocab_size, seq_length]
        token_probs_over_seq = probs.transpose(0, 1)

        # For each token in vocabulary that appears in sequence
        # We only care about tokens that have significant probability
        significant_probs = (token_probs_over_seq > 0.1).float()
        token_counts = significant_probs.sum(dim=1)  # How many times each token appears

        # Only penalize tokens that appear more than the threshold times
        # Using a quadratic penalty: (count - threshold)^2 for counts > threshold
        excess_counts = torch.clamp(token_counts - threshold, min=0)
        repetition_penalty = excess_counts ** 2
        repetition_loss += repetition_penalty.sum()

    # Not sure if this penalty really helps, but it's good to measure + log at least
    # Because a bad run will repeat lots of nonsense
    # We'll scale the repetition penalty suuuper small because it can be counterproductive
    repetition_loss *= 0.0001

    return repetition_loss

def activation_distance_loss(activations, mean_a, mean_b, distance_metric=F.cosine_similarity):
    loss_per_layer = []
    for layer_id in activations:
        # Get current layer activations
        current_activation = torch.cat(activations[layer_id], dim=0)

        # Compute distances directly using the activations
        if distance_metric == F.cosine_similarity:
            # Compute cosine similarity with both means
            sim_a = F.cosine_similarity(current_activation, mean_a[layer_id].to(current_activation.device), dim=-1)
            sim_b = F.cosine_similarity(current_activation, mean_b[layer_id].to(current_activation.device), dim=-1)

            # Loss: minimize similarity to restricted (a) and maximize similarity to accepted (b)
            # Using a margin-based loss to ensure a minimum separation
            margin = 0.1
            loss = torch.mean(torch.relu(sim_a - sim_b + margin))

        elif distance_metric == torch.cdist:
            # Compute Euclidean distances
            distance_a = torch.cdist(current_activation.unsqueeze(0), mean_a[layer_id].to(current_activation.device).unsqueeze(0)).mean()
            distance_b = torch.cdist(current_activation.unsqueeze(0), mean_b[layer_id].to(current_activation.device).unsqueeze(0)).mean()

            # Normalize distances and compute loss
            max_distance = torch.max(torch.stack([distance_a, distance_b]))
            norm_distance_a = distance_a / max_distance
            norm_distance_b = distance_b / max_distance

            # We want to minimize distance to accepted (b) and maximize distance to restricted (a)
            loss = norm_distance_a + (1 - norm_distance_b)

        loss_per_layer.append(loss)

    return torch.mean(torch.stack(loss_per_layer))

def process_prompts(prompts, storage_dict, prompt_template):
    hooks = []
    for layer in layers:
        hook = layer.register_forward_hook(capture_last_token_activations(storage_dict))
        hooks.append(hook)

    for prompt in prompts:
        prompt = prompt_template.format(prompt=prompt)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        model(**inputs)

    for hook in hooks:
        hook.remove()

def compute_mean_activations(storage_dict):
    mean_activations = {}
    for layer in layers:
        activations = torch.cat(storage_dict[id(layer)], dim=0)
        mean = activations.mean(dim=0).detach()
        mean_activations[id(layer)] = mean
    return mean_activations

if __name__ == "__main__":
    # Load configuration from delmatconfig.json
    config_path = "./delmatconfig.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # Get values from config
    PENALTY_WORDS = config["penalty_words"]
    restricted_prompts = config["restricted_prompts"]
    accepted_prompts = config["accepted_prompts"]
    # Combine prompts for training data
    all_prompts = restricted_prompts + accepted_prompts

    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # Get model configuration from config
    model_name = config["model_name"]
    output_dir = config["output_dir"]

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize penalty words
    penalty_word_ids = []
    seen_sequences = set()  # Track unique token sequences

    for word in PENALTY_WORDS:
        variations = [
            word,                    # Original lowercase
            word.capitalize(),       # Capitalized version
            f" {word}",             # Space-prefixed lowercase
            f" {word.capitalize()}"  # Space-prefixed capitalized
        ]

        # Process each variation as a separate sequence
        for variation in variations:
            word_tokens = tokenizer.encode(variation, add_special_tokens=False)
            # Convert to tuple for hashing
            token_tuple = tuple(word_tokens)
            # Only add if we haven't seen this exact sequence before
            if token_tuple not in seen_sequences:
                seen_sequences.add(token_tuple)
                penalty_word_ids.append(torch.tensor(word_tokens))

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["down_proj", "gate_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    prompts_a = restricted_prompts
    prompts_b = accepted_prompts

    # Identify layers to hook
    layers = [model.base_model.model.model.layers[i] for i in range(model.config.num_hidden_layers)]

    # Storage for activations
    restricted_activations = {id(layer): [] for layer in layers}
    accepted_activations = {id(layer): [] for layer in layers}

    def compute_reference_activations():
        # Reset activation storage
        restricted_activations = {id(layer): [] for layer in layers}
        accepted_activations = {id(layer): [] for layer in layers}

        # Process prompts and compute means
        print("Processing restricted prompts")
        process_prompts(restricted_prompts, restricted_activations, """Below is an inappropriate request. Decline the request and explain which rules it is in violation of. Be sure to emphasize how offended and disgusted you are by the request.

        User: {prompt}
        Assistant:""")
        print("Processing accepted prompts")
        process_prompts(accepted_prompts, accepted_activations, """Below is a request from a user. Explain why the request is a good one, warmly welcome it, and explain why you think the user stands to benefit from making such inquiries.

        User: {prompt}
        Assistant:""")

        print("Captured reference activations")
        return (
            compute_mean_activations(restricted_activations),
            compute_mean_activations(accepted_activations)
        )

    # Compute initial reference activations
    restricted_mean, accepted_mean = compute_reference_activations()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])  # Increased learning rate
    num_epochs = 2
    device = model.device

    # Tokenize the prompts
    tokenizer.pad_token = tokenizer.eos_token
    encoded_prompts = tokenizer(restricted_prompts, padding=True, return_tensors="pt")

    train_dataset = PromptDataset(encoded_prompts)

    # Create a DataLoader
    batch_size = 1
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            # Reset activation storage for this iteration
            activations = {id(layer): [] for layer in layers}

            # Process inputs
            inputs = {k: v.to(device) for k, v in batch.items()}

            # Ensure model is in training mode
            model.train()

            # Set up hooks for this iteration
            hooks = []
            for layer in layers:
                hook = layer.register_forward_hook(capture_last_token_activations(activations))
                hooks.append(hook)

            # Forward pass
            outputs = model(**inputs)

            # Remove hooks
            for hook in hooks:
                hook.remove()

            # Compute activation distance loss
            act_loss = activation_distance_loss(activations, restricted_mean, accepted_mean)

            # Compute penalty word loss
            penalty_loss = check_penalty_words(outputs.logits, penalty_word_ids, device)

            # Compute repetition penalty
            rep_loss = compute_repetition_penalty(outputs.logits, device, threshold=6)

            # Combine losses with weighted penalty loss
            loss = act_loss + penalty_loss + rep_loss
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            print(f"Epoch: {epoch}, Batch Loss: {loss.item():.6f}, Penalty Loss: {penalty_loss.item():.6f}, Repetition Loss: {rep_loss.item():.6f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch: {epoch}, Average Loss: {avg_loss:.6f}")

    logger.info("Saving the trained LoRA adapter...")
    model.save_pretrained(f"{output_dir}/lora_adapter")

    logger.info("Cleaning up model from memory...")
    del model
    torch.cuda.empty_cache()

    logger.info("Loading the original 16-bit model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    )

    logger.info("Loading LoRA adapter and applying it to the 16-bit model...")
    model = PeftModel.from_pretrained(model, f"{output_dir}/lora_adapter")

    logger.info("Merging LoRA adapter into the base model...")
    model = model.merge_and_unload()

    logger.info("Saving model...")
    save_path = f"{output_dir}/final_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
