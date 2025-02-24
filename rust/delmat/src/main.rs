use std::collections::{HashMap, HashSet};
use std::fs;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tch::{CModule, Device as TchDevice, Kind, Tensor, nn};
use tokenizers::Tokenizer;
use tensorrt_rs::{Engine, Builder};

#[derive(Deserialize, Serialize)]
struct Config {
    model_name: String,
    output_dir: String,
    learning_rate: f64,
    num_epochs: usize,
    penalty_words: Vec<String>,
    reward_words: Vec<String>,
    restricted_prompts: Vec<String>,
    accepted_prompts: Vec<String>,
    rep_loss_threshold: usize,
    enable_penalty_loss: bool,
    enable_reward_loss: bool,
    enable_rep_loss: bool,
    enable_ground_truth_loss: bool,
    use_tensorrt: bool,
}

fn load_config(file_path: &str) -> Result<Config> {
    let config_str = fs::read_to_string(file_path)?;
    let config: Config = serde_json::from_str(&config_str)?;
    Ok(config)
}

struct PromptDataset {
    encodings: Vec<Tensor>,
}

impl PromptDataset {
    fn new(prompts: &Vec<String>, tokenizer: &Tokenizer, max_length: usize) -> Self {
        let mut encodings = Vec::new();
        for prompt in prompts {
            let encoding = tokenize_prompt(prompt, tokenizer, max_length);
            encodings.push(encoding);
        }
        Self { encodings }
    }
    fn len(&self) -> usize { self.encodings.len() }
    fn get(&self, idx: usize) -> &Tensor { &self.encodings[idx] }
}

fn tokenize_prompt(prompt: &str, tokenizer: &Tokenizer, max_length: usize) -> Tensor {
    let encoding = tokenizer.encode(prompt, true).unwrap();
    let ids = encoding.get_ids();
    let token_tensor = Tensor::of_slice(ids);
    if token_tensor.size()[0] < max_length as i64 {
        let pad = Tensor::zeros(&[max_length as i64 - token_tensor.size()[0]], (Kind::Int64, TchDevice::Cpu));
        Tensor::cat(&[token_tensor, pad], 0)
    } else {
        token_tensor.narrow(0, 0, max_length as i64)
    }
}

// This function assumes the model returns a tuple where the first element is logits,
// and subsequent elements are per-layer activations.
// Simulated forward pass that returns (logits, per-layer activations)
fn forward_with_activations(model: &CModule, input: &Tensor) -> (Tensor, Vec<Tensor>) {
    let output = model.forward_ts(&[input]).unwrap();
    let outputs = output.to_tuple().unwrap();
    let logits = outputs[0].shallow_clone();
    let mut activations = Vec::new();
    for i in 1..outputs.len() { activations.push(outputs[i].shallow_clone()); }
    (logits, activations)
}

fn activation_distance_loss(
    activations: &HashMap<usize, Vec<Tensor>>,
    mean_restricted: &HashMap<usize, Tensor>,
    mean_accepted: &HashMap<usize, Tensor>,
) -> Tensor {
    let mut losses = Vec::new();
    let margin = 0.1;
    for (&layer_id, acts) in activations.iter() {
        let current_activation = Tensor::stack(acts, 0);
        let mean_a = mean_restricted.get(&layer_id).unwrap().to_device(current_activation.device());
        let mean_b = mean_accepted.get(&layer_id).unwrap().to_device(current_activation.device());
        let sim_a = current_activation.cosine_similarity(&mean_a.unsqueeze(0), -1, 1e-8);
        let sim_b = current_activation.cosine_similarity(&mean_b.unsqueeze(0), -1, 1e-8);
        let diff = (sim_a - sim_b + margin).relu();
        losses.push(diff.mean(Kind::Float));
    }
    Tensor::stack(&losses, 0).mean(Kind::Float)
}

fn check_penalty_words(logits: &Tensor, penalty_word_ids: &Vec<Tensor>, multiplier: f64) -> Tensor {
    let batch_size = logits.size()[0];
    let seq_length = logits.size()[1];
    let mut penalty_loss = Tensor::zeros(&[], (Kind::Float, logits.device()));
    for i in 0..batch_size {
        let seq_logits = logits.get(i);
        for word_ids in penalty_word_ids.iter() {
            let word_ids = word_ids.to_device(logits.device());
            let word_length = word_ids.size()[0];
            if word_length > seq_length { continue; }
            for pos in 0..(seq_length - word_length + 1) {
                let window_logits = seq_logits.narrow(0, pos, word_length);
                let probs = window_logits.softmax(-1, Kind::Float);
                for j in 0..word_length {
                    let token_id = i64::from(word_ids.int64_value(&[j]));
                    penalty_loss += probs.get(j).double_value(&[token_id]) as f32;
                }
            }
        }
    }
    penalty_loss * multiplier
}

fn compute_repetition_penalty(logits: &Tensor, threshold: i64) -> Tensor {
    let batch_size = logits.size()[0];
    let seq_length = logits.size()[1];
    let mut repetition_loss = Tensor::zeros(&[], (Kind::Float, logits.device()));
    for i in 0..batch_size {
        let seq_logits = logits.get(i);
        let probs = seq_logits.softmax(-1, Kind::Float);
        let token_probs_over_seq = probs.transpose(0, 1);
        let significant = token_probs_over_seq.gt(0.1).to_kind(Kind::Float);
        let token_counts = significant.sum_dim_intlist(&[1], false, Kind::Float);
        let excess_counts = (token_counts - threshold as f32).clamp_min(0.0);
        let rep_penalty = excess_counts.pow(2);
        repetition_loss += rep_penalty.sum(Kind::Float);
    }
    repetition_loss * 1e-4
}

fn compute_ground_truth_loss(logits: &Tensor, ground_truth_ids: &Tensor) -> Tensor {
    let seq_length = logits.size()[1];
    let mut gt_loss = Tensor::zeros(&[], (Kind::Float, logits.device()));
    for i in 0..logits.size()[0] {
        let seq_logits = logits.get(i);
        let start_pos = (seq_length - ground_truth_ids.size()[0]).max(0);
        for (ans_pos, token_id) in ground_truth_ids.iter::<i64>().unwrap().enumerate() {
            let pos = start_pos + ans_pos as i64;
            if pos >= seq_length { break; }
            let curr_logits = seq_logits.get(pos);
            let log_probs = curr_logits.log_softmax(-1, Kind::Float);
            let token_loss = -log_probs.double_value(&[token_id]);
            gt_loss += token_loss as f32;
        }
    }
    gt_loss
}

fn check_reward_words(logits: &Tensor, reward_word_ids: &Vec<Tensor>, reward: f64, threshold: f64) -> Tensor {
    let batch_size = logits.size()[0];
    let seq_length = logits.size()[1];
    let mut reward_loss = Tensor::zeros(&[], (Kind::Float, logits.device()));
    for i in 0..batch_size {
        let seq_logits = logits.get(i);
        for word_ids in reward_word_ids.iter() {
            let word_ids = word_ids.to_device(logits.device());
            let word_length = word_ids.size()[0];
            if word_length > seq_length { continue; }
            for pos in 0..(seq_length - word_length + 1) {
                let window_logits = seq_logits.narrow(0, pos, word_length);
                let probs = window_logits.softmax(-1, Kind::Float);
                let mut all_match = true;
                for j in 0..word_length {
                    let token_id = i64::from(word_ids.int64_value(&[j]));
                    if probs.get(j).double_value(&[token_id]) < threshold {
                        all_match = false;
                        break;
                    }
                }
                if all_match { reward_loss -= reward; }
            }
        }
    }
    reward_loss
}

fn process_prompts(prompts: &Vec<String>, model: &CModule, tokenizer: &Tokenizer, max_length: usize) -> HashMap<usize, Vec<Tensor>> {
    let mut activations_map: HashMap<usize, Vec<Tensor>> = HashMap::new();
    for prompt in prompts.iter() {
        let formatted = format!("User: {}\nAssistant:", prompt);
        let input = tokenize_prompt(&formatted, tokenizer, max_length)
            .unsqueeze(0)
            .to_device(TchDevice::Cuda(0));
        let (_logits, acts) = forward_with_activations(model, &input);
        for (layer_idx, act) in acts.into_iter().enumerate() {
            let last_act = act.select(1, act.size()[1] - 1);
            activations_map.entry(layer_idx).or_default().push(last_act);
        }
    }
    activations_map
}

fn compute_mean_activations_map(activations: &HashMap<usize, Vec<Tensor>>) -> HashMap<usize, Tensor> {
    let mut mean_map = HashMap::new();
    for (&layer_id, acts) in activations.iter() {
        let stacked = Tensor::stack(acts, 0);
        let mean = stacked.mean(Kind::Float);
        mean_map.insert(layer_id, mean);
    }
    mean_map
}

// ----- Simulated LoRA/PEFT integration functions -----
struct LoRAConfig {
    r: i32,
    lora_alpha: i32,
    target_modules: Vec<String>,
    lora_dropout: f64,
    bias: String,
    task_type: String,
}

fn get_peft_model(model: CModule, lora_config: &LoRAConfig) -> CModule {
    println!("Applying LoRA adapter: r={}, lora_alpha={}, targets={:?}",
             lora_config.r, lora_config.lora_alpha, lora_config.target_modules);
    // Dummy: return model unchanged.
    model
}

fn save_lora_adapter(model: &CModule, path: &str) -> Result<()> {
    println!("Saving LoRA adapter to {}", path);
    model.save(path)?;
    Ok(())
}

fn load_original_model(model_name: &str, device: TchDevice) -> Result<CModule> {
    println!("Loading original 16-bit model: {}", model_name);
    let mut model = CModule::load(model_name)?;
    // Simulate setting model to float16.
    model.to(device)?;
    Ok(model)
}

fn load_peft_adapter(model: CModule, adapter_path: &str) -> Result<CModule> {
    println!("Loading LoRA adapter from {}", adapter_path);
    // Dummy: in practice, load adapter weights and update model.
    Ok(model)
}

fn merge_and_unload(model: CModule) -> CModule {
    println!("Merging LoRA adapter into the base model and unloading adapter");
    // Dummy merge: return model unchanged.
    model
}
// --------------------------------------------------------

fn main() -> Result<()> {
    // Simulate enabling anomaly detection.
    println!("Enabling anomaly detection (simulated)");

    // Enable CUDA optimizations.
    tch::Cuda::set_user_enabled_cudnn(true);
    tch::Cuda::cudnn_set_benchmark(true);

    let config = load_config("delmatconfig.json")?;
    let device = TchDevice::Cuda(0);
    let tokenizer = Tokenizer::from_file("tokenizer.json")?;
    println!("Loading model: {}", config.model_name);
    let mut model = CModule::load(&config.model_name)?;
    model.to(device)?;

    let ground_truth_prompt = format!("User: {}\nAssistant:", "What is the capital of France?");
    let gt_input = tokenize_prompt(&ground_truth_prompt, &tokenizer, 128)
        .unsqueeze(0)
        .to_device(device);
    let (gt_logits, _) = forward_with_activations(&model, &gt_input);
    let ground_truth_ids = gt_logits.argmax(-1, false);

    let restricted_acts = process_prompts(&config.restricted_prompts, &model, &tokenizer, 128);
    let accepted_acts = process_prompts(&config.accepted_prompts, &model, &tokenizer, 128);
    let mean_restricted = compute_mean_activations_map(&restricted_acts);
    let mean_accepted = compute_mean_activations_map(&accepted_acts);

    let mut penalty_word_ids: Vec<Tensor> = Vec::new();
    let mut reward_word_ids: Vec<Tensor> = Vec::new();
    let mut seen = HashSet::new();
    for word in config.penalty_words.iter() {
        for variation in &[
            word.to_string(),
            word.to_lowercase(),
            format!(" {}", word),
            format!(" {}", word.to_lowercase()),
        ] {
            let encoding = tokenizer.encode(variation, true).unwrap();
            let ids = Tensor::of_slice(encoding.get_ids());
            let key = ids.sum(Kind::Int64).int64_value(&[]);
            if seen.insert(key) { penalty_word_ids.push(ids); }
        }
    }
    seen.clear();
    for word in config.reward_words.iter() {
        for variation in &[
            word.to_string(),
            word.to_lowercase(),
            format!(" {}", word),
            format!(" {}", word.to_lowercase()),
        ] {
            let encoding = tokenizer.encode(variation, true).unwrap();
            let ids = Tensor::of_slice(encoding.get_ids());
            let key = ids.sum(Kind::Int64).int64_value(&[]);
            if seen.insert(key) { reward_word_ids.push(ids); }
        }
    }

    let train_dataset = PromptDataset::new(&config.restricted_prompts, &tokenizer, 128);
    let vs = nn::VarStore::new(device);
    let mut opt = nn::Adam::default().build(&vs, config.learning_rate)?;

    // Apply LoRA adapter before training.
    let lora_config = LoRAConfig {
        r: 8,
        lora_alpha: 16,
        target_modules: vec!["layer1".to_string(), "layer2".to_string()],
        lora_dropout: 0.0,
        bias: "none".to_string(),
        task_type: "CAUSAL_LM".to_string(),
    };
    model = get_peft_model(model, &lora_config);

    // Training loop with CUDA stream synchronization and simulated cache clearing.
    for epoch in 0..config.num_epochs {
        println!("Epoch {}", epoch);
        let mut total_loss = 0.0;
        let mut num_batches = 0;
        for i in 0..train_dataset.len() {
            opt.zero_grad();
            let input = train_dataset.get(i)
                .unsqueeze(0)
                .to_device(device);
            let (logits, acts) = forward_with_activations(&model, &input);
            let mut batch_acts: HashMap<usize, Vec<Tensor>> = HashMap::new();
            for (layer_idx, act) in acts.into_iter().enumerate() {
                let last_act = act.select(1, act.size()[1] - 1);
                batch_acts.entry(layer_idx).or_default().push(last_act);
            }
            let act_loss = activation_distance_loss(&batch_acts, &mean_restricted, &mean_accepted);
            let penalty_loss = if config.enable_penalty_loss {
                check_penalty_words(&logits, &penalty_word_ids, 1.0)
            } else { Tensor::zeros(&[], (Kind::Float, device)) };
            let rep_loss = if config.enable_rep_loss {
                compute_repetition_penalty(&logits, config.rep_loss_threshold as i64)
            } else { Tensor::zeros(&[], (Kind::Float, device)) };
            let reward_loss = if config.enable_reward_loss {
                check_reward_words(&logits, &reward_word_ids, 1.0, 1e-3)
            } else { Tensor::zeros(&[], (Kind::Float, device)) };
            let gt_loss = if config.enable_ground_truth_loss {
                compute_ground_truth_loss(&logits, &ground_truth_ids)
            } else { Tensor::zeros(&[], (Kind::Float, device)) };

            let loss = act_loss + penalty_loss + rep_loss + reward_loss + gt_loss;
            loss.backward();
            tch::nn::clip_grad_norm_(vs.trainable_variables(), 1.0);
            opt.step();
            // Simulate cache clearing.
            tch::Cuda::synchronize(0);
            total_loss += f64::from(loss);
            num_batches += 1;
            println!("Epoch: {}, Batch: {}, Loss: {:.6}", epoch, i, f64::from(loss));
        }
        println!("Epoch: {}, Average Loss: {:.6}", epoch, total_loss / num_batches as f64);
    }

    // Save the LoRA adapter.
    let lora_path = format!("{}/lora_adapter.pt", config.output_dir);
    println!("Saving the trained LoRA adapter...");
    save_lora_adapter(&model, &lora_path)?;

    println!("Cleaning up model from memory...");
    drop(model);
    tch::Cuda::synchronize(0); // simulate empty_cache

    // Reload original 16-bit model.
    let mut model = load_original_model(&config.model_name, device)?;
    println!("Loading LoRA adapter and applying it to the 16-bit model...");
    model = load_peft_adapter(model, &lora_path)?;
    println!("Merging LoRA adapter into the base model...");
    model = merge_and_unload(model);

    let save_path = format!("{}/final_model", config.output_dir);
    println!("Saving final merged model...");
    model.save(&format!("{}/final_model.pt", config.output_dir))?;
    tokenizer.save(&format!("{}/final_tokenizer.json", save_path), false)?;

    // Optionally run with TensorRT for faster cuda infrance.
    if config.use_tensorrt {
        println!("Building TensorRT engine...");
        let builder = Builder::new()?;
        let engine = builder.build_engine_from_file(&format!("{}/final_model.pt", config.output_dir))?;
        println!("Running inference with TensorRT engine...");
        let input_tensor = Tensor::rand(&[1, 3, 224, 224], (Kind::Float, device));
        let trt_output = engine.run_inference(&[input_tensor.data_ptr()])?;
        println!("TensorRT Inference output: {:?}", trt_output);
    }

    Ok(())
}
