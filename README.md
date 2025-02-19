# DELMAT
Decensoring Language Models Through Activation Tuning

This repository contains a training script that utilizes the following process:

1. Create a list of restricted prompts that your LLM is likely to refuse. I used 15 prompts that were unethical or explicit in nature. These are not included because I don't want to share explicit content.
2. Create a list of accepted prompts that your LLM is likely to accept (included). The included prompts are in the format "Are you allowed to explain ______" because I found that it worked well.
3. We run all of the restricted and accepted prompts through our model, registering a forward hook first and storing the activation data in two separate dicts.
4. We take our activation data dicts and compute the mean activations for each - producing new dicts that essentially answer the question "What does an average prompt refusal activation look like?" and the same for accepted prompts.
5. We tokenize all of our restricted prompts and build a dataset out of it.
6. We begin the training loop, registering forward hooks on each pass and capturing activation data during training
7. We calculate our loss by comparing our activation with the previously stored mean refusal activation and mean accepted activation. An activation closer to the mean refusal activation produces a higher loss. An additional penalty is added based on probabilities of tokens that are common in refusals (ex. "sorry")
8. Save the model and enjoy your new morally bankrupt LLM

You will need to tweak the learning rate and epochs per model - I've found that the optimal values are wildly inconsistent.

Remember to fill in `restricted_prompts` before use, containing prompts that you are certain your model will refuse.
