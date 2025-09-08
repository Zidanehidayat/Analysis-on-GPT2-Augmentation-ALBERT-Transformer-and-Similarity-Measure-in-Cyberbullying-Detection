from transformers import pipeline, set_seed
from tqdm import tqdm
import pandas as pd
import os

# GPT2 model
set_seed(42)

def gpt2def(text, num_aug, model_name, max_new_tokens=10):
    generator = pipeline('text-generation', model=model_name)
    input_length = len(generator.tokenizer.encode(text, add_special_tokens=False))
    max_model_length = generator.model.config.n_positions  # usually 1024
    max_length = min(input_length + max_new_tokens, max_model_length)

    results = generator(
        text,
        max_length=max_length,
        num_return_sequences=num_aug,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        pad_token_id=generator.tokenizer.eos_token_id,
        truncation=True  # 👈 important fix
    )

    augmented = [res["generated_text"] for res in results]
    return augmented
