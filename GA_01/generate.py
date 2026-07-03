"""
PRODIGY_GA_01 — Generate text from the fine-tuned GPT-2 model.
Run this after finetune_gpt2.py to interactively test the model.
"""

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

MODEL_PATH = "./gpt2-shakespeare-final"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path=MODEL_PATH):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).to(DEVICE)
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, top_k=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    output = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    model, tokenizer = load_model()

    print("Fine-tuned GPT-2 (Shakespeare) — type a prompt, or 'quit' to exit.\n")
    while True:
        prompt = input("Prompt: ")
        if prompt.strip().lower() == "quit":
            break
        result = generate_text(model, tokenizer, prompt)
        print(f"\nGenerated: {result}\n")