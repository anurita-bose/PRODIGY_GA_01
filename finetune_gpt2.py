"""
PRODIGY_GA_01 — Fine-tuning GPT-2 on Shakespeare text
Task: Train a model to generate coherent, contextually relevant text
based on a given prompt, by fine-tuning GPT-2 on a custom dataset.
"""

import torch
import urllib.request
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import Dataset as TorchDataset


# --- Step 1: Check GPU ---
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- Step 2: Download dataset ---
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "shakespeare.txt"

urllib.request.urlretrieve(DATA_URL, DATA_PATH)
with open(DATA_PATH, "r") as f:
    text = f.read()
print(f"Total characters in dataset: {len(text)}")


# --- Step 3: Load tokenizer and base model ---
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)


# --- Step 4: Chunk text into fixed-length token sequences ---
def chunk_text(text, tokenizer, block_size=128):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens) - block_size, block_size):
        chunks.append(tokens[i:i + block_size])
    return chunks


chunks = chunk_text(text, tokenizer)
print(f"Number of training chunks: {len(chunks)}")


# --- Step 5: Wrap chunks in a plain PyTorch Dataset ---
class ShakespeareDataset(TorchDataset):
    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.chunks[idx])
        return {"input_ids": input_ids, "labels": input_ids.clone()}


dataset = ShakespeareDataset(chunks)


# --- Step 6: Text generation helper ---
def generate_text(model, tokenizer, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    output = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


PROMPTS = ["To be, or not to be", "O Romeo, Romeo", "What light through yonder window"]

print("\n--- BEFORE FINE-TUNING ---")
before_outputs = {}
for p in PROMPTS:
    out = generate_text(model, tokenizer, p)
    before_outputs[p] = out
    print(f"\nPrompt: {p}\nOutput: {out}")


# --- Step 7: Training configuration ---
training_args = TrainingArguments(
    output_dir="./gpt2-shakespeare",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=200,
    save_total_limit=2,
    logging_steps=20,
    fp16=torch.cuda.is_available(),
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)


# --- Step 8: Fine-tune ---
trainer.train()


# --- Step 9: Generate after fine-tuning ---
print("\n--- AFTER FINE-TUNING ---")
after_outputs = {}
for p in PROMPTS:
    out = generate_text(model, tokenizer, p)
    after_outputs[p] = out
    print(f"\nPrompt: {p}\nOutput: {out}")


# --- Step 10: Save model ---
trainer.save_model("./gpt2-shakespeare-final")
tokenizer.save_pretrained("./gpt2-shakespeare-final")
print("\nModel saved to ./gpt2-shakespeare-final")


# --- Step 11: Write outputs to file for README/sample_outputs.md ---
with open("sample_outputs.md", "w") as f:
    f.write("# Sample Outputs — Before vs After Fine-tuning\n\n")
    for p in PROMPTS:
        f.write(f"## Prompt: \"{p}\"\n\n")
        f.write(f"**Before fine-tuning:**\n> {before_outputs[p]}\n\n")
        f.write(f"**After fine-tuning:**\n> {after_outputs[p]}\n\n")
        f.write("---\n\n")

print("\nSample outputs written to sample_outputs.md")