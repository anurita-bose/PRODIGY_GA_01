# PRODIGY_GA_01 — Text Generation with GPT-2

## Task
Train a model to generate coherent and contextually relevant text based on a
given prompt. Starting with GPT-2 (a transformer model developed by OpenAI),
this project fine-tunes the model on a custom dataset to produce text that
mimics the style and structure of the training data.

## Dataset
**Tiny Shakespeare** — the complete works of Shakespeare (~1.1M characters),
sourced from [karpathy/char-rnn](https://github.com/karpathy/char-rnn).
Chosen because the archaic style shift is easy to verify visually in
generated INSTANT output.

## Approach
1. Loaded the pretrained `gpt2` model and tokenizer from Hugging Face.
2. Split the Shakespeare corpus into 2,640 fixed-length chunks of 128 tokens each.
3. Wrapped the chunks in a custom PyTorch `Dataset` for causal language modeling.
4. Fine-tuned for 3 epochs (990 steps) using Hugging Face's `Trainer` API with
   mixed precision (`fp16`) on a Colab T4 GPU — training completed in ~5.5 minutes
   (327 seconds).
5. Compared generations from the same prompt before and after fine-tuning.



## Results

**Final training loss:** `3.4635` (after 3 epochs / 990 steps)
**Training runtime:** 327.3 seconds (~24 samples/sec on T4 GPU)

See [`sample_outputs.md`](./sample_outputs.md) for full before/after text
comparison. Summary: the base model's output on the prompt *"To be, or not to
be"* drifted into unrelated, repetitive modern phrasing. After fine-tuning,
outputs adopted clear Shakespearean traits — character-name headers (e.g.
"DUKE VINCENTIO:"), archaic diction ("hath", "unto"), and dramatic/royal
subject matter — directly reflecting the training corpus style.

## How to Run
```bash
pip install transformers torch accelerate
python finetune_gpt2.py    # trains the model, saves it, prints before/after outputs
python generate.py         # interactively generate text from the fine-tuned model
```

**Note:** the fine-tuned model weights (~500MB) are not included in this repo
due to GitHub's file size limits. Run `finetune_gpt2.py` to reproduce them.

## What I Learned
Fine-tuning a pretrained transformer is fundamentally different from training
from scratch — GPT-2 already has strong general language understanding, so a
relatively small dataset (~1.1M characters) and just 3 epochs were enough to
noticeably shift its output style, rather than needing to teach it grammar
and structure from zero. This is the core idea behind transfer learning:
large pretrained models act as reusable "knowledge bases" that can be
specialized cheaply for narrow domains. I also learned that `datasets`
library's automatic tensor formatting can break due to unrelated
`torchvision` version conflicts — solved by writing a plain PyTorch `Dataset`
class instead of relying on Hugging Face's `datasets.Dataset`.

## Tech Stack
- Python, PyTorch
- Hugging Face `transformers` (GPT-2, Trainer API)
- Google Colab (T4 GPU)
