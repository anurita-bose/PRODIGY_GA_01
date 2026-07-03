# Sample Outputs — Before vs After Fine-tuning

## Prompt: "To be, or not to be"

**Before fine-tuning:**
> To be, or not to be, the only way to be the only way to have the same sex, it is not necessary to have the same sex all the time, because this is the only way to have a different sex. If we have a different sex at the beginning of our lives, we will have no more sex. If we have a different sex at the end of our lives, we will have different sex in many different ways. What's wrong with people who are obsessed

**After fine-tuning:**
> To be, or not to be:
> This is the word of the king, for he was the king's son:
> The king that is his son is his son's son:
> And the king that was his son hath the king's son's son:
> So we say, that both are sons' sons.
>
> DUKE VINCENTIO:
> I will hence unto the city: therefore, take the news.
>
> HENRY BOLINGBRO

---

## Observations
The base GPT-2 model's output on this prompt drifts into repetitive, unrelated modern phrasing with no dramatic structure. After fine-tuning on Shakespeare's works, the model adopts clear stylistic markers of the source text: character name headers (e.g. "DUKE VINCENTIO:"), archaic phrasing ("hath", "unto"), and verse-like rhythm with royal/dramatic subject matter — a direct reflection of the training corpus.