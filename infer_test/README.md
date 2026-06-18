# Inference Test Helpers

These small wrappers are for manual checkpoint inspection after a smoke or Slurm run.
They intentionally call the package entrypoints, so inference uses the same GPT-2 tokenizer
and RoPE GPT class as training.

```bash
python infer_test/one_shot.py --checkpoint logs/smoke_d1/checkpoints/final.pt --prompt "Once upon a time"
python infer_test/stream_chat.py --checkpoint logs/smoke_d1/checkpoints/final.pt
```
