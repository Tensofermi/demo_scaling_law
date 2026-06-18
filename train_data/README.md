# Training Data

This repository does not commit raw datasets, processed JSONL files, tokenized `.bin`
streams, or checkpoints.

Default six-source categories:

| category | source label | intended dataset |
| --- | --- | --- |
| story | tinystories | TinyStories |
| code | the_stack_smol_xl | The Stack Smol XL |
| dialogue | open_orca | OpenOrca |
| math | open_web_math | OpenWebMath |
| news | cc_news | CC-News style data |
| encyclopedia | wikipedia_en | English Wikipedia |

Before public release or redistribution, check every upstream dataset card and license.

Pipeline:

```bash
python -m demo_scaling.data.download --manifest train_data/manifests/default_sources.yaml --output train_data/raw
python -m demo_scaling.data.clean_split --input train_data/raw --output train_data/processed/splits --seed 42
python -m demo_scaling.data.metrics --input train_data/processed/splits --output train_data/metrics/doc_metrics.csv --workers 30
python -m demo_scaling.data.build_streams --splits-root train_data/processed/splits --output-root train_data/tokenized/gpt2 --mode all --force
```
