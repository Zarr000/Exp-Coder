# Legacy configuration archive

These YAML files predate the Exp-Coder canonical configuration system.

They are kept for reference and historical analysis only.

They are **NOT** loadable through `src.config.load_model_config` and
**NOT** canonical. Several reference `LlamaTokenizer` / `LlamaForCausalLM`
(HuggingFace classes) and vocabularies (e.g. 32000) that do not match the
canonical Exp-Coder 120M architecture (vocab 50304).

Active canonical files live one directory up:

- `configs/exp_coder_120m.yaml` — canonical model architecture
- `configs/exp_coder_tiny.yaml` — debug/tiny config for fast iteration
- `configs/training.yaml`       — canonical training-run configuration