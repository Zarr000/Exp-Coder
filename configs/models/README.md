# Model Configurations

This directory contains configuration files for different Expera AI model sizes.

## Available Models

| Model | Parameters | Context | Use Case |
|-------|------------|---------|----------|
| expera_coder_120m | 120M | 2048 | Lightweight, fast inference |
| expera_coder_350m | 350M | 2048 | Balanced code generation |
| expera_coder_1b | 1B | 4096 | Complex reasoning tasks |
| expera_coder_3b | 3B | 4096 | Production, long-context |

## Usage

Each YAML file contains:
- Model architecture details
- Performance characteristics
- Hardware requirements
- Capability flags
- Deployment recommendations
- Default generation parameters

## Loading Models

```python
from src.runtime.model_router import ModelRouter

router = ModelRouter()
model = router.route("code-gen", prompt_length=100)
```

## Quantization Options

All models support multiple quantization levels:
- `fp16` - Full precision
- `q4` - 4-bit (75% smaller)
- `q5` - 5-bit (65% smaller)
- `q8` - 8-bit (50% smaller)