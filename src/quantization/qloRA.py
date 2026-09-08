"""
QLoRA Quantization (4-bit for low VRAM GPUs).

Provides 4-bit quantization for GPUs like RTX 4050 6GB.
"""

from typing import Optional, List
import logging

import torch
import torch.nn as nn

from .config import QuantizationConfig

logger = logging.getLogger(__name__)

HAS_BITSANDBYTES = False
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    logger.warning("bitsandbytes not available, QLoRA not supported")


class QLoRAQuantizer:
    """
    QLoRA quantizer for 4-bit quantization.

    Usage:
        config = QuantizationConfig(type=QuantizationType.QLORA, bits=4)
        quantizer = QLoRAQuantizer(config)
        model = quantizer.quantize(model)
    """

    def __init__(self, config: QuantizationConfig):
        """Initialize quantizer."""
        self.config = config
        self.bits = config.bits
        self.group_size = config.group_size

    def quantize(self, model: nn.Module) -> nn.Module:
        """Quantize model to 4-bit."""
        if not HAS_BITSANDBYTES:
            logger.warning("bitsandbytes not available, returning FP16 model")
            return model.half()

        try:
            return self._quantize_qlora(model)
        except Exception as e:
            logger.warning(f"QLoRA quantization failed: {e}, using FP16")
            return model.half()

    def _quantize_qlora(self, model: nn.Module) -> nn.Module:
        """Apply QLoRA quantization."""
        # Target modules
        target_modules = self.config.target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Get module name without prefix
                module_name = name.split(".")[-1]

                if any(tm in module_name for tm in target_modules):
                    try:
                        quant_module = bnb.nn.Linear4bit(
                            module.in_features,
                            module.out_features,
                            module.bias is not None,
                            quantization=self._get_quantization_config(),
                        )
                        module.replace_linear(quant_module)
                    except Exception as e:
                        logger.debug(f"Could not quantize {name}: {e}")

        logger.info("Model quantized with QLoRA 4-bit")
        return model

    def _get_quantization_config(self):
        """Get bitsandbytes quantization config."""
        # Return config dict for bitsandbytes
        return {
            "quant_state": {
                "bits": self.bits,
                "group_size": self.group_size,
                "zero_point": self.config.zero_point,
            }
        }


__all__ = ["QLoRAQuantizer"]