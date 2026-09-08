"""
INT8 Quantization.

Provides INT8 quantization using bitsandbytes or fallback.
"""

from typing import Optional
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

HAS_BITSANDBYTES = False
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    logger.warning("bitsandbytes not available, using PyTorch INT8")


class Int8Quantizer:
    """
    INT8 quantizer.

    Usage:
        quantizer = Intizer()
        model = quantizer.quantize(model)
    """

    def __init__(
        self,
        calibration_method: str = "minmax",
        calibration_samples: int = 512,
    ):
        """Initialize quantizer."""
        self.calibration_method = calibration_method
        self.calibration_samples = calibration_samples

    def quantize(self, model: nn.Module) -> nn.Module:
        """Quantize model to INT8."""
        if HAS_BITSANDBYTES:
            return self._quantize_bitsandbytes(model)
        else:
            return self._quantize_pytorch(model)

    def _quantize_bitsandbytes(self, model: nn.Module) -> nn.Module:
        """Quantize using bitsandbytes."""
        try:
            # Find linear layers
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    # Replace with quantized version
                    quant_module = bnb.nn.Linear8bitLt(
                        module.in_features,
                        module.out_features,
                        module.bias is not None,
                    )
                    module.replace_linear(quant_module)

            logger.info("Quantized model with bitsandbytes INT8")
            return model

        except Exception as e:
            logger.warning(f"bitsandbytes quantization failed: {e}")
            return self._quantize_pytorch(model)

    def _quantize_pytorch(self, model: nn.Module) -> nn.Module:
        """Quantize using PyTorch dynamic quantization."""
        # Dynamic quantization
        model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8,
        )

        logger.info("Quantized model with PyTorch INT8")
        return model


__all__ = ["Int8Quantizer"]