"""
Quantization Manager.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging

import torch
import torch.nn as nn

from .config import QuantizationConfig, QuantizationType

logger = logging.getLogger(__name__)


class QuantizationManager:
    """
    Manages model quantization.

    Usage:
        manager = QuantizationManager()
        model = manager.quantize(model, config)
    """

    def __init__(self, device: str = "cuda"):
        """Initialize manager."""
        self.device = device

    def quantize(
        self,
        model: nn.Module,
        config: QuantizationConfig,
    ) -> nn.Module:
        """
        Quantize model.

        Args:
            model: Model to quantize
            config: Quantization config

        Returns:
            Quantized model
        """
        if config.type == QuantizationType.FP16:
            return self._quantize_fp16(model)
        elif config.type == QuantizationType.BF16:
            return self._quantize_bf16(model)
        elif config.type == QuantizationType.INT8:
            return self._quantize_int8(model, config)
        elif config.type == QuantizationType.QLORA:
            return self._quantize_qlora(model, config)
        else:
            raise ValueError(f"Unknown quantization type: {config.type}")

    def _quantize_fp16(self, model: nn.Module) -> nn.Module:
        """Convert to FP16."""
        return model.half()

    def _quantize_bf16(self, model: nn.Module) -> nn.Module:
        """Convert to BF16."""
        return model.to(dtype=torch.bfloat16)

    def _quantize_int8(
        self,
        model: nn.Module,
        config: QuantizationConfig,
    ) -> nn.Module:
        """Quantize to INT8."""
        try:
            from .qint8 import Int8Quantizer
            quantizer = Int8Quantizer(
                calibration_method=config.calibration_method,
                calibration_samples=config.calibration_samples,
            )
            return quantizer.quantize(model)
        except ImportError:
            logger.warning("bitsandbytes not available, using FP16")
            return model.half()

    def _quantize_qlora(
        self,
        model: nn.Module,
        config: QuantizationConfig,
    ) -> nn.Module:
        """Quantize with QLoRA."""
        try:
            from .qloRA import QLoRAQuantizer
            quantizer = QLoRAQuantizer(config)
            return quantizer.quantize(model)
        except ImportError:
            logger.warning("bitsandbytes not available, using INT8")
            return self._quantize_int8(model, config)

    def dequantize(self, model: nn.Module) -> nn.Module:
        """Dequantize model."""
        try:
            model = model.float()
        except:
            pass
        return model

    def get_model_size(self, model: nn.Module) -> Dict[str, float]:
        """Get model size in MB."""
        size_mb = 0
        for param in model.parameters():
            numel = param.numel()
            if param.dtype == torch.float16 or param.dtype == torch.bfloat16:
                bytes_per_elem = 2
            elif param.dtype == torch.float32:
                bytes_per_elem = 4
            elif param.dtype == torch.int8:
                bytes_per_elem = 1
            else:
                bytes_per_elem = 4
            size_mb += numel * bytes_per_elem

        return {
            "mb": size_mb / (1024 * 1024),
            "gb": size_mb / (1024 * 1024 * 1024),
        }


__all__ = ["QuantizationManager"]