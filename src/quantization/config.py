"""
Quantization Configuration.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List


class QuantizationType(Enum):
    """Quantization types."""
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    QLORA = "qlora"


@dataclass
class QuantizationConfig:
    """
    Quantization configuration.

    Usage:
        # FP16
        config = QuantizationConfig(type=QuantizationType.FP16)

        # INT8
        config = QuantizationConfig(type=QuantizationType.INT8)

        # 4-bit QLoRA (for 6GB GPUs)
        config = QuantizationConfig(
            type=QuantizationType.QLORA,
            bits=4,
            group_size=128,
        )
    """
    type: QuantizationType = QuantizationType.FP16
    bits: int = 16  # For INT4/INT8: 4 or 8
    group_size: int = 128  # Group size for quantization
    zero_point: bool = True  # Use zero point
    desc_act: bool = False  # Quantize activations

    # LoRA-specific
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None  # type: ignore

    # Conversion
    calibration_method: str = "minmax"  # minmax, histogram, entropy
    calibration_samples: int = 512

    def __post_init__(self):
        """Validate config."""
        if self.type in (QuantizationType.INT4, QuantizationType.INT8):
            if self.bits not in (4, 8):
                raise ValueError(f"Invalid bits: {self.bits}")


__all__ = [
    "QuantizationConfig",
    "QuantizationType",
]