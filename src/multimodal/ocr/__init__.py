"""
OCR Pipeline for Expera AI.

OCR abstractions supporting:
- EasyOCR
- TrOCR
- Cloud APIs
- Pluggable backends
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Union

import torch
import numpy as np
from PIL import Image


class OCRBackend(Enum):
    """OCR backends."""
    EASYOCR = "easyocr"
    TROCR = "trocr"
    TESSERACT = "tesseract"
    CLOUD = "cloud"


@dataclass
class OCRConfig:
    """OCR configuration."""
    backend: OCRBackend = OCRBackend.EASYOCR
    languages: List[str] = field(default_factory=lambda: ["en"])
    gpu: bool = True
    model_size: str = "medium"  # small, medium, large
    use_angle_cls: bool = True
    download_enabled: bool = True


@dataclass
class OCRResult:
    """OCR result container."""
    text: str
    confidence: float
    bbox: Optional[List[List[int]]] = None
    words: Optional[List[Dict[str, Any]]] = None


class OCREngine(ABC):
    """Base OCR engine."""

    @abstractmethod
    def recognize(
        self,
        image: Union[Image.Image, np.ndarray],
    ) -> OCRResult:
        """
        Recognize text in image.

        Args:
            image: Input image

        Returns:
            OCRResult with text, confidence, bbox
        """
        pass

    @abstractmethod
    def recognize_batch(
        self,
        images: List[Union[Image.Image, np.ndarray]],
    ) -> List[OCRResult]:
        """
        Recognize text in batch of images.

        Args:
            images: List of images

        Returns:
            List of OCRResults
        """
        pass


class EasyOCREngine(OCREngine):
    """EasyOCR implementation."""

    def __init__(self, config: OCRConfig):
        self.config = config
        self._model = None
        self._reader = None

    def _get_reader(self):
        """Lazy load EasyOCR reader."""
        if self._reader is None:
            try:
                import easyocr
                self._reader = easyocr.Reader(
                    self.config.languages,
                    gpu=self.config.gpu,
                    model_storage_directory="models/ocr",
                    download_enabled=self.config.download_enabled,
                )
            except ImportError:
                # Fallback for when EasyOCR not available
                self._reader = None
        return self._reader

    def recognize(
        self,
        image: Union[Image.Image, np.ndarray],
    ) -> OCRResult:
        """Recognize text with EasyOCR."""
        reader = self._get_reader()

        if reader is None:
            # Fallback: return dummy result
            return OCRResult(
                text="[EasyOCR not available]",
                confidence=0.0,
            )

        # Convert image if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Run OCR
        results = reader.recognize(image)

        # Parse results
        text_parts = []
        bboxes = []
        words = []

        for bbox, text, conf in results:
            text_parts.append(text)
            bboxes.append(bbox)
            words.append({
                "text": text,
                "bbox": bbox,
                "confidence": conf,
            })

        return OCRResult(
            text=" ".join(text_parts),
            confidence=np.mean([w["confidence"] for w in words]) if words else 0.0,
            bbox=bboxes,
            words=words,
        )

    def recognize_batch(
        self,
        images: List[Union[Image.Image, np.ndarray]],
    ) -> List[OCRResult]:
        """Recognize batch."""
        return [self.recognize(img) for img in images]


class TrOCREngine(OCREngine):
    """TrOCR (Transformer OCR) implementation."""

    def __init__(self, config: OCRConfig):
        self.config = config
        self._processor = None
        self._model = None
        self._device = "cuda" if config.gpu and torch.cuda.is_available() else "cpu"

    def _get_models(self):
        """Lazy load TrOCR models."""
        if self._model is None:
            try:
                from transformers import (
                    ViTFeatureExtractor,
                    TrOCRForConditionalGeneration,
                    AutoTokenizer,
                )
                # Load processor and model
                self._processor = ViTFeatureExtractor.from_pretrained(
                    "microsoft/trocr-base-handwritten"
                )
                self._model = TrOCRForConditionalGeneration.from_pretrained(
                    "microsoft/tocr-base-handwritten"
                )
                self._model.to(self._device)
                self._tokenizer = AutoTokenizer.from_pretrained(
                    "microsoft/tocr-base-handwritten"
                )
            except ImportError:
                self._model = None
                self._processor = None
        return self._processor, self._model

    def recognize(
        self,
        image: Union[Image.Image, np.ndarray],
    ) -> OCRResult:
        """Recognize text with TrOCR."""
        processor, model = self._get_models()

        if model is None:
            return OCRResult(
                text="[TrOCR not available]",
                confidence=0.0,
            )

        # Convert to RGB if needed
        if isinstance(image, Image.Image):
            image = image.convert("RGB")

        # Prepare input
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self._device)

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)

        # Decode
        text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return OCRResult(
            text=text,
            confidence=0.9,
        )

    def recognize_batch(
        self,
        images: List[Union[Image.Image, np.ndarray]],
    ) -> List[OCRResult]:
        """Recognize batch."""
        processor, model = self._get_models()

        if model is None:
            return [
                OCRResult(text="[TrOCR not available]", confidence=0.0)
                for _ in images
            ]

        # Convert to RGB
        images = [
            img.convert("RGB") if isinstance(img, Image.Image) else img
            for img in images
        ]

        # Batch process
        pixel_values = processor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self._device)

        with torch.no_grad():
            generated_ids = model.generate(pixel_values)

        texts = self._processor.batch_decode(generated_ids, skip_special_tokens=True)

        return [
            OCRResult(text=text, confidence=0.9)
            for text in texts
        ]


class TesseractOCREngine(OCREngine):
    """Tesseract OCR implementation."""

    def __init__(self, config: OCRConfig):
        self.config = config
        self._api = None

    def _get_api(self):
        """Get Tesseract API."""
        if self._api is None:
            try:
                import pytesseract
                self._api = pytesseract
            except ImportError:
                self._api = None
        return self._api

    def recognize(
        self,
        image: Union[Image.Image, np.ndarray],
    ) -> OCRResult:
        """Recognize text with Tesseract."""
        api = self._get_api()

        if api is None:
            return OCRResult(
                text="[Tesseract not available]",
                confidence=0.0,
            )

        # Convert image if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Get custom config
        lang = "+".join(self.config.languages)
        config = f"-l {lang}"

        # Get data with boxes
        try:
            data = api.image_to_data(
                image,
                output_type=api.Output.DICT,
                config=config,
            )
            words = []
            bboxes = []
            text_parts = []

            for i, text in enumerate(data["text"]):
                if text.strip():
                    text_parts.append(text)
                    bbox = [
                        data["left"][i],
                        data["top"][i],
                        data["left"][i] + data["width"][i],
                        data["top"][i] + data["height"][i],
                    ]
                    bboxes.append(bbox)
                    words.append({
                        "text": text,
                        "bbox": bbox,
                        "confidence": data["conf"][i] / 100.0,
                    })

            return OCRResult(
                text=" ".join(text_parts),
                confidence=np.mean([w["confidence"] for w in words]) if words else 0.0,
                bbox=bboxes,
                words=words,
            )
        except Exception as e:
            return OCRResult(
                text=f"[OCR error: {e}]",
                confidence=0.0,
            )

    def recognize_batch(
        self,
        images: List[Union[Image.Image, np.ndarray]],
    ) -> List[OCRResult]:
        """Recognize batch."""
        return [self.recognize(img) for img in images]


class CloudOCREngine(OCREngine):
    """Cloud API OCR (AWS, GCP, Azure, etc.)."""

    def __init__(self, config: OCRConfig):
        self.config = config
        self._provider = "aws"  # Default

    def recognize(
        self,
        image: Union[Image.Image, np.ndarray],
    ) -> OCRResult:
        """Cloud OCR (would integrate with cloud APIs in practice)."""
        return OCRResult(
            text="[Cloud OCR - configure API keys]",
            confidence=0.0,
        )

    def recognize_batch(
        self,
        images: List[Union[Image.Image, np.ndarray]],
    ) -> List[OCRResult]:
        """Cloud OCR batch."""
        return [self.recognize(img) for img in images]


def create_ocr_engine(
    backend: OCRBackend = OCRBackend.EASYOCR,
    languages: List[str] = ["en"],
    gpu: bool = True,
    **kwargs,
) -> OCREngine:
    """
    Create an OCR engine.

   Args:
        backend: OCR backend type
        languages: Languages to support
        gpu: Use GPU if available
        **kwargs: Additional arguments

    Returns:
        OCREngine instance
    """
    config = OCRConfig(
        backend=backend,
        languages=languages,
        gpu=gpu,
        **kwargs,
    )

    if backend == OCRBackend.EASYOCR:
        return EasyOCREngine(config)
    elif backend == OCRBackend.TROCR:
        return TrOCREngine(config)
    elif backend == OCRBackend.TESSERACT:
        return TesseractOCREngine(config)
    else:
        return CloudOCREngine(config)


__all__ = [
    "OCREngine",
    "OCRConfig",
    "OCRResult",
    "OCRBackend",
    "EasyOCREngine",
    "TrOCREngine",
    "TesseractOCREngine",
    "CloudOCREngine",
    "create_ocr_engine",
]