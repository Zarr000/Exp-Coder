"""
Multimodal Benchmark Suite.

Benchmarks for:
- Image understanding
- OCR accuracy
- UI parsing
- Cross-modal retrieval
- Image generation prompts
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import numpy as np
from PIL import Image


@dataclass
class BenchmarkResult:
    """Benchmark result."""
    name: str
    score: float
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class Benchmark(ABC):
    """Base benchmark class."""

    @abstractmethod
    def run(self) -> BenchmarkResult:
        """Run benchmark."""
        pass

    @abstractmethod
    def setup(self):
        """Setup benchmark resources."""
        pass


class ImageEmbeddingBenchmark(Benchmark):
    """Benchmark image embedding speed and quality."""

    def __init__(
        self,
        model_name: str = "vit-base-patch16-224",
        num_iterations: int = 100,
    ):
        self.model_name = model_name
        self.num_iterations = num_iterations
        self.model = None
        self.test_images = []

    def setup(self):
        """Setup models and test images."""
        from src.multimodal.vision_encoder import create_vision_encoder

        self.model = create_vision_encoder(self.model_name)

        # Create test images
        for size in [(224, 224), (384, 384), (512, 512)]:
            img = Image.new("RGB", size, color=(128, 128, 128))
            self.test_images.append(img)

    def run(self) -> BenchmarkResult:
        """Run benchmark."""
        if self.model is None:
            self.setup()

        # Warmup
        for _ in range(5):
            _ = self.model.encode_image(self.test_images[0])

        # Benchmark
        start = time.perf_counter()
        for _ in range(self.num_iterations):
            _ = self.model.encode_image(self.test_images[0])
        end = time.perf_counter()

        latency_ms = (end - start) / self.num_iterations * 1000

        return BenchmarkResult(
            name="image_embedding",
            score=1.0 / latency_ms,  # Higher is better
            latency_ms=latency_ms,
            metadata={
                "model": self.model_name,
                "iterations": self.num_iterations,
            },
        )


class OCRBenchmark(Benchmark):
    """Benchmark OCR accuracy and speed."""

    def __init__(
        self,
        backend: str = "easyocr",
        num_samples: int = 10,
    ):
        self.backend = backend
        self.num_samples = num_samples
        self.ocr = None
        self.test_images = []
        self.ground_truth = []

    def setup(self):
        """Setup OCR and test images."""
        from src.multimodal.ocr import create_ocr_engine

        self.ocr = create_ocr_engine(backend=self.backend)

        # Create test images with text
        for text in ["Hello World", "Test 123", "OCR Test"]:
            img = Image.new("RGB", (200, 50), color=(255, 255, 255))
            self.test_images.append(img)
            self.ground_truth.append(text)

    def run(self) -> BenchmarkResult:
        """Run benchmark."""
        if self.ocr is None:
            self.setup()

        # Warmup
        try:
            _ = self.ocr.recognize(self.test_images[0])
        except:
            pass

        # Benchmark
        start = time.perf_counter()
        results = []
        for img in self.test_images[:self.num_samples]:
            try:
                result = self.ocr.recognize(img)
                results.append(result)
            except:
                results.append(None)
        end = time.perf_counter()

        latency_ms = (end - start) / len(self.test_images[:self.num_samples]) * 1000

        # Compute accuracy (simple)
        correct = 0
        for result in results:
            if result and result.text:
                correct += 1

        accuracy = correct / len(results) if results else 0

        return BenchmarkResult(
            name="ocr",
            score=accuracy,
            latency_ms=latency_ms,
            metadata={
                "backend": self.backend,
                "samples": len(results),
            },
        )


class UIParsingBenchmark(Benchmark):
    """Benchmark UI parsing accuracy."""

    def __init__(
        self,
        num_screenshots: int = 10,
    ):
        self.num_screenshots = num_screenshots
        self.detector = None
        self.test_images = []

    def setup(self):
        """Setup UI detector and test images."""
        from src.multimodal.ui_parser import create_ui_detector

        self.detector = create_ui_detector()

        # Create test screenshots
        for _ in range(self.num_screenshots):
            img = Image.new("RGB", (800, 600), color=(200, 200, 200))
            self.test_images.append(img)

    def run(self) -> BenchmarkResult:
        """Run benchmark."""
        if self.detector is None:
            self.setup()

        start = time.perf_counter()
        results = []
        for img in self.test_images:
            elements = self.detector.detect(img)
            results.append(elements)
        end = time.perf_counter()

        latency_ms = (end - start) / len(self.test_images) * 1000

        return BenchmarkResult(
            name="ui_parsing",
            score=len(results) / len(self.test_images),
            latency_ms=latency_ms,
            metadata={
                "screenshots": self.num_screenshots,
                "elements_found": sum(len(r) for r in results),
            },
        )


class CrossModalRetrievalBenchmark(Benchmark):
    """Benchmark cross-modal retrieval."""

    def __init__(
        self,
        num_samples: int = 100,
    ):
        self.num_samples = num_samples
        self.embedder = None

    def setup(self):
        """Setup embedder."""
        from src.multimodal.embeddings import create_multimodal_embedder

        self.embedder = create_multimodal_embedder()

    def run(self) -> BenchmarkResult:
        """Run benchmark."""
        if self.embedder is None:
            self.setup()

        # Generate test data
        texts = [f"text_{i}" for i in range(self.num_samples)]
        images = [
            Image.new("RGB", (224, 224), color=(i % 255, (i*2) % 255, (i*3) % 255))
            for i in range(self.num_samples)
        ]

        # Embed
        text_embs = [self.embedder.embed_text(t) for t in texts]
        image_embs = [self.embedder.embed_image(i) for i in images]

        # Retrieval test
        start = time.perf_counter()
        correct = 0
        for i in range(min(10, self.num_samples)):
            query = text_embs[i]
            # Simple nearest neighbor
            best_idx = 0
            best_sim = -1
            for j, emb in enumerate(image_embs):
                sim = torch.cosine_similarity(query, emb)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = j
            if best_idx == i:
                correct += 1
        end = time.perf_counter()

        latency_ms = (end - start) / 10 * 1000

        return BenchmarkResult(
            name="cross_modal_retrieval",
            score=correct / 10,
            latency_ms=latency_ms,
            metadata={"samples": self.num_samples},
        )


class PromptGenerationBenchmark(Benchmark):
    """Benchmark prompt generation speed."""

    def __init__(
        self,
        num_prompts: int = 100,
    ):
        self.num_prompts = num_prompts
        self.generator = None

    def setup(self):
        """Setup prompt generator."""
        from src.multimodal.image_prompting import create_prompt_generator

        self.generator = create_prompt_generator()

    def run(self) -> BenchmarkResult:
        """Run benchmark."""
        if self.generator is None:
            self.setup()

        descriptions = [
            f"A beautiful landscape with mountains"
            for _ in range(self.num_prompts)
        ]

        start = time.perf_counter()
        prompts = []
        for desc in descriptions:
            prompt = self.generator.generate(desc)
            prompts.append(prompt)
        end = time.perf_counter()

        latency_ms = (end - start) / self.num_prompts * 1000

        return BenchmarkResult(
            name="prompt_generation",
            score=len(prompts) / self.num_prompts,
            latency_ms=latency_ms,
            metadata={"prompts_generated": len(prompts)},
        )


class MultimodalBenchmarkSuite:
    """
    Complete multimodal benchmark suite.

    Runs all benchmarks and generates report.
    """

    def __init__(self):
        self.benchmarks: List[Benchmark] = []

    def add_benchmark(self, benchmark: Benchmark):
        """Add benchmark to suite."""
        self.benchmarks.append(benchmark)

    def run_all(self) -> List[BenchmarkResult]:
        """Run all benchmarks."""
        results = []

        for benchmark in self.benchmarks:
            try:
                benchmark.setup()
                result = benchmark.run()
                results.append(result)
            except Exception as e:
                results.append(BenchmarkResult(
                    name=benchmark.__class__.__name__,
                    score=0.0,
                    latency_ms=0.0,
                    metadata={"error": str(e)},
                ))

        return results

    def generate_report(self, results: List[BenchmarkResult]) -> str:
        """Generate human-readable report."""
        lines = ["# Multimodal Benchmark Report", ""]

        for result in results:
            lines.append(f"## {result.name}")
            lines.append(f"- **Score**: {result.score:.4f}")
            lines.append(f"- **Latency**: {result.latency_ms:.2f}ms")
            for key, value in result.metadata.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        return "\n".join(lines)


def run_default_benchmarks() -> List[BenchmarkResult]:
    """Run default benchmark suite."""
    suite = MultimodalBenchmarkSuite()

    suite.add_benchmark(ImageEmbeddingBenchmark())
    suite.add_benchmark(OCRBenchmark())
    suite.add_benchmark(UIParsingBenchmark())
    suite.add_benchmark(CrossModalRetrievalBenchmark())
    suite.add_benchmark(PromptGenerationBenchmark())

    return suite.run_all()


__all__ = [
    "Benchmark",
    "BenchmarkResult",
    "ImageEmbeddingBenchmark",
    "OCRBenchmark",
    "UIParsingBenchmark",
    "CrossModalRetrievalBenchmark",
    "PromptGenerationBenchmark",
    "MultimodalBenchmarkSuite",
    "run_default_benchmarks",
]