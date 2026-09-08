"""
Tests for Expera AI Multimodal Foundation.

Tests:
- Vision Encoder
- Image Understanding
- OCR
- UI Parser
- Image Prompting
- Image Editing
- Cross Attention
- Embeddings
- Benchmarks
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
import numpy as np
from PIL import Image

from src.multimodal.vision_encoder import (
    VisionEncoder,
    VisionEncoderConfig,
    CLIPEncoder,
    ViTEncoder,
    create_vision_encoder,
    EncoderBackend,
)
from src.multimodal.image_understanding import (
    ImageUnderstandingPipeline,
    ImageUnderstandingConfig,
    ImageDescription,
    create_image_embedder,
    UnderstandingBackend,
)
from src.multimodal.ocr import (
    OCREngine,
    OCRConfig,
    OCRResult,
    OCRBackend,
    EasyOCREngine,
    TrOCREngine,
    TesseractOCREngine,
    create_ocr_engine,
)
from src.multimodal.ui_parser import (
    UIElementDetector,
    UIElement,
    UILayout,
    UIParseConfig,
    UIParseBackend,
    LayoutDetector,
    AccessibilityDetector,
    create_ui_detector,
)
from src.multimodal.image_prompting import (
    ImagePromptGenerator,
    ImagePromptConfig,
    GeneratedPrompt,
    PromptBackend,
    StableDiffusionPromptGenerator,
    FluxPromptGenerator,
    PromptOptimizer,
    create_prompt_generator,
)
from src.multimodal.image_editing import (
    ImageEditingParser,
    EditInstruction,
    EditResult,
    EditBackend,
    ImageEditConfig,
    InstructionParser,
    InpaintEditor,
    create_editing_parser,
)
from src.multimodal.cross_attention import (
    CrossAttentionModule,
    CrossAttentionConfig,
    MultimodalTransformer,
    VisionLanguageModel,
    create_multimodal_model,
)
from src.multimodal.embeddings import (
    MultimodalEmbedder,
    MultimodalEmbedderConfig,
    UnifiedEmbeddings,
    ContrastiveLoss,
    create_multimodal_embedder,
)
from src.multimodal.benchmarks import (
    Benchmark,
    BenchmarkResult,
    ImageEmbeddingBenchmark,
    OCRBenchmark,
    MultimodalBenchmarkSuite,
)


class TestVisionEncoder:
    """Test suite for Vision Encoder."""

    @pytest.fixture
    def config(self):
        return VisionEncoderConfig(
            model_name="vit-base-patch16-224",
            embed_dim=768,
            depth=4,
            num_heads=4,
            device="cpu",
        )

    def test_initialization(self, config):
        """Test encoder initializes."""
        encoder = ViTEncoder(config)
        assert encoder.config.embed_dim == 768
        assert encoder.config.depth == 4

    def test_clip_encoder(self, config):
        """Test CLIP encoder."""
        encoder = CLIPEncoder(config)
        assert encoder.config.embed_dim == 768

    def test_preprocess(self, config):
        """Test image preprocessing."""
        encoder = ViTEncoder(config)
        img = Image.new("RGB", (224, 224))
        batch = encoder.preprocess(img)
        assert batch.shape == (1, 3, 224, 224)

    def test_encode_image(self, config):
        """Test image encoding."""
        encoder = ViTEncoder(config)
        img = Image.new("RGB", (224, 224))
        embedding = encoder.encode_image(img)
        assert embedding.shape[0] == 1  # batch
        assert embedding.shape[-1] == 768  # embed_dim

    def test_encode_patches(self, config):
        """Test patch encoding."""
        encoder = ViTEncoder(config)
        img = Image.new("RGB", (224, 224))
        patches = encoder.encode_patches(img)
        assert patches.shape[1] == (224 // 16) ** 2  # num patches
        assert patches.shape[-1] == 768


class TestImageUnderstanding:
    """Test suite for Image Understanding."""

    @pytest.fixture
    def config(self):
        return ImageUnderstandingConfig(
            vision_encoder="vit-base-patch16-224",
            embed_dim=768,
            device="cpu",
        )

    def test_initialization(self, config):
        """Test pipeline initializes."""
        pipeline = ImageUnderstandingPipeline(config)
        assert pipeline.config.embed_dim == 768

    def test_embed_images(self, config):
        """Test image embedding."""
        pipeline = ImageUnderstandingPipeline(config)
        img = Image.new("RGB", (224, 224))
        embedding = pipeline.embed_images(img)
        assert embedding.shape == (1, 768)

    def test_embed_patches(self, config):
        """Test patch embedding."""
        pipeline = ImageUnderstandingPipeline(config)
        img = Image.new("RGB", (224, 224))
        patches = pipeline.embed_patches(img)
        assert patches.shape[-1] == 768

    def test_describe_image(self, config):
        """Test image description."""
        pipeline = ImageUnderstandingPipeline(config)
        img = Image.new("RGB", (224, 224))
        description = pipeline.describe_image(img)
        assert isinstance(description, ImageDescription)
        assert description.text is not None


class TestOCR:
    """Test suite for OCR."""

    @pytest.fixture
    def config(self):
        return OCRConfig(
            backend=OCRBackend.EASYOCR,
            languages=["en"],
            gpu=False,
        )

    def test_ocr_config(self, config):
        """Test OCR config."""
        assert config.backend == OCRBackend.EASYOCR
        assert config.languages == ["en"]

    def test_ocr_result(self):
        """Test OCR result."""
        result = OCRResult(
            text="test",
            confidence=0.9,
        )
        assert result.text == "test"
        assert result.confidence == 0.9

    def test_easyocr_engine(self, config):
        """Test EasyOCR engine creation."""
        engine = EasyOCREngine(config)
        assert engine is not None

    def test_tesseract_engine(self):
        """Test Tesseract engine."""
        config = OCRConfig(backend=OCRBackend.TESSERACT)
        engine = TesseractOCREngine(config)
        assert engine is not None

    def test_create_ocr_engine(self):
        """Test OCR engine factory."""
        engine = create_ocr_engine(backend=OCRBackend.EASYOCR)
        assert isinstance(engine, EasyOCREngine)


class TestUIParser:
    """Test suite for UI Parser."""

    @pytest.fixture
    def config(self):
        return UIParseConfig(
            backend=UIParseBackend.DETR,
            device="cpu",
        )

    def test_ui_element(self):
        """Test UI element."""
        element = UIElement(
            element_type="button",
            text="Click me",
            bbox=[0, 0, 100, 50],
            confidence=0.9,
        )
        assert element.element_type == "button"
        assert element.text == "Click me"

    def test_layout(self):
        """Test UI layout."""
        elements = [
            UIElement(element_type="button", text="OK"),
            UIElement(element_type="input", text="Name"),
        ]
        layout = UILayout(
            elements=elements,
            width=800,
            height=600,
        )
        assert len(layout.elements) == 2
        assert layout.width == 800

    def test_layout_detector(self, config):
        """Test layout detector."""
        detector = LayoutDetector(config)
        img = Image.new("RGB", (800, 600))
        elements = detector.detect(img)
        assert isinstance(elements, list)

    def test_accessibility_detector(self, config):
        """Test accessibility detector."""
        detector = AccessibilityDetector(config)
        img = Image.new("RGB", (800, 600))
        elements = detector.detect(img)
        assert len(elements) > 0


class TestImagePrompting:
    """Test suite for Image Prompting."""

    @pytest.fixture
    def config(self):
        return ImagePromptConfig(
            backend=PromptBackend.STABLE_DIFFUSION,
            quality="high",
        )

    def test_generated_prompt(self):
        """Test generated prompt."""
        prompt = GeneratedPrompt(
            prompt="test prompt",
            negative_prompt="bad",
            parameters={"steps": 25},
        )
        assert prompt.prompt == "test prompt"
        assert prompt.parameters["steps"] == 25

    def test_sd_prompt_generator(self, config):
        """Test SD prompt generator."""
        generator = StableDiffusionPromptGenerator(config)
        result = generator.generate("a cat")
        assert isinstance(result, GeneratedPrompt)
        assert "cat" in result.prompt.lower()

    def test_flux_prompt_generator(self):
        """Test Flux prompt generator."""
        config = ImagePromptConfig(backend=PromptBackend.FLUX)
        generator = FluxPromptGenerator(config)
        result = generator.generate("a dog")
        assert isinstance(result, GeneratedPrompt)

    def test_prompt_optimizer(self):
        """Test prompt optimizer."""
        prompt = PromptOptimizer.enhance("a cat")
        assert "high quality" in prompt.lower()

    def test_create_prompt_generator(self):
        """Test prompt generator factory."""
        generator = create_prompt_generator(backend=PromptBackend.STABLE_DIFFUSION)
        assert isinstance(generator, StableDiffusionPromptGenerator)


class TestImageEditing:
    """Test suite for Image Editing."""

    @pytest.fixture
    def parser(self):
        config = ImageEditConfig(backend=EditBackend.INPAINT)
        return InstructionParser(config)

    def test_edit_instruction(self):
        """Test edit instruction."""
        instruction = EditInstruction(
            operation="remove",
            target="person",
        )
        assert instruction.operation == "remove"

    def test_parse_instruction(self, parser):
        """Test instruction parsing."""
        result = parser.parse("remove the person in the center")
        assert result.operation == "remove"
        assert result.region is not None

    def test_inpaint_editor(self):
        """Test inpaint editor."""
        config = ImageEditConfig(backend=EditBackend.INPAINT)
        editor = InpaintEditor(config)
        assert editor is not None


class TestCrossAttention:
    """Test suite for Cross Attention."""

    @pytest.fixture
    def config(self):
        return CrossAttentionConfig(
            embed_dim=256,
            num_heads=4,
            dropout=0.1,
        )

    def test_cross_attention_config(self, config):
        """Test config."""
        assert config.embed_dim == 256
        assert config.num_heads == 4

    def test_cross_attention_module(self, config):
        """Test cross attention."""
        module = CrossAttentionModule(config)
        q = torch.randn(2, 10, 256)
        k = torch.randn(2, 20, 256)
        v = torch.randn(2, 20, 256)
        output = module(q, k, v)
        assert output.shape == (2, 10, 256)

    def test_multimodal_transformer(self):
        """Test multimodal transformer."""
        transformer = MultimodalTransformer(
            embed_dim=256,
            num_heads=4,
            num_layers=2,
            text_dim=256,
            vision_dim=256,
        )
        text = torch.randn(2, 10, 256)
        vision = torch.randn(2, 20, 256)
        text_out, vision_out = transformer(text, vision)
        assert text_out.shape[0] == 2

    def test_vision_language_model(self):
        """Test VLM."""
        model = VisionLanguageModel(
            vision_embed_dim=768,
            text_embed_dim=768,
            hidden_dim=256,
            num_heads=4,
            vocab_size=1000,
        )
        vision = torch.randn(2, 10, 768)
        text_ids = torch.randint(0, 1000, (2, 20))
        loss, metadata = model(vision_features=vision, text_ids=text_ids, labels=text_ids)
        assert loss is not None or "logits" in metadata


class TestEmbeddings:
    """Test suite for Multimodal Embeddings."""

    @pytest.fixture
    def config(self):
        return MultimodalEmbedderConfig(
            text_dim=512,
            image_dim=768,
            output_dim=256,
        )

    def test_multimodal_embedder_config(self, config):
        """Test config."""
        assert config.output_dim == 256

    def test_multimodal_embedder(self, config):
        """Test embedder."""
        embedder = MultimodalEmbedder(config)
        text_emb = embedder.embed_text("hello")
        assert text_emb.shape[-1] == 256

    def test_unified_embeddings(self, config):
        """Test unified embeddings."""
        unified = UnifiedEmbeddings(config)
        stats = unified.get_index_stats()
        assert stats["text_count"] == 0

    def test_contrastive_loss(self):
        """Test contrastive loss."""
        loss_fn = ContrastiveLoss()
        emb1 = torch.randn(4, 128)
        emb2 = torch.randn(4, 128)
        loss = loss_fn(emb1, emb2)
        assert loss.item() >= 0


class TestBenchmarks:
    """Test suite for Multimodal Benchmarks."""

    def test_benchmark_result(self):
        """Test benchmark result."""
        result = BenchmarkResult(
            name="test",
            score=0.9,
            latency_ms=10.0,
        )
        assert result.name == "test"
        assert result.score == 0.9

    def test_image_embedding_benchmark(self):
        """Test image embedding benchmark."""
        benchmark = ImageEmbeddingBenchmark(
            model_name="vit-base-patch16-224",
            num_iterations=10,
        )
        benchmark.setup()
        result = benchmark.run()
        assert result.name == "image_embedding"

    def test_benchmark_suite(self):
        """Test benchmark suite."""
        suite = MultimodalBenchmarkSuite()
        suite.add_benchmark(ImageEmbeddingBenchmark(num_iterations=5))
        results = suite.run_all()
        assert len(results) > 0


class TestIntegration:
    """Integration tests combining multiple modules."""

    def test_vision_to_text_pipeline(self):
        """Test vision to text pipeline."""
        # Vision encoder -> embedder
        encoder_config = VisionEncoderConfig(device="cpu", embed_dim=768, depth=2, num_heads=4)
        encoder = ViTEncoder(encoder_config)
        img = Image.new("RGB", (224, 224))
        embedding = encoder.encode_image(img)

        # Should produce valid embedding
        assert embedding.shape[-1] == 768

    def test_prompt_to_image_pipeline(self):
        """Test prompt generation pipeline."""
        config = ImagePromptConfig(backend=PromptBackend.STABLE_DIFFUSION, quality="high")
        generator = StableDiffusionPromptGenerator(config)
        prompt = generator.generate("a sunset over mountains")
        assert prompt.prompt is not None
        assert prompt.parameters["steps"] > 0

    def test_full_multimodal_stack(self):
        """Test full multimodal stack."""
        # 1. Vision encoder
        encoder = ViTEncoder(VisionEncoderConfig(device="cpu", embed_dim=256, depth=2, num_heads=4))

        # 2. Image embedder
        embedder = MultimodalEmbedder(MultimodalEmbedderConfig(output_dim=256))

        # 3. Caption
        img = Image.new("RGB", (224, 224))
        vision_emb = encoder.encode_image(img)
        text_emb = embedder.embed_text("a test image")

        # 4. Cross attention
        cross_attn = CrossAttentionModule(CrossAttentionConfig(embed_dim=256, num_heads=4))
        output = cross_attn(text_emb, vision_emb, vision_emb)

        assert output.shape == vision_emb.shape


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])