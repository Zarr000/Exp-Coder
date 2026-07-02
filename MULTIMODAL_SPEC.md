# Phase 5: Multimodal Foundation

## Overview

Transform Expera AI into a multimodal coding and creative assistant capable of processing and generating images, text, and code together. The system provides vision encoders, OCR, UI parsing, image prompt generation, image editing, and unified multimodal embeddings.

## Goals

1. Support vision-language models with CLIP-compatible vision encoders
2. Enable OCR with multiple backends (EasyOCR, TrOCR, Tesseract, cloud APIs)
3. Parse UI elements from screenshots for automation
4. Generate quality prompts for image generation models
5. Parse and execute image editing instructions
6. Provide unified cross-modal embeddings
7. Run benchmarks for all multimodal components

## Architecture

### Module Structure

```
src/multimodal/
├── __init__.py                       # Public exports
├── vision_encoder/                  # Vision encoder abstraction
│   ├── VisionEncoderConfig
│   ├── VisionEncoder (ABC)
│   ├── CLIPEncoder
│   ├── ViTEncoder
│   └── create_vision_encoder()
├── image_understanding/           # Image embedding pipeline
│   ├── ImageUnderstandingConfig
│   ├── ImageUnderstandingPipeline
│   ├── ImageDescription
│   └── create_image_embedder()
├── ocr/                           # OCR with multiple backends
│   ├── OCRBackend (Enum)
│   , OCRConfig
│   , OCRResult
│   , OCREngine (ABC)
│   , EasyOCREngine
│   , TrOCREngine
│   , TesseractOCREngine
│   , CloudOCREngine
│   └── create_ocr_engine()
├── ui_parser/                      # UI detection/parsing
│   , UIParseBackend (Enum)
│   , UIParseConfig
│   , UIElement
│   , UILayout
│   , UIElementDetector (ABC)
│   , LayoutDetector
│   , AccessibilityDetector
│   , OCRBasedDetector
│   , ScreenshotAnalyzer
│   └── create_ui_detector()
├── image_prompting/                 # Image prompt generation
│   , PromptBackend (Enum)
│   , ImagePromptConfig
│   , GeneratedPrompt
│   , ImagePromptGenerator (ABC)
│   , StableDiffusionPromptGenerator
│   , FluxPromptGenerator
│   , MidjourneyPromptGenerator
│   , DALLEPromptGenerator
│   , PromptOptimizer
│   and create_prompt_generator()
├── image_editing/                  # Image editing parser
│   , EditBackend (Enum)
│   , ImageEditConfig
│   , EditInstruction
│   , EditResult
│   , ImageEditingParser (ABC)
│   , InstructionParser
│   , InpaintEditor
│   , OutpaintEditor
│   , EditCommandBuilder
│   and create_editing_parser()
├── cross_attention/               # Cross-modal attention
│   , CrossAttentionConfig
│   , CrossAttentionModule
│   , MultimodalTransformer
│   , VisionLanguageModel
│   , SelfAttention
│   and create_multimodal_model()
├── embeddings/                 # Unified embeddings
│   , MultimodalEmbedderConfig
│   , MultimodalEmbedder
│   , UnifiedEmbeddings
│   , ContrastiveLoss
│   and create_multimodal_embedder()
└── benchmarks/                 # Benchmark suite
    , Benchmark (ABC)
    , BenchmarkResult
    , ImageEmbeddingBenchmark
    , OCRBenchmark
    , UIParsingBenchmark
    , CrossModalRetrievalBenchmark
    , PromptGenerationBenchmark
    and MultimodalBenchmarkSuite
```

## Public APIs

### vision_encoder/__init__.py

```python
class VisionEncoderConfig:
    """Vision encoder configuration."""
    model_name: str = "vit-base-patch16-224"
    image_size: int = 224
    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    backend: EncoderBackend = EncoderBackend.TORCH
    device: str = "cuda" or "cpu"
    dtype: torch.dtype = torch.float32
    use_pretrained: bool = True

class VisionEncoder(ABC, nn.Module):
    """Abstract vision encoder base class."""

    def __init__(self, config: VisionEncoderConfig)

    @abstractmethod
    def encode_image(self, images: Union[Image.Image, List[Image.Image]]) -> Tensor:
        """Encode images to embeddings (batch, embed_dim)."""

    @abstractmethod
    def encode_patches(self, images: ...) -> Tensor:
        """Encode images as patches (batch, num_patches, embed_dim)."""

    def preprocess(self, images: Image.Image) -> Tensor:
        """Preprocess images to tensor (batch, 3, height, width)."""

class CLIPEncoder(VisionEncoder):
    """CLIP vision encoder implementation."""

    def __init__(self, config: VisionEncoderConfig)

    def encode_image(self, images: ...) -> Tensor:
        """Encode to CLIP embeddings (batch, embed_dim)."""

    def encode_patches(self, images: ...) -> Tensor:
        """Encode to patch embeddings."""

class ViTEncoder(VisionEncoder):
    """Vision Transformer encoder."""

    def __init__(self, config: VisionEncoderConfig)

    def encode_image(self, images: ...) -> Tensor:
        """Encode to ViT embeddings."""

    def encode_patches(self, images: ...) -> Tensor:
        """Encode to patch embeddings."""

def create_vision_encoder(
    backend: EncoderBackend = EncoderBackend.TORCH,
    device: Optional[str] = None,
    **kwargs,
) -> VisionEncoder:
    """Create vision encoder instance."""
```

### image_understanding/__init__.py

```python
class ImageUnderstandingConfig:
    """Image understanding configuration."""
    vision_encoder: str = "vit-base-patch16-224"
    embed_dim: int = 768
    use_larger_encoder: bool = False
    backend: UnderstandingBackend = UnderstandingBackend.LOCAL
    device: str = "cuda" or "cpu"
    max_resolution: int = 1024

class ImageDescription:
    """Image description container."""
    text: str
    confidence: float
    tags: List[str]
    attributes: Dict[str, Any]

class ImageUnderstandingPipeline:
    """Image embedding and description pipeline."""

    def __init__(self, config: ImageUnderstandingConfig)

    def embed_images(self, images: Union[Image.Image, List[Image.Image]]) -> Tensor:
        """Get image embeddings (batch, embed_dim)."""

    def embed_patches(self, images: ...) -> Tensor:
        """Get patch embeddings."""

    def describe_image(self, images: Image.Image) -> ImageDescription:
        """Get image description."""

def create_image_embedder(
    backend: UnderstandingBackend = UnderstandingBackend.LOCAL,
    device: Optional[str] = None,
    **kwargs,
) -> ImageUnderstandingPipeline:
    """Create image embedder."""
```

### ocr/__init__.py

```python
class OCRBackend(Enum):
    """OCR backends."""
    EASYOCR = "easyocr"
    TROCR = "trocr"
    TESSERACT = "tesseract"
    CLOUD = "cloud"

class OCRConfig:
    """OCR configuration."""
    backend: OCRBackend = OCRBackend.EASYOCR
    languages: List[str] = ["en"]
    gpu: bool = True
    model_size: str = "medium"
    use_angle_cls: bool = True
    download_enabled: bool = True

class OCRResult:
    """OCR result container."""
    text: str
    confidence: float
    bbox: Optional[List[List[int]]] = None
    words: Optional[List[Dict[str, Any]]] = None

class OCREngine(ABC):
    """OCR engine base class."""

    @abstractmethod
    def recognize(self, image: Union[Image.Image, np.ndarray]) -> OCRResult:
        """Recognize text in image."""

    @abstractmethod
    def recognize_batch(self, images: List[...]) -> List[OCRResult]:
        """Batch recognition."""

class EasyOCREngine(OCREngine):
    """EasyOCR implementation."""

    def __init__(self, config: OCRConfig)
    def recognize(self, image: ...) -> OCRResult
    def recognize_batch(self, images: ...) -> List[OCRResult]

class TrOCREngine(OCREngine):
    """TrOCR (Transformer OCR) implementation."""

    def __init__(self, config: OCRConfig)
    def recognize(self, image: ...) -> OCRResult
    def recognize_batch(self, images: ...) -> List[OCRResult]

class TesseractOCREngine(OCREngine):
    """Tesseract OCR implementation."""

    def __init__(self, config: OCRConfig)
    def recognize(self, image: ...) -> OCRResult
    def recognize_batch(self, images: ...) -> List[OCRResult]

class CloudOCREngine(OCREngine):
    """Cloud API OCR (AWS, GCP, Azure)."""

    def __init__(self, config: OCRConfig)
    def recognize(self, image: ...) -> OCRResult
    def recognize_batch(self, images: ...) -> List[OCRResult]

def create_ocr_engine(
    backend: OCRBackend = OCRBackend.EASYOCR,
    languages: List[str] = ["en"],
    gpu: bool = True,
    **kwargs,
) -> OCREngine:
    """Create OCR engine."""
```

### ui_parser/__init__.py

```python
class UIParseBackend(Enum):
    """UI parsing backends."""
    DETR = "detr"          # Detection Transformer
    OCR = "ocr"            # OCR-based
    ACCESSIBILITY = "accessibility"  # Accessibility tree
    LAYOUT = "layout"      # Layout modeling

class UIParseConfig:
    """UI parsing configuration."""
    backend: UIParseBackend = UIParseBackend.DETR
    device: str = "cuda" or "cpu"
    confidence_threshold: float = 0.5
    max_elements: int = 100

@dataclass
class UIElement:
    """UI element representation."""
    element_type: str  # button, input, text, image, etc.
    text: Optional[str] = None
    bbox: List[float] = [0, 0, 0, 0]  # x1, y1, x2, y2
    confidence: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    children: List["UIElement"]] = field(default_factory=list)
    parent: Optional["UIElement"]] = None

@dataclass
class UILayout:
    """Complete UI layout."""
    elements: List[UIElement] = field(default_factory=list)
    width: int = 0
    height: int = 0
    device: str = "cpu"
    metadata: Dict[str, Any] = field(default_factory=dict)

class UIElementDetector(ABC):
    """Base UI element detector."""

    @abstractmethod
    def detect(self, image: Image.Image) -> List[UIElement]:
        """Detect UI elements in screenshot."""

class LayoutDetector(UIElementDetector):
    """Layout-based UI element detection."""

    def __init__(self, config: UIParseConfig)
    def detect(self, image: Image.Image) -> List[UIElement]

class AccessibilityDetector(UIElementDetector):
    """Accessibility tree-based UI parsing."""

    def __init__(self, config: UIParseConfig)
    def detect(self, image: Image.Image) -> List[UIElement]
    def parse_accessibility_tree(self) -> UILayout

class OCRBasedDetector(UIElementDetector):
    """OCR-based UI element detection."""

    def __init__(self, config: UIParseConfig)
    def detect(self, image: Image.Image) -> List[UIElement]

class ScreenshotAnalyzer:
    """Complete screenshot analysis pipeline."""

    def __init__(self, config: UIParseConfig)

    def analyze(self, image: Image.Image) -> UILayout:
        """Analyze screenshot."""

    def extract_interaction_points(self, layout: UILayout) -> Dict[str, List[Dict[str, Any]]]:
        """Extract clickable/tappable elements."""

def create_ui_detector(
    backend: UIParseBackend = UIParseBackend.DETR,
    device: Optional[str] = None,
    **kwargs,
) -> UIElementDetector:
    """Create UI element detector."""
```

### image_prompting/__init__.py

```python
class PromptBackend(Enum):
    """Image prompt backends."""
    STABLE_DIFFUSION = "stable_diffusion"
    FLUX = "flux"
    MIDJOURNEY = "midjourney"
    DALL_E = "dall_e"

class ImagePromptConfig:
    """Image prompt configuration."""
    backend: PromptBackend = PromptBackend.STABLE_DIFFUSION
    quality: str = "high"  # low, medium, high, ultra
    style: Optional[str] = None

@dataclass
class GeneratedPrompt:
    """Generated prompt container."""
    prompt: str
    negative_prompt: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

class ImagePromptGenerator(ABC):
    """Image prompt generator base class."""

    @abstractmethod
    def generate(self, description: str) -> GeneratedPrompt:
        """Generate prompt from description."""

class StableDiffusionPromptGenerator(ImagePromptGenerator):
    """Stable Diffusion prompt generator."""

    def __init__(self, config: ImagePromptConfig)
    def generate(self, description: str) -> GeneratedPrompt

class FluxPromptGenerator(ImagePromptGenerator):
    """Flux prompt generator."""

    def __init__(self, config: ImagePromptConfig)
    def generate(self, description: str) -> GeneratedPrompt

class MidjourneyPromptGenerator(ImagePromptGenerator):
    """Midjourney prompt generator."""

    def __init__(self, config: ImagePromptConfig)
    def generate(self, description: str) -> GeneratedPrompt

class DALLEPromptGenerator(ImagePromptGenerator):
    """DALL-E prompt generator."""

    def __init__(self, config: ImagePromptConfig)
    def generate(self, description: str) -> GeneratedPrompt

class PromptOptimizer:
    """Prompt optimization utilities."""

    @staticmethod
    def enhance(prompt: str, quality: str = "high") -> str:
        """Enhance prompt with quality modifiers."""

    @staticmethod
    def add_style(prompt: str, style: str) -> str:
        """Add style to prompt."""

def create_prompt_generator(
    backend: PromptBackend = PromptBackend.STABLE_DIFFUSION,
    **kwargs,
) -> ImagePromptGenerator:
    """Create prompt generator."""
```

### image_editing/__init__.py

```python
class EditBackend(Enum):
    """Image editing backends."""
    INPAINT = "inpaint"
    OUTPAINT = "outpaint"
    BLEND = "blend"
    SDXL_INPAINT = "sdxl_inpaint"
    FLUX_EDIT = "flux_edit"

class ImageEditConfig:
    """Image editing configuration."""
    backend: EditBackend = EditBackend.INPAINT
    mask_blur: int = 8
    inpaint_padding: int = 32
    seamless: bool = False

@dataclass
class EditInstruction:
    """Parsed editing instruction."""
    operation: str
    target: str
    region: Optional[Tuple[int, int, int, int]] = None
    replacement: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EditResult:
    """Image editing result."""
    image: Image.Image
    instruction: EditInstruction
    mask: Optional[Image.Image] = None
    success: bool = True
    error: Optional[str] = None

class ImageEditingParser(ABC):
    """Base image editing parser."""

    @abstractmethod
    def parse(self, instruction: str) -> EditInstruction:
        """Parse editing instruction."""

    @abstractmethod
    def execute(self, image: Image.Image, edit: EditInstruction) -> EditResult:
        """Execute editing instruction."""

class InstructionParser(ImageEditingParser):
    """Natural language instruction parser."""

    def __init__(self, config: ImageEditConfig)
    def parse(self, instruction: str) -> EditInstruction
    def execute(self, image: Image.Image, edit: EditInstruction) -> EditResult

class InpaintEditor(ImageEditingParser):
    """Inpainting editor."""

    def __init__(self, config: ImageEditConfig)
    def parse(self, instruction: str) -> EditInstruction
    def execute(self, image: Image.Image, edit: EditInstruction) -> EditResult

class OutpaintEditor(ImageEditingParser):
    """Outpainting editor."""

    def __init__(self, config: ImageEditConfig)
    def parse(self, instruction: str) -> EditInstruction
    def execute(self, image: Image.Image, edit: EditInstruction) -> EditResult

class EditCommandBuilder:
    """Build editing commands for API consumption."""

    @staticmethod
    def build_sd_command(edit: EditInstruction) -> Dict[str, Any]:
        """Build Stable Diffusion command."""

    @staticmethod
    def build_flux_command(edit: EditInstruction) -> Dict[str, Any]:
        """Build Flux command."""

def create_editing_parser(
    backend: EditBackend = EditBackend.INPAINT,
    **kwargs,
) -> ImageEditingParser:
    """Create editing parser."""
```

### cross_attention/__init__.py

```python
class CrossAttentionConfig:
    """Cross attention configuration."""
    embed_dim: int = 768
    num_heads: int = 12
    dropout: float = 0.1
    kv_dim: Optional[int] = None  # If different from embed_dim
    is_cross_attention: bool = True

class CrossAttentionModule(nn.Module):
    """Cross attention module."""

    def __init__(self, config: CrossAttentionConfig)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Cross attention forward.

        Args:
            query: (batch, seq_len_q, embed_dim)
            key: (batch, seq_len_kv, kv_dim)
            value: (batch, seq_len_kv, kv_dim)

        Returns:
            Output (batch, seq_len_q, embed_dim)
        """

class SelfAttention(nn.Module):
    """Custom self-attention module."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1)

    def forward(self, query, key=None, value=None, mask=None) -> Tuple[Tensor, None]:
        """Self attention forward."""

class MultimodalTransformer(nn.Module):
    """Multimodal transformer combining vision and text."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 6,
        text_dim: int = 768,
        vision_dim: int = 768,
        dropout: float = 0.1,
    )

    def forward(
        self,
        text: Tensor,
        vision: Tensor,
        text_mask: Optional[Tensor] = None,
        vision_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Multimodal forward."""

class VisionLanguageModel(nn.Module):
    """Vision-Language Model."""

    def __init__(
        self,
        vision_embed_dim: int = 768,
        text_embed_dim: int = 768,
        hidden_dim: int = 768,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        vocab_size: int = 32000,
        max_seq_len: int = 512,
    )

    def encode_image(self, vision_features: Tensor) -> Tensor:
        """Encode vision features."""

    def generate(
        self,
        vision_features: Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
    ) -> Tensor:
        """Generate text from vision features."""

    def forward(
        self,
        vision_features: Tensor,
        text_ids: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, dict]:
        """Forward pass."""

def create_multimodal_model(
    model_type: str = "vlm",
    **kwargs,
) -> nn.Module:
    """Create multimodal model."""
```

### embeddings/__init__.py

```python
class MultimodalEmbedderConfig:
    """Multimodal embedder configuration."""
    text_dim: int = 768
    image_dim: int = 768
    audio_dim: int = 768
    output_dim: int = 768
    normalize: bool = True
    temperature: float = 0.07
    use_projection: bool = True

class MultimodalEmbedder(nn.Module):
    """Unified multimodal embedder."""

    def __init__(self, config: MultimodalEmbedderConfig)

    def embed_text(self, texts: Union[str, List[str]]) -> Tensor:
        """Embed text (batch, output_dim)."""

    def embed_image(self, images: Union[Image.Image, List[Image.Image]]) -> Tensor:
        """Embed images (batch, output_dim)."""

    def embed_audio(self, audio: Union[Tensor, np.ndarray]) -> Tensor:
        """Embed audio (batch, output_dim)."""

    def forward(
        self,
        text: Optional[Tensor] = None,
        image: Optional[Tensor] = None,
        audio: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass."""

class UnifiedEmbeddings:
    """Unified embeddings manager."""

    def __init__(self, config: MultimodalEmbedderConfig)

    def add_text(self, key: str, text: str) -> Tensor:
        """Add text embedding."""

    def add_image(self, key: str, image: Image.Image) -> Tensor:
        """Add image embedding."""

    def add_audio(self, key: str, audio: Tensor) -> Tensor:
        """Add audio embedding."""

    def retrieve_text_to_image(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> List[tuple]:
        """Retrieve images from text query."""

    def retrieve_image_to_text(
        self,
        query_image: Image.Image,
        top_k: int = 5,
    ) -> List[tuple]:
        """Retrieve text from image query."""

    def compute_similarity(self, emb1: Tensor, emb2: Tensor) -> Tensor:
        """Compute similarity between embeddings."""

    def normalize(self, embeddings: Tensor) -> Tensor:
        """Normalize embeddings."""

    def get_index_stats(self) -> Dict[str, Any]:
        """Get embedding index statistics."""

class ContrastiveLoss(nn.Module):
    """Contrastive loss for multimodal learning."""

    def __init__(self, temperature: float = 0.07)

    def forward(
        self,
        embeddings1: Tensor,
        embeddings2: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute contrastive loss."""

def create_multimodal_embedder(
    output_dim: int = 768,
    normalize: bool = True,
    **kwargs,
) -> MultimodalEmbedder:
    """Create multimodal embedder."""
```

### benchmarks/__init__.py

```python
@dataclass
class BenchmarkResult:
    """Benchmark result container."""
    name: str
    score: float
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class Benchmark(ABC):
    """Benchmark base class."""

    @abstractmethod
    def setup(self):
        """Setup benchmark."""

    @abstractmethod
    def run(self) -> BenchmarkResult:
        """Run benchmark."""

class ImageEmbeddingBenchmark(Benchmark):
    """Image embedding benchmark."""

    def __init__(
        self,
        model_name: str = "vit-base-patch16-224",
        num_iterations: int = 100,
    )

    def setup(self):
        """Setup benchmark."""

    def run(self) -> BenchmarkResult:
        """Run benchmark."""

class OCRBenchmark(Benchmark):
    """OCR benchmark."""

    def __init__(
        self,
        backend: OCRBackend = OCRBackend.EASYOCR,
        num_iterations: int = 100,
    )

    def setup(self):
        """Setup benchmark."""

    def run(self) -> BenchmarkResult:
        """Run benchmark."""

class UIParsingBenchmark(Benchmark):
    """UI parsing benchmark."""

    def __init__(
        self,
        backend: UIParseBackend = UIParseBackend.DETR,
        num_iterations: int = 100,
    )

    def setup(self):
        """Setup benchmark."""

    def run(self) -> BenchmarkResult:
        """Run benchmark."""

class CrossModalRetrievalBenchmark(Benchmark):
    """Cross-modal retrieval benchmark."""

    def __init__(
        self,
        embedder: MultimodalEmbedder,
        num_samples: int = 1000,
    )

    def setup(self):
        """Setup benchmark."""

    def run(self) -> BenchmarkResult:
        """Run benchmark."""

class PromptGenerationBenchmark(Benchmark):
    """Prompt generation benchmark."""

    def __init__(
        self,
        backend: PromptBackend = PromptBackend.STABLE_DIFFUSION,
        num_iterations: int = 100,
    )

    def setup(self):
        """Setup benchmark."""

    def run(self) -> BenchmarkResult:
        """Run benchmark."""

class MultimodalBenchmarkSuite:
    """Complete multimodal benchmark suite."""

    def __init__(self)

    def add_benchmark(self, benchmark: Benchmark):
        """Add benchmark to suite."""

    def run_all(self) -> List[BenchmarkResult]:
        """Run all benchmarks."""
```

## Usage Examples

### Vision Encoding

```python
from src.multimodal.vision_encoder import (
    create_vision_encoder,
    EncoderBackend,
)
from PIL import Image

# Create encoder
encoder = create_vision_encoder(
    backend=EncoderBackend.TORCH,
    device="cuda"
)

# Encode single image
img = Image.open("photo.jpg")
embedding = encoder.encode_image(img)
# embedding.shape: (1, 768)

# Encode batch
batch = [Image.open(f"img_{i}.jpg") for i in range(4)]
embeddings = encoder.encode_image(batch)
# embeddings.shape: (4, 768)
```

### Image Understanding

```python
from src.multimodal.image_understanding import (
    create_image_embedder,
    UnderstandingBackend,
)

# Create embedder
embedder = create_image_embedder(
    backend=UnderstandingBackend.LOCAL,
    device="cuda"
)

# Get embeddings
embeddings = embedder.embed_images(img)

# Get description
description = embedder.describe_image(img)
print(description.text)
```

### OCR

```python
from src.multimodal.ocr import (
    create_ocr_engine,
    OCRBackend,
)

# Create OCR engine
ocr = create_ocr_engine(
    backend=OCRBackend.EASYOCR,
    languages=["en", "de"],
    gpu=True
)

# Recognize text
result = ocr.recognize(screenshot)
print(result.text)
print(result.confidence)

# Batch recognition
results = ocr.recognize_batch([img1, img2, img3])
```

### UI Parsing

```python
from src.multimodal.ui_parser import (
    create_ui_detector,
    UIParseBackend,
    ScreenshotAnalyzer,
)

# Create detector
detector = create_ui_detector(
    backend=UIParseBackend.DETR,
    device="cuda"
)

# Detect elements
elements = detector.detect(screenshot)

# Full analysis
analyzer = ScreenshotAnalyzer(config)
layout = analyzer.analyze(screenshot)
points = analyzer.extract_interaction_points(layout)
# points: {"buttons": [...], "inputs": [...], "links": [...]}
```

### Image Prompt Generation

```python
from src.multimodal.image_prompting import (
    create_prompt_generator,
    PromptBackend,
    PromptOptimizer,
)

# Create generator
generator = create_prompt_generator(
    backend=PromptBackend.STABLE_DIFFUSION,
)

# Generate prompt
result = generator.generate("a cat sitting on a windowsill")
print(result.prompt)
# "a cat sitting on a windowsill, highly detailed, photorealistic"

# Optimize prompt
enhanced = PromptOptimizer.enhance(
    "a sunset over mountains",
    quality="ultra"
)
```

### Image Editing

```python
from src.multimodal.image_editing import (
    create_editing_parser,
    EditBackend,
)

# Create parser
parser = create_editing_parser(
    backend=EditBackend.INPAINT,
)

# Parse instruction
instruction = parser.parse("remove the person in the center")

# Execute
result = parser.execute(image, instruction)
edited_image = result.image
```

### Cross-Modal Embeddings

```python
from src.multimodal.embeddings import (
    create_multimodal_embedder,
    UnifiedEmbeddings,
)

# Create embedder
embedder = create_multimodal_embedder(output_dim=256)

# Embed text and images
text_emb = embedder.embed_text("a cat image")
img_emb = embedder.embed_image(cat_image)

# Unified embeddings with retrieval
unified = UnifiedEmbeddings(config)
unified.add_text("cat_photo", "a cat photo")
unified.add_image("cat_photo", cat_image)

# Cross-modal retrieval
results = unified.retrieve_text_to_image("cat photo")
```

### Benchmarks

```python
from src.multimodal.benchmarks import (
    ImageEmbeddingBenchmark,
    OCRBenchmark,
    MultimodalBenchmarkSuite,
)

# Create suite
suite = MultimodalBenchmarkSuite()

# Add benchmarks
suite.add_benchmark(ImageEmbeddingBenchmark(num_iterations=50))
suite.add_benchmark(OCRBenchmark(backend=OCRBackend.EASYOCR))

# Run all
results = suite.run_all()
for result in results:
    print(f"{result.name}: {result.score:.3f} ({result.latency_ms:.1f}ms)")
```

## Testing

```bash
# Run all multimodal tests
python -m pytest tests/test_multimodal.py -v

# Run specific test class
python -m pytest tests/test_multimodal.py::TestVisionEncoder -v

# Run specific test
python -m pytest tests/test_multimodal.py::TestOCR::test_easyocr_engine -v
```

## Configuration

### Environment Variables

```bash
# Device selection
MULTIMODAL_DEVICE=cuda  # or cpu

# Model caching
MODEL_CACHE_DIR=models/

# OCR models
EASYOCR_MODEL_PATH=models/ocr/easyocr.pth
TESSERACT_DATA=tessdata/
```

### Default Paths

```
models/
├── vision/           # Vision encoder models
│   ├── vit-base-patch16-224/
│   └── clip/
├── ocr/             # OCR models
│   ├── easyocr/
│   └── trocr/
└── multimodal/      # Multimodal models
```

## Notes

1. All vision encoders require images to be preprocessed via `preprocess()` before encoding
2. OCR engines lazy-load models on first use for memory efficiency
3. UI detectors can use multiple backends for enhanced accuracy
4. Prompt generators add quality modifiers automatically based on config
5. CrossAttentionModule handles both 2D and 3D input tensors
6. UnifiedEmbeddings supports cross-modal retrieval with cosine similarity