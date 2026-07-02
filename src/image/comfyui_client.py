"""
ComfyUI Client.

Interface to ComfyUI workflow API.

Usage:
    python -m src.image.comfyui_client --workflow workflow.json --prompt "a cat"
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class ComfyUIConfig:
    """ComfyUI client configuration."""

    api_url: str = "http://localhost:8188"

    queue_after_prompt: bool = True
    auto_download: bool = True


@dataclass
class NodeInput:
    """Input for a ComfyUI node."""

    node_id: str
    name: str
    value: Any


class ComfyUIClient:
    """Client for ComfyUI API."""

    def __init__(self, config: ComfyUIConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        if self._session:
            await self._session.close()

    async def get_history(self, run_id: str) -> dict:
        """Get execution history for a run."""
        async with self._session.get(
            f"{self.config.api_url}/history/{run_id}"
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"History error: {resp.status}")
            return await resp.json()

    async def get_queue(self) -> dict:
        """Get current queue status."""
        async with self._session.get(
            f"{self.config.api_url}/queue"
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Queue error: {resp.status}")
            return await resp.json()

    async def queue_prompt(
        self,
        prompt_data: dict,
        extra_data: Optional[dict] = None,
    ) -> str:
        """
        Queue a prompt for execution.

        Returns:
            Run ID for tracking
        """
        payload = {"prompt": prompt_data, "extra_data": extra_data or {}}

        async with self._session.post(
            f"{self.config.api_url}/prompt", json=payload
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Queue error: {resp.status} - {text}")

            result = await resp.json()
            return result["prompt_id"]

    async def interrupt(self) -> bool:
        """Interrupt current execution."""
        async with self._session.post(
            f"{self.config.api_url}/interrupt"
        ) as resp:
            return resp.status == 200

    async def upload_image(self, image_path: Path) -> dict:
        """Upload an image for use in workflows."""
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        payload = {
            "image": image_data,
            "upload": "image",
            "name": image_path.name,
        }

        async with self._session.post(
            f"{self.config.api_url}/upload/image", json=payload
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Upload error: {resp.status}")
            return await resp.json()

    async def upload_mask(self, image_path: Path) -> dict:
        """Upload a mask image."""
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        payload = {
            "image": image_data,
            "upload": "mask",
            "name": image_path.name,
        }

        async with self._session.post(
            f"{self.config.api_url}/upload/image", json=payload
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Upload error: {resp.status}")
            return await resp.json()

    async def get_objects(self, object_type: str = "node") -> dict:
        """Get available objects (nodes, workflows, etc.)."""
        async with self._session.get(
            f"{self.config.api_url}/objects/{object_type}"
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Objects error: {resp.status}")
            return await resp.json()

    async def await_prompt(
        self,
        run_id: str,
        timeout: int = 300,
    ) -> dict:
        """Wait for a queued prompt to complete."""
        import time

        start_time = time.time()
        while time.time() - start_time < timeout:
            history = await self.get_history(run_id)
            if run_id in history:
                status = history[run_id]
                if status.get("status"):
                    if status["status"].get("completed"):
                        return status
                    if status["status"].get("executed"):
                        return status
            await asyncio.sleep(1)

        raise TimeoutError(f"Prompt {run_id} timed out")

    async def download_image(
        self,
        filename: str,
        output_path: Optional[Path] = None,
    ) -> bytes:
        """Download a generated image."""
        async with self._session.get(
            f"{self.config.api_url}/view", params={"filename": filename}
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Download error: {resp.status}")
            return await resp.read()


# Workflow templates

def text_to_image_workflow(
    model: str,
    clip: str,
    vae: str,
    positive_prompt: str,
    negative_prompt: str,
    sampler: str = "euler",
    steps: int = 20,
    cfg: float = 8.0,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
) -> dict:
    """Build a text-to-image workflow."""
    return {
        "1": {
            "inputs": {"text": positive_prompt, "clip": clip},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Positive Prompt"},
        },
        "2": {
            "inputs": {"text": negative_prompt, "clip": clip},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Negative Prompt"},
        },
        "3": {
            "inputs": {"seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler},
            "class_type": "KSampler",
            "_meta": {"title": "Sampler"},
        },
        "4": {
            "inputs": {"width": width, "height": height},
            "class_type": "EmptyLatentImage",
            "_meta": {"title": "Empty Latent"},
        },
        "5": {
            "inputs": {"model": model, "clip": clip, "positive": "1", "negative": "2"},
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load Checkpoint"},
        },
    }


def inpaint_workflow(
    model: str,
    vae: str,
    clip: str,
    positive_prompt: str,
    negative_prompt: str,
    image_path: str,
    mask_path: str,
    sampler: str = "euler",
    steps: int = 20,
    cfg: float = 8.0,
    denoise: float = 0.7,
) -> dict:
    """Build an inpainting workflow."""
    return {
        "1": {
            "inputs": {"text": positive_prompt, "clip": clip},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Positive Prompt"},
        },
        "2": {
            "inputs": {"text": negative_prompt, "clip": clip},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Negative Prompt"},
        },
        "3": {
            "inputs": {"image": image_path},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Image"},
        },
        "4": {
            "inputs": {"image": mask_path},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Mask"},
        },
        "5": {
            "inputs": {"image": "3", "mask": "4"},
            "class_type": "InvertMask",
            "_meta": {"title": "Invert Mask"},
        },
        "6": {
            "inputs": {"images": "3", "mask": "5", "grow": 10},
            "class_type": "PrepImageForInpaint",
            "_meta": {"title": "Prep Inpaint"},
        },
    }


def upscaling_workflow(
    model: str,
    vae: str,
    image_path: str,
    scale_factor: int = 2,
    method: str = "nearest",
) -> dict:
    """Build an upscaling workflow."""
    return {
        "1": {
            "inputs": {"image": image_path},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Image"},
        },
        "2": {
            "inputs": {"model_name": model, "input_image": "1"},
            "class_type": "UpscaleModel",
            "_meta": {"title": "Upscale Image"},
        },
    }


def image_to_image_workflow(
    model: str,
    vae: str,
    clip: str,
    positive_prompt: str,
    negative_prompt: str,
    image_path: str,
    strength: float = 0.7,
) -> dict:
    """Build an image-to-image workflow."""
    return {
        "1": {
            "inputs": {"text": positive_prompt, "clip": clip},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Positive Prompt"},
        },
        "2": {
            "inputs": {"text": negative_prompt, "clip": clip},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Negative Prompt"},
        },
        "3": {
            "inputs": {"image": image_path},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Image"},
        },
        "4": {
            "inputs": {"image": "3", "strength": strength},
            "class_type": "VAEEncodeForInpaint",
            "_meta": {"title": "Encode Image"},
        },
    }


def generate_controlnet_workflow(
    model: str,
    control_net: str,
    image_path: str,
    prompt: str,
    strength: float = 1.0,
) -> dict:
    """Build a ControlNet workflow."""
    return {
        "1": {
            "inputs": {"image": image_path},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Image"},
        },
        "2": {
            "inputs": {"image": "1", "control_net_name": control_net},
            "class_type": "ControlNetApply",
            "_meta": {"title": "Apply ControlNet"},
        },
    }


async def main():
    parser = argparse.ArgumentParser(description="ComfyUI client")
    parser.add_argument("--workflow", help="Workflow JSON file")
    parser.add_argument("--prompt", help="Text prompt")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--url", default="http://localhost:8188")
    args = parser.parse_args()

    config = ComfyUIConfig(api_url=args.url)

    async with ComfyUIClient(config) as client:
        if args.workflow:
            workflow_path = Path(args.workflow)
            with open(workflow_path) as f:
                workflow = json.load(f)

            run_id = await client.queue_prompt(workflow)
            logger.info(f"Queued: {run_id}")

            result = await client.await_prompt(run_id)

            if args.output:
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)

                for node_id, node_output in result.get("outputs", {}).items():
                    if "images" in node_output:
                        for img in node_output["images"]:
                            image_data = await client.download_image(
                                img["filename"]
                            )
                            out_path = output_dir / img["filename"]
                            with open(out_path, "wb") as f:
                                f.write(image_data)
                            logger.info(f"Saved: {out_path}")

        else:
            queue = await client.get_queue()
            logger.info(f"Queue: {json.dumps(queue, indent=2)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())