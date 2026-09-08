#!/usr/bin/env python3
"""
Interactive Chat Interface for Expera AI.

Provides a local chat interface:
- Interactive terminal chat
- Multiple model support
- System prompts
- Chat history

Usage:
    python scripts/chat.py
    python scripts/chat.py --model checkpoints/final/
    python scripts/chat.py --system "You are a helpful coding assistant."
    python scripts/chat.py --interactive
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.inference import InferencePipeline, GenerationConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ChatSession:
    """Chat session with history."""

    def __init__(
        self,
        system_prompt: str = "You are Expera AI, a helpful AI assistant.",
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
    ):
        """Initialize chat session."""
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = []

        # Load pipeline
        self.pipeline = self._load_pipeline(model_path, tokenizer_path)

        logger.info("Chat session initialized")

    def _load_pipeline(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
    ) -> InferencePipeline:
        """Load inference pipeline."""
        # Load tokenizer if available
        tokenizer = None
        if tokenizer_path:
            from src.tokenizer import SentencePieceTokenizer
            tokenizer = SentencePieceTokenizer(tokenizer_path)

        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_path:
            pipeline = InferencePipeline(
                model_path=model_path,
                tokenizer=tokenizer,
                device=device,
            )
        else:
            # Use default model
            logger.warning("No model specified, using placeholder")
            pipeline = InferencePipeline(
                model=None,
                tokenizer=tokenizer,
                device=device,
            )

        return pipeline

    def chat(
        self,
        user_input: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Process user input and generate response.

        Args:
            user_input: User message
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling

        Returns:
            Assistant response
        """
        # Build prompt with history
        prompt = self._build_prompt(user_input)

        # Generate
        output = self.pipeline.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
        )

        response = output.text if hasattr(output, "text") else str(output)

        # Fallback for empty/undertrained responses
        if not response or not response.strip():
            response = "I'm still learning and can't respond to that yet. Please try again later!"

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response

    def _build_prompt(self, user_input: str) -> str:
        """Build prompt with system and history."""
        # System
        prompt = f"System: {self.system_prompt}\n\n"

        # History
        for msg in self.history[-10:]:  # Last 10 messages
            role = msg["role"]
            content = msg["content"]
            prompt += f"{role.capitalize()}: {content}\n"

        # Current input
        prompt += f"User: {user_input}\nAssistant:"

        return prompt

    def clear_history(self):
        """Clear chat history."""
        self.history = []

    def get_history(self) -> List[Dict[str, str]]:
        """Get chat history."""
        return self.history.copy()

    def save_history(self, path: str):
        """Save chat history to file."""
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    def load_history(self, path: str):
        """Load chat history from file."""
        with open(path, "r") as f:
            self.history = json.load(f)


def interactive_chat(
    model_path: Optional[str] = None,
    system_prompt: str = "You are Expera AI, a helpful AI assistant.",
    tokenizer_path: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """
    Run interactive chat.

    Args:
        model_path: Model checkpoint path
        system_prompt: System prompt
        tokenizer_path: Tokenizer path
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling
    """
    print("=" * 60)
    print("  Expera AI Chat")
    print("=" * 60)
    print(f"System: {system_prompt}")
    print()
    print("Commands:")
    print("  /clear - Clear history")
    print("  /save - Save history")
    print("  /quit - Exit")
    print()

    # Setup session
    session = ChatSession(
        system_prompt=system_prompt,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )

    print("Chat ready! Type your message below.")
    print()

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Commands
            if user_input.startswith("/"):
                cmd = user_input.lower()

                if cmd == "/clear":
                    session.clear_history()
                    print("History cleared.")
                    continue
                elif cmd == "/save":
                    path = f"chat_history_{len(session.get_history())}.json"
                    session.save_history(path)
                    print(f"History saved to {path}")
                    continue
                elif cmd in ["/quit", "/exit"]:
                    print("Goodbye!")
                    break
                else:
                    print(f"Unknown command: {cmd}")
                    continue

            # Generate response
            print("Assistant: ", end="", flush=True)

            response = session.chat(
                user_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            print(response)
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            continue


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Interactive chat with Expera AI")

    # Model
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model checkpoint path"
    )
    parser.add_argument(
        "--tokenizer", "-t",
        type=str,
        default=None,
        help="Tokenizer path"
    )

    # Prompt
    parser.add_argument(
        "--system",
        type=str,
        default="You are Expera AI, a helpful AI assistant.",
        help="System prompt"
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Load system prompt from file"
    )

    # Generation
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", "-T",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold"
    )

    # Mode
    parser.add_argument(
        "--single",
        type=str,
        default=None,
        help="Single prompt mode (non-interactive)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load system prompt from file
    system_prompt = args.system
    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            system_prompt = f.read()

    # Single prompt mode
    if args.single:
        session = ChatSession(
            system_prompt=system_prompt,
            model_path=args.model,
            tokenizer_path=args.tokenizer,
        )

        response = session.chat(
            args.single,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        print(response)
        return

    # Interactive mode
    interactive_chat(
        model_path=args.model,
        system_prompt=system_prompt,
        tokenizer_path=args.tokenizer,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()