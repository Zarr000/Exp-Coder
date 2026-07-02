"""
Code Generator for Expera AI.

Generates code from prompts:
- Multiple language support
- Template-based generation
- Code completion
- File scaffolding

Usage:
    generator = CodeGenerator()
    code = await generator.generate("Create a function to add two numbers")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Optional

logger = logging.getLogger(__name__)


@dataclass
class GenerationRequest:
    """Code generation request."""

    prompt: str
    language: Optional[str] = None
    framework: Optional[str] = None
    template: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.7


# Language templates
CODE_TEMPLATES = {
    "python": {
        "function": "def {name}({params}):\n    \"\"\"{docstring}\"\"\"\n    {body}\n",
        "class": "class {name}:\n    \"\"\"{docstring}\"\"\"\n\n    def __init__(self{init_params}):\n        {init_body}\n\n    def __str__(self):\n        return f\"{name}()\"\n",
    },
    "javascript": {
        "function": "function {name}({params}) {{\n    {docstring}\n    {body}\n}}",
        "class": "class {name} {{\n    constructor({params}) {{\n        {init_body}\n    }}\n}}",
    },
    "typescript": {
        "function": "function {name}({params}): {rettype} {{\n    {docstring}\n    {body}\n}}",
        "class": "class {name} {{\n    constructor({params}) {{\n        {init_body}\n    }}\n}}",
    },
    "rust": {
        "function": "fn {name}({params}) -> {rettype} {{\n    {docstring}\n    {body}\n}}",
        "struct": "struct {name} {{\n    {fields}\n}}\n\nimpl {name} {{\n    new({params}) -> Self {{\n        {init_body}\n    }}\n}}",
    },
    "go": {
        "func": "func {name}({params}) {{\n    {docstring}\n    {body}\n}}",
        "struct": "type {name} struct {{\n    {fields}\n}}",
    },
}


class CodeGenerator:
    """
    Generates code from prompts.

    Features:
    - Template-based generation
    - Language detection
    - Framework support
    - Code completion
    """

    def __init__(self):
        """Initialize code generator."""
        self.templates = CODE_TEMPLATES.copy()
        self._runtime = None

    def set_runtime(self, runtime) -> None:
        """Set inference runtime."""
        self._runtime = runtime

    async def generate(
        self,
        prompt: str,
        language: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Generate code from prompt."""
        # Detect language if not specified
        if language is None:
            language = self._detect_language(prompt)

        # Use runtime if available
        if self._runtime:
            async for chunk in self._runtime.generate(
                f"Generate {language} code: {prompt}",
                max_tokens=max_tokens,
                temperature=temperature,
            ):
                yield chunk
            return

        # Template-based generation
        code = self._generate_from_template(prompt, language)
        yield code

    def _detect_language(self, prompt: str) -> str:
        """Detect programming language from prompt."""
        prompt_lower = prompt.lower()

        # Explicit mentions
        if "python" in prompt_lower:
            return "python"
        if "javascript" in prompt_lower or "js" in prompt_lower:
            return "javascript"
        if "typescript" in prompt_lower or "ts" in prompt_lower:
            return "typescript"
        if "rust" in prompt_lower or "rs" in prompt_lower:
            return "rust"
        if "go" in prompt_lower or "golang" in prompt_lower:
            return "go"
        if "java" in prompt_lower:
            return "java"
        if "c++" in prompt_lower or "cpp" in prompt_lower:
            return "cpp"
        if "c#" in prompt_lower or "csharp" in prompt_lower:
            return "csharp"

        # Default
        return "python"

    def _generate_from_template(self, prompt: str, language: str) -> str:
        """Generate code using templates."""
        # Simple keyword matching
        prompt_lower = prompt.lower()

        # Function detection
        if "function" in prompt_lower or "def " in prompt_lower or "fn " in prompt_lower:
            if language == "python":
                return '''def example_function(param1, param2):
    """Example function."""
    result = param1 + param2
    return result

# Example usage
if __name__ == "__main__":
    print(example_function(1, 2))'''
            elif language == "javascript":
                return '''function exampleFunction(param1, param2) {
    // Example function
    const result = param1 + param2;
    return result;
}

// Example usage
console.log(exampleFunction(1, 2));'''
            elif language == "typescript":
                return '''function exampleFunction(param1: number, param2: number): number {
    // Example function
    return param1 + param2;
}

// Example usage
console.log(exampleFunction(1, 2));'''
            elif language == "rust":
                return '''fn example_function(param1: i32, param2: i32) -> i32 {
    // Example function
    param1 + param2
}

fn main() {
    println!("{}", example_function(1, 2));
}'''
            elif language == "go":
                return '''func ExampleFunction(param1 int, param2 int) int {
    // Example function
    return param1 + param2
}

func main() {
    fmt.Println(ExampleFunction(1, 2))
}'''

        # Class detection
        if "class" in prompt_lower or "object" in prompt_lower:
            if language == "python":
                return '''class ExampleClass:
    """Example class."""

    def __init__(self, value):
        self.value = value

    def process(self):
        """Process data."""
        return self.value * 2

# Example usage
obj = ExampleClass(10)
print(obj.process())'''
            elif language == "javascript":
                return '''class ExampleClass {
    constructor(value) {
        this.value = value;
    }

    process() {
        return this.value * 2;
    }
}

// Example usage
const obj = new ExampleClass(10);
console.log(obj.process());'''

        # Default: simple script
        if language == "python":
            return '''# Generated code
def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()'''
        elif language == "javascript":
            return '''// Generated code
function main() {
    console.log("Hello, World!");
}

main();'''
        elif language == "typescript":
            return '''// Generated code
function main(): void {
    console.log("Hello, World!");
}

main();'''
        elif language == "rust":
            return '''// Generated code
fn main() {
    println!("Hello, World!");
}'''
        elif language == "go":
            return '''// Generated code
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}'''

        return "// Generated code"

    async def complete(
        self,
        code: str,
        max_tokens: int = 128,
    ) -> str:
        """Complete partial code."""
        if self._runtime:
            result = ""
            async for chunk in self._runtime.generate(
                f"Complete this code: {code}",
                max_tokens=max_tokens,
            ):
                result += chunk
            return result

        return ""

    def get_supported_languages(self) -> list[str]:
        """Get list of supported languages."""
        return list(self.templates.keys())


# Export
__all__ = [
    "CodeGenerator",
    "GenerationRequest",
]