"""
Test Generator for Expera AI Coding Agent.

Generates unit tests for code.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TestCase:
    """Represents a test case."""

    name: str
    code: str
    inputs: dict
    expected: str


@dataclass
class TestResult:
    """Result of test generation."""

    success: bool
    tests: list[TestCase] = field(default_factory=list)
    error: Optional[str] = None


class TestGenerator:
    """
    Generates unit tests.

    Supports:
    - Function tests
    - Class tests
    - Integration tests
    - Mock tests
    """

    def __init__(self, root_path: Optional[str] = None) -> None:
        """Initialize test generator."""
        self.root_path = Path(root_path) if root_path else Path.cwd()

    def _get_func_signature(self, node: ast.FunctionDef) -> str:
        """Get function signature."""
        args = [arg.arg for arg in node.args.args]
        defaults = node.args.defaults

        for i in range(len(defaults)):
            args[i] = f"{args[i]}={ast.unparse(defaults[i])}"

        return ", ".join(args)

    def generate_function_test(
        self,
        file_path: str,
        function_name: str,
    ) -> TestCase:
        """Generate test for function."""
        full_path = self.root_path / file_path
        content = full_path.read_text()
        tree = ast.parse(content)

        func_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                func_node = node
                break

        if not func_node:
            return TestCase(
                name=f"test_{function_name}",
                code="",
                inputs={},
                expected="",
            )

        signature = self._get_func_signature(func_node)

        test_code = f'''import pytest
from {file_path.replace("/", ".").replace(".py", "")} import {function_name}


def test_{function_name}():
    """Test {function_name}."""
    # TODO: Add test inputs
    result = {function_name}()
    assert result is not None
'''

        return TestCase(
            name=f"test_{function_name}",
            code=test_code,
            inputs={},
            expected="",
        )

    def generate_class_test(
        self,
        file_path: str,
        class_name: str,
    ) -> list[TestCase]:
        """Generate tests for class."""
        full_path = self.root_path / file_path
        content = full_path.read_text()
        tree = ast.parse(content)

        tests: list[TestCase] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                test_code = f'''import pytest
from {file_path.replace("/", ".").replace(".py", "")} import {class_name}


class Test{class_name}:
    """Tests for {class_name}."""

    def test_init(self):
        """Test initialization."""
        instance = {class_name}()
        assert instance is not None
'''
                tests.append(TestCase(
                    name=f"test_{class_name}_init",
                    code=test_code,
                    inputs={},
                    expected="",
                ))

        return tests

    def generate_integration_test(
        self,
        files: list[str],
        test_name: str,
    ) -> TestCase:
        """Generate integration test."""
        imports = "\n".join(
            f"from {f.replace('/', '.').replace('.py', '')} import *"
            for f in files
        )

        test_code = f'''import pytest


{imports}


class TestIntegration:
    """Integration tests."""

    def test_{test_name}(self):
        """Test {test_name}."""
        # TODO: Add integration test
        pass
'''

        return TestCase(
            name=f"test_{test_name}",
            code=test_code,
            inputs={},
            expected="",
        )

    def generate_tests_for_file(
        self,
        file_path: str,
    ) -> TestResult:
        """Generate all tests for file."""
        tests: list[TestCase] = []
        full_path = self.root_path / file_path

        try:
            content = full_path.read_text()
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    tests.append(self.generate_function_test(
                        file_path,
                        node.name,
                    ))

                elif isinstance(node, ast.ClassDef):
                    tests.extend(self.generate_class_test(
                        file_path,
                        node.name,
                    ))

            return TestResult(success=True, tests=tests)

        except Exception as e:
            return TestResult(success=False, error=str(e))

    def generate_pytest_fixture(
        self,
        class_name: str,
    ) -> str:
        """Generate pytest fixture."""
        return f'''@pytest.fixture
def {class_name.lower()}_instance():
    """Fixture for {class_name}."""
    return {class_name}()
'''

    def generate_mock_test(
        self,
        file_path: str,
        function_name: str,
    ) -> str:
        """Generate test with mocks."""
        return f'''import pytest
from unittest.mock import Mock, patch


@patch("{file_path.replace("/", ".").replace(".py", "")}.{function_name}")
def test_{function_name}_mock(mock_func):
    """Test {function_name} with mock."""
    mock_func.return_value = None
    result = mock_func()
    mock_func.assert_called_once()
'''

    def write_tests(
        self,
        tests: list[TestCase],
        output_path: str,
    ) -> bool:
        """Write tests to file."""
        full_path = self.root_path / output_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        code = "\n\n".join(t.code for t in tests)
        full_path.write_text(code)

        return True

    def generate_test_file(
        self,
        file_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """Generate complete test file."""
        result = self.generate_tests_for_file(file_path)

        if output_path is None:
            p = Path(file_path)
            output_path = str(p.parent / f"test_{p.name}")

        self.write_tests(result.tests, output_path)

        return output_path