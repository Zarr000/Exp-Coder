"""
Repository-aware code preprocessing for Expera AI.

Handles:
1. Repository structure parsing (detect project roots)
2. Import resolution and dependency graph building
3. Context window construction (group related files)
4. Header injection (file path, language tags)
5. Optional: only enabled for code datasets
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path


@dataclass
class RepoFile:
    """A file within a repository."""
    path: str
    language: str
    content: str
    imports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class RepoContext:
    """A group of related files from a repository."""
    repo_name: str
    files: List[RepoFile]
    dependency_graph: Dict[str, List[str]] = field(default_factory=dict)


class RepoPreprocessor:
    """
    Preprocesses code repositories for training.
    
    Detects repository structure, resolves imports,
    and constructs context windows of related files.
    """

    def __init__(
        self,
        enabled: bool = True,
        max_files_per_context: int = 10,
        include_file_headers: bool = True,
        resolve_imports: bool = True,
    ):
        self.enabled = enabled
        self.max_files_per_context = max_files_per_context
        self.include_file_headers = include_file_headers
        self.resolve_imports = resolve_imports

    def detect_repo_root(self, file_path: str) -> Optional[str]:
        """Detect repository root from file path."""
        path = Path(file_path)
        for parent in path.parents:
            markers = ['.git', 'package.json', 'setup.py', 'Cargo.toml',
                       'go.mod', 'pom.xml', 'build.gradle', 'CMakeLists.txt']
            for marker in markers:
                if (parent / marker).exists():
                    return str(parent)
        return None

    def extract_imports(self, content: str, language: str) -> List[str]:
        """Extract import statements from code."""
        imports = []
        patterns = {
            'python': r'(?:from\s+(\S+)\s+import|import\s+(\S+))',
            'javascript': r'(?:import\s+.*?from\s+[\'"]([^\'"]+)[\'"]|require\([\'"]([^\'"]+)[\'"]\))',
            'typescript': r'(?:import\s+.*?from\s+[\'"]([^\'"]+)[\'"]|require\([\'"]([^\'"]+)[\'"]\))',
            'java': r'import\s+([\w.]+)',
            'go': r'import\s+(?:"([^"]+)"|(\w+)\s+"([^"]+)")',
            'rust': r'use\s+([\w:]+)',
        }
        pattern = patterns.get(language)
        if pattern:
            for match in re.finditer(pattern, content):
                groups = [g for g in match.groups() if g]
                if groups:
                    imports.append(groups[0])
        return imports

    def resolve_dependencies(self, files: List[RepoFile]) -> Dict[str, List[str]]:
        """Build dependency graph from imports."""
        graph = {}
        file_paths = {f.path: f for f in files}
        
        for f in files:
            deps = []
            for imp in f.imports:
                # Try to resolve import to a file in the repo
                for fp in file_paths:
                    if imp.replace('.', '/') in fp or imp.split('/')[-1] in fp:
                        deps.append(fp)
                        break
            graph[f.path] = deps
        return graph

    def build_context(self, files: List[RepoFile]) -> List[RepoContext]:
        """Build context windows of related files."""
        if not self.resolve_imports:
            # Simple grouping: just take files in order
            contexts = []
            for i in range(0, len(files), self.max_files_per_context):
                group = files[i:i + self.max_files_per_context]
                contexts.append(RepoContext(
                    repo_name=Path(files[0].path).parent.name if files else "unknown",
                    files=group,
                ))
            return contexts
        
        # Build dependency graph and group connected files
        graph = self.resolve_dependencies(files)
        visited = set()
        contexts = []
        
        for f in files:
            if f.path in visited:
                continue
            
            # BFS to find connected files
            context_files = []
            queue = [f.path]
            while queue and len(context_files) < self.max_files_per_context:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                
                file_obj = next((x for x in files if x.path == current), None)
                if file_obj:
                    context_files.append(file_obj)
                    for dep in graph.get(current, []):
                        if dep not in visited:
                            queue.append(dep)
            
            if context_files:
                contexts.append(RepoContext(
                    repo_name=Path(f.path).parent.name,
                    files=context_files,
                    dependency_graph=graph,
                ))
        
        return contexts

    def format_file_with_header(self, file: RepoFile) -> str:
        """Format a file with header information."""
        if not self.include_file_headers:
            return file.content
        
        header = f"<|file_start|> {file.path} <|language|> {file.language}\n"
        footer = "\n<|file_end|>"
        return header + file.content + footer

    def process(self, files: List[Tuple[str, str, str]]) -> List[str]:
        """
        Process repository files into training contexts.
        
        Args:
            files: List of (file_path, language, content) tuples
            
        Returns:
            List of formatted context strings
        """
        if not self.enabled:
            return [content for _, _, content in files]
        
        repo_files = []
        for path, lang, content in files:
            imports = self.extract_imports(content, lang) if self.resolve_imports else []
            repo_files.append(RepoFile(path=path, language=lang,
                                       content=content, imports=imports))
        
        contexts = self.build_context(repo_files)
        
        results = []
        for ctx in contexts:
            formatted = []
            for f in ctx.files:
                formatted.append(self.format_file_with_header(f))
            results.append('\n'.join(formatted))
        
        return results