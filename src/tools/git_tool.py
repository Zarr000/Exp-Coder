"""
Git Tool for Expera AI.

Provides git operations: commit, branch, diff, log, status, stash, merge, rebase.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class GitResult:
    """Result of git operation."""

    success: bool
    output: str
    error: Optional[str] = None


class GitTool:
    """
    Git operations tool.

    Supports: status, diff, commit, branch, log, stash, merge, rebase.
    """

    def __init__(self, repo_path: Optional[str] = None) -> None:
        """Initialize git tool."""
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()

    async def _run(self, *args: str) -> GitResult:
        """Run git command."""
        cmd = ["git", "-C", str(self.repo_path)] + list(args)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        return GitResult(
            success=proc.returncode == 0,
            output=stdout.decode() if stdout else "",
            error=stderr.decode() if stderr else None,
        )

    async def status(self) -> GitResult:
        """Get git status."""
        return await self._run("status", "--porcelain")

    async def diff(self, staged: bool = False) -> GitResult:
        """Get git diff."""
        if staged:
            return await self._run("diff", "--cached")
        return await self._run("diff")

    async def log(self, limit: int = 10) -> GitResult:
        """Get git log."""
        return await self._run(
            "log", f"-n{limit}", "--oneline", "--graph"
        )

    async def current_branch(self) -> GitResult:
        """Get current branch."""
        return await self._run("branch", "--show-current")

    async def branches(self) -> GitResult:
        """List all branches."""
        return await self._run("branch", "-a")

    async def create_branch(self, name: str, start_point: Optional[str] = None) -> GitResult:
        """Create a new branch."""
        if start_point:
            return await self._run("checkout", "-b", name, start_point)
        return await self._run("branch", name)

    async def checkout(self, ref: str, create: bool = False) -> GitResult:
        """Checkout a branch or commit."""
        if create:
            return await self._run("checkout", "-b", ref)
        return await self._run("checkout", ref)

    async def add(self, paths: list[str]) -> GitResult:
        """Stage files."""
        return await self._run("add", *paths)

    async def commit(self, message: str) -> GitResult:
        """Commit staged changes."""
        return await self._run("commit", "-m", message)

    async def push(
        self,
        remote: str = "origin",
        branch: Optional[str] = None,
        set_upstream: bool = False,
    ) -> GitResult:
        """Push to remote."""
        args = ["push"]
        if set_upstream:
            args.extend(["-u"])
        args.append(remote)
        if branch:
            args.append(branch)
        return await self._run(*args)

    async def pull(self, remote: str = "origin", branch: Optional[str] = None) -> GitResult:
        """Pull from remote."""
        args = ["pull", remote]
        if branch:
            args.append(branch)
        return await self._run(*args)

    async def fetch(self, remote: str = "origin") -> GitResult:
        """Fetch from remote."""
        return await self._run("fetch", remote)

    async def merge(self, branch: str, no_ff: bool = False) -> GitResult:
        """Merge branch."""
        args = ["merge"]
        if no_ff:
            args.append("--no-ff")
        args.append(branch)
        return await self._run(*args)

    async def rebase(self, branch: str, interactive: bool = False) -> GitResult:
        """Rebase onto branch."""
        if interactive:
            return await self._run("rebase", "-i", branch)
        return await self._run("rebase", branch)

    async def stash(self, message: Optional[str] = None, pop: bool = False) -> GitResult:
        """Stash changes."""
        if pop:
            return await self._run("stash", "pop")
        if message:
            return await self._run("stash", "push", "-m", message)
        return await self._run("stash", "push")

    async def reset(self, mode: str = "mixed", target: str = "HEAD") -> GitResult:
        """Reset to commit."""
        return await self._run("reset", f"--{mode}", target)

    async def revert(self, commits: list[str]) -> GitResult:
        """Revert commits."""
        return await self._run("revert", *commits)

    async def cherry_pick(self, commit: str) -> GitResult:
        """Cherry-pick commit."""
        return await self._run("cherry-pick", commit)

    async def tag(self, name: str, message: Optional[str] = None) -> GitResult:
        """Create tag."""
        if message:
            return await self._run("tag", "-a", name, "-m", message)
        return await self._run("tag", name)

    async def describe(self, abbrev: int = 7) -> GitResult:
        """Describe current commit."""
        return await self._run("describe", f"--abbrev={abbrev}")

    async def show(self, ref: str = "HEAD") -> GitResult:
        """Show commit details."""
        return await self._run("show", ref)

    async def blame(self, file: str) -> GitResult:
        """Get blame for file."""
        return await self._run("blame", file)

    async def bisect_start(self, bad: str, good: list[str]) -> GitResult:
        """Start bisect."""
        await self._run("bisect", "start")
        await self._run("bisect", "bad", bad)
        for g in good:
            await self._run("bisect", "good", g)
        return GitResult(success=True, output="Bisect started")

    async def bisect_skip(self, commits: list[str]) -> GitResult:
        """Skip commits in bisect."""
        return await self._run("bisect", "skip", *commits)

    async def bisect_reset(self) -> GitResult:
        """Reset bisect."""
        return await self._run("bisect", "reset")