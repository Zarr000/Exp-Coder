"""Deterministic representative multilingual code/text corpus.

Generates ``benchmarks/data/code_sample/<lang>.<ext>`` files covering Python,
JavaScript, C++, Rust, Go, JSON, YAML, Markdown, shell, and natural text —
with the feature categories the Exp-Coder tokenizer must handle
(identifiers, numbers, strings, operators, indentation, type annotations,
generics, error handling, comments, docstrings).

No network access; fully deterministic.
"""

from pathlib import Path
from typing import Dict, List

OUT = Path(__file__).resolve().parent / "data" / "code_sample"


# ---------------------------------------------------------------------------
# Corpus content (deterministic, representative).
# ---------------------------------------------------------------------------
PYTHON = r'''
"""A small module demonstrating Python constructs."""

import math
import collections
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

MAX_RETRIES = 3
DEFAULT_RATIO = 0.6180339887498949


def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def primes(limit: int) -> list[int]:
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, ok in enumerate(sieve) if ok]


@dataclass
class Stack:
    name: str
    items: List[int] = field(default_factory=list)

    def push(self, item: int) -> None:
        self.items.append(item)

    def pop(self) -> Optional[int]:
        return self.items.pop() if self.items else None

    def size(self) -> int:
        return len(self.items)


def word_frequencies(text: str) -> Dict[str, int]:
    counts = collections.Counter(text.split())
    return {word: count for word, count in counts.most_common(10)}


def safe_divide(a: float, b: float) -> Tuple[float, Optional[str]]:
    try:
        return a / b, None
    except ZeroDivisionError as exc:
        return float("nan"), f"division failed: {exc}"


async def fetch(url: str, timeout: float = 2.5) -> bytes:
    # simplified async fetch placeholder
    await asyncio.sleep(timeout)
    return b"<html/>"


LAMBDA = lambda x, y: x ** 2 + y ** 2
RESULT = [x * 2 for x in range(20) if x % 3 == 0]
ENUM = {"open", "closed", "error"}
NESTED = {"config": {"retries": MAX_RETRIES, "verbose": True}}
'''

JAVASCRIPT = r'''
// Demonstrate common JavaScript/TypeScript patterns.
const MAX_RETRIES = 3;
const ratio = 0.6180339887498949;

function fibonacci(n) {
  if (n <= 1) return n;
  return fibonacci(n - 1) + fibonacci(n - 2);
}

function primes(limit) {
  const sieve = new Array(limit + 1).fill(true);
  sieve[0] = sieve[1] = false;
  for (let i = 2; i <= Math.sqrt(limit); i++) {
    if (sieve[i]) {
      for (let j = i * i; j <= limit; j += i) sieve[j] = false;
    }
  }
  return sieve.map((ok, i) => ok ? i : null).filter(x => x !== null);
}

class Stack {
  constructor(name) {
    this.name = name;
    this.items = [];
  }
  push(item) {
    this.items.push(item);
  }
  pop() {
    return this.items.pop();
  }
  get size() {
    return this.items.length;
  }
}

async function fetchJson(url) {
  const response = await fetch(url);
  const data = await response.json();
  return data;
}

const wordCounts = (text) =>
  text.toLowerCase().split(/\W+/).reduce((acc, w) => {
    acc[w] = (acc[w] || 0) + 1;
    return acc;
  }, {});

const { a, b, ...rest } = { a: 1, b: 2, c: 3 };
const doubled = [1, 2, 3].map((x) => x * 2);
const named = (callable, { retries = MAX_RETRIES } = {}) => callable();
'''

CPP = r'''
// C++ patterns: templates, STL, classes, smart pointers, lambdas.
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

template <typename T>
class Stack {
 public:
  explicit Stack(std::string name) : name_(std::move(name)) {}
  void push(const T& item) { items_.push_back(item); }
  T pop() {
    T v = items_.back();
    items_.pop_back();
    return v;
  }
  size_t Size() const { return items_.size(); }

 private:
  std::string name_;
  std::vector<T> items_;
};

int Fibonacci(int n) {
  if (n <= 1) return n;
  return Fibonacci(n - 1) + Fibonacci(n - 2);
}

std::vector<int> Primes(int limit) {
  std::vector<bool> sieve(limit + 1, true);
  sieve[0] = sieve[1] = false;
  for (int i = 2; i * i <= limit; ++i) {
    if (sieve[i]) {
      for (int j = i * i; j <= limit; j += i) sieve[j] = false;
    }
  }
  std::vector<int> primes;
  for (int i = 2; i <= limit; ++i) {
    if (sieve[i]) primes.push_back(i);
  }
  return primes;
}

int main() {
  auto stack = std::make_unique<Stack<int>>("main");
  stack->push(42);
  std::unordered_map<std::string, int> counts;
  counts[stack->name] = stack->Size();
  auto tripler = [](int x) -> int { return x * 3; };
  std::vector<int> xs{1, 2, 3};
  std::transform(xs.begin(), xs.end(), xs.begin(), tripler);
  return 0;
}
'''

RUST = r'''
// Rust patterns: ownership, Result, structs, iterators, match.
use std::collections::HashMap;
use std::error::Error;

const MAX_RETRIES: u32 = 3;

fn fibonacci(n: u32) -> u32 {
    match n {
        0 | 1 => n,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

fn primes(limit: usize) -> Vec<usize> {
    let mut sieve = vec![true; limit + 1];
    sieve[0] = false;
    sieve[1] = false;
    for i in 2..=((limit as f64).sqrt() as usize) {
        if sieve[i] {
            for j in (i * i..=limit).step_by(i) {
                sieve[j] = false;
            }
        }
    }
    sieve.iter().enumerate().filter(|&(_, ok)| *ok).map(|(i, _)| i).collect()
}

struct Stack<T> {
    name: String,
    items: Vec<T>,
}

impl<T> Stack<T> {
    fn new(name: &str) -> Self {
        Stack { name: name.to_string(), items: Vec::new() }
    }
    fn push(&mut self, item: T) {
        self.items.push(item);
    }
    fn pop(&mut self) -> Option<T> {
        self.items.pop()
    }
}

fn safe_divide(a: f64, b: f64) -> Result<f64, Box<dyn Error>> {
    if b == 0.0 {
        return Err("division by zero".into());
    }
    Ok(a / b)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut counts: HashMap<String, u32> = HashMap::new();
    let mut stack = Stack::new("main");
    stack.push(1u64);
    let n = stack.pop().unwrap_or_default();
    counts.insert("pops".to_string(), n as u32);
    println!("fibonacci(10) = {}", fibonacci(10));
    Ok(())
}
'''

GO = r'''
// Go patterns: goroutines, structs/interfaces, slices, error handling.
package main

import (
    "errors"
    "fmt"
    "sync"
)

const maxRetries = 3

func fibonacci(n int) int {
    if n <= 1 {
        return n
    }
    return fibonacci(n-1) + fibonacci(n-2)
}

type Stack struct {
    name  string
    items []int
}

func (s *Stack) Push(item int) {
    s.items = append(s.items, item)
}

func (s *Stack) Pop() (int, error) {
    if len(s.items) == 0 {
        return 0, errors.New("empty stack")
    }
    last := s.items[len(s.items)-1]
    s.items = s.items[:len(s.items)-1]
    return last, nil
}

func safeDivide(a, b float64) (float64, error) {
    if b == 0 {
        return 0, fmt.Errorf("division by zero")
    }
    return a / b, nil
}

func main() {
    var wg sync.WaitGroup
    ch := make(chan int, 4)
    wg.Add(2)
    go func() {
        defer wg.Done()
        ch <- fibonacci(10)
    }()
    go func() {
        defer wg.Done()
        ch <- len([]int{1, 2, 3})
    }()
    wg.Wait()
    close(ch)
    stack := Stack{name: "main"}
    stack.Push(fibonacci(7))
    fmt.Printf("sum=%d retries=%d\n", <-ch+<-ch, maxRetries)
}
'''

JSON_DATA = r'''
{
  "name": "exp-coder",
  "version": "0.1.0",
  "debug": true,
  "max_retries": 3,
  "ratio": 0.6180339887498949,
  "tags": ["model", "tokenizer", "benchmark"],
  "model": {
    "vocab_size": 50304,
    "hidden_size": 768,
    "num_layers": 12,
    "num_kv_heads": 12,
    "rope_theta": 10000.0
  },
  "nested": {"a": [1, 2, 3], "b": {"c": {"d": "deep string with \"escapes\" and \\n newline"}}},
  "unicode": "héllo wörld 日本語",
  "empty": [],
  "null_value": null
}
'''

YAML_DATA = r'''
# Deployment configuration.
app: exp-coder
port: 8080
replicas: 2
features:
  code_generation: true
  multimodal: false
servers:
  - host: 10.0.0.1
    role: primary
  - host: 10.0.0.2
    role: replica
training:
  learning_rate: 3.0e-4
  warmup_steps: 2000
  gradient_accumulation: 4
notes: "quoted string with colon: inside"
nested:
  very:
    deep:
      value: yes
unicode: "café au lait"
'''

MARKDOWN = r'''
# Exp-Coder Benchmark Notes

## Model
- **Decay**: decoder-only GPT-style transformer
- **RoPE**: rotary positional embeddings
- **GQA**: grouped-query attention

### Tokenizer
1. Byte-level BPE
2. `vocab.json` + `merges.txt`

Inline code: `python -m exp_coder.generate --prompt "hi"`.

```python
def add(a, b):
    "Docstring with *emphasis* and `backticks`."
    return a + b
```

> A blockquote with **bold** and *italic* text.

[links](https://example.com) end with punctuation: code, tokens, workers.

| header_one | header_two |
| ---------- | ---------- |
| cell a     | cell b     |

- unordered list item
- another item with `inline`

1. numbered item
2. second item
'''

SHELL = r'''
#!/usr/bin/env bash
# Demonstrates shell constructs for the tokenizer corpus.
set -euo pipefail

MAX_RETRIES=3
ratio=0.6180339887498949

log() {
  local level="$1"
  shift
  echo "[${level}] $*"
}

fib() {
  local n=$1
  if (( n <= 1 )); then
    echo "$n"
  else
    echo $(( $(fib $((n-1))) + $(fib $((n-2))) ))
  fi
}

for i in 1 2 3; do
  log "info" "iteration ${i} ${ratio}" | tr '[:lower:]' '[:upper:]'
done

files=$(find . -name "*.txt" -print0 | xargs -0 wc -l | tail -1)
printf 'lines: %s\n' "$files"

case "$1" in
  start) echo "starting" ;;
  stop)  echo "stopping" ;;
  *)     echo "unknown: ${1:-}" ;;
esac
'''

TEXT_DATA = r'''
Natural-language documentation and comments used to ensure the tokenizer
handles prose alongside code. This paragraph intentionally contains nested
parentheses (like this (and this)), numbers 42 and 1.5e3, and long
identifiers such as maximum_retry_count_across_all_nodes_backoff.

Notes for training:
- keep comments readable
- docstrings are important signal for code models
- whitespace, indentation, and newlines carry meaning in Python

Cientific notation: 1e10, 2.5e-4, 0x1F, 0b1010, 0o17, 1_000_000.
Punctuation: !@#$%^&*()_+-=[]{};:'",.<>/?\|
'''

CORPUS: Dict[str, str] = {
    "python": PYTHON,
    "javascript": JAVASCRIPT,
    "cpp": CPP,
    "rust": RUST,
    "go": GO,
    "json": JSON_DATA,
    "yaml": YAML_DATA,
    "markdown": MARKDOWN,
    "shell": SHELL,
    "text": TEXT_DATA,
}

EXT: Dict[str, str] = {
    "python": "py",
    "javascript": "js",
    "cpp": "cpp",
    "rust": "rs",
    "go": "go",
    "json": "json",
    "yaml": "yaml",
    "markdown": "md",
    "shell": "sh",
    "text": "txt",
}

# Deterministic identifier/pseudo-randomizing renames used to produce a few
# distinct variants per language (more tokens, still fully reproducible).
# Pairs are (find, replace) applied in order per variant.
VARIANTS: Dict[str, List[tuple]] = {
    "python": [
        [("fibonacci", "quicksort"), ("primes", "matrix_multiply"),
         ("Stack", "Deque"), ("safe_divide", "safe_modulo")],
        [("fibonacci", "dijkstra"), ("primes", "topological_sort"),
         ("word_frequencies", "char_bigrams"), ("LAMBDA", "COMPOSE")],
    ],
    "javascript": [
        [("fibonacci", "quicksort"), ("primes", "matrix_multiply"),
         ("Stack", "Deque"), ("wordCounts", "tokenCounts")],
        [("fibonacci", "dijkstra"), ("primes", "topologicalSort"),
         ("fetchJson", "uploadBlob"), ("doubled", "tripled")],
    ],
    "cpp": [
        [("Fibonacci", "QuickSort"), ("Primes", "MatrixMultiply"),
         ("Stack", "Deque"), ("tripler", "quadrupler")],
        [("Fibonacci", "Dijkstra"), ("Primes", "TopoSort"),
         ("Stack", "RingBuffer"), ("counts", "frequencies")],
    ],
    "rust": [
        [("fibonacci", "quicksort"), ("primes", "matrix_multiply"),
         ("Stack", "Deque"), ("safe_divide", "safe_modulo")],
        [("fibonacci", "dijkstra"), ("primes", "topological_sort"),
         ("Stack", "RingBuffer"), ("counts", "frequencies")],
    ],
    "go": [
        [("fibonacci", "quicksort"), ("Stack", "Deque"),
         ("safeDivide", "safeModulo"), ("maxRetries", "maxAttempts")],
        [("fibonacci", "dijkstra"), ("Stack", "RingBuffer"),
         ("wg", "semaphore"), ("ch", "queue")],
    ],
}


def _variants(lang: str, content: str) -> List[str]:
    versions = [content]
    for renames in VARIANTS.get(lang, []):
        text = content
        for find, repl in renames:
            text = text.replace(find, repl)
        versions.append(text)
    return versions


def generate(out: Path = OUT) -> Dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for lang, content in CORPUS.items():
        target = out / f"{lang}.{EXT[lang]}"
        target.write_text(content.lstrip("\n"), encoding="utf-8")
        counts[lang] = len(content)
        for i, variant in enumerate(_variants(lang, content), start=1):
            vtarget = out / f"{lang}_v{i}.{EXT[lang]}"
            vtarget.write_text(variant.lstrip("\n"), encoding="utf-8")
            counts[f"{lang}_v{i}"] = len(variant)
    (out / "README.txt").write_text(
        "Generated deterministically by benchmarks/make_code_corpus.py.\n",
        encoding="utf-8",
    )
    return counts


def corpus_files(out: Path = OUT) -> List[Path]:
    return sorted(
        p for p in out.glob("*")
        if p.is_file() and p.suffix != ".txt"
    )


if __name__ == "__main__":
    counts = generate()
    total = sum(counts.values())
    print(f"Generated {len(counts)} language files, {total} chars total")
    for lang, n in counts.items():
        print(f"  {lang}: {n} chars")