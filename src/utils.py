"""
JudgeSense utilities — shared helper functions.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


def load_env() -> Dict[str, str]:
    """
    Load environment variables from .env file.
    
    Returns:
        Dictionary of environment variables.
    """
    load_dotenv()
    env = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "HF_TOKEN": os.getenv("HF_TOKEN"),
        "MISTRAL_API_KEY": os.getenv("MISTRAL_API_KEY"),
    }
    return env


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: Path, pretty: bool = True) -> None:
    """Save data to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2 if pretty else None)


def load_json(path: Path) -> Any:
    """Load data from a JSON or JSONL file.

    JSONL files (one JSON object per line) are read line-by-line and returned
    as a list.  Regular JSON files are returned as-is.
    """
    path = Path(path)
    with open(path, 'r', encoding='utf-8') as f:
        if path.suffix.lower() == '.jsonl':
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def load_jsonl(path: Path) -> list:
    """Load JSONL file (one JSON object per line)."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: list, path: Path) -> None:
    """Save list of objects as JSONL (one JSON per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def format_prompt(template: str, **kwargs) -> str:
    """
    Format a prompt template with kwargs.
    
    Example:
        template = "Question: {question}\\nAnswer the question."
        formatted = format_prompt(template, question="What is AI?")
    """
    return template.format(**kwargs)
