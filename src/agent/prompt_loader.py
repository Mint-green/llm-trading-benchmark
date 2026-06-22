"""
Prompt Loader — 从 prompts/active/prompts.py 读取 prompt 配置。

用户编辑 prompts.py 中的变量即可改变 LLM 行为，无需改代码。
"""

from __future__ import annotations
import importlib.util
import sys
from pathlib import Path


# Prompt 文件路径
_PROMPT_FILE = Path(__file__).parent.parent.parent / "prompts" / "active" / "prompts.py"


def _load_module(filepath: Path):
    """动态加载 Python 模块"""
    spec = importlib.util.spec_from_file_location("prompts_config", str(filepath))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PromptLoader:
    """从 prompts.py 读取 prompt 配置"""

    def __init__(self, prompt_file: str | Path | None = None):
        self._file = Path(prompt_file) if prompt_file else _PROMPT_FILE
        self._module = None

    def _get_module(self):
        if self._module is None:
            if not self._file.exists():
                raise FileNotFoundError(f"Prompt file not found: {self._file}")
            self._module = _load_module(self._file)
        return self._module

    def load_system_prompt(self) -> str:
        return self._get_module().SYSTEM_PROMPT

    def load_instruction_template(self) -> str:
        return self._get_module().INSTRUCTION_TEMPLATE

    def load_final_round_instruction(self) -> str:
        return self._get_module().FINAL_ROUND_INSTRUCTION
