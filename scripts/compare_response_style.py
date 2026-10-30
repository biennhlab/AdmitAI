"""Compare the committed and working-tree response prompts with fixed evidence."""

from __future__ import annotations

import json
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import settings
from src.generation.llm_client import LLMClient
from src.generation.prompts import SYSTEM_PROMPT
from src.generation.rag_chain import RAGChain


QUESTIONS = [
    "Học phí chương trình chuẩn năm học 2025–2026 là bao nhiêu?",
    "PTIT có những phương thức tuyển sinh nào?",
    "So sánh học phí chương trình chuẩn và chương trình chất lượng cao.",
]

CONTEXT = """<document id="1" title="Thông báo học phí" published_at="2025-08-01">
Năm học 2025–2026, học phí chương trình chuẩn là 29,6–37,6 triệu đồng mỗi năm tùy ngành. Học phí chương trình chất lượng cao là 49,2 triệu đồng mỗi năm.
</document>

<document id="2" title="Đề án tuyển sinh" published_at="2025-06-01">
Các phương thức tuyển sinh gồm: xét tuyển tài năng; xét tuyển dựa vào chứng chỉ quốc tế; xét tuyển dựa vào kết quả thi tốt nghiệp THPT; và xét tuyển kết hợp.
</document>"""


def committed_prompt() -> str:
    completed = subprocess.run(
        ["git", "show", "HEAD:src/generation/prompts.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    namespace: dict[str, object] = {}
    exec(compile(completed.stdout, "committed-prompts.py", "exec"), namespace)
    return str(namespace["SYSTEM_PROMPT"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--after-only",
        action="store_true",
        help="Run only the working-tree prompt when rechecking an iteration.",
    )
    args = parser.parse_args()
    client = LLMClient(
        api_key=settings.LLM_API_KEY,
        model=settings.LLM_MODEL,
        base_url=settings.LLM_BASE_URL,
        timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
        max_retries=settings.LLM_MAX_RETRIES,
        reasoning_effort=(
            "minimal"
            if "generativelanguage.googleapis.com" in settings.LLM_BASE_URL
            and settings.LLM_MODEL.casefold().startswith("gemma-4-")
            else None
        ),
    )
    prompts = {"after": SYSTEM_PROMPT.format(context=CONTEXT)}
    if not args.after_only:
        prompts = {
            "before": committed_prompt().format(context=CONTEXT),
            **prompts,
        }
    results = []
    for question in QUESTIONS:
        item = {"question": question}
        messages = [{"role": "user", "content": f"<user_question>\n{question}\n</user_question>"}]
        for label, prompt in prompts.items():
            generated = "".join(client.generate_stream(prompt, messages, temperature=0))
            item[label] = generated if label == "before" else RAGChain._clean_answer(generated)
        results.append(item)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
