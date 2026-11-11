"""Answer a question about one drug label, with citations that are checked against the label text.

    python -m clinical_docs_agent.ask metformin "What is the renal dose adjustment?"
    LLM_PROVIDER=anthropic python -m clinical_docs_agent.ask warfarin "Can it be taken with fluconazole?"
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import unicodedata
from dataclasses import dataclass

from .corpus import Label, load_label, select_sections
from .llm import LLMClient, LLMError, get_client
from .schemas import Citation, LabelAnswer

SYSTEM_PROMPT = """You answer questions about a prescription drug using ONLY the FDA label sections provided.

Rules:
- Use only facts stated in the provided sections. Do not use outside knowledge.
- Support every claim with a citation: the exact section heading and a short verbatim quote copied from that section.
- If the sections do not answer the question, set answerable to false, explain briefly, and return no citations.
- Write the answer in your own words, in 1-3 sentences. Do not paste long passages into the answer; quotes go in citations.
- Be precise with doses, thresholds and units.

Example of the expected shape (for a different drug):
{"answerable": true,
 "answer": "Reduce the dose to 5 mg daily when CrCl is below 30 mL/min.",
 "citations": [{"section": "Dosage and administration", "quote": "reduce the dose to 5 mg once daily in patients with CrCl below 30 mL/min"}]}"""


@dataclass
class CheckedCitation:
    citation: Citation
    verified: bool  # the quote really appears in the cited section


@dataclass
class Result:
    label: Label
    answer: LabelAnswer
    citations: list[CheckedCitation]
    provider: str
    model: str
    seconds: float


def _norm(text: str) -> str:
    # NFKC turns "m²" into "m2"; dropping spaces and punctuation makes "m 2" match too.
    return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKC", text).lower())


def verify_citation(label: Label, citation: Citation) -> bool:
    """A cheap faithfulness check: the quoted text must occur in the named section."""
    section = next((v for k, v in label.sections.items() if _norm(k) == _norm(citation.section)), None)
    quote = _norm(citation.quote)
    return bool(section and quote and quote in _norm(section))


def build_prompt(label: Label, question: str, sections: dict[str, str]) -> str:
    parts = [f"Drug label: {label.title} (label date {label.label_date})", ""]
    for name, text in sections.items():
        parts += [f"## {name}", text, ""]
    parts += [f"Question: {question}"]
    return "\n".join(parts)


def answer_question(drug: str, question: str, client: LLMClient | None = None, max_chars: int = 24_000) -> Result:
    client = client or get_client()
    label = load_label(drug)
    sections = select_sections(label, question, max_chars=max_chars)
    start = time.perf_counter()
    answer = client.generate(SYSTEM_PROMPT, build_prompt(label, question, sections), LabelAnswer)
    seconds = time.perf_counter() - start
    checked = [CheckedCitation(c, verify_citation(label, c)) for c in answer.citations]
    return Result(label, answer, checked, client.provider, client.model, seconds)


def format_result(r: Result) -> str:
    lines = [r.answer.answer.strip() or "(The model gave no explanation.)", ""]
    if not r.answer.answerable:
        lines.insert(0, "[Not answered by the label]")
    elif not r.citations:
        lines.insert(0, "[Warning: answer has no citations; treat as unsupported]")
    for i, c in enumerate(r.citations, 1):
        mark = "verified" if c.verified else "NOT FOUND in label"
        lines.append(f'[{i}] {c.citation.section}: "{c.citation.quote}" ({mark})')
    lines += [
        "",
        f"Source: {r.label.title}, label date {r.label.label_date}, {r.label.source}",
        f"Model: {r.provider}/{r.model} in {r.seconds:.1f}s",
        "Not medical advice; check the current official label.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ask a question about one FDA drug label.")
    p.add_argument("drug", help="label file stem in data/docs, e.g. metformin")
    p.add_argument("question")
    p.add_argument("--provider", help="ollama (default) or anthropic; overrides LLM_PROVIDER")
    p.add_argument("--model", help="model id; overrides LLM_MODEL")
    args = p.parse_args(argv)
    try:
        result = answer_question(args.drug, args.question, get_client(args.provider, args.model))
    except (FileNotFoundError, LLMError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(format_result(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
