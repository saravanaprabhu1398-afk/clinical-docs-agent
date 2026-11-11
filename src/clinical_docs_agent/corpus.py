"""Load FDA drug labels from data/docs/*.md and pick the sections relevant to a question."""
from __future__ import annotations

import math
import pathlib
import re
from dataclasses import dataclass, field

DOCS_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "docs"

_STOPWORDS = {
    "a", "an", "and", "are", "can", "do", "does", "for", "how", "i", "in", "is", "it", "of",
    "on", "or", "the", "to", "what", "when", "which", "who", "with", "should", "be", "any",
    "there", "this", "that", "my", "patient", "patients", "take", "taking", "drug", "label",
}


@dataclass
class Label:
    doc_id: str
    meta: dict[str, str]
    title: str
    sections: dict[str, str] = field(default_factory=dict)

    @property
    def generic_name(self) -> str:
        return self.meta.get("generic_name", self.doc_id)

    @property
    def label_date(self) -> str:
        d = self.meta.get("effective_date", "")
        return f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else d

    @property
    def source(self) -> str:
        return self.meta.get("source", "")


def available_drugs(docs_dir: pathlib.Path = DOCS_DIR) -> list[str]:
    return sorted(p.stem for p in docs_dir.glob("*.md"))


def load_label(drug: str, docs_dir: pathlib.Path = DOCS_DIR) -> Label:
    """Load one label by file stem, e.g. 'metformin'."""
    path = docs_dir / f"{drug.lower().strip()}.md"
    if not path.exists():
        raise FileNotFoundError(f"No label for '{drug}'. Try one of: {', '.join(available_drugs(docs_dir)[:10])}, ...")
    return parse_label(path.read_text(), doc_id=path.stem)


def parse_label(text: str, doc_id: str) -> Label:
    _, front, body = text.split("---", 2)
    meta = {}
    for line in front.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"')
    title_match = re.search(r"^# (.+)$", body, re.M)
    sections: dict[str, str] = {}
    for chunk in re.split(r"^## ", body, flags=re.M)[1:]:
        name, _, content = chunk.partition("\n")
        sections[name.strip()] = content.strip()
    return Label(doc_id=doc_id, meta=meta, title=title_match.group(1) if title_match else doc_id, sections=sections)


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS and len(w) > 2}


def select_sections(label: Label, question: str, max_chars: int = 24_000) -> dict[str, str]:
    """Rank sections by how many question words they contain and keep the best ones within max_chars.

    A deliberately simple stand-in for real retrieval (Week 2 replaces it with hybrid search),
    so a small local model with a short context window still sees the relevant text.
    """
    # The drug's own name appears in every section, so it carries no signal.
    q = _terms(question) - _terms(label.title) - _terms(label.generic_name)
    words_by_section = {n: re.findall(r"[a-z0-9]+", c.lower()) for n, c in label.sections.items()}
    n_sections = len(words_by_section)
    # IDF: a word found in few sections (e.g. "fluconazole") counts more than one found everywhere.
    idf = {t: math.log((n_sections + 1) / (0.5 + sum(t in set(ws) for ws in words_by_section.values()))) for t in q}

    def score(item: tuple[str, str]) -> float:
        name, _ = item
        words = words_by_section[name]
        hits = sum(idf[w] for w in words if w in idf)
        name_bonus = 5 * len(q & _terms(name))
        return name_bonus + hits / (1 + len(words) / 2000)

    chosen: dict[str, str] = {}
    used = 0
    for name, content in sorted(label.sections.items(), key=score, reverse=True):
        if used >= max_chars:
            break
        piece = content[: max_chars - used]
        chosen[name] = piece
        used += len(piece)
    return chosen
