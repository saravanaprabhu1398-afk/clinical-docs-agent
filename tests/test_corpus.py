"""Sanity checks on the collected corpus."""
import pathlib, re

DOCS = pathlib.Path(__file__).resolve().parents[1] / "data" / "docs"
REQUIRED = ["doc_id", "generic_name", "set_id", "effective_date", "source"]

def test_corpus_size():
    assert 50 <= len(list(DOCS.glob("*.md"))) <= 200

def test_front_matter_and_sections():
    for p in DOCS.glob("*.md"):
        text = p.read_text()
        fm = text.split("---")[1]
        for key in REQUIRED:
            assert re.search(rf"^{key}: ", fm, re.M), f"{p.name} missing {key}"
        assert "## Indications and usage" in text, f"{p.name} has no indications section"
