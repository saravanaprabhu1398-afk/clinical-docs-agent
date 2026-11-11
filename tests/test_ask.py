"""Offline tests for label loading, section selection and citation checking (no model needed)."""
import pytest

from clinical_docs_agent.ask import answer_question, format_result, verify_citation
from clinical_docs_agent.corpus import load_label, select_sections
from clinical_docs_agent.llm import LLMError, get_client
from clinical_docs_agent.schemas import Citation, LabelAnswer


class FakeClient:
    provider, model = "fake", "fake-1"

    def __init__(self, answer: LabelAnswer):
        self.answer = answer
        self.prompt = None

    def generate(self, system, user, schema):
        self.prompt = user
        return schema.model_validate(self.answer.model_dump())


def test_load_label_reads_front_matter_and_sections():
    label = load_label("metformin")
    assert label.label_date.count("-") == 2
    assert label.source.startswith("https://dailymed")
    assert "Dosage and administration" in label.sections


def test_unknown_drug_raises():
    with pytest.raises(FileNotFoundError):
        load_label("not-a-real-drug")


def test_select_sections_prefers_relevant_and_respects_budget():
    label = load_label("metformin")
    chosen = select_sections(label, "What is the dosage in renal impairment?", max_chars=6000)
    assert next(iter(chosen)) == "Dosage and administration"
    assert sum(map(len, chosen.values())) <= 6000


def test_select_sections_finds_rare_term_not_drug_name():
    label = load_label("warfarin")
    chosen = select_sections(label, "Can warfarin be taken with fluconazole?", max_chars=12000)
    assert "Drug interactions" in list(chosen)[:2]


def test_verify_citation_tolerates_unicode_and_spacing():
    label = load_label("metformin")
    assert verify_citation(label, Citation(section="dosage AND administration", quote="eGFR below 30 mL/minute/1.73 m²"))


def test_verify_citation_accepts_real_quote_and_rejects_invented_one():
    label = load_label("metformin")
    real = Citation(section="Contraindications", quote="Severe renal impairment (eGFR below 30 mL/min/1.73 m2)")
    fake = Citation(section="Contraindications", quote="Safe in all patients with kidney failure")
    wrong_section = Citation(section="No such section", quote="Hypersensitivity to metformin")
    assert verify_citation(label, real)
    assert not verify_citation(label, fake)
    assert not verify_citation(label, wrong_section)


def test_answer_question_end_to_end_with_fake_model():
    client = FakeClient(LabelAnswer(
        answerable=True,
        answer="Contraindicated below eGFR 30.",
        citations=[Citation(section="Contraindications", quote="Hypersensitivity to metformin")],
    ))
    result = answer_question("metformin", "Who should not take it?", client=client)
    assert "Question: Who should not take it?" in client.prompt
    assert result.citations[0].verified
    assert "verified" in format_result(result)


def test_unknown_provider():
    with pytest.raises(LLMError):
        get_client("nope")
