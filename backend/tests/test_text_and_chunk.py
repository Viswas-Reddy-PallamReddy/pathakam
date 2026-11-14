import pytest

from pathakam.ingest.chunk import chunk_pages
from pathakam.ingest.extract import Page
from pathakam.text import detect_language, normalize, query_terms, script_profile, stem_variants


def test_telugu_stems():
    assert stem_variants("రైతులకు") == ["రైతులకు", "రైతు"]
    assert stem_variants("పథకం") == ["పథకం"]
    assert stem_variants("కు") == ["కు"]  # never strip down below the trigram minimum
    assert stem_variants("cards") == ["cards", "card"]


def test_query_terms_split_keeps_vowel_signs():
    assert query_terms("రైతు భరోసా ఎవరికి?") == ["రైతు", "భరోసా", "ఎవరికి", "ఎవరి"]
    assert query_terms("a ? of") == []


def test_detect_language():
    assert detect_language("రైతు భరోసా ఎవరికి వస్తుంది?") == "te"
    assert detect_language("Who is eligible for Rythu Bharosa?") == "en"
    assert detect_language("Rythu Bharosa ki ఎవరు అర్హులు?") == "te"


def test_legacy_font_text_needs_ocr():
    # What a PDF in a pre-Unicode Telugu font typically extracts as.
    garbled = "¿=Ö Ôè·T ¹ø¹ |ü³Ty ®½ ¿=Ö Ôè·T ¹ø¹ |ü³Ty ®½"
    assert script_profile(garbled).needs_ocr
    assert not script_profile("రైతు భరోసా పథకం కింద ప్రతి రైతుకు సహాయం అందుతుంది").needs_ocr
    assert script_profile("   ").needs_ocr  # scanned page with no text layer


def test_normalize_keeps_paragraphs():
    assert normalize("a  b­ c \n\n\n\n d") == "a b c\n\nd"


def test_chunks_respect_size_overlap_and_pages():
    para = "వాక్యం ఒకటి. " * 30
    pages = [Page(1, f"{para}\n\n{para}"), Page(2, "short page")]
    chunks = chunk_pages(pages, max_chars=200, overlap_chars=40)
    assert all(len(c.text) <= 200 for c in chunks)
    assert {c.page for c in chunks} == {1, 2}
    assert chunks[-1].text == "short page"
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    # consecutive chunks on the same page share their boundary sentence
    first, second = chunks[0].text.splitlines(), chunks[1].text.splitlines()
    assert first[-1] == second[0]


def test_overlap_must_be_smaller():
    with pytest.raises(ValueError):
        chunk_pages([Page(1, "x")], max_chars=100, overlap_chars=100)
