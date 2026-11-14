"""Text helpers shared by ingestion and querying: normalisation and script detection."""

import re
import unicodedata
from dataclasses import dataclass

TELUGU_RE = re.compile(r"[ఀ-౿]")
LATIN_RE = re.compile(r"[A-Za-z]")
# Legacy (pre-Unicode) Telugu fonts such as Anu/Eenadu map glyphs onto Latin-1 and private-use
# code points, so their PDFs extract as gibberish like "¿=Ö Ôè·T". These ranges flag that.
LEGACY_RE = re.compile(r"[\u0080-ÿ-]")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass(frozen=True)
class ScriptProfile:
    chars: int
    telugu: float  # share of letters that are Telugu
    latin: float
    legacy: float  # share of suspicious legacy-font characters

    @property
    def needs_ocr(self) -> bool:
        """No usable text (scanned page) or text in a legacy non-Unicode font."""
        return self.chars < 20 or self.legacy > 0.15


def script_profile(text: str) -> ScriptProfile:
    te = len(TELUGU_RE.findall(text))
    la = len(LATIN_RE.findall(text))
    lg = len(LEGACY_RE.findall(text))
    letters = te + la + lg
    if letters == 0:
        return ScriptProfile(len(text.strip()), 0.0, 0.0, 0.0)
    return ScriptProfile(len(text.strip()), te / letters, la / letters, lg / letters)


# Common Telugu case/plural suffixes, longest first. A trigram index matches substrings, so a short
# query word finds its inflected forms ("రైతు" finds "రైతులకు"), but not the reverse. Stripping
# suffixes from query words closes that gap. This is a light stemmer, not a morphological analyser.
TELUGU_SUFFIXES = sorted(
    ["లందరికీ", "లన్నీ", "నుండి", "నుంచి", "యొక్క", "కోసం", "ల్లో", "లలో", "లకు", "లకి", "లతో",
     "లను", "లని", "లపై", "లు", "లో", "కు", "కి", "కీ", "ను", "ని", "తో", "గా", "పై", "మీద"],
    key=len, reverse=True,
)
MIN_STEM = 3  # code points; a trigram index cannot match anything shorter

_QUERY_SPLIT = re.compile(r"[\s.,;:!?\"'()\[\]{}<>|/\\।॥*+=_-]+")


def stem_variants(term: str) -> list[str]:
    """The term plus its suffix-stripped stem(s), each at least MIN_STEM code points long."""
    out = [term]
    if TELUGU_RE.search(term):
        stem = term
        for _ in range(2):  # e.g. రైతులకు -> రైతు, at most two suffixes
            for suf in TELUGU_SUFFIXES:
                if stem.endswith(suf) and len(stem) - len(suf) >= MIN_STEM:
                    stem = stem[: -len(suf)]
                    break
            else:
                break
        if stem != term:
            out.append(stem)
    elif len(term) > 4 and term.lower().endswith("s"):
        out.append(term[:-1])  # cards -> card
    return out


def query_terms(query: str) -> list[str]:
    """Search terms for the trigram index: split on whitespace/punctuation, expand stems, dedupe.

    Python's \\w is not used for splitting because it treats Telugu vowel signs (combining marks)
    as separators and would cut words apart.
    """
    terms: list[str] = []
    for word in _QUERY_SPLIT.split(query):
        if len(word) >= MIN_STEM:
            terms += stem_variants(word)
    return list(dict.fromkeys(terms))


def detect_language(text: str) -> str:
    """'te' if the text is mostly Telugu script, else 'en'. Romanised Telugu counts as 'en'."""
    return "te" if script_profile(text).telugu >= 0.3 else "en"
