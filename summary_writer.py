"""Turns retrieved drug records into clear, conversational clinical summaries.

Used in two places:

* ``format_records_for_prompt`` gives the LLM a clean, readable view of the
  retrieved records (lists split into items, missing fields clearly marked).
* ``compose_summary`` is the deterministic, record-only writer used whenever
  the local LLM is unavailable, so users never see raw field dumps.

Every sentence built here comes only from retrieved record fields; nothing is
inferred or added. ``explain_terms`` returns *general* plain-language meanings
of medical words for display next to the summary (not part of the verified
summary text itself).
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

NO_DATA = "[NO DATA]"
# Exact prefix nli_judge.py recognises as a genuine "nothing found" answer.
NO_RESULTS_MESSAGE = "⚠️ [NO DATA] No relevant clinical records were found to answer your request."

_MISSING_TOKENS = {"", "nan", "none", "null", "n/a", "-", "--"}
_NO_DATA_RE = re.compile(r"\[\s*no[\s_-]*data\s*\]", re.IGNORECASE)

# Words after which a capitalised word continues the same item
# (e.g. "Treatment of Bacterial infections" stays one item).
_CONNECTORS = {
    "of", "and", "or", "with", "the", "for", "to", "in", "on", "from", "due",
    "by", "at", "a", "an", "as", "into", "after", "before", "during",
    "without", "against", "&", "associated", "related", "caused",
}

# Eponyms that should keep their capital letter in running text.
_KEEP_CAPITALISED = {
    "parkinson", "alzheimer", "crohn", "hodgkin", "cushing", "addison",
    "wilson", "paget", "raynaud", "meniere", "tourette", "graves",
    "hashimoto", "down", "gram", "kaposi", "zollinger", "ellison",
}

# Noun-style use phrases -> natural verb phrases ("Pain relief" -> "relieving pain").
_USE_PHRASE_RULES: Sequence[Tuple[str, str]] = (
    (r"^symptomatic (?:treatment|relief) of (.+)$", r"relieving the symptoms of \1"),
    (r"^(?:the )?treatment of (.+)$", r"treating \1"),
    (r"^(?:the )?prevention of (.+)$", r"preventing \1"),
    (r"^(?:the )?relief of (.+)$", r"relieving \1"),
    (r"^(?:the )?management of (.+)$", r"managing \1"),
    (r"^(?:the )?control of (.+)$", r"controlling \1"),
    (r"^(?:the )?reduction of (.+)$", r"reducing \1"),
    (r"^(.+?) relief$", r"relieving \1"),
    (r"^(.+?) prevention$", r"preventing \1"),
    (r"^(.+?) treatment$", r"treating \1"),
)

_INLINE_LIST_LIMIT_USES = 6
_INLINE_LIST_LIMIT_SIDE_EFFECTS = 8
_MAX_OTHER_MATCHES = 2

# (regex, display term, plain-language meaning). General definitions only.
MEDICAL_TERMS: Sequence[Tuple[str, str, str]] = (
    (r"contraindications?", "Contraindication", "a situation or condition in which a medicine should not be used"),
    (r"mucocutaneous candidiasis", "Mucocutaneous candidiasis", "a yeast (fungal) infection of the skin or the moist linings of the body, such as the mouth"),
    (r"candidiasis", "Candidiasis", "a yeast (fungal) infection"),
    (r"palpitations?", "Palpitations", "a noticeable fast, strong, or irregular heartbeat"),
    (r"hypertension", "Hypertension", "high blood pressure"),
    (r"hypotension", "Hypotension", "low blood pressure"),
    (r"hypoglyc(?:a)?emia", "Hypoglycemia", "low blood sugar"),
    (r"hyperglyc(?:a)?emia", "Hyperglycemia", "high blood sugar"),
    (r"dyslipid(?:a)?emia", "Dyslipidemia", "abnormal levels of fats, such as cholesterol, in the blood"),
    (r"dyspepsia", "Dyspepsia", "indigestion or discomfort in the upper stomach"),
    (r"hyperacidity", "Hyperacidity", "too much acid in the stomach"),
    (r"gastro-?(?:o)?esophageal reflux disease|\bgerd\b", "GERD", "stomach acid flowing back into the food pipe, causing heartburn"),
    (r"flatulence", "Flatulence", "excess gas in the stomach or intestines"),
    (r"insomnia", "Insomnia", "difficulty falling or staying asleep"),
    (r"somnolence", "Somnolence", "unusual sleepiness or drowsiness"),
    (r"pruritus", "Pruritus", "itching"),
    (r"urticaria", "Urticaria", "hives: itchy, raised patches on the skin"),
    (r"\bhives\b", "Hives", "itchy, raised, red patches on the skin"),
    (r"o?edema", "Edema", "swelling caused by fluid building up in the body"),
    (r"angioedema", "Angioedema", "swelling under the skin, often around the eyes, lips, or throat"),
    (r"anaphyla(?:xis|ctic)", "Anaphylaxis", "a severe, potentially life-threatening allergic reaction"),
    (r"alopecia", "Alopecia", "hair loss"),
    (r"an(?:a)?emia", "Anemia", "a low level of healthy red blood cells, which can cause tiredness"),
    (r"thrombocytopenia", "Thrombocytopenia", "a low platelet count, which can make bleeding or bruising easier"),
    (r"neutropenia", "Neutropenia", "a low level of a type of white blood cell that fights infection"),
    (r"vertigo", "Vertigo", "a spinning sensation or loss of balance"),
    (r"tachycardia", "Tachycardia", "a faster than normal heart rate"),
    (r"bradycardia", "Bradycardia", "a slower than normal heart rate"),
    (r"arrhythmias?", "Arrhythmia", "an irregular heartbeat"),
    (r"angina", "Angina", "chest pain caused by reduced blood flow to the heart"),
    (r"jaundice", "Jaundice", "yellowing of the skin or the whites of the eyes"),
    (r"rhinitis", "Rhinitis", "irritation inside the nose, causing a runny or blocked nose and sneezing"),
    (r"dysgeusia", "Dysgeusia", "a change in the sense of taste"),
    (r"\banorexia\b", "Anorexia", "loss of appetite (when listed as a medicine side effect)"),
    (r"myalgia", "Myalgia", "muscle pain"),
    (r"arthralgia", "Arthralgia", "joint pain"),
    (r"dermatitis", "Dermatitis", "inflamed, often red and itchy skin"),
    (r"erythema", "Erythema", "redness of the skin"),
    (r"photosensitivity", "Photosensitivity", "skin that is more sensitive to sunlight than usual"),
    (r"hyperkal(?:a)?emia", "Hyperkalemia", "a high level of potassium in the blood"),
    (r"hyponatr(?:a)?emia", "Hyponatremia", "a low level of sodium in the blood"),
    (r"hepatotoxicity", "Hepatotoxicity", "liver damage"),
    (r"hepatocellular insufficiency", "Hepatocellular insufficiency", "reduced liver function"),
    (r"nephrotoxicity", "Nephrotoxicity", "kidney damage"),
    (r"renal impairment", "Renal impairment", "reduced kidney function"),
    (r"creatinine clearance", "Creatinine clearance", "a measure of how well the kidneys filter waste from the blood"),
    (r"rhabdomyolysis", "Rhabdomyolysis", "a breakdown of muscle tissue that can harm the kidneys"),
    (r"(?:peripheral )?neuropathy", "Neuropathy", "nerve damage, often causing numbness or tingling in the hands or feet"),
    (r"extrapyramidal", "Extrapyramidal symptoms", "involuntary muscle movements, such as tremors or stiffness"),
    (r"hypothyroidism", "Hypothyroidism", "an underactive thyroid gland"),
    (r"hyperthyroidism", "Hyperthyroidism", "an overactive thyroid gland"),
    (r"helminthiasis", "Helminthiasis", "an infection with parasitic worms"),
    (r"ocular", "Ocular", "relating to the eyes"),
    (r"injection site reactions?", "Injection site reaction", "pain, redness, or swelling where the injection was given"),
)
_MAX_TERMS = 5


# ── Value helpers ─────────────────────────────────────────────────────────────

def clean_value(value: Any) -> str:
    """Normalise a record field to text; missing / [NO DATA] values become ''."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    if text.lower() in _MISSING_TOKENS or _NO_DATA_RE.fullmatch(text):
        return ""
    return text


def split_items(value: Any) -> List[str]:
    """Split a dataset list field into separate items.

    The dataset joins list items with spaces only ("Pain relief Treatment of
    Fever"), so a capitalised word starts a new item unless it follows a
    connector word ("of", "and", ...) or sits inside parentheses.
    """
    text = clean_value(value).rstrip(".")
    if not text:
        return []

    parts = re.split(r"\s*(?:[;\n•]|,(?![^()]*\)))\s*", text)
    items: List[str] = []
    for part in parts:
        current: List[str] = []
        depth = 0
        for word in part.split():
            previous = current[-1].lower().strip("(),") if current else ""
            starts_new_item = (
                bool(current)
                and depth == 0
                and word[:1].isupper()
                and len(word) > 1
                and previous not in _CONNECTORS
                and not current[-1].endswith("-")
            )
            if starts_new_item:
                items.append(" ".join(current))
                current = []
            current.append(word)
            depth = max(0, depth + word.count("(") - word.count(")"))
        if current:
            items.append(" ".join(current))

    seen, unique = set(), []
    for item in (i.strip(" .") for i in items):
        if item and item.lower() not in seen:
            seen.add(item.lower())
            unique.append(item)
    return unique


def _soft_lower(phrase: str) -> str:
    """Lower-case ordinary capitalised words for running text, keeping
    acronyms (HIV, COVID-19) and eponyms (Parkinson's) intact."""
    words = []
    for word in phrase.split():
        stem = re.match(r"[A-Za-z]+", word)
        stem_lower = stem.group(0).lower() if stem else ""
        if word[:1].isupper() and word[1:] == word[1:].lower() and stem_lower not in _KEEP_CAPITALISED:
            word = word.lower()
        words.append(word)
    return " ".join(words)


def _capitalise_first(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _to_use_phrase(item: str) -> str:
    phrase = _soft_lower(item)
    for pattern, replacement in _USE_PHRASE_RULES:
        new_phrase, count = re.subn(pattern, replacement, phrase, flags=re.IGNORECASE)
        if count:
            return new_phrase
    return phrase


def join_human(items: Sequence[str], conjunction: str = "and") -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {conjunction} {items[1]}"
    return f"{', '.join(items[:-1])}, {conjunction} {items[-1]}"


def _display_name(value: Any) -> str:
    name = clean_value(value)
    if name.isupper() and len(name) > 3:
        name = name.title()
    return name


def _composition_phrase(value: Any) -> str:
    composition = clean_value(value)
    if not composition:
        return ""
    parts = [re.sub(r"\s+", " ", p).strip() for p in composition.split("+")]
    return join_human([p for p in parts if p])


def _bullets(items: Iterable[str]) -> str:
    return "\n".join(f"- {_capitalise_first(i)}" for i in items)


# ── Natural-language summary (LLM-free fallback) ──────────────────────────────

def _intro_paragraph(name: str, uses: List[str], composition: str) -> str:
    subject = f"**{name}**" if name else "this medicine"
    if uses and len(uses) <= _INLINE_LIST_LIMIT_USES:
        intro = f"According to the available records, {subject} is used for {join_human(uses)}."
    elif uses:
        intro = (
            f"According to the available records, {subject} is used for the following:\n\n"
            f"{_bullets(uses)}"
        )
    elif name:
        intro = f"**{name}** was found in the available records."
    else:
        intro = "A matching medicine was found in the available records."

    if composition:
        sentence = f"It contains {composition}."
        # After a bullet list the sentence needs its own paragraph.
        intro = f"{intro}\n\n{sentence}" if "\n- " in intro else f"{intro} {sentence}"
    return intro


def _side_effects_paragraph(side_effects: List[str]) -> str:
    phrases = [_soft_lower(s) for s in side_effects]
    if len(phrases) <= _INLINE_LIST_LIMIT_SIDE_EFFECTS:
        return f"The recorded side effects include {join_human(phrases)}."
    return f"The recorded side effects include:\n\n{_bullets(phrases)}"


def _dosage_safety_paragraph(dosage: str, contraindications: List[str]) -> str:
    sentences = []
    if dosage:
        text = dosage.rstrip(".")
        if len(text) > 1 and text[0].isupper() and text[1].islower():
            text = text[0].lower() + text[1:]
        sentences.append(f"The recorded dosage is {text}.")
    if contraindications:
        conditions = join_human([_soft_lower(c) for c in contraindications])
        sentences.append(f"According to the records, it should not be used in cases of {conditions}.")
    return " ".join(sentences)


def _missing_paragraph(missing: List[str]) -> str:
    if not missing:
        return ""
    detail = "it cannot" if len(missing) == 1 else "these details cannot"
    return (
        f"The available records do not include {join_human(missing, 'or')} information "
        f"for this medicine, so {detail} be confirmed here."
    )


def _other_matches(primary_name: str, others: Sequence[Dict[str, Any]]) -> List[str]:
    lines, seen = [], {primary_name.lower()}
    for record in others:
        name = _display_name(record.get("name"))
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        uses = [_to_use_phrase(u) for u in split_items(record.get("indications"))[:2]]
        if uses:
            lines.append(f"- **{name}** is also listed for {join_human(uses)}.")
        else:
            lines.append(f"- **{name}** also matched your search, but its uses are not recorded.")
        if len(lines) >= _MAX_OTHER_MATCHES:
            break
    return lines


def compose_summary(query: str, contexts: Sequence[Dict[str, Any]]) -> str:
    """Write a friendly, record-grounded Markdown summary of retrieved records.

    ``query`` is accepted for API symmetry with the LLM generator; wording is
    driven purely by the retrieved fields so nothing unsupported is added.
    """
    records = [r for r in (contexts or []) if r]
    if not records:
        return NO_RESULTS_MESSAGE

    primary, others = records[0], records[1:]
    name = _display_name(primary.get("name"))
    uses = [_to_use_phrase(u) for u in split_items(primary.get("indications"))]
    side_effects = split_items(primary.get("side_effects"))
    contraindications = split_items(primary.get("contraindications"))
    dosage = clean_value(primary.get("dosage"))
    composition = _composition_phrase(primary.get("composition"))

    sections = [_intro_paragraph(name, uses, composition)]

    if side_effects:
        sections.append(f"##### Possible side effects\n\n{_side_effects_paragraph(side_effects)}")

    dosage_safety = _dosage_safety_paragraph(dosage, contraindications)
    if dosage_safety:
        sections.append(f"##### Dosage and safety\n\n{dosage_safety}")

    missing = [
        label for label, present in (
            ("usage", bool(uses)),
            ("dosage", bool(dosage)),
            ("contraindication", bool(contraindications)),
            ("side effect", bool(side_effects)),
        ) if not present
    ]
    if missing:
        sections.append(f"##### What the records don't cover\n\n{_missing_paragraph(missing)}")

    other_lines = _other_matches(name, others)
    if other_lines:
        sections.append("##### Other matching medicines\n\n" + "\n".join(other_lines))

    return "\n\n".join(sections)


# ── LLM prompt helpers ────────────────────────────────────────────────────────

def format_records_for_prompt(contexts: Sequence[Dict[str, Any]]) -> str:
    """Readable record listing for the LLM: list items separated by '; ' and
    missing fields shown as NOT RECORDED so the model never guesses them."""
    fields = (
        ("Medicine name", "name", False),
        ("Composition", "composition", False),
        ("Uses", "indications", True),
        ("Dosage", "dosage", False),
        ("Contraindications", "contraindications", True),
        ("Side effects", "side_effects", True),
    )
    blocks = []
    for number, record in enumerate(contexts, start=1):
        lines = [f"Record {number}"]
        for label, key, is_list in fields:
            if is_list:
                value = "; ".join(split_items(record.get(key)))
            else:
                value = clean_value(record.get(key))
            lines.append(f"- {label}: {value or 'NOT RECORDED'}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# ── Plain-language glossary (presentation only) ───────────────────────────────

def explain_terms(text: str, limit: int = _MAX_TERMS) -> List[Tuple[str, str]]:
    """Return (term, plain meaning) pairs for medical words found in ``text``."""
    haystack = (text or "").lower()
    found: List[Tuple[str, str]] = []
    seen_terms = set()
    for pattern, term, meaning in MEDICAL_TERMS:
        if term in seen_terms:
            continue
        if re.search(rf"(?<![a-z]){pattern}(?![a-z])", haystack):
            # Skip generic entries already covered by a more specific one.
            if any(term.lower() in t.lower() or t.lower() in term.lower() for t in seen_terms):
                continue
            seen_terms.add(term)
            found.append((term, meaning))
            if len(found) >= limit:
                break
    return found
