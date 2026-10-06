"""Text-to-speech for PharmaRAG clinical summaries.

Speech runs entirely in the user's browser through the Web Speech API
(``window.speechSynthesis``), embedded via a small Streamlit HTML component.
No extra Python packages, API keys or network calls are needed, and Play /
Pause / Resume / Stop are natively supported.

This module only *reads* the already-generated summary; it never touches the
retrieval or NLI verification pipeline.
"""
from __future__ import annotations

import html
import json
import re
from functools import lru_cache
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

_TEMPLATE_PATH = Path(__file__).with_name("tts_player.html")
_PLAYER_HEIGHT = 76
_MAX_CHUNK_CHARS = 220

NO_DATA_SPOKEN = "no information is available in the retrieved records"

_NO_DATA_RE = re.compile(r"\[\s*NO[\s_-]*DATA\s*\]", re.IGNORECASE)
# "[NO DATA]" used as a leading flag before an explanatory sentence, e.g.
# "⚠️ [NO DATA] No relevant clinical records were found..."
_NO_DATA_PREFIX_RE = re.compile(r"^[^\w\[]*\[\s*NO[\s_-]*DATA\s*\]\s+(?=\w)", re.IGNORECASE | re.MULTILINE)
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # pictographs, emoticons, transport, symbols
    "\u2600-\u27BF"          # misc symbols & dingbats (⚠, ✅, ✔ ...)
    "\u2B00-\u2BFF"          # arrows / stars
    "\uFE0E\uFE0F\u200D"     # variation selectors, zero-width joiner
    "]+"
)


def summary_to_speech_text(summary: object) -> str:
    """Convert a Markdown/HTML clinical summary into clean, speakable English.

    Strips HTML tags, Markdown syntax and emoji, and rewrites ``[NO DATA]``
    markers so missing fields are announced as unavailable rather than skipped
    or read literally.
    """
    text = str(summary or "")

    # HTML -> plain text
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</\s*(p|div|li|h[1-6])\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)

    text = _EMOJI_RE.sub(" ", text)

    # Markdown blocks
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)          # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)           # links
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)  # headings
    text = re.sub(r"^\s*>+\s?", "", text, flags=re.MULTILINE)       # blockquotes
    text = re.sub(r"^\s*([-*_]\s*){3,}$", "", text, flags=re.MULTILINE)  # rules
    text = re.sub(r"^\s*[-*+•]\s+", "", text, flags=re.MULTILINE)   # bullets
    text = re.sub(r"^\s*\|?(\s*:?-{2,}:?\s*\|)+\s*$", "", text, flags=re.MULTILINE)  # table rules
    text = text.replace("|", ", ")

    # [NO DATA] handling (before bracket/emphasis cleanup)
    text = _NO_DATA_PREFIX_RE.sub("", text)
    text = _NO_DATA_RE.sub(NO_DATA_SPOKEN, text)
    text = re.sub(r"(?<![\w/])N/A(?![\w/])", "not available", text)

    # Inline emphasis / code
    text = re.sub(r"(\*{1,3}|_{1,3})(?=\S)(.+?)(?<=\S)\1", r"\2", text)
    text = re.sub(r"`+([^`]*)`+", r"\1", text)
    text = re.sub(r"[*#`~^]+", " ", text)
    text = re.sub(r"(?<!\w)_+|_+(?!\w)", " ", text)

    # Sentence-ify each line so field labels get a natural pause.
    sentences = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip(" ,;")
        if not line:
            continue
        if not re.search(r"[.!?:;]$", line):
            line += "."
        elif line.endswith(":"):
            line = line[:-1] + "."
        sentences.append(line)

    spoken = " ".join(sentences)
    spoken = re.sub(r"\s+([.,;:!?])", r"\1", spoken)
    spoken = re.sub(r"([.!?])\1+", r"\1", spoken)
    return spoken.strip()


def split_into_chunks(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    """Split speech into sentence-sized chunks.

    Some browsers (notably Chrome) silently stop long utterances after ~15 s,
    so the player queues short chunks instead of one long string.
    """
    chunks: list[str] = []
    buffer = ""
    for sentence in re.split(r"(?<=[.!?;])\s+", text):
        sentence = sentence.strip()
        while len(sentence) > max_chars:
            cut = sentence.rfind(",", 0, max_chars)
            if cut < max_chars // 2:
                cut = sentence.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.append(sentence[: cut + 1].strip())
            sentence = sentence[cut + 1 :].strip()
        if not sentence:
            continue
        if buffer and len(buffer) + len(sentence) + 1 > max_chars:
            chunks.append(buffer)
            buffer = sentence
        else:
            buffer = f"{buffer} {sentence}".strip()
    if buffer:
        chunks.append(buffer)
    return chunks


@lru_cache(maxsize=1)
def _load_template() -> str:
    return _TEMPLATE_PATH.read_text(encoding="utf-8")


def _build_player_html(title: str, emphasis: str | None, level: int, chunks: list[str]) -> str:
    tag = "h5" if level >= 5 else "h4"
    title_html = html.escape(title)
    plain_title = title
    if emphasis:
        title_html += f" <em>{html.escape(emphasis)}</em>"
        plain_title += f" {emphasis}"
    chunks_json = json.dumps(chunks).replace("</", "<\\/")
    return (
        _load_template()
        .replace("{{TAG}}", tag)
        .replace("{{TITLE_ATTR}}", html.escape(plain_title, quote=True))
        .replace("{{TITLE_HTML}}", title_html)
        .replace("{{CHUNKS_JSON}}", chunks_json)
    )


def render_summary_heading_with_tts(
    title: str,
    summary: object,
    *,
    emphasis: str | None = None,
    level: int = 4,
) -> None:
    """Render a summary heading with a speaker button that reads ``summary`` aloud."""
    try:
        chunks = split_into_chunks(summary_to_speech_text(summary))
        components.html(_build_player_html(title, emphasis, level, chunks), height=_PLAYER_HEIGHT)
    except Exception as exc:  # never let TTS break the clinical summary itself
        heading = "#" * (5 if level >= 5 else 4)
        st.markdown(f"{heading} {title}" + (f" *{emphasis}*" if emphasis else ""))
        st.error(f"🔇 Text-to-speech could not be initialised: {exc}")
