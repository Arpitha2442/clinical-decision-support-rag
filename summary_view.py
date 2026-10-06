"""Streamlit presentation for generated clinical summaries."""
from __future__ import annotations

import html
import re

import streamlit as st

from summary_writer import explain_terms

_NO_RESULTS_RE = re.compile(r"^\W*\[\s*NO[\s_-]*DATA\s*\]\s*", re.IGNORECASE)

SUMMARY_DISCLAIMER = (
    "This summary reflects only the retrieved records and is not medical advice. "
    "Please confirm any treatment decision with a doctor or pharmacist."
)
NO_RESULTS_HINT = "Try searching with a different medicine name, condition, or symptom."


def is_no_results(summary: str) -> bool:
    return bool(_NO_RESULTS_RE.match(summary or ""))


def render_clinical_summary(summary: str, *, key: str) -> None:
    """Render a summary inside the dark clinical card with a plain-language
    glossary and the medical disclaimer. ``key`` must start with
    ``summary_card`` so the card styles in app.py apply."""
    with st.container(key=key):
        if is_no_results(summary):
            message = _NO_RESULTS_RE.sub("", summary).strip()
            st.markdown(f"{message} {NO_RESULTS_HINT}")
        else:
            st.markdown(summary)

            terms = explain_terms(summary)
            if terms:
                items = "".join(
                    f"<li><b>{html.escape(term)}</b>: {html.escape(meaning)}</li>"
                    for term, meaning in terms
                )
                st.markdown(
                    '<div class="summary-terms">'
                    '<div class="summary-terms-title">In simpler terms</div>'
                    f"<ul>{items}</ul></div>",
                    unsafe_allow_html=True,
                )

        st.markdown(
            f'<div class="summary-note">ℹ️ {html.escape(SUMMARY_DISCLAIMER)}</div>',
            unsafe_allow_html=True,
        )
