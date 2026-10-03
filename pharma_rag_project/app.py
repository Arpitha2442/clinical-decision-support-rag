import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rag_engine import RAGEngine
from prescription_scanner import (
    scan_prescription_image,
    format_rag_drug_card,
    ScanStatus,
)
from tts import render_summary_heading_with_tts

# Page Setup
st.set_page_config(
    page_title="PharmaRAG | Clinical Support",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme Interface Styling
st.markdown("""
<style>
    .stApp { background-color: #0F172A; color: #F8FAFC; }
    .hero-container {
        background: linear-gradient(135deg, #1E3A8A 0%, #0D9488 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        color: #FFFFFF;
        margin-bottom: 1.5rem;
    }
    .hero-title { font-size: 2.2rem; font-weight: 700; margin: 0; color: #FFFFFF !important; }
    .hero-subtitle { font-size: 1.05rem; margin-top: 0.4rem; color: #E2E8F0 !important; }
    .clinical-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        color: #F1F5F9 !important;
    }
    .rx-ocr-box {
        background-color: #0F2942;
        border: 1px solid #1E4976;
        border-left: 4px solid #38BDF8;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.95rem;
        color: #BAE6FD;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .rx-drug-card {
        background: linear-gradient(135deg, #1E293B 0%, #162032 100%);
        border: 1px solid #334155;
        border-top: 3px solid #0D9488;
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        color: #F1F5F9 !important;
    }
    .rx-badge-uncertain {
        background: #451A03;
        border: 1px solid #92400E;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        color: #FDE68A;
        margin-bottom: 1rem;
    }
    .rx-badge-empty {
        background: #1C1917;
        border: 1px solid #57534E;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        color: #A8A29E;
        margin-bottom: 1rem;
    }
    .rx-candidate-chip {
        display: inline-block;
        background: #164E63;
        border: 1px solid #0E7490;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        margin: 0.2rem;
        font-size: 0.85rem;
        color: #67E8F9;
    }
    .rx-conf-bar-wrap {
        background: #1E293B;
        border-radius: 6px;
        height: 10px;
        margin-top: 4px;
        overflow: hidden;
    }
    .stMarkdown, p, span, h1, h2, h3, h4, h5, h6 { color: #F8FAFC; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    return RAGEngine()

with st.spinner("⚡ Initializing PharmaRAG Neural Indexes..."):
    rag = load_engine()

# Sidebar Panel
with st.sidebar:
    st.image("https://img.icons8.com/color/96/pill.png", width=56)
    st.title("PharmaRAG Engine")
    st.caption("v2.5 | Clinical Intelligence System")
    st.divider()
    st.markdown("### ⚙️ System Status")
    dense_status = "🟢" if getattr(rag.retriever, "collection", None) is not None else "🟡 (unavailable — using sparse only)"
    st.success(f"{dense_status} **Dense Search:** ChromaDB + MiniLM") if dense_status.startswith("🟢") else st.warning(f"{dense_status} **Dense Search:** ChromaDB + MiniLM")
    bm25_status = "🟢" if getattr(rag.retriever, "bm25", None) is not None else "🟡 (unavailable — using keyword fallback)"
    st.success(f"{bm25_status} **Sparse Search:** BM25 Keywords") if bm25_status.startswith("🟢") else st.warning(f"{bm25_status} **Sparse Search:** BM25 Keywords")
    st.success("🟢 **Fusion Strategy:** RRF (k=60)")
    nli_status = "🟢" if getattr(rag.nli_judge, "model", None) is not None else "🟡 (unavailable — unverified fallback)"
    st.success(f"{nli_status} **Safety Layer:** NLI Cross-Encoder") if nli_status.startswith("🟢") else st.warning(f"{nli_status} **Safety Layer:** NLI Cross-Encoder")
    st.success("🟢 **Guardrail:** Indication vs. Side-Effect Filter")
    try:
        from prescription_scanner import _EASYOCR_AVAILABLE, _TESSERACT_AVAILABLE
        if _EASYOCR_AVAILABLE:
            st.success("🟢 **OCR Engine:** EasyOCR (deep learning, no binary needed)")
        elif _TESSERACT_AVAILABLE:
            st.warning("🟡 **OCR Engine:** Tesseract (binary must be on PATH)")
        else:
            st.error("🔴 **OCR Engine:** Unavailable — run: pip install easyocr")
    except Exception:
        st.warning("🟡 **OCR Engine:** Status check failed (torchvision mismatch)")
    st.divider()
    
    with st.expander("📊 Dataset Overview"):
        st.write(f"**Indexed Records:** {len(rag.passages)} Passages")
        st.write("**Data Format:** Standardized CSV")
        if st.button("🔄 Refresh Cache", use_container_width=True):
            st.cache_resource.clear()
            st.toast("Cache cleared successfully!", icon="🧹")

# Hero Banner
st.markdown("""
<div class="hero-container">
    <div class="hero-title">💊 PharmaRAG Decision Support System</div>
    <div class="hero-subtitle">Interactive Hybrid Retrieval Engine with Real-Time NLI Verification</div>
</div>
""", unsafe_allow_html=True)

st.warning("⚠️ **Clinical Disclaimer:** This application is built strictly for clinical research and decision support. Always consult a certified medical professional before administering treatments.")

# Main Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "🔍 Search Clinical Records",
    "📷 Prescription Scanner",
    "📊 Offline Model Comparison",
])

# TAB 1: Main Assistant
with tab1:
    st.subheader("🔍 Search Clinical Records")

    if "query_input" not in st.session_state:
        st.session_state.query_input = "uses of paracetamol"

    def update_query_from_chip():
        if st.session_state.selected_chip:
            st.session_state.query_input = st.session_state.selected_chip

    preset_queries = [
        "What is the treatment for High cholesterol?",
        "uses of paracetamol",
        "Medication that causes nausea",
        "What is the indication for Albendazole?"
    ]

    st.pills(
        label="⚡ Dynamic Sample Queries (Click to fill):",
        options=preset_queries,
        selection_mode="single",
        key="selected_chip",
        on_change=update_query_from_chip
    )

    col_search, col_btn = st.columns([4, 1])
    with col_search:
        user_query = st.text_input(
            label="Clinical Query",
            label_visibility="collapsed",
            key="query_input",
            placeholder="Search by drug name, condition, or contraindication..."
        )
    with col_btn:
        search_triggered = st.button("🚀 Search Records", type="primary", use_container_width=True)

    st.divider()

    if search_triggered or user_query:
        if not user_query.strip():
            st.warning("Please enter a query to run the search engine.")
        else:
            retrieved_docs = rag.retrieve(user_query)
            llm_summary = rag.generate_llm_response(user_query, retrieved_docs)
            
            context_text = "\n".join([str(d) for d in retrieved_docs])
            nli_result = rag.verify_claim(llm_summary, context_text)

            t_summary, t_nli, t_sources = st.tabs([
                "📋 Clinical Summary", 
                "🛡️ NLI Safety Audit", 
                "🔍 Retrieved Sources"
            ])

            with t_summary:
                render_summary_heading_with_tts(
                    "Generated Summary for:",
                    llm_summary,
                    emphasis=f"\"{user_query}\"",
                )
                st.markdown(f'<div class="clinical-card">{llm_summary}</div>', unsafe_allow_html=True)

            with t_nli:
                st.markdown("#### 🛡️ NLI Fact Verification Metrics")
                if nli_result.get("is_safe"):
                    st.success(f"✅ **{nli_result.get('status', 'PASSED')}:** The generated summary is verified and grounded in source records.")
                else:
                    st.warning(f"⚠️ **{nli_result.get('status', 'FLAGGED')}:** Statement cannot be fully grounded in retrieved data.")
                st.metric("Entailment Score", f"{nli_result.get('score', 0):.2f}")
                if "all_scores" in nli_result:
                    st.caption(f"NLI class scores: {nli_result['all_scores']}")

            with t_sources:
                st.markdown("#### 🔍 Retrieved Context Records")
                for idx, doc in enumerate(retrieved_docs, start=1):
                    with st.expander(f"Record #{idx} - {doc.get('name', 'Medication')}"):
                        st.json(doc)

# TAB 2: Prescription Scanner
with tab2:
    st.subheader("📷 AI-Powered Prescription & Packaging Scanner")
    st.markdown(
        "Upload a photo of a **handwritten prescription** or **medicine packaging** "
        "(PNG, JPG, JPEG). The system will extract text via OCR, identify medicine "
        "names, and retrieve clinical information from the dataset.\n\n"
        "> ⚕️ **Clinical Safety Notice:** Uncertain or unclear handwriting will always "
        "be flagged for manual verification. This system will never auto-confirm an "
        "ambiguous medicine name."
    )

    uploaded_file = st.file_uploader(
        label="Upload Prescription Image",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
        help="Supports PNG, JPG, and JPEG formats. Max recommended size: 10 MB.",
        key="rx_uploader",
    )

    if uploaded_file is not None:
        col_img, col_meta = st.columns([1, 1])
        with col_img:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        with col_meta:
            st.markdown(f"**Filename:** `{uploaded_file.name}`")
            st.markdown(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
            st.markdown(f"**Type:** `{uploaded_file.type}`")

        st.divider()

        scan_btn = st.button(
            "🔬 Scan & Analyse Prescription",
            type="primary",
            use_container_width=True,
            key="rx_scan_btn",
        )

        if scan_btn:
            with st.spinner("🔍 Running OCR and analysing prescription image..."):
                image_bytes = uploaded_file.getvalue()
                scan_result = scan_prescription_image(
                    image_bytes=image_bytes,
                    filename=uploaded_file.name,
                )
            st.session_state["rx_result"] = scan_result

    # ── Render results (persisted in session_state so re-runs keep them) ──
    if "rx_result" in st.session_state:
        result = st.session_state["rx_result"]

        # ── Status: backend / file errors ──────────────────────────────────
        if result.status == ScanStatus.INVALID_FILE:
            st.error(
                f"❌ **Invalid File:** {result.error_message}",
                icon="🚫",
            )

        elif result.status == ScanStatus.BACKEND_UNAVAILABLE:
            st.error("❌ **OCR Backend Unavailable**", icon="🔧")
            st.code(result.error_message, language="text")
            st.info(
                "Until Tesseract-OCR is installed, you can still use the "
                "**🔍 Search Clinical Records** tab to query medicines by name."
            )

        elif result.status == ScanStatus.OCR_EMPTY:
            st.markdown(
                f'<div class="rx-badge-empty">🔇 <b>No Text Detected</b><br>{result.error_message}</div>',
                unsafe_allow_html=True,
            )

        else:
            # ── OCR extracted text block ──────────────────────────────────
            backend = getattr(result, 'ocr_backend', 'unknown')
            st.markdown(
                f"#### 📄 Extracted Prescription Text "
                f"<span style='font-size:0.75rem; background:#134E4A; color:#5EEAD4; "
                f"border-radius:12px; padding:2px 10px; margin-left:8px; vertical-align:middle;'>"
                f"via {backend}</span>",
                unsafe_allow_html=True,
            )

            # Confidence meter
            if result.mean_confidence >= 0:
                conf = result.mean_confidence
                conf_color = (
                    "#22C55E" if conf >= 80
                    else "#F59E0B" if conf >= 60
                    else "#EF4444"
                )
                st.markdown(
                    f"""
                    <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
                      <span style="color:#94A3B8; font-size:0.85rem;">OCR Confidence</span>
                      <div class="rx-conf-bar-wrap" style="flex:1;">
                        <div style="height:10px; width:{conf}%; background:{conf_color}; border-radius:6px;"></div>
                      </div>
                      <span style="color:{conf_color}; font-weight:700; font-size:0.95rem;">{conf:.1f}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Uncertainty warning (must come before showing candidates)
            if result.status == ScanStatus.OCR_UNCERTAIN:
                st.markdown(
                    f'<div class="rx-badge-uncertain">'
                    f'⚠️ <b>Low Confidence Warning</b><br>'
                    f'{result.error_message}<br><br>'
                    f'<b>Please verify the identified medicine names below before relying on retrieved information.</b>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Raw OCR text
            st.markdown(
                f'<div class="rx-ocr-box">{result.extracted_text}</div>',
                unsafe_allow_html=True,
            )

            # Low-confidence tokens highlight
            if result.low_confidence_tokens:
                with st.expander("⚠️ Low-confidence OCR tokens (verify these)"):
                    st.markdown(
                        " ".join(
                            f'<span class="rx-candidate-chip" style="background:#3B1515; border-color:#7C2D12; color:#FCA5A5;">{t}</span>'
                            for t in result.low_confidence_tokens
                        ),
                        unsafe_allow_html=True,
                    )

            st.divider()

            # ── Medicine candidates ───────────────────────────────────────
            st.markdown("#### 💊 Identified Medicine Name Candidates")

            if not result.medicine_candidates:
                st.info(
                    "No medicine name candidates could be extracted from the text. "
                    "You can copy text from the box above and search manually in the "
                    "**🔍 Search Clinical Records** tab."
                )
            else:
                if result.status == ScanStatus.OCR_UNCERTAIN:
                    st.warning(
                        "⚠️ These candidates come from **low-confidence OCR output**. "
                        "Verify them manually before proceeding."
                    )

                # Display chips for all candidates
                chips_html = " ".join(
                    f'<span class="rx-candidate-chip">{c}</span>'
                    for c in result.medicine_candidates
                )
                st.markdown(chips_html, unsafe_allow_html=True)
                st.markdown("")

                # Let user select which candidate to look up
                selected_candidate = st.selectbox(
                    "Select a candidate to retrieve clinical information:",
                    options=result.medicine_candidates,
                    key="rx_selected_candidate",
                )

                lookup_btn = st.button(
                    f'📚 Retrieve Info for "{selected_candidate}"',
                    key="rx_lookup_btn",
                    type="secondary",
                )

                if lookup_btn and selected_candidate:
                    with st.spinner(
                        f'Searching RAG database for "{selected_candidate}"...'
                    ):
                        retrieved_docs = rag.retrieve(selected_candidate)
                        llm_summary = rag.generate_llm_response(
                            f"Clinical information about {selected_candidate}",
                            retrieved_docs,
                        )
                        context_text = "\n".join([str(d) for d in retrieved_docs])
                        nli_result = rag.verify_claim(llm_summary, context_text)
                    st.session_state["rx_lookup_docs"] = retrieved_docs
                    st.session_state["rx_lookup_summary"] = llm_summary
                    st.session_state["rx_lookup_nli"] = nli_result
                    st.session_state["rx_lookup_query"] = selected_candidate

            # ── Retrieved drug information ────────────────────────────────
            if "rx_lookup_docs" in st.session_state:
                lookup_query = st.session_state.get("rx_lookup_query", "")
                lookup_docs = st.session_state["rx_lookup_docs"]
                lookup_summary = st.session_state["rx_lookup_summary"]
                lookup_nli = st.session_state["rx_lookup_nli"]

                st.divider()
                st.markdown(f"#### 🏥 Clinical Information: *{lookup_query}*")

                rx_t1, rx_t2, rx_t3 = st.tabs([
                    "📋 Clinical Summary",
                    "🛡️ NLI Safety Audit",
                    "🔍 Retrieved Sources",
                ])

                with rx_t1:
                    if lookup_docs:
                        for doc in lookup_docs:
                            card_md = format_rag_drug_card(doc)
                            st.markdown(
                                f'<div class="rx-drug-card">{card_md}</div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.warning(
                            "No matching records found in the pharmaceutical dataset. "
                            "Try verifying the medicine name and searching again."
                        )

                    render_summary_heading_with_tts(
                        "🤖 AI-Generated Summary",
                        lookup_summary,
                        level=5,
                    )
                    st.markdown(
                        f'<div class="clinical-card">{lookup_summary}</div>',
                        unsafe_allow_html=True,
                    )

                with rx_t2:
                    st.markdown("#### 🛡️ NLI Fact Verification")
                    if lookup_nli.get("is_safe"):
                        st.success(
                            f"✅ **{lookup_nli.get('status', 'PASSED')}:** "
                            "The generated summary is verified and grounded in source records."
                        )
                    else:
                        st.warning(
                            f"⚠️ **{lookup_nli.get('status', 'FLAGGED')}:** "
                            "Statement cannot be fully grounded in retrieved data."
                        )
                    st.metric("Entailment Score", f"{lookup_nli.get('score', 0):.2f}")
                    if "all_scores" in lookup_nli:
                        st.caption(f"NLI class scores: {lookup_nli['all_scores']}")

                with rx_t3:
                    st.markdown("#### 🔍 Retrieved Context Records")
                    for idx, doc in enumerate(lookup_docs, start=1):
                        with st.expander(f"Record #{idx} — {doc.get('name', 'Medication')}"):
                            st.json(doc)

# TAB 3: Evaluation Benchmark Dashboard
with tab3:
    st.subheader("📊 Offline Benchmark & Architecture Comparisons")
    eval_file = "eval_results.csv"

    col_run, col_info = st.columns([1, 3])
    with col_run:
        run_now = st.button("▶️ Run Evaluation Now", use_container_width=True)
    with col_info:
        st.caption("Runs the live RAG engine against the benchmark set and scores it (BERTScore F1, NLI faithfulness, groundedness).")

    df_eval = None
    if run_now:
        with st.spinner("Running benchmark against the live RAG engine..."):
            from evaluate import run_evaluation
            df_eval = run_evaluation(rag_engine=rag, output_csv=eval_file)
        st.success("✅ Evaluation complete.")
    elif os.path.exists(eval_file):
        df_eval = pd.read_csv(eval_file)

    if df_eval is not None:
        st.dataframe(df_eval, use_container_width=True)
        
        st.markdown("---")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.set_theme(style="darkgrid")
        
        df_melted = df_eval.melt(
            id_vars="Model Variant", 
            value_vars=["BERTScore F1", "Faithfulness (%)", "Groundedness (%)"],
            var_name="Metric", 
            value_name="Score"
        )
        
        sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model Variant", ax=ax, palette="Set2")
        ax.set_ylim(0, 100)
        st.pyplot(fig)
    else:
        st.info("Click **▶️ Run Evaluation Now** above, or run `python evaluate.py` from the terminal, to generate benchmark metrics here.")