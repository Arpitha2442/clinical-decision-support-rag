import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rag_engine import RAGEngine

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
    st.success("🟢 **Dense Search:** ChromaDB Vector")
    st.success("🟢 **Sparse Search:** BM25 Keywords")
    st.success("🟢 **Fusion Strategy:** RRF (k=60)")
    st.success("🟢 **Safety Layer:** NLI Judge Active")
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
tab1, tab2 = st.tabs(["🔍 Search Clinical Records", "📊 Offline Model Comparison"])

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
            is_verified = rag.verify_claim(llm_summary, context_text)

            t_summary, t_nli, t_sources = st.tabs([
                "📋 Clinical Summary", 
                "🛡️ NLI Safety Audit", 
                "🔍 Retrieved Sources"
            ])

            with t_summary:
                st.markdown(f"#### Generated Summary for: *\"{user_query}\"*")
                st.markdown(f'<div class="clinical-card">{llm_summary}</div>', unsafe_allow_html=True)

            with t_nli:
                st.markdown("#### 🛡️ NLI Fact Verification Metrics")
                if is_verified:
                    st.success("✅ **PASSED:** The generated summary is verified and grounded in source records.")
                else:
                    st.warning("⚠️ **FLAGGED:** Statement cannot be fully grounded in retrieved data.")

            with t_sources:
                st.markdown("#### 🔍 Retrieved Context Records")
                for idx, doc in enumerate(retrieved_docs, start=1):
                    with st.expander(f"Record #{idx} - {doc.get('Nom', 'Medication')}"):
                        st.json(doc)

# TAB 2: Evaluation Benchmark Dashboard
with tab2:
    st.subheader("📊 Offline Benchmark & Architecture Comparisons")
    eval_file = "eval_results.csv"
    
    if os.path.exists(eval_file):
        df_eval = pd.read_csv(eval_file)
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
        st.info("Run `python evaluate.py` to pre-generate benchmark metric charts here.")