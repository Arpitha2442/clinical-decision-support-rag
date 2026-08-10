import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "folder"

import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(
    page_title="Clinical Decision Support RAG",
    page_icon="💊",
    layout="wide"
)

# Clear cached resources to guarantee loading updated logic
@st.cache_resource
def load_rag_engine():
    return RAGEngine()

rag = load_rag_engine()

# --- HEADER & DISCLAIMER ---
st.title("💊 Clinical Decision Support RAG System")
st.caption("Hybrid Search (BM25 + ChromaDB) with Indication Verification & Guardrails")

st.info(
    "⚠️ **Disclaimer:** This application is for decision support research based on pharmaceutical records. "
    "Always consult a licensed medical professional before administering medications."
)

st.divider()

# --- INPUT SECTION ---
query = st.text_input(
    "Enter clinical query or symptom:",
    value="What is the treatment for vertigo?",
    placeholder="e.g. uses of paracetamol, treatment for vertigo, medication that causes nausea..."
)

if st.button("Search Records", type="primary"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Retrieving records & verifying indications..."):
            # 1. Retrieve hybrid context
            contexts = rag.retrieve(query, top_k=3)
            
            # 2. Generate response with guardrails
            response = rag.generate_llm_response(query, contexts)
            
            # 3. Verify factual entailment via NLI model
            context_block = "\n".join(contexts)
            is_factually_supported = rag.verify_claim(response, context_block)

        # --- OUTPUT SECTION ---
        st.subheader(f"📋 Clinical Summary")
        st.markdown(response)

        st.divider()

        # NLI Verification Badge
        if "⚠️ No matching treatment found" not in response:
            if is_factually_supported:
                st.success("✅ **NLI Guardrail:** Result verified against database context.")
            else:
                st.warning("⚠️ **NLI Guardrail Warning:** Response contains unverified statements.")

        # Context Expander
        with st.expander("🔍 View Retrieved Raw Database Context (Top-3 Hybrid Hits)"):
            for idx, ctx in enumerate(contexts, 1):
                st.markdown(f"**Record #{idx}:**")
                st.code(ctx, language="text")