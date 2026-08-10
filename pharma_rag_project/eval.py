import pandas as pd
from rag_engine import RAGEngine

# 1. Initialize RAG Engine
print("Initializing RAG Engine for Realistic Performance Evaluation...")
engine = RAGEngine(csv_path="dataset.csv")

# 2. Benchmark Test Suite with Edge Cases (Harder Queries)
test_cases = [
    # Standard exact match
    {"query": "What are the side effects of Amoxicillin?", "expected_keyword": "Amoxicillin"},
    {"query": "What is Aceclofenac used for?", "expected_keyword": "Aceclofenac"},
    
    # Symptom/Intent matching
    {"query": "What medicine is used for treating cough?", "expected_keyword": "Cough"},
    {"query": "What treatment is recommended for diphtheria or pertussis?", "expected_keyword": "Diphtheria"},
    
    # Interaction / Contraindication queries
    {"query": "Can Ibuprofen cause upper abdominal pain or flatulence?", "expected_keyword": "Flatulence"},
    {"query": "Are there side effects like tremors or changes in serum aminotransferase levels?", "expected_keyword": "Tremors"},
    
    # Colloquial phrasing / Complex phrasing
    {"query": "Give me details on Althrocin Kid 125mg Tablet.", "expected_keyword": "Althrocin"},
    {"query": "What are the common stomach issues associated with pain relief capsules?", "expected_keyword": "Pain"},
    
    # Hard Edge Cases (May fail retrieval depending on dataset depth)
    {"query": "What is the exact pediatric dosage for severe renal impairment patients?", "expected_keyword": "Renal"},
    {"query": "Is Doliprane safe for pregnant women in their third trimester?", "expected_keyword": "Pregnancy"}
]

# 3. Run Evaluation Loop
results = []
correct_retrievals = 0
grounded_responses = 0

print("\n--- RUNNING 10-CASE BENCHMARK EVALUATION ---")
for test in test_cases:
    query = test["query"]
    expected = test["expected_keyword"]
    
    # Evaluate Retrieval Precision
    retrieved_docs = engine.retrieve(query, top_k=2)
    context_text = " ".join(retrieved_docs)
    retrieval_success = expected.lower() in context_text.lower()
    
    if retrieval_success:
        correct_retrievals += 1
        
    # Evaluate NLI Faithfulness
    top_doc = retrieved_docs[0]
    generated_answer = f"Based on clinical records: {top_doc}"
    is_grounded = engine.verify_claim(generated_answer, context_text)
    
    # Only count grounded if both retrieval succeeded and NLI passed
    if is_grounded and retrieval_success:
        grounded_responses += 1
        
    results.append({
        "Query": query[:40] + "...",
        "Retrieval": "✅ PASS" if retrieval_success else "❌ FAIL",
        "NLI Safety": "✅ GROUNDED" if (is_grounded and retrieval_success) else "⚠️ UNVERIFIED"
    })

# 4. Display Results Summary
total = len(test_cases)
retrieval_acc = (correct_retrievals / total) * 100
faithfulness_rate = (grounded_responses / total) * 100

df_results = pd.DataFrame(results)
print("\n" + df_results.to_string(index=False))

print("\n================ BENCHMARK METRICS SUMMARY ================")
print(f" Total Test Cases Evaluated : {total}")
print(f" Retrieval Precision@2      : {retrieval_acc:.1f}%  (Target: >90%)")
print(f" NLI Faithfulness Rate     : {faithfulness_rate:.1f}%  (Target: >90%)")
print("===========================================================")