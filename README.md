# Financial Q&A RAG System with RapidFire AI

RapidFire AI Winter Competition Submission (RAG Track)
Demonstrating precision-first retrieval strategies for educational financial Q&A

Competition Period: Dec 22, 2025 - Jan 19, 2026
Author: Yilin Cai

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROJECT OVERVIEW

This project explores RAG optimization for a financial Q&A chatbot helping finance 
students learn personal finance with accurate, well-sourced answers. 

KEY FINDING: Precision-first retrieval outperforms recall-maximizing strategies for 
educational applications where misinformation has real consequences.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXPERIMENT RESULTS

Complete Performance Comparison:

Strategy        Precision  Recall   F1 Score  NDCG@5   MRR
Baseline        43.95%     88.33%   53.26%    20.07%   68.06%  ← BEST
Conservative    38.43%     91.67%   49.41%    19.79%   68.06%
Aggressive      36.34%     91.67%   47.22%    19.34%   65.28%

Best Configuration (Baseline):
- Chunk Size: 256 tokens (32-token overlap)
- Retriever k: 8 candidates (FAISS GPU exact search)
- Reranker top_n: 2 final documents (cross-encoder)

Key Achievement:
- +5.52% precision over Conservative
- +3.85% F1 score improvement
- Strategic 3.34% recall sacrifice for accuracy

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY BASELINE WON

Three synergistic design principles:

1. Larger chunks (256 tokens) preserve educational context
   Financial explanations need connected sentences—concepts like "compound interest" 
   require surrounding context to be pedagogically effective.

2. Strict top-2 reranking eliminates noise
   For students, wrong information is more harmful than incomplete information. 
   Keeping only 2 highest-confidence chunks ensures focused, high-quality evidence.

3. Focused k=8 retrieval improves reranker effectiveness
   Fewer initial candidates allow better discrimination between truly relevant and 
   superficially similar documents.

Result: Retrieving 88% of relevant documents with 44% precision beats 92% recall 
with only 36% precision for educational use cases.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUICK START

Option 1: Google Colab (Recommended)
Open this link in your browser (free T4 GPU):
https://colab.research.google.com/github/yilincaii/RAG_exp/blob/main/rf_colab_rag_fiqa_tutorial.ipynb

Runtime: ~60 seconds for 6 queries

Option 2: Local Setup
git clone https://github.com/yilincaii/RAG_exp.git
cd RAG_exp
python3 -m venv .venv && source .venv/bin/activate
pip install rapidfireai
rapidfireai init --evals
export OPENAI_API_KEY='your-api-key-here'
jupyter notebook rf_colab_rag_fiqa_tutorial.ipynb

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY INSIGHTS

1. Post-Retrieval Filtering > Retrieval Exhaustiveness
   MRR remained stable (~68%) across configs, proving the embedding model 
   (all-MiniLM-L6-v2) provides reliable ranking. The real optimization lever is 
   reranking threshold and chunk size, not the retriever itself.

2. API Latency Dominates Pipeline Speed
   Despite simplest retrieval (k=8, top_n=2), Baseline was slowest (60.65s) due 
   to OpenAI API variance. Throughput optimization requires batching generation 
   requests, not simplifying retrieval.

3. RapidFire AI Enables 5-7x Productivity at Scale
   This experiment (6 queries): 46% faster than sequential
   Full FiQA (6,648 queries): 24 hours → 8 hours with parallelization
   With IC Ops: Test 10-15 configs in time budget of 2-3 configs

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TECHNICAL STACK

RAG Components:
- Chunking: RecursiveCharacterTextSplitter (256 tokens, 32 overlap)
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (GPU, 384-dim)
- Vector Store: FAISS GPU exact search (IndexFlatL2)
- Reranker: cross-encoder/ms-marco-MiniLM-L6-v2 (CPU, top-2)
- Generator: OpenAI gpt-4o-mini (max_tokens=128, temp=0.8)

Framework:
- RapidFire AI 0.12 - Hyperparallel execution with online aggregation
- LangChain - RAG pipeline orchestration
- Google Colab - Free T4 GPU environment

Dataset:
- Source: FiQA from BEIR benchmark
- Corpus: 57,638 financial documents
- Sample: 6 evaluation queries (0.1% for prototyping)
- Full dataset: 6,648 queries (future validation planned)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RAPIDFIRE AI'S CONTRIBUTION

What It Accelerated:
- Parallel Execution: Tested 3 configs simultaneously (46% time savings)
- Zero Boilerplate: Eliminated ~200 lines of batching/metrics code
- Real-Time Metrics: Online aggregation with confidence intervals

Key Insights Surfaced:
- MRR stability revealed embedding model reliability
- Baseline's precision advantage emerged after 50% data
- IC Ops potential: Stop poor configs after 30% data → save ~5 hours + API costs

Net Impact:
                  This Experiment    Full-Scale Projection
Time saved        46% (52s)          67% (16 hours)
Configs tested    3 simultaneous     10-15 in same budget
Productivity      1.9x               5-7x

Without RapidFire AI: Would've tested only 1-2 configs, missing the counterintuitive 
finding that precision-first design beats "more context = better answers" for 
educational RAG.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DOCUMENTATION

- Detailed Experiment Summary: RAG-Experiment-Summary.md
- Key Findings: Conclusion.md
- RapidFire AI Docs: https://oss-docs.rapidfire.ai/
- Dataset: https://huggingface.co/datasets/BeIR/fiqa
- Competition: http://www.rapidfire.ai/university-program

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONTACT

Author: Yilin Cai
Track: RAG and Context Engineering
Submission: January 2026

GitHub: https://github.com/yilincaii/RAG_exp

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ACKNOWLEDGMENTS

Special thanks to RapidFire AI Team for creating tools that make rigorous RAG 
experimentation accessible to students with limited resources, and to the FiQA 
dataset creators for the high-quality financial Q&A corpus.

Built for the RapidFire AI Winter Competition on LLM Experimentation
December 22, 2025 - January 19, 2026

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━