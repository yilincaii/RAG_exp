# FiQA RAG Context Optimization

RAG pipeline optimization for financial Q&A targeting finance students.

## Results

**Best Configuration:** Baseline (chunk=256, k=8, top_n=2)
- **Precision:** 43.95% (+7.6% vs alternatives)
- **F1 Score:** 53.26% (+3.9% vs Conservative)
- **NDCG@5:** 20.07% (best ranking quality)

## Key Findings

Precision-first design outperforms recall-maximizing for educational content.
Larger chunks + strict reranking = higher quality answers for students.

## Links

- [Full Summary](./summary.md)
- [Notebook](./notebooks/fiqa_rag_context_optimization.ipynb)
- [Competition Details](https://www.rapidfire.ai/university-program)

## Tech Stack

- **Dataset:** FiQA (BEIR)
- **Generator:** gpt-4o-mini
- **Embeddings:** all-MiniLM-L6-v2
- **Framework:** RapidFire AI
- **GitHub Repository Structure:**
```
fiqa-rag-optimization/
├── notebooks.ipynb
│   └── fiqa_rag_context_optimization.ipynb
├── logs/
│   └── rapidfire_experiment_log.txt
├── visualizations_and_screenshots/
│   ├── ic_ops_realtime_table.png
│   ├── rag_comprehensive_analysis.png
│   └── rag_preprocessing_log.png
├    └── metrics_analysis.png
│   
├── summary.md (this document)
└── README.md
```