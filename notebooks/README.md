# 📓 Notebooks & Interactive Exploration

This directory is designated for Jupyter notebooks used for interactive exploration, data inspection, RAG retrieval testing, and model evaluation.

## Suggested Notebook Workflows
- `01_dataset_exploration.ipynb`: Inspect distribution of medical questions, answers, and specialties.
- `02_rag_retrieval_testing.ipynb`: Test semantic search and reranking on custom medical queries.
- `03_model_evaluation.ipynb`: Interactive side-by-side comparison of baseline vs. fine-tuned responses.

## Setting Up the Kernel
Ensure you select the project's virtual environment:
```bash
# Register venv as Jupyter kernel (optional):
.\venv\Scripts\python.exe -m ipykernel install --user --name chatrbot2 --display-name "Python (chatrbot2)"
```
