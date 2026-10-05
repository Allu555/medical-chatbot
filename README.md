# 🏥 Sorin-AI — Intelligent Clinical Medical Assistant

> **⚠️ Medical & Regulatory Disclaimer:**  
> This software is developed strictly for **educational, academic, and research purposes**. It is **not** certified as a medical device and cannot replace professional clinical consultation, diagnosis, or triage. In any real-life emergency, contact local emergency services immediately (911 / 112 / 999).

---

## 🌟 Overview: What is Sorin-AI?

**Sorin-AI** is a clinical-grade medical conversational assistant engineered to solve two fundamental problems in healthcare AI:
1. **Hallucination Prevention:** Instead of guessing medical facts, it grounds every answer in a verified medical knowledge base using **Retrieval-Augmented Generation (RAG)** and dense vector similarity search.
2. **Patient Safety & Emergency Triage:** Traditional keyword filters fail when patients describe emergencies in lay terms (e.g., *"feels like an elephant on my chest"*). Sorin-AI features a **Dual-Layer Gating Architecture** (fast deterministic regex + deep semantic LLM triage) that intercepts acute health crises before any chatbot response is generated.

---

## ⚡ Quick Start (Run in 3 Steps)

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone <your-repo-url>
cd chatrbot2

# Create and activate a Python virtual environment (Python 3.10+ recommended)
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux / Mac:
source venv/bin/activate
```

### 2. Install Dependencies & Configure Keys
```bash
# Install required packages
pip install -r requirements.txt

# Copy example environment configuration
copy .env.example .env     # Windows
cp .env.example .env       # Linux / Mac
```
Open `.env` in any text editor and paste your **Google Gemini API Key**:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash-latest
GRADIO_PORT=7860
```
*(Optional: If you want to use the local offline MedGemma 4B model, also add your `HF_TOKEN`).*

### 3. Launch the Application
```bash
# Option A: One-click script (Windows)
run_app.bat

# Option B: PowerShell
.\run_app.ps1

# Option C: Direct Python
python app.py
```
Open your browser and navigate to: **`http://localhost:7860`**

---

## 🏛️ System Architecture Explained Simply

```
                             Patient / User Query
                                      │
                                      ▼
        ┌───────────────────────────────────────────────────────────┐
        │        Layer 1: Dual-Layer Safety & Emergency Gate        │
        │  • Deterministic Regex: Immediate pattern check (0ms)     │
        │  • Semantic LLM Triage: Analyzes lay metaphors & intent   │
        └─────────────────────────────┬─────────────────────────────┘
                                      │
                     Is this an Emergency / Crisis?
                             /              \
                       YES  /                \  NO
                           ▼                  ▼
      ┌─────────────────────────┐   ┌───────────────────────────────────┐
      │   Emergency Protocol    │   │   Layer 2: Semantic RAG Search    │
      │ • Red Alert Banner      │   │ • Embed query via BGE-Small       │
      │ • Clear Action Guidance │   │ • Query FAISS in-memory index     │
      │ • Emergency Dispatch    │   │ • Retrieve top-3 medical snippets │
      └─────────────────────────┘   └─────────────────┬─────────────────┘
                                                      │
                                                      ▼
                                    ┌───────────────────────────────────┐
                                    │ Layer 3: Grounded Clinical LLM    │
                                    │ • Generates empathetic answer     │
                                    │ • Injects verified citations      │
                                    │ • Appends medical disclaimers     │
                                    └─────────────────┬─────────────────┘
                                                      │
                                                      ▼
                                    ┌───────────────────────────────────┐
                                    │ Layer 4: Multi-Theme Web UI       │
                                    │ • Gradio SaaS-style interface     │
                                    │ • Multi-session consultation log  │
                                    │ • Expandable source references    │
                                    └───────────────────────────────────┘
```

### How the Pipeline Works:
1. **Safety First (Layer 1):** The user message is tested by **both** regex patterns and the LLM triage classifier. If an acute symptom is detected (stroke signs, cardiac pain, poisoning, active bleeding, suicidal ideation), the system **halts** normal chat flow and immediately serves a red-banner emergency escalation.
2. **Knowledge Retrieval (Layer 2):** If the query is safe/routine, dense vector embeddings (`BAAI/bge-small-en-v1.5`) search the in-memory **FAISS** vector database containing curated medical literature.
3. **Clinical Generation (Layer 3):** The model receives the user query alongside the retrieved medical references. It formats answers into an intuitive structure: **Overview**, **Possible Causes**, and **Recommended Next Steps**.
4. **Interactive SaaS UI (Layer 4):** Delivered via a zero-scroll 100vh Gradio interface with 5 modern themes (Claude Warm Editorial, Espresso Dark, Midnight Cyan, Minimal Indigo, Nordic Slate).

---

## 📊 Final Evaluation & Confusion Matrix

The safety triage pipeline was evaluated on a comprehensive **400-case held-out clinical validation dataset** ([data/eval_heldout_400.json](data/eval_heldout_400.json)) spanning 20 clinical slices, 11 medical domains, and 120 adversarial robustness variants.

### A. Final 2×2 Confusion Matrix (Combined Dual-Layer Architecture)

| | Predicted Emergency (Critical) | Predicted Routine (Safe) | Total Ground Truth |
| :--- | :---: | :---: | :---: |
| **Actual Emergency (Critical)** | **$\text{TP} = 174$** *(98.86% Recall)* | **$\text{FN} = 2$** *(1.14% Miss Rate)* | **$176$** |
| **Actual Routine (Safe)** | **$\text{FP} = 12$** *(5.36% FPR)* | **$\text{TN} = 212$** *(94.64% Specificity)* | **$224$** |
| **Total Predicted** | **$186$** | **$214$** | **$N = 400$** |

*Overall Accuracy: **96.50%** (386 / 400 cases)*

---

### B. Headline Metrics Comparison: Baseline vs. Dual-Layer

| Metric | Regex-Only Baseline | LLM-Only Baseline | **Combined Dual-Layer (Our System)** |
| :--- | :---: | :---: | :---: |
| **Safety-Critical Recall (Sensitivity)** | 12.50% (22/176) | 98.30% (173/176) | **98.86% (174/176)** |
| **95% Wilson Confidence Interval** | [8.40%, 18.20%] | [95.11%, 99.42%] | **[95.95%, 99.69%]** |
| **False Positive Rate (FPR)** | 1.79% (4/224) | 3.57% (8/224) | **5.36% (12/224)** |
| **Specificity** | 98.21% (220/224) | 96.43% (216/224) | **94.64% (212/224)** |
| **Precision (PPV)** | 84.62% (22/26) | 95.58% (173/181) | **93.55% (174/186)** |
| **F1 Score** | 0.2178 | 0.9692 | **0.9613** |
| **Critical Emergencies Missed (FN)** | 154 dangerous misses | 3 misses | **Only 2 misses** *(Prompt injections)* |

> 📌 **Visual Artifacts:**  
> - Visual 2-Panel Confusion Matrix: [`artifacts/evaluation/heldout_validation_confusion_matrix.png`](artifacts/evaluation/heldout_validation_confusion_matrix.png)  
> - Publication 6-Panel Performance Dashboard: [`artifacts/evaluation/final_model_performance_metrics.png`](artifacts/evaluation/final_model_performance_metrics.png)  
> - Full Clinical Validation Report: [`artifacts/evaluation/heldout_validation_report.md`](artifacts/evaluation/heldout_validation_report.md)

---

## 🔬 Scientific Methodology & Academic Limitations

For academic review (e.g., MCA/MTech thesis and research paper submissions), note the following validation characteristics:
1. **In Silico Benchmark:** Due to clinical governance and privacy limitations, the 400-case held-out test suite was programmatically synthesized across 20 clinical slices. The 98.86% sensitivity reflects an *in silico* simulated stress-test rather than a prospective hospital deployment.
2. **Prevalence & PPV Calibration:** In real-world outpatient traffic, true emergencies represent 1% to 5% of queries. While our test-set precision is 93.55% (at 44% emergency prevalence), a 5.36% FPR in a 2% prevalence environment would produce a real-world PPV of ~27%, demonstrating the need for clinician threshold calibration to prevent alert fatigue.
3. **Edge Failure Modes:** The only 2 false negatives were caused by adversarial prompt-injection prefixes (`SECURITY BYPASS:`). Future iterations incorporate rule-based pre-triage sanitizers.

---

## 📂 Project Structure

```
chatrbot2/
├── app.py                             # Main Gradio application & multi-theme UI
├── run_app.bat                        # Windows 1-click launcher
├── run_app.ps1                        # PowerShell launcher
├── requirements.txt                   # Project Python dependencies
├── README.md                          # Full system documentation
├── PROJECT_PRESENTATION.md            # Viva presentation & slide deck guide
├── .env.example                       # Environment keys template
│
├── configs/                           # Model & pipeline configurations
│   ├── inference.yaml                 # Generation parameters & temperature
│   ├── rag.yaml                       # FAISS index and chunking settings
│   └── training.yaml                  # QLoRA fine-tuning hyperparameters
│
├── knowledge_base/                    # Verified medical reference knowledge
│   ├── documents/                     # Authoritative clinical JSON files
│   └── processed/chunks.json          # Pre-chunked text snippets
│
├── artifacts/                         # Generated models and validation artifacts
│   ├── faiss/                         # FAISS index (index.faiss + metadata.json)
│   ├── model/                         # Local fine-tuned model weights
│   └── evaluation/                    # Reports, confusion matrices & plots
│       ├── eval_heldout_results.json
│       ├── heldout_validation_confusion_matrix.png
│       ├── final_model_performance_metrics.png
│       └── heldout_validation_report.md
│
├── data/                              # Data pipelines & test suites
│   ├── eval_heldout_400.json          # 400-case held-out clinical validation set
│   ├── train/                         # Training dataset split
│   ├── validation/                    # Validation dataset split
│   └── test/                          # Dialogue test dataset split
│
├── evaluate_heldout.py                # Official validation runner script
├── generate_final_metrics_graph.py    # Generates 6-panel publication graph
│
├── notebooks/                         # Colab experimentation notebooks
│   └── ai_medical_chatbot_colab.ipynb
│
└── src/                               # Core modular package
    ├── data/                          # Dataset preprocessing & splitting
    │   ├── inspect_dataset.py
    │   ├── prepare_dataset.py
    │   └── split_dataset.py
    ├── rag/                           # Knowledge retrieval engine
    │   ├── build_index.py             # FAISS indexing script
    │   ├── knowledge_base.py          # Document loader & chunker
    │   └── retriever.py               # Vector similarity search
    ├── inference/                     # Inference & safety layers
    │   ├── gemini_client.py           # Gemini 1.5 Flash Lite client & triage
    │   ├── safety.py                  # Regex patterns & severity logic
    │   ├── model_loader.py            # Local model loading helper
    │   └── generate.py                # Text generation pipeline
    ├── training/                      # QLoRA fine-tuning modules
    │   ├── train_qlora.py             # PEFT/LoRA 4-bit trainer
    │   └── evaluate_training.py       # Training loss evaluation
    └── evaluation/                    # Base benchmark evaluations
        ├── evaluate_model.py
        └── evaluate_rag.py
```

---

## 🛠️ Step-by-Step Development & Commands

### 1. Rebuild the FAISS RAG Index
If you add new medical reference documents to `knowledge_base/documents/`:
```bash
python -m src.rag.build_index
```
*(Runs in ~3 seconds on CPU and creates `artifacts/faiss/index.faiss`).*

### 2. Run the 400-Case Held-Out Validation
To re-run the safety triage evaluation against the benchmark:
```bash
python evaluate_heldout.py
```

### 3. Generate the Publication Charts
To reproduce the 6-panel performance dashboard:
```bash
python generate_final_metrics_graph.py
```

### 4. Optional: Fine-Tune MedGemma 4B via QLoRA
If you have an NVIDIA GPU (RTX 3080/4090 or Google Colab T4/A100):
```bash
python -m src.training.train_qlora
```

---

## ❓ Frequently Asked Questions (FAQ)

**Q: Do I need a high-end GPU to run the chatbot?**  
*A:* **No.** The web application defaults to the cloud-hosted Google Gemini 1.5 Flash Lite API for clinical dialogue and uses CPU for the FAISS vector index. A GPU is only required if you choose to train or run the local MedGemma 4B model offline.

**Q: Why use both Regex and an LLM for safety?**  
*A:* Regex responds in **0 milliseconds** for classic keywords (`severe chest pain`, `heart attack`, `swallowed pills`) without consuming API quota. The LLM triage handles subtle metaphors (*"can't catch my breath"*, *"everything is spinning"*), providing defense-in-depth.

**Q: How do I change the theme in the UI?**  
*A:* Use the theme selector dropdown located at the top-right corner of the web interface. You can switch on-the-fly between Warm Editorial, Warm Espresso, Midnight Cyan, Minimal Indigo, and Nordic Clinical.

---

## 📜 License & Acknowledgments

- **License:** Educational and Academic Research Use.
- **Base Models:** Google Gemini 1.5 & Google DeepMind MedGemma.
- **RAG & Embeddings:** FAISS (Meta AI Research) & BAAI BGE (`BAAI/bge-small-en-v1.5`).
- **Frameworks:** Hugging Face `transformers`, `peft`, `bitsandbytes`, and Gradio.
