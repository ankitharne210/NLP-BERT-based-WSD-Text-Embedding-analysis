# WSD with BERT — BITS Pilani DSECLZG530 Assignment 2

A complete Streamlit application for **Contextual Word Sense Disambiguation using BERT**.

## Quick-start (step by step)

### Step 1 — Prerequisites
- Python 3.9, 3.10, or 3.11
- pip (Python package manager)
- ~4 GB free disk space (for BERT model weights)
- Internet connection (first run downloads model weights from Hugging Face)

---

### Step 2 — Clone / download the project

```bash
# If you have git:
git clone <your-repo-url>
cd nlp_wsd_app

# Or just place the folder anywhere and cd into it:
cd path/to/nlp_wsd_app
```

---

### Step 3 — Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

---

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on PyTorch:** If you have a GPU (CUDA), replace the `torch` line in requirements.txt with
> the CUDA-enabled wheel from https://pytorch.org/get-started/locally/.
> CPU-only is fine for this assignment.

---

### Step 5 — Download NLTK data (one-time)

```python
# Run in Python once:
import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")
```

Or simply launch the app — it downloads these automatically on first Task 2 visit.

---

### Step 6 — Launch the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

---

## Using the application — Task-by-task guide

### 🏠 Home (Data Input)
1. Open the **Home** page from the sidebar.
2. Choose one of:
   - **Upload File** — upload your own CSV or JSON (see format below)
   - **Fetch from URL** — paste a public CSV/JSON link (e.g. GitHub raw)
   - **Use Sample Data** — loads 5 words × 2 senses × 10 sentences (100 instances)
3. Confirm the dataset preview and sense distribution appear correctly.

**CSV format:**
```
word,sentence,sense,sense_label
bank,"She deposited her salary into the bank account.",financial_institution,bank.n.01
bank,"They camped on the grassy bank of the river.",river_bank,bank.n.09
```

---

### 📚 Task 1 — Dataset Preparation
1. Go to **Task 1** in the sidebar.
2. Review the assignment requirement checklist (green = satisfied).
3. Inspect per-word statistics and sense distribution chart.
4. Set the **test proportion** (default 20%) and click **Create Train/Test Split**.
5. Download train/test CSVs if needed.

---

### 🧠 Task 2 — WordNet Analysis
1. Go to **Task 2**.
2. Use the **Synset Browser** to explore all synsets for any word.
3. Expand each word's panel to see:
   - Sense definitions and WordNet examples
   - Semantic distinction explanation
   - Sample sentences from your dataset with the target word highlighted
4. Review the **Dataset Sense → WordNet Mapping Table** at the bottom.

---

### 🤖 Task 3 — BERT Embeddings
1. Go to **Task 3**.
2. Select the BERT variant (default: `bert-base-uncased`), hidden layer, and pooling strategy.
3. Click **Extract Embeddings** — this downloads the model on first run (~440 MB) and processes all sentences.
4. Once done, choose **PCA** or **t-SNE** and a word filter, then inspect the scatter plot.
5. Check the **Cosine Similarity** box plot to quantify sense separation.

> **Tip:** `distilbert-base-uncased` is ~40% faster if you need quick results.

---

### 🎯 Task 4 — WSD Classifier
1. Go to **Task 4**.
2. Select a classifier method (Nearest Centroid is a good baseline; SVM often performs best).
3. Click **Train & Evaluate** — metrics (Accuracy, Precision, Recall, F1) appear instantly.
4. Review per-sense and per-word breakdowns, correct/incorrect predictions.
5. Try the **Live WSD Demo** at the bottom — type any sentence and predict the sense in real time.

---

### 🔍 Task 5 — Error Analysis
1. Go to **Task 5**.
2. View the error overview: total errors, per-word breakdown, confusion pairs.
3. Browse categorised errors (similar senses, ambiguous context, domain-specific, short context).
4. Read the **BERT vs. static embeddings** comparison table and discussion.
5. Review the **Limitations** section for assignment discussion points.
6. Download the Markdown error report for inclusion in your submission.

---

## Project structure

```
nlp_wsd_app/
├── app.py                  # Streamlit entry point & navigation
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── pages/
    ├── __init__.py
    ├── page_home.py        # Home – data input
    ├── page_task1.py       # Task 1 – dataset prep & split
    ├── page_task2.py       # Task 2 – WordNet analysis
    ├── page_task3.py       # Task 3 – BERT embeddings & viz
    ├── page_task4.py       # Task 4 – WSD classifier & eval
    └── page_task5.py       # Task 5 – error analysis & discussion
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: transformers` | Run `pip install -r requirements.txt` |
| Model download very slow | Use `distilbert-base-uncased` — it's 250 MB vs 440 MB |
| `No module named 'nltk'` | `pip install nltk` |
| App not opening in browser | Navigate to http://localhost:8501 manually |
| CUDA out of memory | App runs on CPU by default — no GPU needed |
| Embeddings lost after page refresh | Re-extract in Task 3 — session state resets on reload |

---

## Assignment mapping

| App section | Assignment Task |
|---|---|
| Home – data input | General Requirement |
| Task 1 | Task 1 – WSD Dataset Preparation |
| Task 2 | Task 2 – WordNet Sense Analysis |
| Task 3 | Task 3 – BERT Contextual Representation |
| Task 4 | Task 4 – Automatic WSD System & Evaluation |
| Task 5 | Task 5 – Error Analysis & Comparative Discussion |
NLP-BERT-based-WSD-Text-Embedding-analysis
