"""
Task 4 – Automatic WSD System & Quantitative Evaluation
Nearest-centroid, cosine-NN, and classical sklearn classifiers.
"""

import streamlit as st
import numpy as np
import pandas as pd


def desc(text: str):
    st.markdown(
        f'<div style="background:#f1f5f9;border-left:3px solid #4fc3f7;'
        f'border-radius:6px;padding:0.7rem 1rem;margin:-0.4rem 0 1rem;'
        f'font-size:0.87rem;color:#334155;">{text}</div>',
        unsafe_allow_html=True,
    )


def render():
    st.markdown("""
    <div class="task-header">
      <h2>🎯 Task 4 – WSD Classifier & Quantitative Evaluation</h2>
      <p>Train a WSD method on BERT embeddings and measure its accuracy on the held-out test set.</p>
    </div>
    """, unsafe_allow_html=True)

    desc(
        "<b>Purpose:</b> This is the core disambiguation task. A WSD classifier takes the BERT embedding of a target "
        "word in a new, unseen sentence and predicts which sense is intended. "
        "The classifier is trained on the training split (from Task 1) and evaluated on the test split it has never seen. "
        "Five methods are available, ranging from a simple geometry-based baseline (Nearest Centroid) "
        "to more expressive machine-learning classifiers (SVM, Logistic Regression). "
        "The evaluation reports standard NLP metrics — Accuracy, Precision, Recall, and F1-score — "
        "both overall and broken down by sense and by word, as required by the assignment."
    )

    if "bert_embeddings" not in st.session_state:
        st.markdown('<div class="warn-box">⚠️ No BERT embeddings found. Complete <b>Task 3</b> first.</div>', unsafe_allow_html=True)
        return
    if "train_df" not in st.session_state:
        st.markdown('<div class="warn-box">⚠️ No train/test split found. Run <b>Task 1 → Train/Test Split</b> first.</div>', unsafe_allow_html=True)
        return

    df       = st.session_state["dataset"]
    train_df = st.session_state["train_df"]
    test_df  = st.session_state["test_df"]
    all_emb  = np.array(st.session_state["bert_embeddings"])

    sentence_to_idx = {row["sentence"]: i for i, (_, row) in enumerate(df.iterrows())}

    def get_embeddings(subset):
        indices = [sentence_to_idx.get(s, -1) for s in subset["sentence"]]
        valid   = [(i, idx) for i, idx in enumerate(indices) if idx >= 0]
        rows    = [i for i, _ in valid]
        emb_idx = [idx for _, idx in valid]
        return all_emb[emb_idx], subset.iloc[rows].reset_index(drop=True)

    train_emb, train_aligned = get_embeddings(train_df)
    test_emb,  test_aligned  = get_embeddings(test_df)

    if len(train_emb) == 0 or len(test_emb) == 0:
        st.error("Could not align embeddings with train/test sentences. Ensure the dataset and embeddings were generated in the same session.")
        return

    # ── Classifier selection ──────────────────────────────────────────────────
    st.markdown("### ⚙️ Classifier Selection")
    desc(
        "Choose the WSD method to train. All methods use the BERT embeddings from Task 3 as input features. "
        "<b>Nearest Centroid</b> computes the mean embedding vector for each sense in the training set (the centroid), "
        "then assigns test instances to the nearest centroid — a transparent, interpretable baseline. "
        "<b>Cosine 1-NN</b> finds the single most similar training instance to each test instance using cosine distance. "
        "<b>Logistic Regression</b> learns a linear decision boundary in embedding space. "
        "<b>SVM (RBF kernel)</b> finds the maximum-margin boundary and handles non-linear separability. "
        "<b>KNN (k=5)</b> uses a majority vote among the 5 most similar training examples. "
        "SVM and Logistic Regression typically perform best; Nearest Centroid is the most interpretable."
    )
    clf_choice = st.selectbox(
        "WSD method",
        [
            "Nearest Centroid (per-sense mean)",
            "Cosine Nearest Neighbour (1-NN)",
            "Logistic Regression",
            "SVM (RBF kernel)",
            "K-Nearest Neighbours (k=5)",
        ],
    )

    if st.button("🏋️ Train & Evaluate"):
        _train_and_evaluate(clf_choice, train_emb, train_aligned, test_emb, test_aligned, df)
    elif "classifier_results" in st.session_state:
        st.markdown('<div class="info-box">ℹ️ Results from the last run are shown below. Select a different classifier and re-run to compare.</div>', unsafe_allow_html=True)
        _display_results(st.session_state["classifier_results"]["predictions"], df)

    if "classifier_results" in st.session_state:
        st.divider()
        _live_demo(train_aligned, train_emb, clf_choice)


# ── Training & evaluation ─────────────────────────────────────────────────────

def _train_and_evaluate(clf_choice, train_emb, train_aligned, test_emb, test_aligned, df):
    try:
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        st.error("Install: pip install scikit-learn")
        return

    y_train = train_aligned["sense"].tolist()
    y_test  = test_aligned["sense"].tolist()

    with st.spinner("Training …"):
        if clf_choice == "Nearest Centroid (per-sense mean)":
            clf = NearestCentroid(); clf.fit(train_emb, y_train); y_pred = clf.predict(test_emb)
        elif clf_choice == "Cosine Nearest Neighbour (1-NN)":
            sims = cosine_similarity(test_emb, train_emb)
            y_pred = [y_train[i] for i in np.argmax(sims, axis=1)]
        elif clf_choice == "Logistic Regression":
            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42); clf.fit(train_emb, y_train); y_pred = clf.predict(test_emb)
        elif clf_choice == "SVM (RBF kernel)":
            clf = SVC(kernel="rbf", C=1.0, random_state=42); clf.fit(train_emb, y_train); y_pred = clf.predict(test_emb)
        else:
            clf = KNeighborsClassifier(n_neighbors=5, metric="cosine"); clf.fit(train_emb, y_train); y_pred = clf.predict(test_emb)

    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    preds = test_aligned.copy()
    preds["predicted"] = y_pred
    preds["correct"]   = [p == t for p, t in zip(y_pred, y_test)]

    st.session_state["classifier_results"] = {
        "accuracy": acc, "report": report, "predictions": preds, "method": clf_choice,
    }
    _display_results(preds, df)


def _display_results(preds, df):
    results = st.session_state["classifier_results"]
    acc     = results["accuracy"]
    report  = results["report"]

    # ── Overall metrics ───────────────────────────────────────────────────────
    st.markdown(f"### 📊 Evaluation Results — *{results['method']}*")
    desc(
        "These four metrics give a complete picture of WSD performance on the held-out test set. "
        "<b>Accuracy</b> is the fraction of test instances correctly classified. "
        "<b>Precision</b> (macro-averaged) measures how often the predicted sense is correct. "
        "<b>Recall</b> (macro-averaged) measures how well the classifier finds all instances of each sense. "
        "<b>F1-score</b> is the harmonic mean of precision and recall — the primary metric for imbalanced WSD tasks. "
        "Macro averaging treats every sense equally regardless of frequency, which is appropriate for WSD evaluation."
    )
    c1, c2, c3, c4 = st.columns(4)
    macro = report.get("macro avg", {})
    c1.metric("Accuracy",  f"{acc*100:.1f}%")
    c2.metric("Precision", f"{macro.get('precision', 0)*100:.1f}%")
    c3.metric("Recall",    f"{macro.get('recall',    0)*100:.1f}%")
    c4.metric("F1-score",  f"{macro.get('f1-score',  0)*100:.1f}%")

    # ── Per-class table ───────────────────────────────────────────────────────
    st.markdown("#### Per-sense Classification Report")
    desc(
        "This table breaks down the evaluation metrics for each individual sense label. "
        "It reveals which senses the classifier handles well and which it struggles with. "
        "<b>Support</b> is the number of test instances for that sense. "
        "Low precision for a sense means the classifier is over-predicting it (false positives). "
        "Low recall means the classifier is missing instances of that sense (false negatives). "
        "This per-sense view is required by the assignment."
    )
    class_rows = []
    for label, vals in report.items():
        if label in ("accuracy", "macro avg", "weighted avg") or not isinstance(vals, dict):
            continue
        class_rows.append({"Sense": label, "Precision": f"{vals.get('precision',0):.3f}",
                            "Recall": f"{vals.get('recall',0):.3f}",
                            "F1": f"{vals.get('f1-score',0):.3f}",
                            "Support": int(vals.get("support", 0))})
    if class_rows:
        st.dataframe(pd.DataFrame(class_rows), use_container_width=True)

    # ── Per-word accuracy ─────────────────────────────────────────────────────
    st.markdown("#### Per-word Accuracy")
    desc(
        "This table shows how well the classifier performs on each ambiguous word separately. "
        "Some words have senses that are very distinct in context (easy for BERT) while others "
        "have subtly overlapping usages (harder). Per-word accuracy helps identify which words "
        "contribute most to errors, guiding further data collection or model tuning."
    )
    word_rows = []
    for word in sorted(preds["word"].unique()):
        sub = preds[preds["word"] == word]
        wa  = sub["correct"].sum() / len(sub)
        word_rows.append({"Word": word, "Correct": sub["correct"].sum(), "Total": len(sub), "Accuracy": f"{wa*100:.1f}%"})
    st.dataframe(pd.DataFrame(word_rows), use_container_width=True)

    # ── Correct predictions sample ────────────────────────────────────────────
    st.markdown("#### ✅ Sample Correct Predictions")
    desc(
        "These are examples where the classifier predicted the right sense. "
        "Examining correct predictions alongside the sentence helps you understand what contextual cues "
        "the model is implicitly relying on — strong, unambiguous context words that clearly signal one sense."
    )
    for _, r in preds[preds["correct"]].head(5).iterrows():
        st.markdown(f"> **{r['word']}** | Gold: `{r['sense']}` | Pred: `{r['predicted']}`  \n> _{r['sentence']}_")

    # ── Incorrect predictions sample ─────────────────────────────────────────
    st.markdown("#### ❌ Sample Incorrect Predictions")
    desc(
        "These are examples where the classifier predicted the wrong sense. "
        "Reading these cases carefully is the foundation of the error analysis in Task 5 — "
        "you can identify patterns such as genuinely ambiguous sentences, misleading context words, "
        "or senses that are semantically very close to each other."
    )
    wrong = preds[~preds["correct"]].head(5)
    if len(wrong) == 0:
        st.success("No incorrect predictions on the test set!")
    else:
        for _, r in wrong.iterrows():
            st.markdown(f"> **{r['word']}** | Gold: `{r['sense']}` | Pred: `{r['predicted']}`  \n> _{r['sentence']}_")

    st.download_button("⬇️ Download predictions CSV", preds.to_csv(index=False).encode(), "wsd_predictions.csv", "text/csv")


# ── Live inference demo ───────────────────────────────────────────────────────

def _live_demo(train_aligned, train_emb, clf_choice):
    st.markdown("### 🔮 Live WSD Demo")
    desc(
        "This interactive panel lets you test the WSD system on brand-new sentences you write yourself. "
        "Type any sentence that contains one of the target words (e.g. <em>'He sat on the bank fishing all day.'</em>), "
        "specify the target word, and click Predict. "
        "The app encodes your sentence through BERT, extracts the target word's embedding, "
        "and uses 1-nearest-neighbour cosine similarity against the training set to predict the sense. "
        "The top-3 nearest training sentences are shown so you can inspect what evidence the prediction is based on. "
        "This demonstrates that the trained system generalises beyond the fixed test set."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        demo_sentence = st.text_area("Input sentence", "The company decided to bank its profits in a secure account.", height=80)
    with col2:
        demo_word = st.text_input("Target word", "bank")

    if st.button("🔍 Predict Sense"):
        if not demo_sentence.strip() or not demo_word.strip():
            st.warning("Please enter both a sentence and the target word.")
            return

        embedding_model = st.session_state.get("embedding_model", "bert-base-uncased")
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        except ImportError:
            st.error("transformers and torch required for live demo.")
            return

        with st.spinner("Encoding …"):
            tokenizer = AutoTokenizer.from_pretrained(embedding_model)
            model = AutoModel.from_pretrained(embedding_model, output_hidden_states=True)
            model.eval()
            encoded = tokenizer(demo_sentence, return_tensors="pt", truncation=True, max_length=512)
            tokens  = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
            with torch.no_grad():
                out = model(**encoded)
            last_hidden = out.hidden_states[-1][0]
            word_lower  = demo_word.lower()
            word_idx    = [j for j, t in enumerate(tokens) if word_lower in t.replace("##", "").lower()]
            if not word_idx:
                word_idx = list(range(1, len(tokens) - 1))
            query_emb = last_hidden[word_idx].mean(dim=0).numpy().reshape(1, -1)

        sims    = cos_sim(query_emb, train_emb)[0]
        best    = np.argmax(sims)
        best_sim = sims[best]
        predicted_sense = train_aligned.iloc[best]["sense"]

        st.markdown("#### Prediction")
        st.markdown(f'<div class="success-box"><b>Predicted sense:</b> <code>{predicted_sense}</code> &nbsp; (cosine similarity to nearest training example: {best_sim:.3f})</div>', unsafe_allow_html=True)

        st.markdown("**Top-3 nearest training instances:**")
        desc("These are the three training sentences whose BERT embeddings are most similar to your input. They show what the model is 'comparing against' to make its prediction.")
        for idx in np.argsort(sims)[::-1][:3]:
            row = train_aligned.iloc[idx]
            st.markdown(f"- `{row['sense']}` (sim={sims[idx]:.3f}) — _{row['sentence'][:90]}_")
