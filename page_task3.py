"""
Task 3 – BERT Contextual Representation
Extract BERT embeddings, compare across senses, PCA/t-SNE visualization.
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
      <h2>🤖 Task 3 – BERT Contextual Representations</h2>
      <p>Extract per-instance BERT embeddings for the target word, visualize with PCA/t-SNE, and measure sense separation.</p>
    </div>
    """, unsafe_allow_html=True)

    desc(
        "<b>Purpose:</b> The core hypothesis of contextual WSD is that BERT produces <em>different</em> vector "
        "representations for the same word depending on the surrounding sentence. This task makes that hypothesis "
        "concrete: it runs every sentence in your dataset through a pretrained BERT model, extracts the hidden-state "
        "vector specifically at the target word's token position, and stores one embedding per instance. "
        "These embeddings are the numerical features that the WSD classifier in Task 4 will learn from. "
        "The visualisations here let you visually verify that same-sense instances cluster together in embedding space, "
        "providing evidence that contextual representations encode sense information."
    )

    df = st.session_state.get("dataset")
    if df is None:
        st.markdown('<div class="warn-box">⚠️ No dataset loaded. Go to <b>Home</b> first.</div>', unsafe_allow_html=True)
        return

    # ── Model selection ───────────────────────────────────────────────────────
    st.markdown("### ⚙️ BERT Model Settings")
    desc(
        "These controls determine <em>how</em> BERT embeddings are extracted. "
        "<b>BERT variant</b> selects the pretrained model — <code>bert-base-uncased</code> (12 layers, 768-dim) "
        "is a good balance of speed and quality; <code>bert-large-uncased</code> (24 layers, 1024-dim) is more powerful but slower. "
        "<b>Hidden layer</b> controls which of BERT's internal layers is used as the representation — "
        "research shows that layers 9–12 tend to capture the most semantic information useful for WSD. "
        "<b>Token pooling</b> decides what to do when a word is split into multiple subword tokens by BERT's tokeniser "
        "(e.g. 'banking' → ['bank', '##ing']) — mean pooling averages all subword vectors, which is generally robust."
    )
    col1, col2 = st.columns(2)
    with col1:
        model_name = st.selectbox(
            "BERT variant",
            ["bert-base-uncased", "bert-base-cased", "distilbert-base-uncased", "bert-large-uncased"],
            help="bert-base-uncased recommended for speed; bert-large for accuracy.",
        )
    with col2:
        layer_choice = st.selectbox(
            "Hidden layer to extract",
            ["Last layer", "Second-to-last layer", "Sum of last 4 layers", "Concatenation of last 4 layers"],
        )

    pooling = st.radio(
        "Token pooling strategy for multi-token words",
        ["Mean pooling", "First token (##)", "Max pooling"],
        horizontal=True,
    )

    st.divider()

    # ── Extraction ────────────────────────────────────────────────────────────
    st.markdown("### 🧲 Extract BERT Embeddings")
    desc(
        "Click this button to run the full dataset through the selected BERT model. "
        "For each sentence, the model performs a forward pass and reads off the hidden-state vector at the target word's position. "
        "This produces one embedding vector per row in your dataset (e.g. 768 numbers for bert-base). "
        "Extraction runs on CPU by default and takes roughly 1–3 minutes for 100 instances on bert-base. "
        "The embeddings are stored in the session and reused by all visualisations and the classifier in Task 4. "
        "You only need to re-run this if you change the model settings above."
    )
    st.markdown('<div class="info-box">Ensure <code>transformers</code> and <code>torch</code> are installed. '
                'First run downloads model weights (~440 MB for bert-base) from Hugging Face.</div>', unsafe_allow_html=True)

    if st.button("🚀 Extract Embeddings"):
        _extract_embeddings(df, model_name, layer_choice, pooling)
    elif "bert_embeddings" in st.session_state:
        st.markdown('<div class="success-box">✅ Embeddings already extracted. Scroll down to visualize or re-extract with different settings above.</div>', unsafe_allow_html=True)

    # ── Visualization ─────────────────────────────────────────────────────────
    if "bert_embeddings" in st.session_state:
        st.divider()
        _render_visualization(df)
        st.divider()
        _render_cosine_comparison(df)


# ── Embedding extraction ──────────────────────────────────────────────────────

def _extract_embeddings(df, model_name, layer_choice, pooling):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        st.error("Install dependencies: pip install transformers torch")
        return

    progress = st.progress(0, text="Loading tokenizer and model …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()

    embeddings = []
    n = len(df)

    with torch.no_grad():
        for i, (_, row) in enumerate(df.iterrows()):
            sentence = row["sentence"]
            word     = row["word"]

            encoded = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
            tokens  = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
            outputs = model(**{k: v for k, v in encoded.items()})
            hidden_states = outputs.hidden_states

            if layer_choice == "Last layer":
                layer_tensor = hidden_states[-1][0]
            elif layer_choice == "Second-to-last layer":
                layer_tensor = hidden_states[-2][0]
            elif layer_choice == "Sum of last 4 layers":
                layer_tensor = sum(hidden_states[-4:])[0]
            else:
                layer_tensor = torch.cat([hidden_states[-4+i][0] for i in range(4)], dim=-1)

            word_lower   = word.lower()
            word_indices = [j for j, tok in enumerate(tokens) if word_lower in tok.replace("##", "").lower()]
            if not word_indices:
                word_indices = list(range(1, len(tokens) - 1))

            target_vecs = layer_tensor[word_indices]
            if pooling == "Mean pooling":
                embedding = target_vecs.mean(dim=0).numpy()
            elif pooling == "First token (##)":
                embedding = target_vecs[0].numpy()
            else:
                embedding = target_vecs.max(dim=0).values.numpy()

            embeddings.append(embedding)
            progress.progress((i + 1) / n, text=f"Processing {i+1}/{n} instances …")

    progress.empty()
    st.session_state["bert_embeddings"]      = np.array(embeddings)
    st.session_state["embedding_labels"]     = df["sense"].tolist()
    st.session_state["embedding_words"]      = df["word"].tolist()
    st.session_state["embedding_sentences"]  = df["sentence"].tolist()
    st.session_state["embedding_model"]      = model_name
    st.markdown('<div class="success-box">✅ Embeddings extracted successfully.</div>', unsafe_allow_html=True)


# ── Visualization ─────────────────────────────────────────────────────────────

def _render_visualization(df):
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import plotly.express as px
    except ImportError:
        st.error("Install: pip install scikit-learn plotly")
        return

    st.markdown("### 📉 Dimensionality Reduction & Visualization")
    desc(
        "BERT embeddings live in a very high-dimensional space (768 dimensions for bert-base). "
        "To inspect them visually we project them down to 2 dimensions using either "
        "<b>PCA</b> (Principal Component Analysis — fast, linear, deterministic; good for a first look) or "
        "<b>t-SNE</b> (t-Distributed Stochastic Neighbour Embedding — non-linear, better at revealing local cluster structure, but slower and non-deterministic). "
        "Each dot in the scatter plot is one sentence; colour = sense label. "
        "If same-colour dots cluster together and different-colour dots are separated, "
        "it confirms that BERT's contextual representations encode sense information — "
        "a key requirement of the assignment. Use the word filter to examine individual words more closely."
    )

    col1, col2 = st.columns(2)
    with col1:
        method = st.selectbox("Reduction method", ["PCA", "t-SNE"])
    with col2:
        word_filter = st.selectbox("Focus on word", ["All words"] + sorted(df["word"].unique().tolist()))

    emb       = np.array(st.session_state["bert_embeddings"])
    labels    = st.session_state["embedding_labels"]
    words     = st.session_state["embedding_words"]
    sentences = st.session_state["embedding_sentences"]

    if word_filter != "All words":
        mask       = [w == word_filter for w in words]
        emb_f      = emb[mask]
        labels_f   = [l for l, m in zip(labels, mask) if m]
        words_f    = [w for w, m in zip(words, mask) if m]
        sentences_f = [s for s, m in zip(sentences, mask) if m]
    else:
        emb_f, labels_f, words_f, sentences_f = emb, labels, words, sentences

    if len(emb_f) < 3:
        st.warning("Not enough instances for visualization.")
        return

    with st.spinner(f"Running {method} …"):
        if emb_f.shape[1] > 50 and method == "t-SNE":
            pca50 = PCA(n_components=min(50, emb_f.shape[0]-1), random_state=42)
            emb_f = pca50.fit_transform(emb_f)

        if method == "PCA":
            reducer    = PCA(n_components=2, random_state=42)
            coords     = reducer.fit_transform(emb_f)
            explained  = reducer.explained_variance_ratio_
            axis_labels = (f"PC1 ({explained[0]*100:.1f}% var)", f"PC2 ({explained[1]*100:.1f}% var)")
        else:
            perplexity  = min(30, len(emb_f) - 1)
            reducer     = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
            coords      = reducer.fit_transform(emb_f)
            axis_labels = ("t-SNE dim 1", "t-SNE dim 2")

    plot_df = pd.DataFrame({
        "x": coords[:, 0], "y": coords[:, 1],
        "sense": labels_f, "word": words_f,
        "sentence": [s[:80] + "…" if len(s) > 80 else s for s in sentences_f],
    })

    fig = px.scatter(
        plot_df, x="x", y="y",
        color="sense",
        symbol="word" if word_filter == "All words" else None,
        hover_data={"sentence": True, "word": True, "x": False, "y": False},
        title=f"{method} of BERT embeddings – {word_filter}",
        labels={"x": axis_labels[0], "y": axis_labels[1]},
        template="plotly_white", height=550,
    )
    fig.update_traces(marker=dict(size=9, opacity=0.85))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 💬 How to interpret this plot")
    desc(
        "<b>Tight, well-separated clusters of the same colour</b> → BERT successfully encodes sense information; "
        "a simple nearest-centroid or KNN classifier should perform well. "
        "<b>Overlapping or scattered clusters</b> → the senses share similar linguistic contexts, "
        "making WSD harder; a more powerful classifier or additional data may help. "
        "<b>Elongated clusters</b> → a linear decision boundary (e.g. Logistic Regression) may be suboptimal; "
        "consider SVM with an RBF kernel. "
        + (f"PCA retained <b>{sum(explained[:2])*100:.1f}%</b> of total variance in 2 dimensions — "
           "the higher this is, the more faithful the 2D projection is to the true high-dimensional structure."
           if method == "PCA" else
           "t-SNE distances between clusters are not directly interpretable as actual distances in embedding space, "
           "but cluster shape and separation are meaningful.")
    )


def _render_cosine_comparison(df):
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        import plotly.graph_objects as go
    except ImportError:
        return

    st.markdown("### 📐 Intra- vs Inter-sense Cosine Similarity")
    desc(
        "Cosine similarity measures the angle between two vectors — a value of 1.0 means they point in exactly "
        "the same direction (maximally similar), while 0.0 means they are orthogonal (completely dissimilar). "
        "This analysis computes cosine similarity for every pair of instances in the selected word's embedding set, "
        "then separates the scores into two groups: <b>same-sense pairs</b> (intra-sense) and "
        "<b>cross-sense pairs</b> (inter-sense). "
        "If BERT's representations are sense-aware, same-sense pairs should show consistently <em>higher</em> similarity "
        "than cross-sense pairs. The gap between the two box plots is the key number — larger gaps mean cleaner sense separation."
    )

    emb    = np.array(st.session_state["bert_embeddings"])
    labels = st.session_state["embedding_labels"]
    words  = st.session_state["embedding_words"]

    word_filter = st.selectbox("Select word for cosine analysis", sorted(df["word"].unique().tolist()), key="cosine_word")
    mask    = [w == word_filter for w in words]
    emb_w   = emb[mask]
    labels_w = [l for l, m in zip(labels, mask) if m]

    if len(emb_w) < 4:
        st.warning("Need at least 4 instances for this word.")
        return

    sim_matrix = cosine_similarity(emb_w)
    intra_sims, inter_sims = [], []
    for i in range(len(labels_w)):
        for j in range(i + 1, len(labels_w)):
            s = sim_matrix[i, j]
            (intra_sims if labels_w[i] == labels_w[j] else inter_sims).append(s)

    fig = go.Figure()
    if intra_sims:
        fig.add_trace(go.Box(y=intra_sims, name="Same-sense (intra)", marker_color="#10b981"))
    if inter_sims:
        fig.add_trace(go.Box(y=inter_sims, name="Cross-sense (inter)", marker_color="#ef4444"))
    fig.update_layout(
        title=f"Cosine similarity distribution for '{word_filter}'",
        yaxis_title="Cosine Similarity", template="plotly_white", height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    if intra_sims and inter_sims:
        intra_mean = np.mean(intra_sims)
        inter_mean = np.mean(inter_sims)
        gap = intra_mean - inter_mean
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean intra-sense sim", f"{intra_mean:.3f}")
        c2.metric("Mean inter-sense sim", f"{inter_mean:.3f}")
        c3.metric("Separation gap", f"{gap:.3f}", delta="Good" if gap > 0.05 else "Low")
        if gap > 0.05:
            st.markdown('<div class="success-box">✅ Positive gap confirms BERT embeddings separate senses meaningfully for this word.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warn-box">⚠️ Small gap — the senses may be semantically close or context insufficient. Consider using more data or a different BERT layer.</div>', unsafe_allow_html=True)
