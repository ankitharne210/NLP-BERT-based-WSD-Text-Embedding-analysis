"""
Task 5 – Error Analysis & Comparative Discussion
Analyse ≥10 incorrect predictions; compare BERT vs static representations.
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
      <h2>🔍 Task 5 – Error Analysis & Comparative Discussion</h2>
      <p>Examine incorrect WSD predictions in depth and compare BERT contextual embeddings with static word representations.</p>
    </div>
    """, unsafe_allow_html=True)

    desc(
        "<b>Purpose:</b> Achieving a high accuracy score is only part of the assignment. This task requires you to "
        "<em>understand</em> where and why the system fails, which reveals deeper insights about what WSD is actually hard. "
        "Errors are categorised into four interpretable types — similar senses, ambiguous context, domain-specific usage, "
        "and short context — so you can identify patterns rather than treating every mistake as random noise. "
        "The second half of the task addresses a key theoretical question: why are BERT's contextual embeddings "
        "more appropriate for WSD than traditional static embeddings like Word2Vec or GloVe? "
        "Both sections directly correspond to assignment requirements."
    )

    results = st.session_state.get("classifier_results")
    if results is None:
        st.markdown('<div class="warn-box">⚠️ No classifier results found. Complete <b>Task 4</b> first.</div>', unsafe_allow_html=True)
        return

    preds = results["predictions"]
    wrong = preds[~preds["correct"]].copy()

    # ── Error statistics ──────────────────────────────────────────────────────
    st.markdown("### 📉 Error Overview")
    desc(
        "This panel summarises the overall volume of errors and where they are concentrated. "
        "The total error rate gives a headline failure figure. The per-word breakdown identifies which target words "
        "the system finds most difficult — a high error rate for a particular word usually means its senses overlap "
        "heavily in contextual usage or that the training data for that word is insufficient. "
        "The confusion pair table shows which senses are most often mixed up with each other."
    )

    total   = len(preds)
    n_wrong = len(wrong)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total test instances", total)
    c2.metric("Incorrect predictions", n_wrong)
    c3.metric("Error rate", f"{n_wrong/total*100:.1f}%")

    st.markdown("#### Errors per Word")
    err_by_word   = wrong.groupby("word").size().reset_index(name="errors")
    total_by_word = preds.groupby("word").size().reset_index(name="total")
    err_table     = err_by_word.merge(total_by_word, on="word")
    err_table["error_rate"] = (err_table["errors"] / err_table["total"] * 100).round(1).astype(str) + "%"
    st.dataframe(err_table, use_container_width=True)

    st.markdown("#### Most Common Confusion Pairs")
    desc(
        "Each row shows a gold sense that was predicted as a different sense, and how many times that confusion occurred. "
        "The most frequent confusion pairs are the most important to analyse — they typically involve senses "
        "that are genuinely hard to distinguish from context alone."
    )
    conf = wrong.groupby(["sense", "predicted"]).size().reset_index(name="count").sort_values("count", ascending=False)
    if len(conf):
        st.dataframe(conf.rename(columns={"sense": "Gold Sense", "predicted": "Predicted Sense"}), use_container_width=True)

    st.divider()

    # ── Detailed error analysis ───────────────────────────────────────────────
    st.markdown("### 🧐 Detailed Error Analysis")
    desc(
        "Errors are automatically categorised into four types based on heuristic rules applied to the sentence and the confusion pair. "
        "<b>Similar Senses</b>: the gold and predicted senses are known to be semantically close (e.g. bank as financial institution vs river bank — both denote physical places associated with money or water). "
        "<b>Ambiguous Context</b>: the sentence does not contain strong disambiguating cues — a human reader might also struggle. "
        "<b>Domain-specific</b>: the sentence uses the word in a specialised domain context (medical, legal, scientific) that BERT's general-domain pre-training may not cover well. "
        "<b>Short Context</b>: the sentence is very short (fewer than 8 words), providing limited surrounding context for BERT to attend to. "
        "Review each tab and include representative examples in your written submission."
    )

    if len(wrong) == 0:
        st.success("No errors to analyse — perfect accuracy on the test set!")
    else:
        error_categories = _categorise_errors(wrong)
        tabs = st.tabs(["📋 All Errors", "🔁 Similar Senses", "❓ Ambiguous Context", "🌍 Domain-specific", "📏 Short Context"])

        with tabs[0]:
            desc("Complete list of all misclassified test instances. Use this table to manually review each error before categorising them in your write-up.")
            st.dataframe(
                wrong[["word", "sentence", "sense", "predicted"]]
                .rename(columns={"sense": "gold", "predicted": "predicted_sense"})
                .reset_index(drop=True),
                use_container_width=True,
            )
            st.caption(f"Showing all {len(wrong)} errors.")

        with tabs[1]:
            desc("Errors where the gold and predicted senses are semantically close — the boundary between these senses is inherently fuzzy, and even human annotators sometimes disagree on them.")
            _show_error_category(error_categories["similar_senses"])

        with tabs[2]:
            desc("Errors where the sentence does not contain strong contextual cues for either sense. These are the hardest cases — any WSD system, static or contextual, would likely struggle here.")
            _show_error_category(error_categories["ambiguous_context"])

        with tabs[3]:
            desc("Errors involving domain-specific language that was underrepresented in BERT's pre-training corpus. Fine-tuning BERT on domain-specific text would likely fix many of these.")
            _show_error_category(error_categories["domain_specific"])

        with tabs[4]:
            desc("Errors from very short sentences (fewer than 8 words). BERT's self-attention mechanism benefits from longer surrounding context — short sentences give it little to work with.")
            _show_error_category(error_categories["short_context"])

    st.divider()

    # ── Static vs contextual comparison ──────────────────────────────────────
    st.markdown("### ⚖️ BERT Contextual vs. Static Representations (Word2Vec / GloVe)")
    desc(
        "This section directly addresses the assignment question: <em>why are contextual representations more suitable for WSD than static ones?</em> "
        "Static embeddings (Word2Vec, GloVe, FastText) assign a single fixed vector to each word type, "
        "regardless of context — so <em>bank</em> always has the same representation whether it appears next to "
        "'deposit' or 'river'. WSD with static embeddings requires external context-aggregation tricks that are inherently lossy. "
        "BERT, by contrast, computes a fresh representation for every word <em>in every sentence</em>, "
        "encoding the influence of every surrounding word via the self-attention mechanism. "
        "The table below compares the two approaches across key technical dimensions."
    )
    _render_comparison_table()

    st.markdown("#### Why contextual representations matter for WSD — in detail")
    st.markdown("""
    **Static embeddings (Word2Vec, GloVe):**
    - Each word type has exactly **one fixed vector** regardless of context.
    - *bank* has the same representation whether it appears in a financial or geographic sentence.
    - WSD must rely on averaging context-word vectors or hand-crafted features applied externally — both are coarse.
    - Averaging context window vectors can wash out the very signal needed to distinguish senses.

    **BERT contextual embeddings:**
    - The embedding for each word token is computed dynamically from the **entire surrounding sentence** via multi-head self-attention.
    - BERT learns to shift the representation based on context: *bank* near *deposit*, *loan*, *interest* lands in one region of embedding space; near *river*, *mud*, *shore* it lands in another.
    - This is directly observable in the PCA/t-SNE plots in Task 3 — same-sense instances cluster together without any task-specific training.
    - Nearest-centroid and cosine-similarity WSD work well on BERT embeddings precisely because the sense information is already encoded in the vector.
    - Fine-tuning BERT on labelled WSD data (e.g. SemCor) further specialises these representations and yields state-of-the-art WSD accuracy.
    """)

    st.divider()

    # ── Limitations ───────────────────────────────────────────────────────────
    st.markdown("### ⚠️ Limitations of the Implemented WSD Approach")
    desc(
        "Every WSD system has limitations. Acknowledging them honestly demonstrates critical thinking and is "
        "explicitly required by the assignment. Each limitation below points to a concrete way the system could "
        "be improved with additional effort or resources."
    )
    limitations = [
        ("**Small dataset**",
         "Only 20 sentences per word were used. Statistical evaluation on such a small test set is unreliable — "
         "a single correct or incorrect prediction can shift accuracy by 5–10 percentage points. "
         "Larger datasets (e.g. SemCor with 200,000+ sense-tagged instances) would produce more stable metrics."),
        ("**Layer selection sensitivity**",
         "BERT's representation quality for WSD varies substantially by layer. Layer 11–12 of bert-base "
         "typically performs best for semantic tasks, but this varies by word and sense. "
         "A systematic ablation across all 12 layers was not performed in this implementation."),
        ("**Domain mismatch**",
         "BERT-base was pre-trained on Wikipedia and BooksCorpus. Specialised domains "
         "(medical, legal, scientific) use words in ways that diverge from general usage, "
         "and BERT's embeddings may not capture these differences as reliably."),
        ("**Nearest-centroid assumptions**",
         "The centroid method assumes that same-sense embeddings form a compact, roughly spherical cluster. "
         "When clusters are elongated, non-convex, or overlapping, the centroid decision boundary is suboptimal "
         "and a kernel-based method like SVM would be more appropriate."),
        ("**Coarse-grained senses only**",
         "The two-sense-per-word setup is a simplification. Real WSD benchmarks (SemEval-2007, SemEval-2015) "
         "use fine-grained WordNet senses that are far harder to separate — e.g. *bank.n.01* (financial institution) "
         "vs *bank.n.02* (depository financial institution) are nearly indistinguishable from context."),
        ("**No fine-tuning**",
         "A BERT model fine-tuned on a WSD corpus (e.g. SemCor + OMSTI) would substantially outperform "
         "the out-of-the-box pretrained representations used here. Fine-tuning teaches the model "
         "to produce representations that are maximally discriminative for sense disambiguation specifically."),
        ("**Subword tokenisation**",
         "When the target word is split into multiple BPE subword tokens, the implementation averages the "
         "subword vectors. This is a reasonable heuristic but may dilute the signal for morphologically "
         "complex words or rare words split into many pieces."),
    ]
    for title, body in limitations:
        with st.expander(title):
            st.markdown(body)

    # ── Export report ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📄 Export Error Analysis Report")
    desc(
        "Download a formatted Markdown report containing the error summary, per-word breakdown, "
        "confusion pairs, and the complete error table. You can include this in your assignment submission "
        "or convert it to PDF for the report section."
    )
    if len(wrong) > 0:
        report_lines = ["# WSD Error Analysis Report\n",
                        f"Total test instances: {total}\n",
                        f"Errors: {n_wrong} ({n_wrong/total*100:.1f}%)\n\n",
                        "## Error Table\n",
                        wrong[["word","sentence","sense","predicted"]].to_markdown(index=False),
                        "\n\n## Confusion Pairs\n",
                        conf.to_markdown(index=False) if len(conf) else "None"]
        st.download_button("⬇️ Download error report (Markdown)", "\n".join(report_lines).encode(), "error_analysis.md", "text/markdown")
    else:
        st.info("No errors to export — the classifier achieved perfect accuracy on the test set.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _categorise_errors(wrong: pd.DataFrame) -> dict:
    similar_pairs = {
        frozenset({"financial_institution", "river_bank"}),
        frozenset({"illumination", "not_heavy"}),
        frozenset({"season", "coil_device"}),
        frozenset({"living_organism", "industrial_facility"}),
        frozenset({"bird", "lifting_machine"}),
    }
    cats = {"similar_senses": [], "ambiguous_context": [], "domain_specific": [], "short_context": []}
    for _, r in wrong.iterrows():
        sentence = r["sentence"]
        pair     = frozenset({r["sense"], r["predicted"]})
        if len(sentence.split()) < 8:
            cats["short_context"].append(r)
        elif pair in similar_pairs:
            cats["similar_senses"].append(r)
        elif any(kw in sentence.lower() for kw in ["technical", "scientific", "medical", "legal", "research"]):
            cats["domain_specific"].append(r)
        else:
            cats["ambiguous_context"].append(r)
    return {k: pd.DataFrame(v) for k, v in cats.items()}


def _show_error_category(df_cat: pd.DataFrame):
    if len(df_cat) == 0:
        st.info("No errors in this category.")
        return
    st.caption(f"{len(df_cat)} instance(s) in this category")
    for _, r in df_cat.iterrows():
        st.markdown(
            f"**{r['word']}** | Gold: `{r['sense']}` → Predicted: `{r['predicted']}`  \n"
            f"> {r['sentence']}"
        )
        st.markdown("---")


def _render_comparison_table():
    comparison = {
        "Aspect": [
            "Representation type",
            "Context sensitivity",
            "Same word → same vector?",
            "Vocabulary coverage",
            "WSD suitability",
            "Pre-training objective",
            "Inference speed",
            "Fine-tuning capability",
        ],
        "Word2Vec / GloVe (Static)": [
            "One fixed vector per word type",
            "None — context is ignored at inference",
            "Yes (always identical)",
            "Fixed vocabulary from training corpus",
            "Low — all senses conflated into one vector",
            "Skip-gram / co-occurrence matrix",
            "Very fast (single vector lookup)",
            "Not applicable — weights are frozen",
        ],
        "BERT (Contextual)": [
            "Dynamic vector per word token per sentence",
            "High — full sentence encoded via self-attention",
            "No — vector varies by surrounding context",
            "Sub-word BPE — handles rare/unseen words",
            "High — sense-specific regions in embedding space",
            "Masked Language Modelling + Next Sentence Prediction",
            "Slower — full forward pass per sentence",
            "Yes — can fine-tune on WSD corpora (e.g. SemCor)",
        ],
    }
    st.dataframe(pd.DataFrame(comparison).set_index("Aspect"), use_container_width=True)
