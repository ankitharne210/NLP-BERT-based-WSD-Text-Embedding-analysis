"""
Task 2 – WordNet Sense Analysis
Synset lookup, definitions, examples, and sense mapping.
"""

import streamlit as st
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
      <h2>🧠 Task 2 – WordNet Sense Analysis</h2>
      <p>Identify synsets for each ambiguous word, read definitions and examples, and map dataset labels to WordNet.</p>
    </div>
    """, unsafe_allow_html=True)

    desc(
        "<b>Purpose:</b> WordNet is the lexical database used as the gold standard for sense inventories in WSD research. "
        "This task connects your dataset's sense labels to official WordNet synsets, giving each sense a precise, "
        "linguistically grounded definition. This is important for three reasons: (1) it documents what each sense "
        "actually means, (2) it justifies why the senses are distinguishable, and (3) it provides the sense inventory "
        "your classifier will predict over. Every sense in your dataset must map to at least one WordNet synset."
    )

    df = st.session_state.get("dataset")
    if df is None:
        st.markdown('<div class="warn-box">⚠️ No dataset loaded. Go to <b>Home</b> first.</div>', unsafe_allow_html=True)
        return

    try:
        import nltk
        from nltk.corpus import wordnet as wn
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    except ImportError:
        st.error("nltk is not installed. Run: pip install nltk")
        return

    words = sorted(df["word"].unique().tolist())

    # ── Global synset browser ─────────────────────────────────────────────────
    st.markdown("### 🔍 WordNet Synset Browser")
    desc(
        "Select any word from your dataset to see <em>all</em> synsets WordNet recognises for it — "
        "not just the two senses in your dataset. Each row shows the synset's unique ID, "
        "its part-of-speech (n=noun, v=verb, a=adjective, r=adverb), "
        "the formal definition, usage examples from WordNet's corpus, and the set of lemmas (synonyms) that belong to it. "
        "Use this to understand the full sense space of each word before deciding which two senses to focus on."
    )
    sel_word = st.selectbox("Select word to analyse", words)

    if sel_word:
        synsets = wn.synsets(sel_word)
        if not synsets:
            st.warning(f"No WordNet synsets found for '{sel_word}'.")
        else:
            st.markdown(f"Found **{len(synsets)}** synsets for *{sel_word}*:")
            rows = []
            for syn in synsets:
                rows.append({
                    "Synset ID":  syn.name(),
                    "POS":        syn.pos(),
                    "Definition": syn.definition(),
                    "Examples":   " | ".join(syn.examples()) if syn.examples() else "—",
                    "Lemmas":     ", ".join([l.name() for l in syn.lemmas()]),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.divider()

    # ── Per-word detailed sense panels ────────────────────────────────────────
    st.markdown("### 📋 Sense-by-sense Analysis (All Words)")
    desc(
        "This section provides a detailed breakdown for every word in your dataset. "
        "Expand any word to see: the sense labels used in your dataset and their corresponding WordNet synset IDs; "
        "the official WordNet definition and example sentences for each sense; "
        "an explanation of the semantic distinction between senses (i.e. what makes them different); "
        "and sample sentences from your own dataset with the target word highlighted in blue. "
        "This forms the core of the written analysis required by the assignment."
    )

    for word in words:
        with st.expander(f"🔤 **{word}** — click to expand"):
            sub = df[df["word"] == word]
            dataset_senses = sub["sense"].unique().tolist()
            dataset_synset_ids = sub["sense_label"].unique().tolist()

            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("**Dataset senses:**")
                for s in dataset_senses:
                    st.markdown(f"- `{s}`")
                st.markdown("**Mapped WordNet synset IDs:**")
                for sid in dataset_synset_ids:
                    st.markdown(f"- `{sid}`")

            with col2:
                st.markdown("**WordNet definitions:**")
                for sid in dataset_synset_ids:
                    try:
                        syn = wn.synset(sid)
                        sense_name = sub[sub["sense_label"] == sid]["sense"].iloc[0]
                        st.markdown(f"**{sense_name}** (`{sid}`)")
                        st.markdown(f"> {syn.definition()}")
                        if syn.examples():
                            st.markdown("_WordNet examples:_")
                            for ex in syn.examples()[:2]:
                                st.markdown(f"  - _{ex}_")
                    except Exception:
                        st.markdown(f"`{sid}` — unable to retrieve from WordNet")

            if len(dataset_senses) >= 2:
                st.markdown("**Semantic distinction:**")
                st.info(_get_distinction(word, dataset_senses))

            st.markdown("**Sample sentences from dataset** _(target word highlighted)_:")
            for sense in dataset_senses:
                sense_rows = sub[sub["sense"] == sense]
                st.markdown(f"*{sense}* ({len(sense_rows)} instances):")
                for _, row in sense_rows.head(2).iterrows():
                    highlighted = _highlight(row["sentence"], word)
                    st.markdown(f"  → {highlighted}", unsafe_allow_html=True)

    st.divider()

    # ── Sense mapping table ───────────────────────────────────────────────────
    st.markdown("### 🗺️ Dataset Sense → WordNet Mapping Table")
    desc(
        "This table provides a consolidated view of the complete sense inventory used in this project. "
        "For each word, it lists every dataset sense label alongside its WordNet synset ID and the official definition. "
        "This mapping is the formal documentation required by the assignment to show that your dataset labels "
        "are grounded in a recognised lexical resource rather than ad-hoc labels."
    )
    mapping_rows = []
    for _, row in df[["word", "sense", "sense_label"]].drop_duplicates().iterrows():
        try:
            syn = wn.synset(row["sense_label"])
            defn = syn.definition()
        except Exception:
            defn = "N/A"
        mapping_rows.append({
            "Word":          row["word"],
            "Dataset Sense": row["sense"],
            "WordNet ID":    row["sense_label"],
            "Definition":    defn,
        })
    mapping_df = pd.DataFrame(mapping_rows).sort_values(["Word", "Dataset Sense"])
    st.dataframe(mapping_df, use_container_width=True)


def _highlight(sentence: str, word: str) -> str:
    import re
    return re.sub(
        rf"\b({re.escape(word)})\b",
        r"<b style='color:#0f3460'>\1</b>",
        sentence,
        flags=re.IGNORECASE,
    )


def _get_distinction(word: str, senses: list) -> str:
    distinctions = {
        "bank": "The word 'bank' can denote a financial institution (where money is deposited/borrowed) "
                "or a geographic landform (the sloped ground alongside a body of water). "
                "Context words like 'deposit', 'loan', 'interest' signal the financial sense, "
                "while 'river', 'mud', 'shore' signal the geographic sense.",
        "light": "'Light' as illumination refers to electromagnetic radiation or a device that emits it. "
                 "'Light' as an adjective means having low weight or density. "
                 "Verbal context ('switch on', 'glowing', 'photon') vs. adjectival context ('carry', 'lift', 'fabric') distinguishes the two.",
        "spring": "As a season, 'spring' follows winter and is associated with warmth and growth. "
                  "As a mechanical device, 'spring' is a coil that stores elastic energy. "
                  "Temporal phrases ('every spring', 'spring rain') signal the season; mechanical words ('coil', 'tension', 'snap') signal the device.",
        "plant": "'Plant' as a living organism refers to any photosynthesising life form. "
                 "'Plant' as an industrial facility refers to a factory or processing installation. "
                 "Biological context ('water', 'soil', 'photosynthesis') vs. industrial context ('workers', 'factory', 'production') is the key distinction.",
        "crane": "As a bird, 'crane' is a large wading bird with long legs and neck. "
                 "As a machine, 'crane' is a lifting apparatus used in construction or logistics. "
                 "Biological cues ('flew', 'nest', 'wetlands') vs. mechanical cues ('operator', 'construction', 'load') separate the senses.",
    }
    return distinctions.get(word, f"The senses of '{word}' are distinguished by the semantic context in which the word appears: {', '.join(senses)}.")
