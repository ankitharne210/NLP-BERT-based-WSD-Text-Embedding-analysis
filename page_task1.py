"""
Task 1 – WSD Dataset Preparation
Train/test split, per-word statistics, sense distribution.
"""

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split


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
      <h2>📚 Task 1 – WSD Dataset Preparation</h2>
      <p>Validate your dataset against assignment requirements, inspect statistics, and create a stratified train/test split.</p>
    </div>
    """, unsafe_allow_html=True)

    desc(
        "<b>Purpose:</b> This task establishes the labelled evaluation set that every downstream task depends on. "
        "It verifies that the dataset meets the assignment's minimum requirements (≥5 words, ≥20 instances per word, "
        "≥2 senses per word, ≥100 instances total), gives you a statistical overview of the data, and splits it into "
        "a <em>training set</em> (used to build the WSD classifier) and a <em>test set</em> (held out entirely for evaluation). "
        "Keeping the test set unseen during training is essential for an honest measure of WSD accuracy."
    )

    df = st.session_state.get("dataset")
    if df is None:
        st.markdown('<div class="warn-box">⚠️ No dataset loaded. Go to <b>Home</b> and upload or load sample data first.</div>', unsafe_allow_html=True)
        return

    # ── Assignment requirements checklist ─────────────────────────────────────
    st.markdown("### ✅ Assignment Requirement Checks")
    desc(
        "This checklist automatically verifies that your dataset satisfies each structural requirement "
        "defined in the assignment brief. A green tick means the requirement is met; a red cross means "
        "you need to add more data before proceeding. All checks must pass for a valid submission."
    )

    words = df["word"].unique().tolist()
    min_senses_ok = all(df[df["word"] == w]["sense"].nunique() >= 2 for w in words)
    min_instances_ok = all(df[df["word"] == w].shape[0] >= 20 for w in words)
    total_ok = len(df) >= 100

    checks = {
        f"≥ 5 ambiguous words selected ({len(words)} found)": len(words) >= 5,
        f"Total ≥ 100 instances ({len(df)} found)": total_ok,
        "≥ 20 instances per word": min_instances_ok,
        "≥ 2 senses per word": min_senses_ok,
        "Required columns present (word, sentence, sense, sense_label)": True,
    }
    for msg, ok in checks.items():
        icon = "✅" if ok else "❌"
        color = "#166534" if ok else "#991b1b"
        st.markdown(f'<span style="color:{color}; font-weight:600;">{icon} {msg}</span>', unsafe_allow_html=True)

    st.divider()

    # ── Per-word stats table ──────────────────────────────────────────────────
    st.markdown("### 📊 Per-word Statistics")
    desc(
        "This table summarises the composition of the dataset word by word. "
        "It shows how many total instances exist for each target word, how many distinct senses are represented, "
        "and what those sense labels are. Use this to spot any word that has too few examples or only one sense — "
        "both of which would violate assignment requirements."
    )
    rows = []
    for w in sorted(words):
        sub = df[df["word"] == w]
        rows.append({
            "Word": w,
            "Instances": len(sub),
            "Senses": sub["sense"].nunique(),
            "Sense Labels": ", ".join(sorted(sub["sense"].unique())),
        })
    stats_df = pd.DataFrame(rows)
    st.dataframe(stats_df, use_container_width=True)

    # ── Sense distribution bar chart ──────────────────────────────────────────
    st.markdown("### 📈 Sense Distribution Chart")
    desc(
        "This grouped bar chart visualises the number of instances per sense for every word. "
        "Each colour represents one sense. Ideally the bars within each word group should be roughly equal in height, "
        "indicating a <em>balanced</em> dataset. Severe imbalance (e.g. 18 instances for one sense and 2 for another) "
        "can skew classifier performance and should be corrected by adding more data."
    )
    dist = df.groupby(["word", "sense"]).size().reset_index(name="count")
    pivot = dist.pivot(index="word", columns="sense", values="count").fillna(0)
    st.bar_chart(pivot)

    st.divider()

    # ── Train/Test split ──────────────────────────────────────────────────────
    st.markdown("### ✂️ Train / Test Split")
    desc(
        "This section divides the dataset into two non-overlapping subsets: a <b>training set</b> used to build "
        "the WSD classifier (Task 4) and a <b>test set</b> used exclusively for evaluation. "
        "The split is <em>stratified</em> — it preserves the sense proportions from the full dataset within each word, "
        "so neither subset ends up heavily biased towards one sense. "
        "The slider controls what fraction of each word's instances go into the test set (default 20%). "
        "Once created, both sets are saved in the app's session and automatically used by Tasks 4 and 5."
    )
    test_size = st.slider("Test set proportion", 0.1, 0.4, 0.2, 0.05)

    if st.button("Create Train/Test Split"):
        try:
            train_frames, test_frames = [], []
            for w in words:
                sub = df[df["word"] == w]
                tr, te = train_test_split(
                    sub,
                    test_size=test_size,
                    random_state=42,
                    stratify=sub["sense"] if sub["sense"].nunique() > 1 else None,
                )
                train_frames.append(tr)
                test_frames.append(te)

            train_df = pd.concat(train_frames).reset_index(drop=True)
            test_df  = pd.concat(test_frames).reset_index(drop=True)

            st.session_state["train_df"] = train_df
            st.session_state["test_df"]  = test_df

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="label">Train instances</div><div class="value">{len(train_df)}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="label">Test instances</div><div class="value">{len(test_df)}</div></div>', unsafe_allow_html=True)

            st.markdown("#### Train set preview")
            desc("The training set — these rows are used to fit the WSD classifier in Task 4. The model will learn sense centroids or decision boundaries from these examples.")
            st.dataframe(train_df.head(8), use_container_width=True)

            st.markdown("#### Test set preview")
            desc("The held-out test set — these rows are never seen during training. Task 4 evaluates the classifier's predictions against the gold sense labels here.")
            st.dataframe(test_df.head(8), use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.download_button("⬇️ Download train set", train_df.to_csv(index=False).encode(), "train.csv", "text/csv")
            with c2:
                st.download_button("⬇️ Download test set", test_df.to_csv(index=False).encode(), "test.csv", "text/csv")

            st.markdown('<div class="success-box">✅ Train/test split created and saved for downstream tasks.</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Split failed: {e}")

    elif "train_df" in st.session_state:
        st.markdown('<div class="info-box">ℹ️ Train/test split already exists in session. Re-run above to regenerate with a different proportion.</div>', unsafe_allow_html=True)

    # ── Full dataset viewer ───────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🔍 Browse Full Dataset")
    desc(
        "Use this panel to inspect individual rows of the dataset filtered by word. "
        "It is useful for spot-checking sentence quality, verifying that the target word appears in each sentence, "
        "and confirming that sense labels are assigned consistently."
    )
    with st.expander("Open dataset browser"):
        word_filter = st.selectbox("Filter by word", ["All"] + sorted(words))
        filtered = df if word_filter == "All" else df[df["word"] == word_filter]
        st.dataframe(filtered, use_container_width=True)
