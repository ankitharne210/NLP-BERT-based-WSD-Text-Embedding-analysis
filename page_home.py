"""
Home page – data upload / URL fetch / sample data loader.
"""

import streamlit as st
import pandas as pd
import requests
import io


SAMPLE_DATA = [
    # bank – financial institution
    {"word": "bank", "sentence": "She deposited her salary into the bank account this morning.", "sense": "financial_institution", "sense_label": "bank.n.01"},
    {"word": "bank", "sentence": "The bank refused to approve his loan application.", "sense": "financial_institution", "sense_label": "bank.n.01"},
    {"word": "bank", "sentence": "Interest rates offered by the bank dropped by half a percent.", "sense": "financial_institution", "sense_label": "bank.n.01"},
    {"word": "bank", "sentence": "He went to the bank to exchange his foreign currency.", "sense": "financial_institution", "sense_label": "bank.n.01"},
    {"word": "bank", "sentence": "The central bank raised its benchmark interest rate.", "sense": "financial_institution", "sense_label": "bank.n.01"},
    {"word": "bank", "sentence": "She opened a savings account at the national bank.", "sense": "financial_institution", "sense_label": "bank.n.01"},
    {"word": "bank", "sentence": "The bank offers free checking accounts to students.", "sense": "financial_institution", "sense_label": "bank.n.01"},
    {"word": "bank", "sentence": "ATMs outside the bank are available 24 hours a day.", "sense": "financial_institution", "sense_label": "bank.n.01"},
    {"word": "bank", "sentence": "The investment bank underwrote the IPO last week.", "sense": "financial_institution", "sense_label": "bank.n.01"},
    {"word": "bank", "sentence": "He transferred funds through the bank's online portal.", "sense": "financial_institution", "sense_label": "bank.n.01"},
    # bank – river bank
    {"word": "bank", "sentence": "They set up camp on the grassy bank of the river.", "sense": "river_bank", "sense_label": "bank.n.09"},
    {"word": "bank", "sentence": "The children played along the muddy bank all afternoon.", "sense": "river_bank", "sense_label": "bank.n.09"},
    {"word": "bank", "sentence": "Willows lined the steep bank beside the stream.", "sense": "river_bank", "sense_label": "bank.n.09"},
    {"word": "bank", "sentence": "Erosion is gradually wearing away the river bank.", "sense": "river_bank", "sense_label": "bank.n.09"},
    {"word": "bank", "sentence": "Fishermen perched along the bank waiting for a bite.", "sense": "river_bank", "sense_label": "bank.n.09"},
    {"word": "bank", "sentence": "The flood waters rose above the bank of the creek.", "sense": "river_bank", "sense_label": "bank.n.09"},
    {"word": "bank", "sentence": "He sat on the bank and watched the canoe drift by.", "sense": "river_bank", "sense_label": "bank.n.09"},
    {"word": "bank", "sentence": "Wildflowers covered the sunny bank of the brook.", "sense": "river_bank", "sense_label": "bank.n.09"},
    {"word": "bank", "sentence": "The otter slid down the muddy bank into the water.", "sense": "river_bank", "sense_label": "bank.n.09"},
    {"word": "bank", "sentence": "They built a fire on the bank to keep warm at night.", "sense": "river_bank", "sense_label": "bank.n.09"},
    # light – illumination
    {"word": "light", "sentence": "She switched on the light to read her book.", "sense": "illumination", "sense_label": "light.n.01"},
    {"word": "light", "sentence": "The light from the lamp cast long shadows on the wall.", "sense": "illumination", "sense_label": "light.n.01"},
    {"word": "light", "sentence": "Natural light flooded the room when she opened the curtains.", "sense": "illumination", "sense_label": "light.n.01"},
    {"word": "light", "sentence": "The streetlight flickered in the fog.", "sense": "illumination", "sense_label": "light.n.01"},
    {"word": "light", "sentence": "Solar panels convert light into electrical energy.", "sense": "illumination", "sense_label": "light.n.01"},
    {"word": "light", "sentence": "He used a flashlight because there was no other light.", "sense": "illumination", "sense_label": "light.n.01"},
    {"word": "light", "sentence": "The theater went dark before a single spotlight cut through the light.", "sense": "illumination", "sense_label": "light.n.01"},
    {"word": "light", "sentence": "Photographers prefer soft light early in the morning.", "sense": "illumination", "sense_label": "light.n.01"},
    {"word": "light", "sentence": "The emergency light activated when the power cut out.", "sense": "illumination", "sense_label": "light.n.01"},
    {"word": "light", "sentence": "Light travels at roughly 300,000 kilometers per second.", "sense": "illumination", "sense_label": "light.n.01"},
    # light – not heavy
    {"word": "light", "sentence": "The bag was surprisingly light for all its contents.", "sense": "not_heavy", "sense_label": "light.s.01"},
    {"word": "light", "sentence": "She chose a light fabric for the summer dress.", "sense": "not_heavy", "sense_label": "light.s.01"},
    {"word": "light", "sentence": "The children's lunch was light but nutritious.", "sense": "not_heavy", "sense_label": "light.s.01"},
    {"word": "light", "sentence": "He prefers a light meal before running.", "sense": "not_heavy", "sense_label": "light.s.01"},
    {"word": "light", "sentence": "The feather was so light it floated in the breeze.", "sense": "not_heavy", "sense_label": "light.s.01"},
    {"word": "light", "sentence": "She wore a light jacket because the evening was mild.", "sense": "not_heavy", "sense_label": "light.s.01"},
    {"word": "light", "sentence": "The suitcase felt light after removing the books.", "sense": "not_heavy", "sense_label": "light.s.01"},
    {"word": "light", "sentence": "A light breeze cooled the beach in the afternoon.", "sense": "not_heavy", "sense_label": "light.s.01"},
    {"word": "light", "sentence": "Carbon fibre is valued because it is extremely light.", "sense": "not_heavy", "sense_label": "light.s.01"},
    {"word": "light", "sentence": "She carried only a light backpack on the day hike.", "sense": "not_heavy", "sense_label": "light.s.01"},
    # spring – season
    {"word": "spring", "sentence": "Flowers bloom across the park every spring.", "sense": "season", "sense_label": "spring.n.01"},
    {"word": "spring", "sentence": "The birds return to nest when spring arrives.", "sense": "season", "sense_label": "spring.n.01"},
    {"word": "spring", "sentence": "Spring rains replenish the reservoirs after the dry winter.", "sense": "season", "sense_label": "spring.n.01"},
    {"word": "spring", "sentence": "Farmers begin planting in early spring.", "sense": "season", "sense_label": "spring.n.01"},
    {"word": "spring", "sentence": "She loves the fresh air that comes with spring.", "sense": "season", "sense_label": "spring.n.01"},
    {"word": "spring", "sentence": "They planned a camping trip for spring break.", "sense": "season", "sense_label": "spring.n.01"},
    {"word": "spring", "sentence": "The garden looks magnificent in spring.", "sense": "season", "sense_label": "spring.n.01"},
    {"word": "spring", "sentence": "Longer days are a welcome sign that spring has come.", "sense": "season", "sense_label": "spring.n.01"},
    {"word": "spring", "sentence": "The market fills with fresh produce every spring.", "sense": "season", "sense_label": "spring.n.01"},
    {"word": "spring", "sentence": "Spring temperatures in the valley are ideal for hiking.", "sense": "season", "sense_label": "spring.n.01"},
    # spring – coil / device
    {"word": "spring", "sentence": "The mattress spring broke and poked through the fabric.", "sense": "coil_device", "sense_label": "spring.n.04"},
    {"word": "spring", "sentence": "He replaced the worn spring in the clock mechanism.", "sense": "coil_device", "sense_label": "spring.n.04"},
    {"word": "spring", "sentence": "The spring in the mouse trap snapped shut instantly.", "sense": "coil_device", "sense_label": "spring.n.04"},
    {"word": "spring", "sentence": "A coiled spring stores elastic potential energy.", "sense": "coil_device", "sense_label": "spring.n.04"},
    {"word": "spring", "sentence": "She stretched the spring to test its tension.", "sense": "coil_device", "sense_label": "spring.n.04"},
    {"word": "spring", "sentence": "The pen stopped working because its spring was bent.", "sense": "coil_device", "sense_label": "spring.n.04"},
    {"word": "spring", "sentence": "He adjusted the valve spring to improve engine performance.", "sense": "coil_device", "sense_label": "spring.n.04"},
    {"word": "spring", "sentence": "The toy car moves when you release the wound spring.", "sense": "coil_device", "sense_label": "spring.n.04"},
    {"word": "spring", "sentence": "Compressing the spring requires significant force.", "sense": "coil_device", "sense_label": "spring.n.04"},
    {"word": "spring", "sentence": "The old sofa had a broken spring that poked out.", "sense": "coil_device", "sense_label": "spring.n.04"},
    # plant – living organism
    {"word": "plant", "sentence": "The plant needs watering twice a week in summer.", "sense": "living_organism", "sense_label": "plant.n.02"},
    {"word": "plant", "sentence": "She bought a fern plant for her office desk.", "sense": "living_organism", "sense_label": "plant.n.02"},
    {"word": "plant", "sentence": "The tropical plant thrives in humid conditions.", "sense": "living_organism", "sense_label": "plant.n.02"},
    {"word": "plant", "sentence": "Photosynthesis allows the plant to convert sunlight into food.", "sense": "living_organism", "sense_label": "plant.n.02"},
    {"word": "plant", "sentence": "The gardener pruned every plant along the path.", "sense": "living_organism", "sense_label": "plant.n.02"},
    {"word": "plant", "sentence": "Each plant in the greenhouse was carefully labelled.", "sense": "living_organism", "sense_label": "plant.n.02"},
    {"word": "plant", "sentence": "The invasive plant displaced native species in the meadow.", "sense": "living_organism", "sense_label": "plant.n.02"},
    {"word": "plant", "sentence": "She repotted the plant because its roots had outgrown the pot.", "sense": "living_organism", "sense_label": "plant.n.02"},
    {"word": "plant", "sentence": "A healthy plant depends on proper soil nutrients.", "sense": "living_organism", "sense_label": "plant.n.02"},
    {"word": "plant", "sentence": "The plant flowered for the first time this spring.", "sense": "living_organism", "sense_label": "plant.n.02"},
    # plant – industrial facility
    {"word": "plant", "sentence": "The chemical plant emitted smoke into the sky.", "sense": "industrial_facility", "sense_label": "plant.n.01"},
    {"word": "plant", "sentence": "Workers at the manufacturing plant went on strike.", "sense": "industrial_facility", "sense_label": "plant.n.01"},
    {"word": "plant", "sentence": "The power plant supplies electricity to three counties.", "sense": "industrial_facility", "sense_label": "plant.n.01"},
    {"word": "plant", "sentence": "They modernised the assembly plant to boost efficiency.", "sense": "industrial_facility", "sense_label": "plant.n.01"},
    {"word": "plant", "sentence": "The nuclear plant underwent a scheduled safety inspection.", "sense": "industrial_facility", "sense_label": "plant.n.01"},
    {"word": "plant", "sentence": "Environmental regulators fined the plant for illegal dumping.", "sense": "industrial_facility", "sense_label": "plant.n.01"},
    {"word": "plant", "sentence": "The new car plant will employ 2,000 people.", "sense": "industrial_facility", "sense_label": "plant.n.01"},
    {"word": "plant", "sentence": "Production at the plant halted due to a machinery fault.", "sense": "industrial_facility", "sense_label": "plant.n.01"},
    {"word": "plant", "sentence": "The water treatment plant purifies supply for the whole city.", "sense": "industrial_facility", "sense_label": "plant.n.01"},
    {"word": "plant", "sentence": "Engineers toured the steel plant to assess capacity.", "sense": "industrial_facility", "sense_label": "plant.n.01"},
    # crane – bird
    {"word": "crane", "sentence": "A crane flew gracefully over the wetlands at dusk.", "sense": "bird", "sense_label": "crane.n.04"},
    {"word": "crane", "sentence": "The whooping crane is an endangered species in North America.", "sense": "bird", "sense_label": "crane.n.04"},
    {"word": "crane", "sentence": "We spotted a pair of cranes nesting near the lake.", "sense": "bird", "sense_label": "crane.n.04"},
    {"word": "crane", "sentence": "The crane stretched its long neck to reach the fish.", "sense": "bird", "sense_label": "crane.n.04"},
    {"word": "crane", "sentence": "A sandhill crane walked slowly through the reeds.", "sense": "bird", "sense_label": "crane.n.04"},
    {"word": "crane", "sentence": "Origami artists often fold the crane as a symbol of peace.", "sense": "bird", "sense_label": "crane.n.04"},
    {"word": "crane", "sentence": "The crane's call echoed across the misty marsh.", "sense": "bird", "sense_label": "crane.n.04"},
    {"word": "crane", "sentence": "The migratory crane returns south every autumn.", "sense": "bird", "sense_label": "crane.n.04"},
    {"word": "crane", "sentence": "Ornithologists tracked the crane using satellite tags.", "sense": "bird", "sense_label": "crane.n.04"},
    {"word": "crane", "sentence": "The crane spread its wings and soared above the river.", "sense": "bird", "sense_label": "crane.n.04"},
    # crane – lifting machine
    {"word": "crane", "sentence": "The construction crane lifted steel beams to the tenth floor.", "sense": "lifting_machine", "sense_label": "crane.n.01"},
    {"word": "crane", "sentence": "A tower crane dominates the skyline of any major building site.", "sense": "lifting_machine", "sense_label": "crane.n.01"},
    {"word": "crane", "sentence": "The crane operator carefully manoeuvred the load into place.", "sense": "lifting_machine", "sense_label": "crane.n.01"},
    {"word": "crane", "sentence": "They used a mobile crane to unload the heavy cargo.", "sense": "lifting_machine", "sense_label": "crane.n.01"},
    {"word": "crane", "sentence": "The port crane broke down, delaying shipments by two days.", "sense": "lifting_machine", "sense_label": "crane.n.01"},
    {"word": "crane", "sentence": "Safety regulations require crane operators to be certified.", "sense": "lifting_machine", "sense_label": "crane.n.01"},
    {"word": "crane", "sentence": "Workers attached the cable to the crane hook.", "sense": "lifting_machine", "sense_label": "crane.n.01"},
    {"word": "crane", "sentence": "The demolition crane swung its wrecking ball at the wall.", "sense": "lifting_machine", "sense_label": "crane.n.01"},
    {"word": "crane", "sentence": "A floating crane was used to raise the sunken vessel.", "sense": "lifting_machine", "sense_label": "crane.n.01"},
    {"word": "crane", "sentence": "The crane's boom extended sixty metres into the air.", "sense": "lifting_machine", "sense_label": "crane.n.01"},
]


def desc(text: str):
    """Render a light description box below a section heading."""
    st.markdown(
        f'<div style="background:#f1f5f9;border-left:3px solid #4fc3f7;'
        f'border-radius:6px;padding:0.7rem 1rem;margin:-0.4rem 0 1rem;'
        f'font-size:0.87rem;color:#334155;">{text}</div>',
        unsafe_allow_html=True,
    )


def render():
    st.markdown("""
    <div class="task-header">
      <h2>🏠 Home – Data Input</h2>
      <p>Upload your WSD dataset, fetch it from a URL, or use built-in sample data to begin the pipeline.</p>
    </div>
    """, unsafe_allow_html=True)

    desc(
        "This is the starting point of the entire WSD pipeline. Before any task can run, the app needs "
        "a labelled dataset of sentences — each one containing a target ambiguous word, a context sentence, "
        "and a sense label. You can supply this data in three ways: upload a local CSV/JSON file, provide "
        "a public URL pointing to one, or load the pre-built sample dataset (5 words × 2 senses × 10 sentences = 100 instances). "
        "All downstream tasks (Tasks 1–5) read from the dataset loaded here, so complete this step first."
    )

    st.markdown("""
    ### About this application
    This tool implements a complete **Word Sense Disambiguation (WSD)** pipeline for your BITS Pilani
    NLP Assignment 2 (DSECLZG530). Work through **Tasks 1–5** using the sidebar, each corresponding
    to an assignment section.

    **Required CSV columns:** `word`, `sentence`, `sense`, `sense_label`
    """)

    # ── Data source tabs ──────────────────────────────────────────────────────
    st.markdown("### 📥 Choose Your Data Source")
    desc(
        "Select the tab that matches how you want to supply data. "
        "<b>Upload File</b> is for local files on your machine. "
        "<b>Fetch from URL</b> lets you point to any public CSV/JSON link (e.g. a raw GitHub URL). "
        "<b>Use Sample Data</b> loads a ready-made dataset instantly — useful for a first run or demo."
    )

    tab_upload, tab_url, tab_sample = st.tabs(["📁 Upload File", "🌐 Fetch from URL", "🗂️ Use Sample Data"])

    # ─ Upload ─
    with tab_upload:
        st.markdown("#### Upload a local CSV or JSON file")
        desc(
            "Upload a <b>CSV</b> or <b>JSON</b> file from your computer. "
            "The file must contain the four required columns: <code>word</code>, <code>sentence</code>, "
            "<code>sense</code>, and <code>sense_label</code>. "
            "Once uploaded, the app validates the columns and shows a preview with basic statistics. "
            "See the format guide at the bottom of this page for an example."
        )
        uploaded = st.file_uploader("Choose file", type=["csv", "json"])

        if uploaded:
            try:
                if uploaded.name.endswith(".csv"):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_json(uploaded)

                required = {"word", "sentence", "sense", "sense_label"}
                if not required.issubset(df.columns):
                    st.error(f"Missing columns. Found: {list(df.columns)}. Required: {list(required)}")
                else:
                    st.session_state["dataset"] = df
                    st.markdown('<div class="success-box">✅ Dataset loaded successfully.</div>', unsafe_allow_html=True)
                    _show_data_preview(df)
            except Exception as e:
                st.error(f"Could not parse file: {e}")

    # ─ URL ─
    with tab_url:
        st.markdown("#### Fetch from a public URL")
        desc(
            "Paste a direct link to a publicly accessible CSV or JSON file. "
            "This is ideal when your dataset is hosted on GitHub (use the <em>Raw</em> link), "
            "Google Drive (public share link ending in <code>/export?format=csv</code>), "
            "or any HTTP/HTTPS endpoint. The app downloads the file, parses it, and validates columns "
            "exactly as it would for an uploaded file."
        )
        url = st.text_input("Dataset URL", placeholder="https://raw.githubusercontent.com/.../dataset.csv")
        if st.button("Fetch from URL") and url:
            with st.spinner("Downloading …"):
                try:
                    resp = requests.get(url, timeout=15)
                    resp.raise_for_status()
                    if url.endswith(".json"):
                        df = pd.read_json(io.StringIO(resp.text))
                    else:
                        df = pd.read_csv(io.StringIO(resp.text))

                    required = {"word", "sentence", "sense", "sense_label"}
                    if not required.issubset(df.columns):
                        st.error(f"Missing columns. Found: {list(df.columns)}. Required: {list(required)}")
                    else:
                        st.session_state["dataset"] = df
                        st.markdown('<div class="success-box">✅ Dataset fetched successfully.</div>', unsafe_allow_html=True)
                        _show_data_preview(df)
                except Exception as e:
                    st.error(f"Failed to fetch: {e}")

    # ─ Sample ─
    with tab_sample:
        st.markdown("#### Load the built-in sample dataset")
        desc(
            "This option loads a pre-built dataset with <b>5 ambiguous English words</b> "
            "(<em>bank, light, spring, plant, crane</em>), each with exactly <b>2 senses</b> and "
            "<b>10 sentences per sense</b> — giving 100 labelled instances in total. "
            "This satisfies all minimum requirements of the assignment and is the recommended "
            "starting point for testing the full pipeline before swapping in your own data."
        )
        if st.button("Load Sample Dataset"):
            df = pd.DataFrame(SAMPLE_DATA)
            st.session_state["dataset"] = df
            st.markdown('<div class="success-box">✅ Sample dataset loaded – 5 words, 2 senses each, 100 instances.</div>', unsafe_allow_html=True)
            _show_data_preview(df)

    # ── Dataset format guide ──────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📋 Expected Data Format")
    desc(
        "This section describes the exact structure your data file must follow. "
        "Each row represents one labelled instance: a single sentence that contains the target word used in a specific sense. "
        "The <code>sense</code> column holds a short human-readable label you define (e.g. <em>financial_institution</em>), "
        "while <code>sense_label</code> holds the corresponding WordNet synset ID (e.g. <em>bank.n.01</em>). "
        "Both are needed — the human label is used for display and evaluation, and the synset ID is used in Task 2 to look up definitions."
    )
    with st.expander("Show format details and CSV example"):
        st.markdown("""
        | Column | Type | Description |
        |---|---|---|
        | `word` | str | Target ambiguous word (e.g. `bank`) |
        | `sentence` | str | Full context sentence containing the word |
        | `sense` | str | Human-readable sense label (e.g. `financial_institution`) |
        | `sense_label` | str | WordNet synset ID (e.g. `bank.n.01`) |

        **CSV example:**
        ```
        word,sentence,sense,sense_label
        bank,"She deposited her salary into the bank account.",financial_institution,bank.n.01
        bank,"They camped on the grassy bank of the river.",river_bank,bank.n.09
        ```
        """)

    # ── Download sample CSV ───────────────────────────────────────────────────
    st.markdown("### ⬇️ Download Sample Template")
    desc(
        "Download the built-in sample dataset as a CSV file. "
        "You can use this as a template to understand the required format, "
        "then replace the rows with your own sentences and sense labels."
    )
    sample_df = pd.DataFrame(SAMPLE_DATA)
    csv_bytes = sample_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download sample CSV template", csv_bytes, "wsd_sample_dataset.csv", "text/csv")


def _show_data_preview(df: pd.DataFrame):
    words = df["word"].unique().tolist()
    senses = df["sense"].unique().tolist()

    st.markdown("### 📊 Dataset Summary")
    desc(
        "A quick snapshot of the loaded dataset. These four numbers confirm whether the dataset "
        "meets the assignment's minimum requirements at a glance — at least 5 words, at least 100 total instances, "
        "and a reasonable average per word."
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="label">Total Instances</div><div class="value">{len(df)}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="label">Unique Words</div><div class="value">{len(words)}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="label">Unique Senses</div><div class="value">{len(senses)}</div></div>', unsafe_allow_html=True)
    with col4:
        avg = round(len(df) / max(len(words), 1), 1)
        st.markdown(f'<div class="metric-card"><div class="label">Avg per Word</div><div class="value">{avg}</div></div>', unsafe_allow_html=True)

    st.markdown("#### First 10 rows")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("#### Sense distribution")
    desc(
        "This table shows how many instances exist for each word-sense combination. "
        "A balanced distribution (roughly equal counts per sense) helps train a fair WSD classifier. "
        "Imbalanced counts can bias the classifier towards the majority sense."
    )
    dist = df.groupby(["word", "sense"]).size().reset_index(name="count")
    st.dataframe(dist, use_container_width=True)
