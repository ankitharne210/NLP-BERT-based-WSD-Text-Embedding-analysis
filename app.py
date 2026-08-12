"""
NLP Assignment 2 – DSECLZG530
Contextual Word Sense Disambiguation using BERT
BITS Pilani M.Tech NLP Assignment
"""

import streamlit as st

st.set_page_config(
    page_title="WSD with BERT | DSECLZG530",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .task-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: #e2e8f0;
    padding: 1.4rem 1.8rem;
    border-radius: 12px;
    margin-bottom: 1.2rem;
    border-left: 4px solid #4fc3f7;
  }
  .task-header h2 { margin: 0; font-size: 1.25rem; font-weight: 700; color: #fff; }
  .task-header p  { margin: 0.3rem 0 0; font-size: 0.85rem; color: #94a3b8; }

  .metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
  }
  .metric-card .label { font-size: 0.75rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }
  .metric-card .value { font-size: 2rem; font-weight: 700; color: #0f3460; margin: .2rem 0 0; }

  .info-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.88rem;
    color: #1e40af;
    margin-bottom: 1rem;
  }
  .warn-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.88rem;
    color: #92400e;
    margin-bottom: 1rem;
  }
  .success-box {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.88rem;
    color: #166534;
    margin-bottom: 1rem;
  }

  .stButton > button {
    background: #0f3460;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: .55rem 1.4rem;
    transition: background .2s;
  }
  .stButton > button:hover { background: #1a4a8a; }

  .sidebar-badge {
    display: inline-block;
    background: #0f3460;
    color: white;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-left: 6px;
  }
  code { font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔤 WSD with BERT")
    st.caption("DSECLZG530 · NLP Assignment 2")
    st.divider()

    pages = {
        "🏠  Home & Data Input":        "home",
        "📚  Task 1 – Dataset Prep":    "task1",
        "🧠  Task 2 – WordNet Analysis": "task2",
        "🤖  Task 3 – BERT Embeddings":  "task3",
        "🎯  Task 4 – WSD Classifier":   "task4",
        "🔍  Task 5 – Error Analysis":   "task5",
    }

    selected = st.radio("Navigate", list(pages.keys()), label_visibility="collapsed")
    page = pages[selected]

    st.divider()
    st.caption("M.Tech · BITS Pilani WILP\nDSECLZG530 – Natural Language Processing")

    # Session state status indicators
    if "dataset" in st.session_state and st.session_state["dataset"] is not None:
        st.success("✅ Dataset loaded")
    else:
        st.info("⬆️ Upload data on Home page")

    if "bert_embeddings" in st.session_state:
        st.success("✅ BERT embeddings ready")

    if "classifier_results" in st.session_state:
        st.success("✅ WSD classifier trained")


# ─── Route to pages ───────────────────────────────────────────────────────────
if page == "home":
    from pages import page_home
    page_home.render()

elif page == "task1":
    from pages import page_task1
    page_task1.render()

elif page == "task2":
    from pages import page_task2
    page_task2.render()

elif page == "task3":
    from pages import page_task3
    page_task3.render()

elif page == "task4":
    from pages import page_task4
    page_task4.render()

elif page == "task5":
    from pages import page_task5
    page_task5.render()
