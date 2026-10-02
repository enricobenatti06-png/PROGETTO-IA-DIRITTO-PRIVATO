import streamlit as st

from chat import show_chat


st.set_page_config(
    page_title="Assistente di Diritto Privato",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# STILE
# =========================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    [data-testid="stChatMessage"] {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }

    section[data-testid="stSidebar"] {
        padding-top: 1rem;
    }

    h1 {
        margin-bottom: 0.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# APPLICAZIONE
# =========================

st.title("⚖️ Assistente di Diritto Privato")

st.caption(
    "Assistente giuridico basato su fonti normative, "
    "dottrina, giurisprudenza e materiale didattico."
)


show_chat(
    user=type(
        "DemoUser",
        (),
        {
            "id": "demo-user",
            "email": "demo@progetto.local"
        }
    )()
)
