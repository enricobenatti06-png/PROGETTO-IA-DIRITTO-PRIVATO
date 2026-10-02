import streamlit as st

from auth import (
    is_logged_in,
    get_current_user,
    show_auth_page,
    show_user_sidebar,
    load_user_session,
    supabase
)

from chat import show_chat


st.set_page_config(
    page_title="Assistente di Diritto Privato",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


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


def restore_session():
    if is_logged_in():
        return True

    try:
        session_response = supabase.auth.get_session()

        if session_response is None:
            return False

        session = session_response

        if hasattr(session_response, "session"):
            session = session_response.session

        if session is None:
            return False

        user = getattr(session, "user", None)

        if user is None:
            return False

        load_user_session(user)

        return True

    except Exception:
        return False


logged_in = restore_session()


if not logged_in:
    show_auth_page()
    st.stop()


user = get_current_user()

show_user_sidebar()


st.title("⚖️ Assistente di Diritto Privato")

st.caption(
    "Assistente giuridico basato su fonti normative, "
    "dottrina, giurisprudenza e materiale didattico."
)


show_chat(user)
