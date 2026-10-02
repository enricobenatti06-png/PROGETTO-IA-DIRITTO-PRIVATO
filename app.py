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


# ============================================================
# CONFIGURAZIONE PAGINA
# ============================================================

st.set_page_config(
    page_title="Assistente di Diritto Privato",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# STILE
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------------------------------------------
       CONTENITORE PRINCIPALE
       ------------------------------------------------------- */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }


    /* -------------------------------------------------------
       CHAT
       ------------------------------------------------------- */

    [data-testid="stChatMessage"] {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }


    /* -------------------------------------------------------
       SIDEBAR
       ------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        padding-top: 1rem;
    }


    /* -------------------------------------------------------
       TITOLO
       ------------------------------------------------------- */

    h1 {
        margin-bottom: 0.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# RIPRISTINO SESSIONE SUPABASE
# ============================================================

def restore_session():
    """
    Cerca di recuperare una sessione Supabase già esistente.

    Questo permette all'utente di non dover effettuare
    nuovamente il login ad ogni refresh della pagina,
    quando la sessione è ancora valida.
    """

    # Se Streamlit ha già l'utente
    # nella session state, non facciamo nulla.

    if is_logged_in():

        return True


    try:

        session_response = (
            supabase
            .auth
            .get_session()
        )


        if session_response is None:

            return False


        # In base alla versione del client Supabase,
        # get_session() può restituire direttamente
        # la sessione oppure un oggetto che la contiene.

        session = session_response


        if hasattr(
            session_response,
            "session"
        ):

            session = session_response.session


        if session is None:

            return False


        user = getattr(
            session,
            "user",
            None
        )


        if user is None:

            return False


        # Carica utente + profilo + ruolo
        load_user_session(
            user
        )


        return True


    except Exception:

        return False


# ============================================================
# CONTROLLO AUTENTICAZIONE
# ============================================================

logged_in = restore_session()


# ============================================================
# UTENTE NON AUTENTICATO
# ============================================================

if not logged_in:

    show_auth_page()

    st.stop()


# ============================================================
# UTENTE AUTENTICATO
# ============================================================

user = get_current_user()


# ============================================================
# SIDEBAR UTENTE
# ============================================================

show_user_sidebar()


# ============================================================
# HEADER PRINCIPALE
# ============================================================

st.title(
    "⚖️ Assistente di Diritto Privato"
)

st.caption(
    "Assistente giuridico basato su fonti normative, "
    "dottrina, giurisprudenza e materiale didattico."
)


# ============================================================
# CHAT
# ============================================================

show_chat(
    user
)
