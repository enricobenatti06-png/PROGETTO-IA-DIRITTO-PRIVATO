import streamlit as st

from chat import show_chat


st.set_page_config(
    page_title="Assistente di Diritto Privato",
    page_icon="⚖️",
    layout="wide"
)


# =====================================
# CONTROLLO AUTENTICAZIONE
# =====================================

if "user" not in st.session_state:

    st.title("Assistente di Diritto Privato")

    st.write(
        "Effettua il login per accedere alle conversazioni."
    )

    st.stop()


# =====================================
# UTENTE AUTENTICATO
# =====================================

user = st.session_state.user


# =====================================
# CHAT
# =====================================

show_chat(user)
