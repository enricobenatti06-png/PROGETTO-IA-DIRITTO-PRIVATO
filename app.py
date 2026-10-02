import streamlit as st

from rag import answer_question


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
# INTERFACCIA
# =========================

st.title("⚖️ Assistente di Diritto Privato")

st.caption(
    "Assistente giuridico basato sul Manuale Galgano "
    "e sul recupero semantico dei contenuti."
)


# =========================
# STATO DELLA CHAT
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================
# MESSAGGI PRECEDENTI
# =========================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        if message["role"] == "assistant":
            st.markdown(message["content"])

            documents = message.get("documents", [])

            if documents:

                with st.expander("📚 Fonti utilizzate"):

                    for index, document in enumerate(
                        documents,
                        start=1
                    ):

                        source = document.get(
                            "source",
                            "Fonte non specificata"
                        )

                        text = document.get(
                            "text",
                            ""
                        )

                        st.markdown(
                            f"**{index}. {source}**"
                        )

                        if text:

                            preview = text[:500]

                            if len(text) > 500:
                                preview += "..."

                            st.caption(preview)

        else:
            st.write(message["content"])


# =========================
# INPUT
# =========================

prompt = st.chat_input(
    "Scrivi la tua domanda di diritto privato..."
)


if prompt:

    prompt = prompt.strip()

    if prompt:

        # DOMANDA UTENTE
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        with st.chat_message("user"):
            st.write(prompt)


        # RAG
        with st.chat_message("assistant"):

            with st.spinner(
                "Sto consultando il Manuale Galgano..."
            ):

                try:

                    result = answer_question(
                        prompt,
                        top_k=5
                    )

                    answer = result.get(
                        "answer",
                        "Non è stato possibile generare una risposta."
                    )

                    documents = result.get(
                        "documents",
                        []
                    )

                    st.markdown(answer)


                    # FONTI
                    if documents:

                        with st.expander(
                            "📚 Fonti utilizzate"
                        ):

                            for index, document in enumerate(
                                documents,
                                start=1
                            ):

                                source = document.get(
                                    "source",
                                    "Fonte non specificata"
                                )

                                text = document.get(
                                    "text",
                                    ""
                                )

                                st.markdown(
                                    f"**{index}. {source}**"
                                )

                                if text:

                                    preview = text[:500]

                                    if len(text) > 500:
                                        preview += "..."

                                    st.caption(preview)


                    # SALVA NELLA SESSIONE
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "documents": documents
                        }
                    )


                except Exception as e:

                    error_message = (
                        "Si è verificato un errore "
                        "durante l'elaborazione della domanda."
                    )

                    st.error(error_message)

                    st.caption(
                        f"{type(e).__name__}: {str(e)}"
                    )
