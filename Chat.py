import streamlit as st

from database import (
    get_conversations,
    create_conversation,
    get_messages,
    create_message,
    update_conversation_title
)

from rag import answer_question


# ============================================================
# CONFIGURAZIONE
# ============================================================

MAX_TITLE_LENGTH = 60


# ============================================================
# SESSION STATE
# ============================================================

def initialize_chat_state():

    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None

    if "chat_mode" not in st.session_state:
        st.session_state.chat_mode = "private"


# ============================================================
# TITOLO
# ============================================================

def generate_title(question):

    question = question.strip()

    if not question:
        return "Nuova conversazione"

    if len(question) <= MAX_TITLE_LENGTH:
        return question

    return question[:MAX_TITLE_LENGTH].rstrip() + "..."


# ============================================================
# NUOVA CONVERSAZIONE
# ============================================================

def start_new_conversation(user_id):

    conversation = create_conversation(
        user_id,
        "Nuova conversazione"
    )

    st.session_state.conversation_id = conversation["id"]

    st.session_state.chat_mode = "private"

    st.rerun()


# ============================================================
# SIDEBAR
# ============================================================

def show_conversation_sidebar(user):

    user_id = user.id

    with st.sidebar:

        st.title("Conversazioni")


        # ----------------------------------------------------
        # NUOVA CHAT
        # ----------------------------------------------------

        if st.button(
            "＋ Nuova conversazione",
            use_container_width=True
        ):

            start_new_conversation(user_id)


        st.divider()


        # ----------------------------------------------------
        # CONVERSAZIONI PRIVATE
        # ----------------------------------------------------

        st.subheader("Le mie conversazioni")

        private_conversations = get_conversations(
            user_id=user_id,
            public_only=False
        )


        if not private_conversations:

            st.caption(
                "Nessuna conversazione."
            )


        for conversation in private_conversations:

            conversation_id = conversation["id"]

            title = conversation.get(
                "title",
                "Nuova conversazione"
            )


            if st.button(
                title,
                key=f"private_{conversation_id}",
                use_container_width=True
            ):

                st.session_state.conversation_id = conversation_id
                st.session_state.chat_mode = "private"

                st.rerun()


        # ----------------------------------------------------
        # CONVERSAZIONI PUBBLICHE
        # ----------------------------------------------------

        st.divider()

        st.subheader("Conversazioni pubbliche")

        public_conversations = get_conversations(
            user_id=user_id,
            public_only=True
        )


        if not public_conversations:

            st.caption(
                "Nessuna conversazione pubblica."
            )


        for conversation in public_conversations:

            conversation_id = conversation["id"]

            title = conversation.get(
                "title",
                "Conversazione pubblica"
            )


            if st.button(
                f"🌐 {title}",
                key=f"public_{conversation_id}",
                use_container_width=True
            ):

                st.session_state.conversation_id = conversation_id
                st.session_state.chat_mode = "public"

                st.rerun()


# ============================================================
# VISUALIZZAZIONE MESSAGGI
# ============================================================

def show_messages(conversation_id):

    messages = get_messages(
        conversation_id
    )


    for message in messages:

        role = message.get(
            "role",
            "assistant"
        )

        content = message.get(
            "content",
            ""
        )


        if role == "user":

            with st.chat_message("user"):

                st.write(content)


        elif role == "assistant":

            with st.chat_message("assistant"):

                st.markdown(content)


        elif role == "system":

            with st.chat_message("assistant"):

                st.caption(content)


    return messages


# ============================================================
# RAG
# ============================================================

def ask_rag(question):

    try:

        result = answer_question(
            question,
            top_k=5
        )

        return result

    except Exception as e:

        return {
            "answer": (
                "Si è verificato un errore durante "
                "l'elaborazione della domanda."
            ),
            "documents": [],
            "context": "",
            "error": str(e)
        }


# ============================================================
# FONTI
# ============================================================

def show_sources(documents):

    if not documents:
        return

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

                st.caption(
                    preview
                    + ("..." if len(text) > 500 else "")
                )


# ============================================================
# INVIO MESSAGGIO
# ============================================================

def process_message(
    conversation_id,
    prompt
):

    # --------------------------------------------------------
    # SALVA DOMANDA
    # --------------------------------------------------------

    create_message(
        conversation_id,
        "user",
        prompt
    )


    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    result = ask_rag(
        prompt
    )


    answer = result.get(
        "answer",
        "Non è stato possibile generare una risposta."
    )


    # --------------------------------------------------------
    # SALVA RISPOSTA
    # --------------------------------------------------------

    create_message(
        conversation_id,
        "assistant",
        answer
    )


    return result


# ============================================================
# CHAT PRINCIPALE
# ============================================================

def show_chat(user):

    initialize_chat_state()


    user_id = user.id


    # ========================================================
    # SIDEBAR
    # ========================================================

    show_conversation_sidebar(
        user
    )


    # ========================================================
    # NESSUNA CONVERSAZIONE
    # ========================================================

    conversation_id = st.session_state.conversation_id


    if conversation_id is None:

        st.title(
            "⚖️ Assistente di Diritto Privato"
        )

        st.write(
            "Benvenuto nell'assistente di diritto privato."
        )

        st.info(
            "Crea una nuova conversazione per iniziare."
        )

        return


    # ========================================================
    # TITOLO
    # ========================================================

    messages = show_messages(
        conversation_id
    )


    # ========================================================
    # INPUT
    # ========================================================

    prompt = st.chat_input(
        "Scrivi la tua domanda di diritto privato..."
    )


    if prompt:

        prompt = prompt.strip()


        if not prompt:
            return


        # ----------------------------------------------------
        # PRIMA DOMANDA
        # ----------------------------------------------------

        is_first_message = (
            len(messages) == 0
        )


        # ----------------------------------------------------
        # GENERAZIONE
        # ----------------------------------------------------

        with st.spinner(
            "Sto consultando le fonti giuridiche..."
        ):

            result = process_message(
                conversation_id,
                prompt
            )


        # ----------------------------------------------------
        # TITOLO
        # ----------------------------------------------------

        if is_first_message:

            title = generate_title(
                prompt
            )

            update_conversation_title(
                conversation_id,
                title
            )


        # ----------------------------------------------------
        # MOSTRA FONTI
        # ----------------------------------------------------

        documents = result.get(
            "documents",
            []
        )

        show_sources(
            documents
        )


        st.rerun()
