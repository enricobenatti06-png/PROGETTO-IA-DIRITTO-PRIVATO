import streamlit as st

from database import (
    get_my_conversations,
    get_public_conversations,
    get_conversation,
    create_conversation,
    get_messages,
    create_message,
    update_conversation_title
)

from rag import answer_question


MAX_TITLE_LENGTH = 60


# =========================
# SESSIONE CHAT
# =========================

def initialize_chat_state():

    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None

    if "chat_mode" not in st.session_state:
        st.session_state.chat_mode = "private"


# =========================
# TITOLO
# =========================

def generate_title(question):

    question = question.strip()

    if not question:
        return "Nuova conversazione"

    if len(question) <= MAX_TITLE_LENGTH:
        return question

    return (
        question[:MAX_TITLE_LENGTH]
        .rstrip()
        + "..."
    )


# =========================
# NUOVA CONVERSAZIONE
# =========================

def start_new_conversation(user_id):

    conversation = create_conversation(
        user_id=user_id,
        title="Nuova conversazione",
        is_public=False
    )

    st.session_state.conversation_id = conversation["id"]
    st.session_state.chat_mode = "private"

    st.rerun()


# =========================
# SIDEBAR
# =========================

def show_conversation_sidebar(user):

    user_id = user.id

    with st.sidebar:

        st.title("Conversazioni")

        if st.button(
            "＋ Nuova conversazione",
            use_container_width=True
        ):
            start_new_conversation(user_id)

        st.divider()

        # -------------------------
        # PRIVATE
        # -------------------------

        st.subheader("Le mie conversazioni")

        private_conversations = get_my_conversations(
            user_id
        )

        if not private_conversations:
            st.caption(
                "Nessuna conversazione privata."
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

                st.session_state.conversation_id = (
                    conversation_id
                )

                st.session_state.chat_mode = "private"

                st.rerun()

        st.divider()

        # -------------------------
        # PUBLIC
        # -------------------------

        st.subheader("Conversazioni pubbliche")

        public_conversations = (
            get_public_conversations()
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

                st.session_state.conversation_id = (
                    conversation_id
                )

                st.session_state.chat_mode = "public"

                st.rerun()


# =========================
# MESSAGGI
# =========================

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


# =========================
# RAG
# =========================

def ask_rag(question):

    try:

        return answer_question(
            question,
            top_k=5
        )

    except Exception as e:

        return {
            "answer": (
                "Si è verificato un errore "
                "durante l'elaborazione della domanda."
            ),
            "documents": [],
            "context": "",
            "error": str(e)
        }


# =========================
# FONTI
# =========================

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
                    + (
                        "..."
                        if len(text) > 500
                        else ""
                    )
                )


# =========================
# PROCESSA DOMANDA
# =========================

def process_message(
    conversation_id,
    prompt
):

    create_message(
        conversation_id,
        "user",
        prompt
    )

    result = ask_rag(prompt)

    answer = result.get(
        "answer",
        "Non è stato possibile generare una risposta."
    )

    create_message(
        conversation_id,
        "assistant",
        answer
    )

    return result


# =========================
# CHAT PRINCIPALE
# =========================

def show_chat(user):

    initialize_chat_state()

    user_id = user.id

    show_conversation_sidebar(user)

    conversation_id = (
        st.session_state.conversation_id
    )

    if conversation_id is None:

        st.title(
            "⚖️ Assistente di Diritto Privato"
        )

        st.write(
            "Benvenuto nell'assistente "
            "di diritto privato."
        )

        st.info(
            "Crea una nuova conversazione "
            "per iniziare."
        )

        return

    # Controllo accesso alla conversazione

    conversation = get_conversation(
        conversation_id,
        user_id
    )

    if conversation is None:

        st.session_state.conversation_id = None

        st.error(
            "Conversazione non disponibile."
        )

        st.rerun()

    # Titolo conversazione

    st.subheader(
        conversation.get(
            "title",
            "Conversazione"
        )
    )

    # Messaggi

    messages = show_messages(
        conversation_id
    )

    # Input

    prompt = st.chat_input(
        "Scrivi la tua domanda di diritto privato..."
    )

    if prompt:

        prompt = prompt.strip()

        if not prompt:
            return

        # Una conversazione pubblica appartenente
        # ad altro utente è in sola lettura.

        is_owner = (
            conversation.get("user_id")
            == user_id
        )

        if (
            conversation.get("is_public", False)
            and not is_owner
        ):

            st.warning(
                "Questa è una conversazione pubblica "
                "in sola lettura."
            )

            return

        is_first_message = (
            len(messages) == 0
        )

        with st.spinner(
            "Sto consultando le fonti giuridiche..."
        ):

            result = process_message(
                conversation_id,
                prompt
            )

        if is_first_message:

            title = generate_title(
                prompt
            )

            update_conversation_title(
                conversation_id,
                title
            )

        show_sources(
            result.get(
                "documents",
                []
            )
        )

        st.rerun()
