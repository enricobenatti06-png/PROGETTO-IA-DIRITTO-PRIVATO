import streamlit as st
from supabase import create_client


@st.cache_resource
def get_supabase():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )


supabase = get_supabase()


# =========================
# PROFILO
# =========================

def get_profile(user_id):
    response = (
        supabase
        .table("profiles")
        .select("*")
        .eq("id", user_id)
        .single()
        .execute()
    )

    return response.data


def update_profile(user_id, data):
    response = (
        supabase
        .table("profiles")
        .update(data)
        .eq("id", user_id)
        .execute()
    )

    return response.data


# =========================
# CONVERSAZIONI
# =========================

def get_my_conversations(user_id):
    response = (
        supabase
        .table("conversations")
        .select("*")
        .eq("user_id", user_id)
        .eq("is_public", False)
        .order("updated_at", desc=True)
        .execute()
    )

    return response.data or []


def get_public_conversations():
    response = (
        supabase
        .table("conversations")
        .select("*")
        .eq("is_public", True)
        .order("updated_at", desc=True)
        .execute()
    )

    return response.data or []


def get_conversation(conversation_id, user_id):
    response = (
        supabase
        .table("conversations")
        .select("*")
        .eq("id", conversation_id)
        .or_(
            f"user_id.eq.{user_id},is_public.eq.true"
        )
        .single()
        .execute()
    )

    return response.data


def create_conversation(
    user_id,
    title="Nuova conversazione",
    is_public=False
):
    data = {
        "user_id": user_id,
        "title": title,
        "is_public": is_public
    }

    response = (
        supabase
        .table("conversations")
        .insert(data)
        .execute()
    )

    if not response.data:
        raise Exception(
            "Impossibile creare la conversazione."
        )

    return response.data[0]


def update_conversation_title(
    conversation_id,
    title
):
    response = (
        supabase
        .table("conversations")
        .update({"title": title})
        .eq("id", conversation_id)
        .execute()
    )

    return response.data


def update_conversation_visibility(
    conversation_id,
    is_public
):
    response = (
        supabase
        .table("conversations")
        .update({"is_public": is_public})
        .eq("id", conversation_id)
        .execute()
    )

    return response.data


def delete_conversation(conversation_id):
    response = (
        supabase
        .table("conversations")
        .delete()
        .eq("id", conversation_id)
        .execute()
    )

    return response.data


# =========================
# MESSAGGI
# =========================

def get_messages(conversation_id):
    response = (
        supabase
        .table("messages")
        .select("*")
        .eq("conversation_id", conversation_id)
        .order("created_at", desc=False)
        .execute()
    )

    return response.data or []


def create_message(
    conversation_id,
    role,
    content
):
    data = {
        "conversation_id": conversation_id,
        "role": role,
        "content": content
    }

    response = (
        supabase
        .table("messages")
        .insert(data)
        .execute()
    )

    if not response.data:
        raise Exception(
            "Impossibile salvare il messaggio."
        )

    return response.data[0]


def conversation_exists(conversation_id):
    response = (
        supabase
        .table("conversations")
        .select("id")
        .eq("id", conversation_id)
        .execute()
    )

    return bool(response.data)


def count_messages(conversation_id):
    response = (
        supabase
        .table("messages")
        .select("id", count="exact")
        .eq("conversation_id", conversation_id)
        .execute()
    )

    return response.count or 0
