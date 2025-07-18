"""
Conversational Streamlit application for Speaker Preference AI
"""

import streamlit as st
import requests
import json
import time
from typing import Dict, Any, List, Optional

from components.speaker_table import display_speaker_table

# Configure Streamlit page
st.set_page_config(
    page_title="Conversational Speaker AI",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"


# --- State Management Initialization ---
def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "displayed_speakers" not in st.session_state:
        st.session_state.displayed_speakers = []
    if "last_query_analysis" not in st.session_state:
        st.session_state.last_query_analysis = {}
    if "last_explanation" not in st.session_state:
        st.session_state.last_explanation = ""
    if "last_recommendation" not in st.session_state:
        st.session_state.last_recommendation = ""
    if "is_first_search" not in st.session_state:
        st.session_state.is_first_search = True


# --- API Communication ---
def call_search_api(
    query: str, history: List[Dict[str, str]]
) -> Optional[Dict[str, Any]]:
    """Call the backend search API for initial searches."""
    try:
        payload = {"query": query, "max_results": 15, "conversation_history": history}
        response = requests.post(f"{API_BASE_URL}/search", json=payload, timeout=1000)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
        return None


def call_refine_api(
    query: str, history: List[Dict[str, str]], current_speakers: List[Dict]
) -> Optional[Dict[str, Any]]:
    """Call the backend refine API for conversational refinements."""
    try:
        payload = {
            "query": query,
            "max_results": 15,
            "conversation_history": history,
            "current_speakers": current_speakers,
        }
        response = requests.post(f"{API_BASE_URL}/refine", json=payload, timeout=1000)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Refine API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Failed to connect to refine API: {e}")
        return None


def check_api_health() -> bool:
    """Check if API is healthy and ready"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("system_ready", False)
        return False
    except Exception:
        return False


def detect_search_type(prompt: str, has_current_speakers: bool) -> str:
    """Determine if this should be a new search or refinement"""
    if not has_current_speakers:
        return "new_search"

    # Keywords that suggest refinement rather than new search
    refinement_keywords = [
        "remove",
        "delete",
        "take out",
        "eliminate",
        "only show",
        "filter",
        "narrow down",
        "from",
        "based in",
        "located in",
        "in the region",
        "first",
        "top",
        "best",
        "except",
        "without",
    ]

    prompt_lower = prompt.lower()
    if any(keyword in prompt_lower for keyword in refinement_keywords):
        return "refinement"

    # Default to new search for clarity
    return "new_search"


def convert_speakers_for_api(speakers: List[Dict]) -> List[Dict]:
    """Convert SpeakerResult objects to dict format for API"""
    converted = []
    for speaker in speakers:
        if hasattr(speaker, "__dict__"):
            # It's a Pydantic model, convert to dict
            converted.append(speaker.__dict__)
        else:
            # It's already a dict
            converted.append(speaker)
    return converted


# --- Main Application Logic ---
def main():
    """Main Streamlit application flow."""
    st.title("🎤 Conversational Speaker Preference AI")
    st.markdown("### Find the perfect speakers for your event through conversation.")

    # Check API health first
    if not check_api_health():
        st.error("🚨 **Backend Service Not Ready**")
        st.info("Please ensure the FastAPI server is running and fully initialized.")
        st.stop()

    initialize_session_state()

    # Display the chat history
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Display the current speaker results if they exist
    if st.session_state.displayed_speakers:
        st.markdown("---")
        st.subheader("Current Speaker Recommendations")
        if st.session_state.last_explanation:
            st.markdown(st.session_state.last_explanation)
        if st.session_state.last_recommendation:
            st.info(
                f"**⭐ Top Recommendation:** {st.session_state.last_recommendation}"
            )

        display_speaker_table(st.session_state.displayed_speakers)

    # User input
    if prompt := st.chat_input(
        "What are you looking for? Or, refine your last search."
    ):
        # Add user message to history immediately
        st.session_state.conversation_history.append(
            {"role": "user", "content": prompt}
        )

        # Determine if this is a new search or refinement
        search_type = detect_search_type(
            prompt, bool(st.session_state.displayed_speakers)
        )

        # Process the user's prompt
        api_response = None

        with st.spinner("Thinking..."):
            if search_type == "new_search" or st.session_state.is_first_search:
                # Use regular search API
                api_response = call_search_api(
                    prompt, st.session_state.conversation_history[:-1]
                )
                st.session_state.is_first_search = False
            else:
                # Use refinement API
                current_speakers_dict = convert_speakers_for_api(
                    st.session_state.displayed_speakers
                )
                api_response = call_refine_api(
                    prompt,
                    st.session_state.conversation_history[:-1],
                    current_speakers_dict,
                )

        # Process API response and update state
        if api_response:
            # Check for a structured error from the backend
            if api_response.get("error"):
                response_content = f"❌ {api_response.get('message', 'I can only help with speaker-related queries.')}"
                if api_response.get("suggestion"):
                    response_content += (
                        f"\n\n**Suggestion:** {api_response.get('suggestion')}"
                    )

                # Add AI response to conversation history
                st.session_state.conversation_history.append(
                    {"role": "assistant", "content": response_content}
                )
            else:
                # It's a successful search result - update all state
                st.session_state.displayed_speakers = api_response.get("speakers", [])
                st.session_state.last_query_analysis = api_response.get(
                    "query_analysis", {}
                )
                st.session_state.last_explanation = api_response.get(
                    "explanation", "Here are the speakers I found."
                )
                st.session_state.last_recommendation = api_response.get(
                    "recommendation", ""
                )

                # Add the AI's explanation to chat history
                ai_message = st.session_state.last_explanation
                st.session_state.conversation_history.append(
                    {"role": "assistant", "content": ai_message}
                )
        else:
            # Handle case where API call fails completely
            error_message = "I'm having trouble connecting to my services. The search couldn't be updated. Please try again in a moment."
            st.session_state.conversation_history.append(
                {"role": "assistant", "content": error_message}
            )

        # Rerun to refresh the display with updated conversation and speakers
        st.rerun()


# --- Entry Point ---
if __name__ == "__main__":
    main()
