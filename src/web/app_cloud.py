"""
Cloud-deployed Streamlit application for Speaker Preference AI
"""

import streamlit as st
import requests
import json
import logging
from typing import Dict, Any, List, Optional
import os

# Import your existing components
from .components.speaker_table import display_speaker_table
from core.config_cloud import cloud_config

# Configure Streamlit page
st.set_page_config(
    page_title="Speaker Preference AI",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# API Client for cloud deployment
class CloudAPIClient:
    """API client for cloud deployment"""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 30

    def check_health(self) -> bool:
        """Check if API is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def search_speakers(
        self, query: str, conversation_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """Search for speakers"""
        try:
            payload = {
                "query": query,
                "conversation_history": conversation_history or [],
                "max_results": 15,
            }

            response = self.session.post(
                f"{self.base_url}/api/v1/search",
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return {"error": True, "message": "API request failed"}

        except Exception as e:
            logger.error(f"Search request failed: {e}")
            return {"error": True, "message": f"Connection error: {str(e)}"}

    def refine_speakers(
        self,
        query: str,
        conversation_history: List[Dict] = None,
        current_speakers: List[Dict] = None,
    ) -> Dict[str, Any]:
        """Refine speaker search"""
        try:
            payload = {
                "query": query,
                "conversation_history": conversation_history or [],
                "current_speakers": current_speakers or [],
                "max_results": 15,
            }

            response = self.session.post(
                f"{self.base_url}/api/v1/refine",
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return {"error": True, "message": "API request failed"}

        except Exception as e:
            logger.error(f"Refine request failed: {e}")
            return {"error": True, "message": f"Connection error: {str(e)}"}


# Initialize API client
api_client = CloudAPIClient(cloud_config.API_BASE_URL)


def initialize_session_state():
    """Initialize Streamlit session state"""
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "displayed_speakers" not in st.session_state:
        st.session_state.displayed_speakers = []
    if "last_explanation" not in st.session_state:
        st.session_state.last_explanation = ""
    if "last_recommendation" not in st.session_state:
        st.session_state.last_recommendation = ""
    if "is_first_search" not in st.session_state:
        st.session_state.is_first_search = True


def detect_search_type(query: str, has_current_speakers: bool) -> str:
    """Detect if this is a new search or refinement"""
    refinement_keywords = [
        "only",
        "just",
        "remove",
        "exclude",
        "filter",
        "narrow",
        "refine",
        "based in",
        "from",
        "located in",
        "except",
        "without",
        "not",
    ]

    query_lower = query.lower()

    if has_current_speakers and any(
        keyword in query_lower for keyword in refinement_keywords
    ):
        return "refinement"

    return "new_search"


def convert_speakers_for_api(speakers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert speakers to API format"""
    return [
        {
            "speaker_id": speaker.get("speaker_id"),
            "name": speaker.get("name"),
            "job_title": speaker.get("job_title"),
            "company": speaker.get("company"),
            "speaking_topics": speaker.get("speaking_topics", []),
            "bio": speaker.get("bio"),
            "specializations": speaker.get("specializations"),
            "audiences": speaker.get("audiences"),
            "centers": speaker.get("centers"),
            "similarity_score": speaker.get("similarity_score", 0.0),
            "rerank_score": speaker.get("rerank_score"),
        }
        for speaker in speakers
    ]


def process_user_message(prompt: str):
    """Process user message and update conversation state"""

    # Add user message to history immediately
    st.session_state.conversation_history.append({"role": "user", "content": prompt})

    # Show the user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Determine search type
    search_type = detect_search_type(prompt, bool(st.session_state.displayed_speakers))

    # Process with spinner
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Make API call
            if search_type == "new_search" or st.session_state.is_first_search:
                api_response = api_client.search_speakers(
                    prompt, st.session_state.conversation_history[:-1]
                )
                st.session_state.is_first_search = False
            else:
                current_speakers_dict = convert_speakers_for_api(
                    st.session_state.displayed_speakers
                )
                api_response = api_client.refine_speakers(
                    prompt,
                    st.session_state.conversation_history[:-1],
                    current_speakers_dict,
                )

        # Process API response
        if api_response:
            if api_response.get("error"):
                response_content = f"❌ {api_response.get('message', 'I can only help with speaker-related queries.')}"
                if api_response.get("suggestion"):
                    response_content += (
                        f"\n\n**Suggestion:** {api_response.get('suggestion')}"
                    )

                # Show and save error response
                st.markdown(response_content)
                st.session_state.conversation_history.append(
                    {"role": "assistant", "content": response_content}
                )
            else:
                # Update all state for successful response
                st.session_state.displayed_speakers = api_response.get("speakers", [])
                st.session_state.last_explanation = api_response.get(
                    "explanation", "Here are the speakers I found."
                )
                st.session_state.last_recommendation = api_response.get(
                    "recommendation", ""
                )

                # Show AI response immediately
                ai_message = st.session_state.last_explanation
                st.markdown(ai_message)

                # Add to conversation history
                st.session_state.conversation_history.append(
                    {"role": "assistant", "content": ai_message}
                )

                # Force rerun to show updated speaker table
                st.rerun()
        else:
            error_message = "I'm having trouble connecting to the backend service. Please try again in a moment."
            st.markdown(error_message)
            st.session_state.conversation_history.append(
                {"role": "assistant", "content": error_message}
            )


def main():
    """Main Streamlit application flow"""
    st.title("🎤 Conversational Speaker Preference AI")
    st.markdown("### Find the perfect speakers for your event through conversation.")

    # Show API connection status
    with st.sidebar:
        st.markdown("### Connection Status")
        if api_client.check_health():
            st.success("✅ Connected to backend")
        else:
            st.error("❌ Backend unavailable")
            st.info(f"Trying to connect to: {cloud_config.API_BASE_URL}")

    # Check API health
    if not api_client.check_health():
        st.error("🚨 **Backend Service Not Available**")
        st.info("The backend service is currently unavailable. Please try again later.")
        st.info(f"Backend URL: {cloud_config.API_BASE_URL}")
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
        process_user_message(prompt)


if __name__ == "__main__":
    main()
