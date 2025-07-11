"""
Main Streamlit application for Speaker Preference AI
"""

import streamlit as st
import requests
import json
import time
from typing import Dict, Any

from components.speaker_table import display_speaker_table

# Configure Streamlit page
st.set_page_config(
    page_title="Speaker Preference AI",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

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

def get_system_stats() -> Dict[str, Any]:
    """Get system statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception:
        return {}

def search_speakers(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Search for speakers using the API"""
    try:
        payload = {
            "query": query,
            "max_results": max_results
        }
        
        response = requests.post(
            f"{API_BASE_URL}/search",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Search failed: {response.status_code} - {response.text}")
            return {}
            
    except Exception as e:
        st.error(f"Error searching speakers: {str(e)}")
        return {}

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🎤 Speaker Preference AI")
    st.markdown("### Find the Perfect Speakers for Your Event")
    st.markdown("---")
    
    # Check API health
    if not check_api_health():
        st.error("🚨 **API Service Unavailable**")
        st.info("Please ensure the FastAPI server is running on http://localhost:8000")
        
        with st.expander("🔧 How to Start the API Server"):
            st.code("""
# In your terminal, navigate to the project root and run:
cd src/api
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or alternatively:
python src/api/main.py
            """)
        st.stop()
    
    # System stats in sidebar
    with st.sidebar:
        st.header("📊 System Status")
        stats = get_system_stats()
        if stats:
            st.metric("Total Speakers", stats.get('total_speakers', 0))
            st.write(f"**Database:** {stats.get('collection_name', 'Unknown')}")
        else:
            st.write("Stats unavailable")
    
    # Main search interface
    st.subheader("🔍 Search for Speakers")
    
    # Search input
    query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., 'cloud computing experts', 'data center specialists', 'executive speakers with business focus'",
        help="Describe what kind of speaker you're looking for. Be as specific as possible about topics, expertise, or audience type."
    )
    
    # Search button
    if st.button("🔍 Search Speakers", type="primary", use_container_width=True):
        if query.strip():
            with st.spinner("🤖 AI is analyzing your query and finding the best speakers..."):
                # Perform search
                results = search_speakers(query.strip())
                
                if results:
                    # Store results in session state
                    st.session_state['search_results'] = results
                    st.session_state['last_query'] = query.strip()
                else:
                    st.error("No results found. Please try a different query.")
        else:
            st.warning("Please enter a search query.")
    
    # Display results if available
    if 'search_results' in st.session_state and st.session_state['search_results']:
        results = st.session_state['search_results']
        
        st.markdown("---")
        st.subheader(f"🎯 Search Results for: '{st.session_state.get('last_query', '')}'")
        
        # Search metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Speakers Found", results.get('total_results', 0))
        with col2:
            st.metric("Search Time", f"{results.get('search_time_ms', 0)}ms")
        with col3:
            intent = results.get('query_analysis', {}).get('intent', 'Unknown')
            st.write(f"**Query Intent:** {intent}")
        
        # AI Explanation
        st.subheader("🧠 AI Analysis & Explanation")
        explanation = results.get('explanation', '')
        if explanation:
            st.markdown(explanation)
        
        # AI Recommendation
        recommendation = results.get('recommendation', '')
        if recommendation:
            st.subheader("⭐ AI Recommendation")
            st.info(recommendation)
        
        # Speaker Results Table
        st.subheader("👥 Speaker Results")
        speakers = results.get('speakers', [])
        if speakers:
            display_speaker_table(speakers)
        else:
            st.warning("No speakers found in results.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>🤖 Powered by NVIDIA AI Services | Built with Streamlit & FastAPI</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()