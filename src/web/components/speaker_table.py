"""
Speaker table component for displaying search results
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any

def display_speaker_table(speakers: List[Dict[str, Any]]):
    """Display speakers in a formatted table"""
    
    if not speakers:
        st.warning("No speakers found for your query.")
        return
    
    # Convert to DataFrame for better display
    table_data = []
    for i, speaker in enumerate(speakers, 1):
        # Format speaking topics
        topics = speaker.get('speaking_topics', [])
        if isinstance(topics, list):
            topics_str = ", ".join(topics[:3])  # Show first 3 topics
            if len(topics) > 3:
                topics_str += f" (+{len(topics)-3} more)"
        else:
            topics_str = str(topics)
        
        # Truncate bio for display
        bio = speaker.get('bio', '')
        if len(bio) > 200:
            bio = bio[:200] + "..."
        
        table_data.append({
            "Rank": i,
            "Speaker ID": speaker.get('speaker_id', ''),
            "Name": speaker.get('name', ''),
            "Job Title": speaker.get('job_title', ''),
            "Company": speaker.get('company', 'N/A'),
            "Speaking Topics": topics_str,
            "Biography": bio,
            "Specializations": speaker.get('specializations', 'N/A'),
            "Target Audiences": speaker.get('audiences', 'N/A')
        })
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Display table with custom styling
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Speaker ID": st.column_config.TextColumn("Speaker ID", width="medium"),
            "Name": st.column_config.TextColumn("Name", width="medium"),
            "Job Title": st.column_config.TextColumn("Job Title", width="large"),
            "Company": st.column_config.TextColumn("Company", width="medium"),
            "Speaking Topics": st.column_config.TextColumn("Speaking Topics", width="large"),
            "Biography": st.column_config.TextColumn("Biography", width="large"),
            "Specializations": st.column_config.TextColumn("Specializations", width="large"),
            "Target Audiences": st.column_config.TextColumn("Target Audiences", width="medium")
        }
    )
    
    # Show expandable details for each speaker
    st.subheader("📋 Detailed Speaker Information")
    
    for i, speaker in enumerate(speakers, 1):
        with st.expander(f"🎤 {speaker.get('name', 'Unknown')} - {speaker.get('job_title', 'Unknown Title')}"):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("**Speaker ID:**", speaker.get('speaker_id', 'N/A'))
                st.write("**Company:**", speaker.get('company', 'N/A'))
                st.write("**Target Audiences:**", speaker.get('audiences', 'N/A'))
            
            with col2:
                st.write("**Specializations:**")
                st.write(speaker.get('specializations', 'N/A'))
            
            st.write("**Complete Biography:**")
            st.write(speaker.get('bio', 'No biography available.'))
            
            st.write("**All Speaking Topics:**")
            topics = speaker.get('speaking_topics', [])
            if isinstance(topics, list):
                for topic in topics:
                    st.write(f"• {topic}")
            else:
                st.write(topics)