"""
Data processor for loading, merging, and preparing speaker data for vector search
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import ast
from pathlib import Path
import logging
from datetime import datetime

from utils.data_cleaner import load_and_clean_data
from .config import config

logger = logging.getLogger(__name__)

class SpeakerDataProcessor:
    """Processes speaker and topic data for AI-powered search"""

    def __init__(self):
        self.speakers_df: Optional[pd.DataFrame] = None
        self.topics_df: Optional[pd.DataFrame] = None
        self.merged_df: Optional[pd.DataFrame] = None
        self.speaker_profiles: List[Dict[str, Any]] = []

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and clean speaker and topic data"""
        logger.info("Loading speaker and topic data...")

        try:
            self.speakers_df, self.topics_df = load_and_clean_data()

            logger.info(f"Loaded {len(self.speakers_df)} speakers and {len(self.topics_df)} topic entries")
            return self.speakers_df, self.topics_df

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def merge_data(self) -> pd.DataFrame:
        """Merge speaker and topic data with LEFT JOIN to preserve all speakers"""
        if self.speakers_df is None or self.topics_df is None:
            self.load_data()

        logger.info("Merging speaker and topic data...")

        # CHANGE: Use RIGHT JOIN to preserve ALL speakers, even without topics
        # This ensures we don't lose speakers who might not have topic mappings
        self.merged_df = self.speakers_df.merge(
            self.topics_df,
            left_on='id',
            right_on='speaker_id',
            how='left',  # Keep all speakers
            suffixes=('_speaker', '_topic')
        )

        logger.info(f"Merged data contains {len(self.merged_df)} records")

        # Log speakers without topics (but don't exclude them)
        speakers_without_topics = self.speakers_df[
            ~self.speakers_df['id'].isin(self.topics_df['speaker_id'])
        ]

        if len(speakers_without_topics) > 0:
            logger.info(f"Found {len(speakers_without_topics)} speakers without topic mappings (will still be included)")

        return self.merged_df

    def _safe_parse_list(self, value: Any) -> List[str]:
        """Safely parse string representations of lists"""
        if pd.isna(value) or value == '' or value == 'nan':
            return []

        if isinstance(value, list):
            return value

        if isinstance(value, str):
            # Handle different list formats
            value = value.strip()

            # Handle empty brackets
            if value in ['[]', '[""]', "['']"]:
                return []

            try:
                # Try to parse as JSON/Python literal
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
                else:
                    return [str(parsed).strip()]
            except:
                # Fallback: split by common delimiters
                if '|' in value:
                    return [item.strip() for item in value.split('|') if item.strip()]
                elif ',' in value:
                    return [item.strip() for item in value.split(',') if item.strip()]
                else:
                    return [value.strip()] if value.strip() else []

        return [str(value).strip()] if str(value).strip() != 'nan' else []

    def _safe_get_value(self, row: pd.Series, field: str, default: Any = '') -> Any:
        """Safely get value from row, handling NaN and missing fields"""
        try:
            if field not in row.index:
                return default

            value = row[field]

            # Handle NaN values
            if pd.isna(value):
                return default

            # Handle string 'nan' values
            if isinstance(value, str) and value.lower() in ['nan', 'null', 'none', '']:
                return default

            return value

        except Exception:
            return default

    def _safe_get_boolean(self, row: pd.Series, field: str, default: bool = False) -> bool:
        """Safely get boolean value from row"""
        try:
            value = self._safe_get_value(row, field, default)

            if isinstance(value, bool):
                return value

            if isinstance(value, str):
                return value.lower() in ['true', 't', '1', 'yes', 'y', 'active']

            if isinstance(value, (int, float)):
                return bool(value)

            return default

        except Exception:
            return default

    def create_speaker_profiles(self) -> List[Dict[str, Any]]:
        """Create enriched speaker profiles for embedding - INCLUSIVE approach"""
        if self.merged_df is None:
            self.merge_data()

        logger.info("Creating speaker profiles...")

        speaker_profiles = []
        processed_speakers = set()

        # Process speakers with topics first
        for speaker_id, group in self.merged_df.groupby('id'):
            if pd.isna(speaker_id) or speaker_id in processed_speakers:
                continue

            try:
                processed_speakers.add(speaker_id)
                profile = self._create_single_profile(speaker_id, group)
                if profile:
                    speaker_profiles.append(profile)

            except Exception as e:
                logger.warning(f"Error processing speaker {speaker_id}: {e}")
                # Don't skip - try to create minimal profile
                try:
                    minimal_profile = self._create_minimal_profile(speaker_id, group.iloc[0])
                    if minimal_profile:
                        speaker_profiles.append(minimal_profile)
                except Exception as e2:
                    logger.warning(f"Failed to create minimal profile for speaker {speaker_id}: {e2}")
                    continue

        # Process speakers without topics (if any missed in merge)
        for _, speaker_row in self.speakers_df.iterrows():
            speaker_id = speaker_row['id']
            if pd.isna(speaker_id) or speaker_id in processed_speakers:
                continue

            try:
                processed_speakers.add(speaker_id)
                profile = self._create_profile_from_speaker_only(speaker_id, speaker_row)
                if profile:
                    speaker_profiles.append(profile)

            except Exception as e:
                logger.warning(f"Error processing speaker-only {speaker_id}: {e}")
                continue

        self.speaker_profiles = speaker_profiles
        logger.info(f"Created {len(speaker_profiles)} speaker profiles")

        # Save processed profiles
        self._save_processed_data()

        return speaker_profiles

    def _create_single_profile(self, speaker_id: int, group: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Create profile for speaker with topic data"""
        first_row = group.iloc[0]

        # Basic speaker information - with safe extraction
        profile = {
            # "speaker_id": int(speaker_id),
            "id": int(speaker_id),
            "name": str(self._safe_get_value(first_row, "name", "Unknown")).strip(),
            "email": str(self._safe_get_value(first_row, "email", "")).strip(),
            "job_title": str(self._safe_get_value(first_row, "job_title", "")).strip(),
            "company": str(self._safe_get_value(first_row, "company", "")).strip(),
            "active": self._safe_get_boolean(
                first_row, "active", True
            ),  # Default to True if unknown
            "bio": str(self._safe_get_value(first_row, "bio", "")).strip(),
            "specializations": str(
                self._safe_get_value(first_row, "specializations", "")
            ).strip(),
            "certifications": str(
                self._safe_get_value(first_row, "certifications", "")
            ).strip(),
            "work_experience": str(
                self._safe_get_value(first_row, "work_experience", "")
            ).strip(),
            "education": str(self._safe_get_value(first_row, "education", "")).strip(),
            "industry_specialty": str(
                self._safe_get_value(first_row, "industry_specialty", "")
            ).strip(),
            "centers": str(
                self._safe_get_value(first_row, "centers_original", "")
            ).strip(),
        }

        # Parse list fields safely
        profile['audiences'] = self._safe_parse_list(self._safe_get_value(first_row, 'audiences', []))
        profile['topics_general'] = self._safe_parse_list(self._safe_get_value(first_row, 'topics', []))
        profile['language'] = self._safe_parse_list(self._safe_get_value(first_row, 'language', []))
        profile['industry'] = self._safe_parse_list(self._safe_get_value(first_row, 'industry', []))

        # Collect speaking topics from topic mappings
        speaker_topics = []
        approved_topics = []
        tech_levels = []

        for _, topic_row in group.iterrows():
            topic_name = str(self._safe_get_value(topic_row, 'topic_name', '')).strip()
            if topic_name and topic_name not in ['nan', 'None', '']:
                speaker_topics.append(topic_name)

                # Track approval status
                approved = self._safe_get_value(topic_row, 'approved', '')
                if approved and str(approved).strip().lower() == 'primary':
                    approved_topics.append(topic_name)

                # Track technical levels
                tech_level = self._safe_get_value(topic_row, 'tech_level', '')
                if tech_level and str(tech_level).strip() not in ['nan', 'None', '']:
                    tech_levels.append(str(tech_level).strip())

        profile['speaking_topics'] = list(set(speaker_topics))
        profile['primary_topics'] = list(set(approved_topics))
        profile['tech_levels'] = list(set(tech_levels))

        # Create searchable text
        profile['searchable_text'] = self._create_searchable_text(profile)

        # Calculate completeness
        profile['completeness_score'] = self._calculate_completeness_score(profile)

        # Only exclude if absolutely no useful information
        if self._is_profile_useful(profile):
            return profile

        return None

    def _create_profile_from_speaker_only(self, speaker_id: int, speaker_row: pd.Series) -> Optional[Dict[str, Any]]:
        """Create profile for speaker without topic mappings"""
        profile = {
            'id': int(speaker_id),
            'name': str(self._safe_get_value(speaker_row, 'name', 'Unknown')).strip(),
            'email': str(self._safe_get_value(speaker_row, 'email', '')).strip(),
            'job_title': str(self._safe_get_value(speaker_row, 'job_title', '')).strip(),
            'company': str(self._safe_get_value(speaker_row, 'company', '')).strip(),
            'active': self._safe_get_boolean(speaker_row, 'active', True),
            'bio': str(self._safe_get_value(speaker_row, 'bio', '')).strip(),
            'specializations': str(self._safe_get_value(speaker_row, 'specializations', '')).strip(),
            'certifications': str(self._safe_get_value(speaker_row, 'certifications', '')).strip(),
            'work_experience': str(self._safe_get_value(speaker_row, 'work_experience', '')).strip(),
            'education': str(self._safe_get_value(speaker_row, 'education', '')).strip(),
            'industry_specialty': str(self._safe_get_value(speaker_row, 'industry_specialty', '')).strip(),
            'centers': str(self._safe_get_value(speaker_row, 'centers_original', '')).strip(),
        }

        # Parse list fields
        profile['audiences'] = self._safe_parse_list(self._safe_get_value(speaker_row, 'audiences', []))
        profile['topics_general'] = self._safe_parse_list(self._safe_get_value(speaker_row, 'topics', []))
        profile['language'] = self._safe_parse_list(self._safe_get_value(speaker_row, 'language', []))
        profile['industry'] = self._safe_parse_list(self._safe_get_value(speaker_row, 'industry', []))

        # No topic mappings available
        profile['speaking_topics'] = []
        profile['primary_topics'] = []
        profile['tech_levels'] = []

        # Create searchable text
        profile['searchable_text'] = self._create_searchable_text(profile)

        # Calculate completeness
        profile['completeness_score'] = self._calculate_completeness_score(profile)

        if self._is_profile_useful(profile):
            return profile

        return None

    def _create_minimal_profile(self, speaker_id: int, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Create minimal profile when all else fails"""
        name = str(self._safe_get_value(row, 'name', 'Unknown')).strip()

        # Skip if name is completely unusable
        if name in ['Unknown', 'nan', 'None', '??', '?? .', '']:
            return None

        profile = {
            'speaker_id': int(speaker_id),
            'name': name,
            'email': '',
            'job_title': str(self._safe_get_value(row, 'job_title', '')).strip(),
            'company': '',
            'active': True,  # Default to active for minimal profiles
            'bio': str(self._safe_get_value(row, 'bio', '')).strip(),
            'specializations': '',
            'certifications': '',
            'work_experience': '',
            'education': '',
            'industry_specialty': '',
            'audiences': [],
            'topics_general': [],
            'language': [],
            'industry': [],
            'speaking_topics': [],
            'primary_topics': [],
            'tech_levels': [],
            'centers': '',
            'searchable_text': '',
            'completeness_score': 0.1
        }

        profile['searchable_text'] = self._create_searchable_text(profile)

        return profile

    def _is_profile_useful(self, profile: Dict[str, Any]) -> bool:
        """Check if profile has enough information to be useful"""
        # Must have a real name
        if not profile['name'] or profile['name'] in ['Unknown', 'nan', 'None', '??', '?? .']:
            return False

        # Must have at least one of: bio, job_title, speaking_topics, or specializations
        useful_fields = [
            profile['bio'],
            profile['job_title'],
            profile['specializations'],
        ]

        has_useful_content = any(field and field.strip() not in ['', 'nan', 'None'] for field in useful_fields)
        has_topics = len(profile['speaking_topics']) > 0 or len(profile['topics_general']) > 0

        return has_useful_content or has_topics

    def _create_searchable_text(self, profile: Dict[str, Any]) -> str:
        """Create comprehensive searchable text for each speaker"""

        text_components = []

        # Basic info
        if profile['name'] and profile['name'] not in ['Unknown', 'nan']:
            text_components.append(f"Speaker: {profile['name']}")

        if profile['job_title']:
            text_components.append(f"Job Title: {profile['job_title']}")

        if profile['company']:
            text_components.append(f"Company: {profile['company']}")

        # Bio and experience
        if profile['bio']:
            text_components.append(f"Biography: {profile['bio']}")

        if profile['work_experience']:
            text_components.append(f"Work Experience: {profile['work_experience']}")

        if profile['education']:
            text_components.append(f"Education: {profile['education']}")

        # Expertise areas
        if profile['specializations']:
            text_components.append(f"Specializations: {profile['specializations']}")

        if profile['industry_specialty']:
            text_components.append(f"Industry Expertise: {profile['industry_specialty']}")

        if profile['certifications']:
            text_components.append(f"Certifications: {profile['certifications']}")

        # Speaking topics
        if profile['speaking_topics']:
            text_components.append(f"Speaking Topics: {', '.join(profile['speaking_topics'])}")

        if profile['primary_topics']:
            text_components.append(f"Primary Expertise: {', '.join(profile['primary_topics'])}")

        # Audiences and technical levels
        if profile['audiences']:
            text_components.append(f"Target Audiences: {', '.join(profile['audiences'])}")

        if profile['tech_levels']:
            text_components.append(f"Technical Levels: {', '.join(profile['tech_levels'])}")

        # Additional fields
        if profile['topics_general']:
            text_components.append(f"General Topics: {', '.join(profile['topics_general'])}")

        if profile['language']:
            text_components.append(f"Languages: {', '.join(profile['language'])}")

        if profile['industry']:
            text_components.append(f"Industries: {', '.join(profile['industry'])}")

        if profile['centers']:
            text_components.append(f"Centers: {profile['centers']}")

        return "\n".join(text_components)

    def _calculate_completeness_score(self, profile: Dict[str, Any]) -> float:
        """Calculate how complete a speaker profile is (0.0 to 1.0) - More lenient"""

        score = 0.0

        # Essential fields - more lenient scoring
        if profile['name'] and profile['name'] not in ['Unknown', 'nan']:
            score += 0.25  # Reduced from 0.3

        if profile['bio'] and len(profile['bio']) > 10:
            score += 0.20

        if profile['speaking_topics'] and len(profile['speaking_topics']) > 0:
            score += 0.15
        elif profile['topics_general'] and len(profile['topics_general']) > 0:
            score += 0.10  # Partial credit for general topics

        # Important fields
        if profile['job_title']:
            score += 0.15

        if profile['specializations']:
            score += 0.10

        if profile['audiences'] and len(profile['audiences']) > 0:
            score += 0.05

        # Bonus for additional information
        if profile['work_experience']:
            score += 0.03

        if profile['certifications']:
            score += 0.02

        if profile['centers']:
            score += 0.05
        return min(score, 1.0)

    def _save_processed_data(self):
        """Save processed data to files"""
        try:
            processed_dir = Path(config.data.processed_data_dir)
            processed_dir.mkdir(parents=True, exist_ok=True)

            # Save speaker profiles as JSON
            profiles_file = processed_dir / "speaker_profiles.json"
            with open(profiles_file, 'w', encoding='utf-8') as f:
                json.dump(self.speaker_profiles, f, indent=2, default=str)

            # Save merged dataframe as CSV
            if self.merged_df is not None:
                merged_file = processed_dir / "merged_speakers_topics.csv"
                self.merged_df.to_csv(merged_file, index=False, encoding='utf-8')

            logger.info(f"Saved processed data to {processed_dir}")

        except Exception as e:
            logger.warning(f"Failed to save processed data: {e}")

    def get_active_speakers(self) -> List[Dict[str, Any]]:
        """Get speakers (now more inclusive with active status)"""
        if not self.speaker_profiles:
            self.create_speaker_profiles()

        # More lenient active filtering - include if active is True or unknown
        active_speakers = [
            profile for profile in self.speaker_profiles 
            if profile.get('active', True)  # Default to True if not specified
        ]

        logger.info(f"Found {len(active_speakers)} active speakers out of {len(self.speaker_profiles)} total")
        return active_speakers

    def get_all_speakers(self) -> List[Dict[str, Any]]:
        """Get all speakers regardless of active status"""
        if not self.speaker_profiles:
            self.create_speaker_profiles()

        return self.speaker_profiles

    def get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about the processed data"""
        if not self.speaker_profiles:
            self.create_speaker_profiles()

        all_speakers = self.get_all_speakers()
        active_speakers = self.get_active_speakers()

        # Calculate statistics
        total_speakers = len(all_speakers)
        active_count = len(active_speakers)

        # Topic statistics
        all_topics = []
        speakers_with_topics = 0
        for profile in all_speakers:
            topics = profile.get('speaking_topics', []) + profile.get('topics_general', [])
            all_topics.extend(topics)
            if len(topics) > 0:
                speakers_with_topics += 1

        unique_topics = list(set(all_topics))

        # Completeness statistics
        completeness_scores = [p.get('completeness_score', 0) for p in all_speakers]
        avg_completeness = np.mean(completeness_scores) if completeness_scores else 0

        # Job title distribution (top 10)
        job_titles = [p.get('job_title', 'Unknown') for p in all_speakers if p.get('job_title')]
        job_title_counts = pd.Series(job_titles).value_counts().head(10).to_dict()

        # Bio statistics
        speakers_with_bio = len([p for p in all_speakers if p.get('bio') and len(p['bio']) > 10])

        return {
            'total_speakers': total_speakers,
            'active_speakers': active_count,
            'inactive_speakers': total_speakers - active_count,
            'speakers_with_topics': speakers_with_topics,
            'speakers_with_bio': speakers_with_bio,
            'unique_topics': len(unique_topics),
            'total_topic_assignments': len(all_topics),
            'average_completeness': round(avg_completeness, 3),
            'top_job_titles': job_title_counts,
            'sample_topics': unique_topics[:10] if unique_topics else []
        }

# Utility function for easy access
def process_speaker_data() -> SpeakerDataProcessor:
    """Process speaker data and return processor instance"""
    processor = SpeakerDataProcessor()
    processor.create_speaker_profiles()
    return processor

if __name__ == "__main__":
    # Test the data processor
    from ..utils.logging_config import setup_logging
    
    setup_logging("INFO")
    
    processor = SpeakerDataProcessor()
    profiles = processor.create_speaker_profiles()
    
    print(f"\n📊 Processing Complete!")
    print(f"Created {len(profiles)} speaker profiles")
    
    # Show statistics
    stats = processor.get_data_statistics()
    print(f"\n📈 Data Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Show sample profile
    all_speakers = processor.get_all_speakers()
    if all_speakers:
        sample_speaker = all_speakers[0]
        print(f"\n👤 Sample Speaker Profile:")
        print(f"Name: {sample_speaker['name']}")
        print(f"Job Title: {sample_speaker['job_title']}")
        print(f"Topics: {sample_speaker['speaking_topics'][:3]}...")
        print(f"Completeness: {sample_speaker['completeness_score']:.2%}")
