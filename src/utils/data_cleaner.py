"""Data cleaning utilities for handling encoding and data quality issues"""

import pandas as pd
import chardet
import os
from typing import Tuple, Optional
from core.config import config

def detect_file_encoding(file_path: str) -> str:
    """Detect the encoding of a file"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            
            print(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
            return encoding
    except Exception as e:
        print(f"Error detecting encoding: {e}")
        return 'utf-8'

def read_csv_with_encoding(file_path: str, encoding: Optional[str] = None) -> pd.DataFrame:
    """Read CSV file with proper encoding handling"""
    
    if encoding is None:
        encoding = detect_file_encoding(file_path)
    
    # Try the detected/provided encoding first
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        print(f"✅ Successfully read {file_path} with encoding: {encoding}")
        return df
    except Exception as e:
        print(f"❌ Failed to read with {encoding}: {e}")
    
    # Fallback to common encodings
    fallback_encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for fallback_encoding in fallback_encodings:
        try:
            df = pd.read_csv(file_path, encoding=fallback_encoding)
            print(f"✅ Successfully read {file_path} with fallback encoding: {fallback_encoding}")
            return df
        except Exception as e:
            print(f"❌ Failed with {fallback_encoding}: {str(e)[:100]}...")
            continue
    
    # Last resort: try with error handling
    try:
        df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
        print(f"⚠️  Read {file_path} with errors ignored")
        return df
    except Exception as e:
        raise Exception(f"Unable to read {file_path} with any encoding: {e}")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize data"""
    
    # Remove completely empty rows
    df = df.dropna(how='all')
    
    # Clean string columns
    string_columns = df.select_dtypes(include=['object']).columns
    
    for col in string_columns:
        if col in df.columns:
            # Remove leading/trailing whitespace
            df[col] = df[col].astype(str).str.strip()
            
            # Replace 'nan' strings with actual NaN
            df[col] = df[col].replace(['nan', 'NaN', 'NULL', 'null', ''], pd.NA)
    
    return df

def load_and_clean_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean both speaker and topic data"""
    
    print("📂 Loading speaker data...")
    speakers_df = read_csv_with_encoding(config.data.speakers_file)
    speakers_df = clean_data(speakers_df)
    
    print("📂 Loading topics data...")
    topics_df = read_csv_with_encoding(config.data.topics_file)
    topics_df = clean_data(topics_df)
    
    # Basic data validation
    print(f"✅ Loaded {len(speakers_df)} speakers")
    print(f"✅ Loaded {len(topics_df)} topic entries")
    
    # Check for required columns
    required_speaker_cols = ['id', 'name', 'active']
    required_topic_cols = ['topic_id', 'speaker_id', 'topic_name']
    
    missing_speaker_cols = [col for col in required_speaker_cols if col not in speakers_df.columns]
    missing_topic_cols = [col for col in required_topic_cols if col not in topics_df.columns]
    
    if missing_speaker_cols:
        print(f"⚠️  Missing speaker columns: {missing_speaker_cols}")
    
    if missing_topic_cols:
        print(f"⚠️  Missing topic columns: {missing_topic_cols}")
    
    return speakers_df, topics_df

if __name__ == "__main__":
    # Test the data cleaning
    speakers_df, topics_df = load_and_clean_data()
    
    print("\n📊 Data Summary:")
    print(f"Speakers: {len(speakers_df)} rows, {len(speakers_df.columns)} columns")
    print(f"Topics: {len(topics_df)} rows, {len(topics_df.columns)} columns")
    
    print("\n📋 Speaker columns:")
    print(list(speakers_df.columns))
    
    print("\n📋 Topics columns:")
    print(list(topics_df.columns))