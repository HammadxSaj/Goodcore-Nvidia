"""Test script to verify initial setup"""

import sys
import os
sys.path.append('src')

from core.config import config
from utils.logging_config import setup_logging

def test_configuration():
    """Test configuration loading"""
    print("🧪 Testing configuration...")
    
    # Test configuration validation
    is_valid = config.validate()
    print(f"Configuration valid: {is_valid}")
    
    # Print configuration
    print("\n📋 Current Configuration:")
    config_dict = config.to_dict()
    for section, values in config_dict.items():
        print(f"\n{section.upper()}:")
        for key, value in values.items():
            print(f"  {key}: {value}")
    
    return is_valid

def test_logging():
    """Test logging setup"""
    print("\n🧪 Testing logging...")
    
    logger = setup_logging(log_level="INFO")
    logger.info("Logging test successful!")
    
    print("✅ Logging configuration working")
    return True

def detect_encoding(file_path):
    """Detect file encoding"""
    import chardet
    
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            return result['encoding']
    except:
        return 'utf-8'

def test_data_files():
    """Test data files existence and encoding"""
    print("\n🧪 Testing data files...")
    
    speakers_exists = os.path.exists(config.data.speakers_file)
    topics_exists = os.path.exists(config.data.topics_file)
    
    print(f"Speakers file exists: {speakers_exists}")
    print(f"Topics file exists: {topics_exists}")
    
    if speakers_exists and topics_exists:
        import pandas as pd
        
        try:
            # Detect encodings
            speakers_encoding = detect_encoding(config.data.speakers_file)
            topics_encoding = detect_encoding(config.data.topics_file)
            
            print(f"Speakers file encoding: {speakers_encoding}")
            print(f"Topics file encoding: {topics_encoding}")
            
            # Try reading with detected encoding
            speakers_df = pd.read_csv(config.data.speakers_file, encoding=speakers_encoding)
            topics_df = pd.read_csv(config.data.topics_file, encoding=topics_encoding)
            
            print(f"Speakers data: {len(speakers_df)} rows, {len(speakers_df.columns)} columns")
            print(f"Topics data: {len(topics_df)} rows, {len(topics_df.columns)} columns")
            
            # Show sample data
            print("\n📊 Sample speakers data:")
            print(speakers_df.head(2)[['id', 'name', 'job_title', 'active']].to_string())
            
            print("\n📊 Sample topics data:")
            print(topics_df.head(2)[['topic_id', 'speaker_id', 'topic_name', 'speaker_name']].to_string())
            
            return True
            
        except Exception as e:
            print(f"❌ Error reading data files: {e}")
            
            # Try alternative encodings
            encodings_to_try = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings_to_try:
                try:
                    print(f"Trying encoding: {encoding}")
                    speakers_df = pd.read_csv(config.data.speakers_file, encoding=encoding)
                    topics_df = pd.read_csv(config.data.topics_file, encoding=encoding)
                    
                    print(f"✅ Success with encoding: {encoding}")
                    print(f"Speakers data: {len(speakers_df)} rows")
                    print(f"Topics data: {len(topics_df)} rows")
                    
                    return True
                except Exception as enc_error:
                    print(f"❌ Failed with {encoding}: {str(enc_error)[:100]}...")
                    continue
            
            return False
    
    return False

if __name__ == "__main__":
    print("🚀 Running initial setup tests...\n")
    
    config_ok = test_configuration()
    logging_ok = test_logging()
    data_ok = test_data_files()
    
    if all([config_ok, logging_ok, data_ok]):
        print("\n✅ All tests passed! Ready to proceed.")
    else:
        print("\n❌ Some tests failed. Please check the issues above.")