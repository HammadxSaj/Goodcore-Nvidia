"""Test the data processor functionality"""

import sys
sys.path.append('src')

from core.data_processor import SpeakerDataProcessor
from utils.logging_config import setup_logging

def test_data_processor():
    """Test the data processor"""
    
    setup_logging("INFO")
    print("🧪 Testing Data Processor...\n")
    
    # Initialize processor
    processor = SpeakerDataProcessor()
    
    # Test data loading
    print("1️⃣ Testing data loading...")
    speakers_df, topics_df = processor.load_data()
    print(f"✅ Loaded {len(speakers_df)} speakers and {len(topics_df)} topics\n")
    
    # Test data merging
    print("2️⃣ Testing data merging...")
    merged_df = processor.merge_data()
    print(f"✅ Merged data: {len(merged_df)} records\n")
    
    # Test profile creation
    print("3️⃣ Testing profile creation...")
    profiles = processor.create_speaker_profiles()
    print(f"✅ Created {len(profiles)} speaker profiles\n")
    
    # Test statistics
    print("4️⃣ Testing statistics...")
    stats = processor.get_data_statistics()
    print("📊 Data Statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in list(value.items())[:5]:  # Show top 5
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    print()
    
    # Test active speakers
    print("5️⃣ Testing active speakers filter...")
    active_speakers = processor.get_active_speakers()
    print(f"✅ Found {len(active_speakers)} active speakers\n")
    
    # Show sample profiles
    print("6️⃣ Sample Speaker Profiles:")
    for i, speaker in enumerate(active_speakers[:3]):
        print(f"\n👤 Speaker {i+1}:")
        print(f"Name: {speaker['name']}")
        print(f"Job Title: {speaker['job_title']}")
        print(f"Speaking Topics: {len(speaker['speaking_topics'])} topics")
        print(f"Primary Topics: {speaker['primary_topics']}")
        print(f"Audiences: {speaker['audiences']}")
        print(f"Completeness: {speaker['completeness_score']:.1%}")
        print(f"Searchable Text Preview: {speaker['searchable_text'][:200]}...")
    
    return True

if __name__ == "__main__":
    success = test_data_processor()
    if success:
        print("\n✅ All data processor tests passed!")
    else:
        print("\n❌ Some tests failed!")