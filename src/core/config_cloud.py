"""
Cloud configuration for Streamlit deployment
"""

import os


class CloudConfig:
    """Configuration for cloud deployment"""

    # Your remote server's API URL
    API_BASE_URL = os.getenv("API_BASE_URL", "http://YOUR_REMOTE_SERVER_IP:8000")

    # Health check endpoint
    HEALTH_ENDPOINT = f"{API_BASE_URL}/api/v1/health"

    # Search endpoints
    SEARCH_ENDPOINT = f"{API_BASE_URL}/api/v1/search"
    REFINE_ENDPOINT = f"{API_BASE_URL}/api/v1/refine"

    # Stats endpoint
    STATS_ENDPOINT = f"{API_BASE_URL}/api/v1/stats"


# Global instance
cloud_config = CloudConfig()
