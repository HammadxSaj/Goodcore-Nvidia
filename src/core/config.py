import os
from dataclasses import dataclass, field
from typing import Dict, Any, List
import yaml
from pathlib import Path

@dataclass
class NVIDIAConfig:
    """NVIDIA services configuration"""
    # base_url: str = "http://localhost:8000"
    base_url: str = "http://localhost:11434"  # Updated to Ollama URL
    embedding_model: str = "snowflake/arctic-embed-l"
    llm_model: str = "qwen3:32b"  # Updated to use Qwen3 model
    reranker_model: str = "nvidia/llama-3.2-nemoretriever-500m-rerank-v2"
    api_key: str = ""
    timeout: int = 1000
    max_retries: int = 3
    ollama_base_url: str = "http://localhost:11434"
    use_ollama_for_llm: bool = False
    vllm_base_url: str = "http://localhost:8001"  # Default vLLM URL
    use_vllm_for_llm: bool = True  # Set to True to use

@dataclass
class VectorDBConfig:
    """Vector database configuration"""
    persist_directory: str = "./data/vector_store/chroma_db"
    collection_name: str = "speakers"
    distance_metric: str = "cosine"
    embedding_dimension: int = 1024

@dataclass
class DataConfig:
    """Data processing configuration"""
    speakers_file: str = "./data/raw/speakersdata.csv"
    topics_file: str = "./data/raw/speakertopics.csv"
    processed_data_dir: str = "./data/processed"
    batch_size: int = 100 

@dataclass
class APIConfig:
    """API server configuration"""
    host: str = "0.0.0.0"
    port: int = 8002
    reload: bool = False
    log_level: str = "INFO"
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

@dataclass
class AppConfig:
    """Application-level configuration"""
    max_results: int = 10
    similarity_threshold: float = 0.3
    cache_ttl: int = 3600  # 1 hour
    log_level: str = "INFO"
    debug: bool = False

class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.nvidia = NVIDIAConfig()
        self.vectordb = VectorDBConfig()
        self.data = DataConfig()
        self.api = APIConfig()
        self.app = AppConfig()
        
    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables"""
        config = cls()
        
        # NVIDIA configuration
        config.nvidia.base_url = os.getenv("NVIDIA_BASE_URL", config.nvidia.base_url)
        config.nvidia.embedding_model = os.getenv("NVIDIA_EMBEDDING_MODEL", config.nvidia.embedding_model)
        config.nvidia.llm_model = os.getenv("NVIDIA_LLM_MODEL", config.nvidia.llm_model)
        config.nvidia.api_key = os.getenv("NVIDIA_API_KEY", config.nvidia.api_key)
        
        # Vector DB configuration
        config.vectordb.persist_directory = os.getenv("VECTORDB_PATH", config.vectordb.persist_directory)
        config.vectordb.collection_name = os.getenv("COLLECTION_NAME", config.vectordb.collection_name)
        
        # Data configuration
        config.data.speakers_file = os.getenv("SPEAKERS_FILE", config.data.speakers_file)
        config.data.topics_file = os.getenv("TOPICS_FILE", config.data.topics_file)
        
        # API configuration
        config.api.host = os.getenv("API_HOST", config.api.host)
        config.api.port = int(os.getenv("API_PORT", config.api.port))
        config.api.log_level = os.getenv("LOG_LEVEL", config.api.log_level)
        
        # App configuration
        config.app.max_results = int(os.getenv("MAX_RESULTS", config.app.max_results))
        config.app.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        return config
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file"""
        config = cls()
        
        if not os.path.exists(config_path):
            return config
            
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Update configuration from YAML
        if 'nvidia' in yaml_config:
            for key, value in yaml_config['nvidia'].items():
                if hasattr(config.nvidia, key):
                    setattr(config.nvidia, key, value)
        
        if 'vectordb' in yaml_config:
            for key, value in yaml_config['vectordb'].items():
                if hasattr(config.vectordb, key):
                    setattr(config.vectordb, key, value)
        
        if 'data' in yaml_config:
            for key, value in yaml_config['data'].items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)
        
        if 'api' in yaml_config:
            for key, value in yaml_config['api'].items():
                if hasattr(config.api, key):
                    setattr(config.api, key, value)
        
        if 'app' in yaml_config:
            for key, value in yaml_config['app'].items():
                if hasattr(config.app, key):
                    setattr(config.app, key, value)
        
        return config
    
    def validate(self) -> bool:
        """Validate configuration"""
        errors = []
        
        # Check required files exist
        if not os.path.exists(self.data.speakers_file):
            errors.append(f"Speakers file not found: {self.data.speakers_file}")
        
        if not os.path.exists(self.data.topics_file):
            errors.append(f"Topics file not found: {self.data.topics_file}")
        
        # Create directories if they don't exist
        os.makedirs(self.vectordb.persist_directory, exist_ok=True)
        os.makedirs(self.data.processed_data_dir, exist_ok=True)
        
        if errors:
            for error in errors:
                print(f"Configuration Error: {error}")
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'nvidia': self.nvidia.__dict__,
            'vectordb': self.vectordb.__dict__,
            'data': self.data.__dict__,
            'api': self.api.__dict__,
            'app': self.app.__dict__
        }

# Global configuration instance
config = Config.from_env()