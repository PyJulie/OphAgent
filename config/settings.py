"""
Global settings for OphAgent.
All paths, API keys, model configs, and deployment parameters live here.
Uses pydantic-settings for env-var overrides via .env file.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_ROOT = PROJECT_ROOT / "models_weights"
LOG_ROOT = PROJECT_ROOT / "logs"


class LLMSettings(BaseSettings):
    # Primary LLM backbone (OpenAI GPT-5 by default)
    provider: str = "openai"                  # "openai" | "gemini" | "local"
    model_id: str = "gpt-5"
    api_key: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.4   # paper §6.3: 0.4 for diagnostic generation tasks
    # Local model via Ollama (used when provider="local")
    local_model_url: str = "http://localhost:11434/v1"
    local_model_name: str = "qwen2.5:72b"


class KnowledgeBaseSettings(BaseSettings):
    # FAISS vector store
    faiss_index_path: Path = DATA_ROOT / "vector_store" / "ophagent.index"
    faiss_metadata_path: Path = DATA_ROOT / "vector_store" / "ophagent_meta.jsonl"
    # Embedding model for text (paper §2.4.3 specifies MedCPT, 768-dim)
    text_embed_model: str = "ncbi/MedCPT-Query-Encoder"
    # Embedding model for images (CLIP)
    image_embed_model: str = "ViT-B/32"
    # Local data archive
    local_data_root: Path = DATA_ROOT / "local_archive"
    # Textbook corpus
    textbook_root: Path = DATA_ROOT / "textbooks"
    # Top-k for RAG retrieval
    retrieval_top_k: int = 5
    # Chunk size for text splitting
    chunk_size: int = 512
    chunk_overlap: int = 64


class ModelSchedulerSettings(BaseSettings):
    # On-demand Conda scheduling
    conda_base: str = os.environ.get("CONDA_BASE", "/opt/conda")
    conda_timeout: int = 120          # seconds before subprocess timeout
    # Docker / FastAPI pre-loaded services
    docker_registry: str = "localhost:5000"
    fastapi_base_url: str = "http://localhost"
    service_startup_timeout: int = 30
    # Port allocation range for dynamic services
    port_range_start: int = 8100
    port_range_end: int = 8200


class ToolSettings(BaseSettings):
    tool_registry_path: Path = PROJECT_ROOT / "config" / "tool_registry.yaml"
    deployment_config_path: Path = PROJECT_ROOT / "config" / "deployment.yaml"
    # Grad-CAM
    gradcam_layer_name: str = "layer4"
    # OCR backend: "tesseract" | "easyocr"
    ocr_backend: str = "easyocr"
    # Web search
    search_max_results: int = 10
    search_domains: List[str] = ["pubmed.ncbi.nlm.nih.gov", "arxiv.org"]


class TrainingSettings(BaseSettings):
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 50
    warmup_epochs: int = 5       # paper §2.2.2: 5-epoch linear warmup; convert to steps at runtime
    weight_decay: float = 1e-2  # paper §2.2.2: 1×10⁻² (strong regularisation for medical imaging)
    mixed_precision: bool = True     # AMP
    gradient_checkpointing: bool = False
    save_every_n_epochs: int = 5
    checkpoint_root: Path = MODEL_ROOT / "checkpoints"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_prefix="OPHAGENT_",
        extra="ignore",
    )

    project_root: Path = PROJECT_ROOT
    data_root: Path = DATA_ROOT
    model_root: Path = MODEL_ROOT
    log_root: Path = LOG_ROOT

    llm: LLMSettings = Field(default_factory=LLMSettings)
    knowledge_base: KnowledgeBaseSettings = Field(default_factory=KnowledgeBaseSettings)
    scheduler: ModelSchedulerSettings = Field(default_factory=ModelSchedulerSettings)
    tools: ToolSettings = Field(default_factory=ToolSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)

    # Session
    session_history_limit: int = 50   # max turns kept in short-term memory
    debug: bool = False
    log_level: str = "INFO"

    def ensure_dirs(self) -> None:
        """Create all necessary directories if they don't exist."""
        dirs = [
            self.data_root,
            self.model_root,
            self.log_root,
            self.knowledge_base.faiss_index_path.parent,
            self.knowledge_base.local_data_root,
            self.knowledge_base.textbook_root,
            self.training.checkpoint_root,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
