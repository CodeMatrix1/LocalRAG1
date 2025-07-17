from huggingface_hub import snapshot_download
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

snapshot_download(
    repo_id="microsoft/phi-2",
    local_dir="models/phi-2",
    allow_patterns=["*.safetensors", "*.json", "*.txt"],  # ✅ add .safetensors
    resume_download=True
)