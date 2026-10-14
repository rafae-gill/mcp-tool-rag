"""mcp-tool-rag: permission-aware tool search for MCP."""

from .catalog import Tool, load_catalog, split_identifier
from .retrievers import BM25, Embeddings, EnrichedEmbeddings, Hybrid, tokenize
from .router import ToolRouter, allowed_for

__version__ = "0.0.1"

__all__ = ["Tool", "load_catalog", "split_identifier", "BM25", "Embeddings", "EnrichedEmbeddings", "Hybrid",
           "tokenize", "ToolRouter", "allowed_for"]
