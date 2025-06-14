from typing import Union
from pathlib import Path
from langchain.embeddings import HuggingFaceEmbeddings


class Embed_Retrive:
    def __init__(self, embed_model: HuggingFaceEmbeddings, query: str, db_path: Union[str, Path]):
        self.embedding_model = embed_model
        
    def