import os
import sys
import pickle
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

_embedding_model = None

def get_embedding_model(model_name: str = 'all-MiniLM-L6-v2'):
    """Lazy-loaded shared SentenceTransformer singleton to prevent RAM duplication on cloud deployment."""
    global _embedding_model
    if _embedding_model is None:
        try:
            import torch
            torch.set_num_threads(1)
        except ImportError:
            pass
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model

