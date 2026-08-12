import os
import sys
import numpy as np
import chromadb
import uuid
import mlflow
from chromadb.config import Settings
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import get_embedding_model

@dataclass
class DataTransformationConfig:
    persist_directory = os.path.join('data', 'vector_store')
    collection_name = "pdf_documents"

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        self.model_name = 'all-MiniLM-L6-v2'

    @property
    def model(self):
        return get_embedding_model(self.model_name)

    def initiate_data_transformation(self, chunks, clear_existing: bool = True):
        logging.info("Initiating data transformation with MLflow tracking")
        try:
            if not chunks:
                logging.warning("No chunks provided for transformation. Skipping indexing.")
                return None

            # Set MLflow experiment
            try:
                mlflow.set_experiment("RAG_Indexing_Pipeline")
            except Exception as me:
                logging.warning(f"Could not set MLflow experiment: {me}")

            if mlflow.active_run():
                try:
                    mlflow.end_run()
                except Exception as me:
                    logging.warning(f"Could not end active MLflow run: {me}")
            
            with mlflow.start_run():
                # Log Parameters
                try:
                    mlflow.log_param("chunk_size", 1000)
                    mlflow.log_param("chunk_overlap", 200)
                    mlflow.log_param("model_name", self.model_name)
                    mlflow.log_param("collection_name", self.config.collection_name)
                except Exception as me:
                    logging.warning(f"MLflow log_param warning: {me}")

                texts = [chunk.page_content for chunk in chunks]
                embeddings = self.model.encode(texts, show_progress_bar=True)
                
                os.makedirs(self.config.persist_directory, exist_ok=True)
                
                # Fixed: Disabling telemetry and explicitly setting embedding_function=None 
                # to avoid loading unused native bindings that cause 'RustBindingsAPI' errors.
                client = chromadb.PersistentClient(
                    path=self.config.persist_directory,
                    settings=Settings(anonymized_telemetry=False)
                )

                if clear_existing:
                    try:
                        client.delete_collection(name=self.config.collection_name)
                        logging.info(f"Purged previous vector store collection '{self.config.collection_name}' for new document.")
                    except Exception as ce:
                        logging.info(f"No existing collection to delete or delete skipped: {ce}")
                
                # Get or create fresh collection
                collection = client.get_or_create_collection(
                    name=self.config.collection_name,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=None
                )
                
                ids = [f"doc_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(chunks))]
                metadatas = [dict(chunk.metadata) for chunk in chunks]
                
                embeddings_list = embeddings.tolist()
                
                # Batch indexing to avoid potential large payload issues
                batch_size = 100
                for i in range(0, len(chunks), batch_size):
                    end = min(i + batch_size, len(chunks))
                    collection.add(
                        ids=ids[i:end],
                        embeddings=embeddings_list[i:end],
                        metadatas=metadatas[i:end],
                        documents=texts[i:end]
                    )
                
                # Log Metrics
                try:
                    mlflow.log_metric("num_chunks", len(chunks))
                    mlflow.log_metric("avg_chunk_length", np.mean([len(t) for t in texts]))
                except Exception as me:
                    logging.warning(f"MLflow log_metric warning: {me}")
                
                logging.info(f"Successfully indexed {len(chunks)} chunks and tracked with MLflow")
                
                return self.config.persist_directory

        except Exception as e:
            raise CustomException(e, sys)
