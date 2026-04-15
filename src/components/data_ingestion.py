import os
import sys
from dataclasses import dataclass
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    pass

class DataIngestion:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.ingestion_config = DataIngestionConfig()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def initiate_data_ingestion(self, pdf_file_path):
        logging.info("Entered the data ingestion method or component")
        try:
            if not os.path.exists(pdf_file_path):
                raise FileNotFoundError(f"Selected file {pdf_file_path} not found.")

            loader = PyMuPDFLoader(pdf_file_path)
            documents = loader.load()
            
            if not documents:
                logging.warning(f"No pages could be loaded from {pdf_file_path}")
                return []

            logging.info(f"Successfully loaded {len(documents)} pages from {pdf_file_path}")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)
            
            non_empty_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
            
            logging.info(f"Split documents into {len(non_empty_chunks)} non-empty chunks with size {self.chunk_size}")

            return non_empty_chunks

        except Exception as e:
            raise CustomException(e, sys)
