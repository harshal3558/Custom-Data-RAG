import os
import sys
import chromadb
import mlflow
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.exception import CustomException
from src.logger import logging

# Load environment variables at module level as well
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

class PredictionPipeline:
    def __init__(self):
        self.persist_directory = os.path.join('data', 'vector_store')
        self.collection_name = "pdf_documents"
        self.model_name = 'all-MiniLM-L6-v2'
        self.model = SentenceTransformer(self.model_name)
        
        # Explicitly load .env from the root directory to be safe
        env_path = os.path.join(os.getcwd(), '.env')
        load_dotenv(dotenv_path=env_path)
        
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            api_key = api_key.strip().strip('"').strip("'")
            # Set it back to environment variable so ChatGroq can also find it automatically
            os.environ["GROQ_API_KEY"] = api_key
            logging.info(f"GROQ_API_KEY loaded successfully. Starts with: {api_key[:4]}... Ends with: {api_key[-4:]}")
        else:
            logging.error(f"GROQ_API_KEY not found in environment variables (checked {env_path}).")
        
        model_name = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
        
        self.llm = ChatGroq(
            api_key=api_key,
            groq_api_key=api_key,  # For compatibility with various versions
            model_name=model_name
        )
        
        # Safely set up MLflow experiment
        try:
            mlflow.set_experiment("RAG_Inference_Monitoring")
        except Exception as e:
            logging.warning(f"Could not initialize MLflow experiment: {e}")

    def predict(self, query):
        try:
            # Start the inference process
            logging.info(f"Checking for vector store at {self.persist_directory}")
            if not os.path.exists(self.persist_directory):
                return "The document index has not been created yet. Please upload a PDF first."

            from chromadb.config import Settings
            client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Check if collection exists
            try:
                collection = client.get_collection(
                    name=self.collection_name,
                    embedding_function=None
                )
            except Exception:
                return "The document collection has not been initialized. Please upload and index a document."
            
            # Start MLflow run if possible
            try:
                active_run = mlflow.start_run()
                mlflow.log_param("query", query)
                mlflow.log_param("model_name", self.model_name)
            except Exception:
                active_run = None

            query_embedding = self.model.encode([query])[0]
            
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=5
            )
            
            contexts = results['documents'][0]
            if not contexts:
                answer = "I couldn't find any relevant information in the uploaded PDF to answer that."
            else:
                context_text = "\n\n".join(contexts)
                
                prompt_template = ChatPromptTemplate.from_template("""
                Answer the user's question based strictly on the provided context. 
                If the context doesn't contain the answer, say that you don't know based on the provided PDF.
                
                Context:
                {context}
                
                Question: {question}
                """)
                
                chain = prompt_template | self.llm
                response = chain.invoke({"context": context_text, "question": query})
                answer = response.content
            
            # Log metrics to MLflow if run was started
            if active_run:
                try:
                    mlflow.log_metric("answer_length", len(answer))
                    mlflow.log_metric("num_retrieved_docs", len(contexts))
                    mlflow.end_run()
                except Exception:
                    pass

            return answer

        except Exception as e:
            raise CustomException(e, sys)
