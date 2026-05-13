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

    def predict(self, query, chat_history=None):
        try:
            # Start the inference process
            logging.info(f"Checking for vector store at {self.persist_directory}")
            if not os.path.exists(self.persist_directory):
                return {"answer": "The document index has not been created yet. Please upload a PDF first.", "sources": []}

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
                return {"answer": "The document collection has not been initialized. Please upload and index a document.", "sources": []}
            
            # Start MLflow run if possible
            import time
            start_time = time.time()
            
            try:
                active_run = mlflow.start_run()
                mlflow.log_param("query", query)
                mlflow.log_param("model_name", self.model_name)
            except Exception:
                active_run = None

            # 1. Condense Question (Rewriting based on history)
            standalone_query = query
            if chat_history and len(chat_history) > 0:
                condense_template = ChatPromptTemplate.from_template("""
                Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
                
                Chat History:
                {history}
                
                Follow Up Input: {question}
                
                Standalone Question:""")
                
                condense_chain = condense_template | self.llm
                history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-3:]])
                rewritten = condense_chain.invoke({"history": history_str, "question": query})
                standalone_query = rewritten.content
                logging.info(f"Rewritten query: {standalone_query}")

            # 2. Retrieval
            retrieval_start = time.time()
            query_embedding = self.model.encode([standalone_query])[0]
            
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=8,
                include=['documents', 'distances', 'metadatas']
            )
            retrieval_end = time.time()
            retrieval_duration = retrieval_end - retrieval_start
            
            contexts = results['documents'][0]
            distances = results['distances'][0]
            metadatas = results['metadatas'][0]
            
            # Filter results by similarity threshold (Distance < 0.6 means Sim > 0.4 for L2/Cosine varies)
            # For cosine, distance is 0-2 (0 is identical). Sim = 1 - Dist.
            # Let's keep only those with similarity > 0.3 to filter out completely irrelevant noise.
            filtered_contexts = []
            sources = []
            for i in range(len(contexts)):
                score = 1 - distances[i]
                if score > 0.18: # Lowered threshold slightly for better recall
                    filtered_contexts.append(contexts[i])
                    sources.append({
                        "content": contexts[i],
                        "metadata": metadatas[i],
                        "score": score
                    })

            avg_score = sum([s['score'] for s in sources]) / len(sources) if sources else 0
            
            context_text = ""
            if not filtered_contexts:
                answer = "I couldn't find any relevant information in the documents to answer your question. Could you please be more specific?"
                llm_duration = 0
            else:
                context_text = "\n\n".join(filtered_contexts)
                
                # Conversational Prompt
                history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-5:]]) if chat_history else ""

                prompt_template = ChatPromptTemplate.from_template("""
                You are "GroqRAG Turbo", an advanced AI specialized in analyzing PDF documents.
                The user has provided a document, and you have been given relevant snippets from it as "Context".
                
                YOUR TASK:
                1. Use the provided Context and Conversation History to answer the Question.
                2. If the context contains a Table of Contents or index, use it to understand the structure, but look for the actual answer in the other snippets.
                3. If the answer is absolutely not present in the context, politely state that the current document doesn't contain that information.
                4. NEVER say you don't have access to the PDF, because the Context below IS the PDF data.

                Conversation History:
                {history}

                Context from PDF:
                {context}
                
                User Question: {question}
                
                Direct Answer:""")
                
                llm_start = time.time()
                chain = prompt_template | self.llm
                response = chain.invoke({
                    "context": context_text, 
                    "question": standalone_query,
                    "history": history_text
                })
                answer = response.content
                llm_end = time.time()
                llm_duration = llm_end - llm_start
            
            total_duration = time.time() - start_time
            
            # Crude token estimation
            estimated_tokens = (len(context_text) + len(query) + len(answer)) // 4

            # Log metrics to MLflow if run was started
            if active_run:
                try:
                    mlflow.log_metric("retrieval_time_sec", retrieval_duration)
                    mlflow.log_metric("llm_time_sec", llm_duration)
                    mlflow.log_metric("total_time_sec", total_duration)
                    mlflow.log_metric("mean_retrieval_score", avg_score)
                    mlflow.log_metric("answer_length", len(answer))
                    mlflow.log_metric("num_retrieved_docs", len(filtered_contexts))
                    mlflow.log_metric("estimated_tokens", estimated_tokens)
                    mlflow.end_run()
                except Exception as e:
                    logging.warning(f"Failed to log metrics to MLflow: {e}")
                    pass

            return {"answer": answer, "sources": sources}

        except Exception as e:
            raise CustomException(e, sys)
