"""
prediction_pipeline.py — RAG Prediction Pipeline

Uses LLMGateway for LLM construction (single source of truth),
ConversationMemory for chat history, and cleans up duplicated
env-var loading / key-stripping that now lives in gateway.py.
"""
import os
import sys
import time
import chromadb
import mlflow
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from src.components.gateway import LLMGateway
from src.exception import CustomException
from src.logger import logging

load_dotenv()


class PredictionPipeline:
    def __init__(self):
        self.persist_directory = os.path.join('data', 'vector_store')
        self.collection_name = "pdf_documents"
        self.embedding_model_name = 'all-MiniLM-L6-v2'
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # LLM via gateway (handles key loading + retry)
        self.gateway = LLMGateway()
        self.llm = self.gateway.get_llm()

        try:
            mlflow.set_experiment("RAG_Inference_Monitoring")
        except Exception as e:
            logging.warning(f"Could not initialise MLflow experiment: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, query: str, chat_history: list = None) -> dict:
        """
        Run retrieval-augmented generation for *query*.

        Parameters
        ----------
        query        : The (already sanitised) user question.
        chat_history : Optional list[{"role": str, "content": str}] from
                       ConversationMemory or the request body. When provided
                       the query is first condensed into a standalone question.

        Returns
        -------
        {"answer": str, "sources": list[dict]}
        """
        try:
            logging.info(f"PredictionPipeline.predict called. Vector store: {self.persist_directory}")

            if not os.path.exists(self.persist_directory):
                return {
                    "answer": "The document index has not been created yet. Please upload a PDF first.",
                    "sources": [],
                }

            client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )

            try:
                collection = client.get_collection(
                    name=self.collection_name,
                    embedding_function=None,
                )
            except Exception:
                return {
                    "answer": "The document collection has not been initialised. Please upload and index a document.",
                    "sources": [],
                }

            start_time = time.time()

            # MLflow run (best-effort)
            active_run = None
            try:
                active_run = mlflow.start_run()
                mlflow.log_param("query", query)
                mlflow.log_param("embedding_model", self.embedding_model_name)
            except Exception:
                active_run = None

            # 1. Condense question using chat history
            standalone_query = self._condense_query(query, chat_history or [])

            # 2. Retrieval
            retrieval_start = time.time()
            query_embedding = self.embedding_model.encode([standalone_query])[0]
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=8,
                include=['documents', 'distances', 'metadatas'],
            )
            retrieval_duration = time.time() - retrieval_start

            contexts   = results['documents'][0]
            distances  = results['distances'][0]
            metadatas  = results['metadatas'][0]

            # Filter by similarity threshold (cosine distance → similarity = 1 - dist)
            filtered_contexts, sources = [], []
            for i, (ctx, dist, meta) in enumerate(zip(contexts, distances, metadatas)):
                score = 1 - dist
                if score > 0.18:
                    filtered_contexts.append(ctx)
                    sources.append({"content": ctx, "metadata": meta, "score": score})

            avg_score = (
                sum(s['score'] for s in sources) / len(sources) if sources else 0.0
            )

            # 3. LLM generation
            llm_duration = 0
            if not filtered_contexts:
                answer = (
                    "I couldn't find any relevant information in the documents to answer your question. "
                    "Could you please be more specific?"
                )
            else:
                context_text = "\n\n".join(filtered_contexts)
                history_text = (
                    "\n".join(
                        f"{m['role'].capitalize()}: {m['content']}"
                        for m in (chat_history or [])[-5:]
                    )
                    if chat_history
                    else ""
                )

                prompt = ChatPromptTemplate.from_template("""
You are "GroqRAG Turbo", an advanced AI specialised in analysing PDF documents.
The user has provided a document, and you have been given relevant snippets from it as "Context".

YOUR TASK:
1. Use the provided Context and Conversation History to answer the Question.
2. If the context contains a Table of Contents or index, use it to understand the structure,
   but look for the actual answer in the other snippets.
3. If the answer is absolutely not present in the context, politely state that the current
   document doesn't contain that information.
4. NEVER say you don't have access to the PDF — the Context below IS the PDF data.

Conversation History:
{history}

Context from PDF:
{context}

User Question: {question}

Direct Answer:""")

                llm_start = time.time()
                chain = prompt | self.llm
                response = self.gateway.invoke_with_retry(
                    chain,
                    {"context": context_text, "question": standalone_query, "history": history_text},
                )
                answer = response.content
                llm_duration = time.time() - llm_start

            total_duration = time.time() - start_time
            estimated_tokens = (
                len("\n\n".join(filtered_contexts)) + len(query) + len(answer)
            ) // 4

            # Log metrics to MLflow
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
                    logging.warning(f"Failed to log MLflow metrics: {e}")

            return {
                "answer": answer,
                "sources": sources,
                "_meta": {
                    "latency_sec": round(total_duration, 3),
                    "num_docs": len(filtered_contexts),
                    "avg_score": round(avg_score, 4),
                },
            }

        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _condense_query(self, query: str, chat_history: list) -> str:
        """Rewrite a follow-up question as a standalone question using history."""
        if not chat_history:
            return query
        try:
            condense_prompt = ChatPromptTemplate.from_template("""
Given the following conversation and a follow-up question, rephrase the follow-up question
to be a standalone question.

Chat History:
{history}

Follow Up Input: {question}

Standalone Question:""")
            history_str = "\n".join(
                f"{m['role']}: {m['content']}" for m in chat_history[-3:]
            )
            chain = condense_prompt | self.llm
            result = self.gateway.invoke_with_retry(
                chain, {"history": history_str, "question": query}
            )
            standalone = result.content.strip()
            logging.info(f"Condensed query: {standalone}")
            return standalone
        except Exception as e:
            logging.warning(f"Query condensation failed, using original: {e}")
            return query
