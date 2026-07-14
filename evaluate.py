import os
import sys
from src.components.model_evaluator import ModelEvaluator
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.logger import logging

def run_offline_evaluation():
    logging.info("Starting Offline RAG Evaluation using ModelEvaluator...")
    try:
        pipeline = PredictionPipeline()
        evaluator = ModelEvaluator()
        
        # Runs the evaluation loop, logs metrics to MLflow, and saves the CSV results
        results_file = evaluator.initiate_model_evaluation(pipeline=pipeline)
        
        print("\n" + "="*35)
        print("Evaluation Complete!")
        print(f"Results saved to: {results_file}")
        print("All metrics logged to MLflow successfully.")
        print("="*35)
        
    except Exception as e:
        print(f"Evaluation script failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_offline_evaluation()
