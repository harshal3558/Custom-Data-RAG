import os
import sys
from dotenv import load_dotenv

# Load environment at the very top
load_dotenv()

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.exception import CustomException
from src.logger import logging

app = Flask(__name__)

# Config for file uploads
UPLOAD_FOLDER = os.path.join('data', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB limit

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            logging.info(f"File saved to {file_path}. Starting ingestion...")
            
            # Step 1: Ingestion (Loading + Chunking)
            ingestion = DataIngestion()
            chunks = ingestion.initiate_data_ingestion(file_path)
            
            if not chunks:
                return jsonify({"error": "No text could be extracted from the PDF. It might be scanned or empty."}), 400

            # Step 2: Transformation (Embedding + Indexing)
            transformation = DataTransformation()
            transformation.initiate_data_transformation(chunks)
            
            return jsonify({"success": f"File '{filename}' uploaded and indexed successfully!"})
        else:
            return jsonify({"error": "Only PDF files are allowed."}), 400

    except Exception as e:
        logging.error(f"Error in upload_file: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def get_answer():
    try:
        data = request.json
        user_query = data.get("query")
        
        if not user_query:
            return jsonify({"error": "No query provided"}), 400
        
        # Step 3: Prediction (Retrieval + LLM)
        prediction_pipeline = PredictionPipeline()
        answer = prediction_pipeline.predict(user_query)
        
        return jsonify({"answer": answer})

    except Exception as e:
        logging.error(f"Error in get_answer: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)