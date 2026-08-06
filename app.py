import os
import sys
import time
from dotenv import load_dotenv

# Load environment at the very top
load_dotenv()

from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename

# Core pipeline
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.pipeline.prediction_pipeline import PredictionPipeline

# New enterprise components
from src.components.memory import ConversationMemory
from src.components.guardrails import Guardrails, GuardrailViolation
from src.components.governance import GovernanceLayer
from src.components.security import SecurityLayer
from src.components.observability import ObservabilityLayer
from src.components.model_evaluator import ModelEvaluator

from src.exception import CustomException
from src.logger import logging

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24).hex())

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
UPLOAD_FOLDER = os.path.join('data', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB

# ------------------------------------------------------------------
# Lazy-Loaded Singletons (Prevents Gunicorn startup timeout)
# ------------------------------------------------------------------
_memory = None
_guardrails = None
_governance = None
_security = None
_observability = None
_prediction_pipeline = None
_evaluator = None

def get_memory():
    global _memory
    if _memory is None:
        _memory = ConversationMemory()
    return _memory

def get_guardrails():
    global _guardrails
    if _guardrails is None:
        _guardrails = Guardrails()
    return _guardrails

def get_governance():
    global _governance
    if _governance is None:
        _governance = GovernanceLayer()
    return _governance

def get_security():
    global _security
    if _security is None:
        _security = SecurityLayer()
    return _security

def get_observability():
    global _observability
    if _observability is None:
        _observability = ObservabilityLayer()
    return _observability

def get_prediction_pipeline():
    global _prediction_pipeline
    if _prediction_pipeline is None:
        _prediction_pipeline = PredictionPipeline()
    return _prediction_pipeline

def get_evaluator():
    global _evaluator
    if _evaluator is None:
        _evaluator = ModelEvaluator()
    return _evaluator

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

def _get_session_id() -> str:
    if 'sid' not in session:
        import uuid
        session['sid'] = str(uuid.uuid4())
    return session['sid']

# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    try:
        status = get_observability().health_check()
        http_code = 200 if status.get("overall") == "healthy" else 207
        return jsonify(status), http_code
    except Exception as e:
        return jsonify({"overall": "error", "detail": str(e)}), 500


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

            logging.info(f"File saved to {file_path}. Starting ingestion…")

            ingestion = DataIngestion()
            chunks = ingestion.initiate_data_ingestion(file_path)

            if not chunks:
                return jsonify({
                    "error": "No text could be extracted from the PDF. It might be scanned or empty."
                }), 400

            transformation = DataTransformation()
            transformation.initiate_data_transformation(chunks)

            return jsonify({"success": f"File '{filename}' uploaded and indexed successfully!"})
        else:
            return jsonify({"error": "Only PDF files are allowed."}), 400

    except Exception as e:
        logging.error(f"Error in upload_file: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/query', methods=['POST'])
def get_answer():
    session_id = _get_session_id()
    hashed_sid = get_security().hash_session_id(session_id)
    start = time.time()
    error_str = None

    try:
        data = request.json or {}
        raw_query = data.get("query", "")

        # ── 1. Security ───────────────────────────────────────────────
        clean_query = get_security().sanitise_input(raw_query)
        ok, sec_reason = get_security().scan_for_injection(clean_query)
        if not ok:
            logging.warning(f"Security block [{hashed_sid}]: {sec_reason}")
            return jsonify(get_security().get_blocked_response(sec_reason)), 400

        # ── 2. Memory ─────────────────────────────────────────────────
        chat_history = get_memory().get_history(session_id)

        # ── 3. Guardrails ─────────────────────────────────────────────
        try:
            safe_query = get_guardrails().check_input(clean_query)
        except GuardrailViolation as gv:
            logging.warning(f"Guardrail input block [{hashed_sid}]: {gv}")
            return jsonify(get_guardrails().get_violation_response(str(gv))), 400

        # ── 4. Prediction pipeline ─────────────────────────────────────
        result = get_prediction_pipeline().predict(safe_query, chat_history=chat_history)
        raw_answer = result.get("answer", "")
        sources    = result.get("sources", [])
        meta       = result.get("_meta", {})

        # ── 5. Guardrails Output ──────────────────────────────────────
        safe_answer = get_guardrails().check_output(raw_answer)

        # ── 6. Governance ─────────────────────────────────────────────
        policy_passed, policy_reason = get_governance().enforce_policy(
            safe_query, safe_answer, sources
        )
        get_governance().audit_log({
            "session_id": hashed_sid,
            "query": safe_query,
            "answer": safe_answer,
            "num_sources": len(sources),
            "policy_passed": policy_passed,
            "policy_reason": policy_reason,
        })

        # ── 7. Evaluation ─────────────────────────────────────────────
        context_text = "\n\n".join([s.get("content", "") for s in sources])
        eval_scores = {}
        if sources:
            try:
                eval_scores = get_evaluator().evaluate_single(safe_query, safe_answer, context_text)
            except Exception as ee:
                logging.warning(f"Live LLM evaluation scoring failed: {ee}")

        # ── 8. Observability ──────────────────────────────────────────
        latency = time.time() - start
        get_observability().record_request(
            session_id=hashed_sid,
            query=safe_query,
            answer=safe_answer,
            latency=latency,
            num_docs=meta.get("num_docs", len(sources)),
            avg_score=meta.get("avg_score", 0.0),
            eval_scores=eval_scores
        )

        # ── 9. Memory Storage ─────────────────────────────────────────
        get_memory().add_turn(session_id, "user",      safe_query)
        get_memory().add_turn(session_id, "assistant", safe_answer)

        return jsonify({"answer": safe_answer, "sources": sources})

    except Exception as e:
        error_str = str(e)
        latency = time.time() - start
        get_observability().record_request(
            session_id=hashed_sid,
            query=raw_query if 'raw_query' in dir() else "",
            answer="",
            latency=latency,
            error=error_str,
        )
        logging.error(f"Error in get_answer [{hashed_sid}]: {e}")
        return jsonify({"error": error_str}), 500


@app.route('/reset', methods=['POST'])
def reset_database():
    try:
        session_id = _get_session_id()
        get_memory().clear(session_id)

        persist_directory = os.path.join('data', 'vector_store')
        if os.path.exists(persist_directory):
            import shutil
            shutil.rmtree(persist_directory)
            logging.info("Vector store deleted successfully.")
            return jsonify({"success": "Database cleared successfully! You can now upload fresh documents."})
        else:
            return jsonify({"success": "Database was already empty."})

    except Exception as e:
        logging.error(f"Error in reset_database: {e}")
        return jsonify({"error": f"Could not reset database: {str(e)}. It might be in use."}), 500


@app.route('/metrics', methods=['GET'])
def metrics_summary():
    try:
        summary = get_observability().get_summary(n=100)
        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)