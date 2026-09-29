import os
import tempfile
import traceback

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from pipeline import _parse_chain, run_pipeline

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


@app.get("/")
def index():
    chain = []
    try:
        chain = _parse_chain(os.getenv("LLM_CHAIN", ""))
    except ValueError:
        pass
    return render_template("index.html", chain=chain)


@app.post("/analyze")
def analyze():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "No file uploaded"}), 400

    suffix = os.path.splitext(file.filename)[1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        file.save(tmp.name)
        tmp.close()
        response = run_pipeline(tmp.name)
        if response is None:
            return jsonify({"response": None, "filename": file.filename})
        return jsonify({"response": response, "filename": file.filename})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


if __name__ == "__main__":
    app.run(debug=True, port=5000)
