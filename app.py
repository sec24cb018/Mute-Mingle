"""
app.py
-------
Mute Mingle backend.

Endpoints
  GET  /                              landing page (create / join a session)
  POST /api/session                   create a new session, returns {code}
  GET  /customer/<code>               customer (sign-to-speak) view
  GET  /staff/<code>                  staff (reply) view
  POST /api/predict                   {landmarks: [[x,y,z], ...21]} -> gesture prediction
  POST /api/session/<code>/message    {sender, text, gesture?} -> append to transcript
  GET  /api/session/<code>/messages?since=<id>  -> new messages after id (polling)

Real-time sync between the two roles is done with short HTTP polling rather
than WebSockets, so the whole project runs on plain Flask with no extra
server dependencies.
"""
import json
import os
import time
import uuid
import string
import random
import threading

from flask import Flask, request, jsonify, render_template, abort
import joblib

from model.feature_extraction import landmarks_to_features

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the trained gesture classifier
# ---------------------------------------------------------------------------
clf = joblib.load(os.path.join(MODEL_DIR, "gesture_model.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
with open(os.path.join(MODEL_DIR, "labels.json")) as f:
    LABELS = json.load(f)

CONFIDENCE_THRESHOLD = 0.65  # tuned against model/model_metrics.json — see README "Accuracy" section

# ---------------------------------------------------------------------------
# In-memory session store: { code: {"messages": [...], "created": ts} }
# Fine for a demo / single-process deployment; swap for Redis or a DB to
# scale beyond one server process.
# ---------------------------------------------------------------------------
SESSIONS = {}
LOCK = threading.Lock()


def new_code():
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=5))


def get_session(code):
    with LOCK:
        return SESSIONS.get(code)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/customer/<code>")
def customer_view(code):
    code = code.upper()
    if not get_session(code):
        abort(404, description="Session not found")
    return render_template("customer.html", code=code, gestures=LABELS)


@app.route("/staff/<code>")
def staff_view(code):
    code = code.upper()
    if not get_session(code):
        abort(404, description="Session not found")
    return render_template("staff.html", code=code)


# ---------------------------------------------------------------------------
# Session API
# ---------------------------------------------------------------------------
@app.route("/api/session", methods=["POST"])
def create_session():
    with LOCK:
        code = new_code()
        while code in SESSIONS:
            code = new_code()
        SESSIONS[code] = {"messages": [], "created": time.time()}
    return jsonify({"code": code})


@app.route("/api/session/<code>/message", methods=["POST"])
def post_message(code):
    code = code.upper()
    session = get_session(code)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    data = request.get_json(force=True) or {}
    sender = data.get("sender")
    text = (data.get("text") or "").strip()
    gesture = data.get("gesture")

    if sender not in ("customer", "staff") or not text:
        return jsonify({"error": "Invalid message"}), 400

    msg = {
        "id": str(uuid.uuid4()),
        "sender": sender,
        "text": text,
        "gesture": gesture,
        "ts": time.time(),
    }
    with LOCK:
        session["messages"].append(msg)

    return jsonify({"ok": True, "message": msg})


@app.route("/api/session/<code>/messages")
def get_messages(code):
    code = code.upper()
    session = get_session(code)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    since = request.args.get("since")
    msgs = session["messages"]
    if since:
        idx = next((i for i, m in enumerate(msgs) if m["id"] == since), None)
        msgs = msgs[idx + 1:] if idx is not None else msgs
    return jsonify({"messages": msgs})


# ---------------------------------------------------------------------------
# Gesture prediction API
# ---------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True) or {}
    points = data.get("landmarks")
    if not points or len(points) < 21:
        return jsonify({"error": "Expected 21 landmarks"}), 400

    try:
        feats = landmarks_to_features(points)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    feats_scaled = scaler.transform([feats])
    proba = clf.predict_proba(feats_scaled)[0]
    classes = clf.classes_
    best_idx = proba.argmax()
    best_key = classes[best_idx]
    confidence = float(proba[best_idx])

    if confidence < CONFIDENCE_THRESHOLD:
        return jsonify({"gesture": None, "confidence": confidence})

    meta = LABELS[best_key]
    return jsonify({
        "gesture": best_key,
        "label": meta["label"],
        "phrase": meta["phrase"],
        "confidence": confidence,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
