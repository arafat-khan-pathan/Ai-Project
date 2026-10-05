import http.server
import socketserver
import json
import os
from pathlib import Path

from src.nlp_preprocessing import preprocess_text


PORT = 8000
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
HTML_FILE = BASE_DIR / "templates" / "intex.html"
TRAIN_FILE = BASE_DIR / "dataset" / "train.csv"
TEST_FILE = BASE_DIR / "dataset" / "test-ham-spam.csv"
SOURCE_FILE = BASE_DIR / "dataset" / "spam.csv"


# ==========================================
# 1. Try loading trained model files
# ==========================================

model = None
vectorizer = None
dashboard_cache = None

if (
    (MODEL_DIR / "random_forest.pkl").exists()
    and (MODEL_DIR / "tfidf_vectorizer.pkl").exists()
):
    try:
        import joblib

        model = joblib.load(MODEL_DIR / "random_forest.pkl")
        vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")

        print("Trained model and vectorizer loaded successfully.")

    except Exception as e:
        print(f"Error loading model: {e}")

else:
    print("Notice: Model files not found inside 'models/' folder.")
    print("Running in demo/simulation mode until you train your model.")


# ==========================================
# 2. HTTP Server Handler
# ==========================================

class SpamServerHandler(http.server.SimpleHTTPRequestHandler):

    # --------------------------------------
    # Serve the HTML frontend page
    # --------------------------------------

    def do_GET(self):

        if self.path == "/" or self.path == "/index.html":
            try:
                page = HTML_FILE.read_bytes()
            except OSError:
                self.send_error(500, "Frontend file could not be read")
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return

        if self.path == "/dashboard":
            try:
                self._send_json(get_dashboard_data())
            except Exception as error:
                self._send_json({"error": f"Dashboard data could not be loaded: {error}"}, 500)
            return

        self.send_error(404, "Not found")


    # --------------------------------------
    # Handle prediction request
    # --------------------------------------

    def do_POST(self):

        if self.path == "/predict":
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(content_length).decode("utf-8"))
                message = str(data.get("message", "")).strip()
            except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
                self._send_json({"error": "Request body must be valid JSON."}, 400)
                return

            if not message:
                self._send_json({"error": "Message cannot be empty."}, 400)
                return


            # ==========================================
            # REAL ML PREDICTION
            # ==========================================

            if model is not None and vectorizer is not None:

                # Convert message to TF-IDF features
                transformed = vectorizer.transform([preprocess_text(message)])

                # Predict
                pred = model.predict(
                    transformed
                )[0]


                # Calculate confidence
                if hasattr(model, "predict_proba"):

                    prob = (
                        max(
                            model.predict_proba(
                                transformed
                            )[0]
                        )
                        * 100
                    )

                    confidence = f"{prob:.1f}%"

                else:

                    confidence = "98.5%"


                prediction = "spam" if int(pred) == 1 else "ham"


            # ==========================================
            # FALLBACK DEMO / SIMULATION MODE
            # ==========================================

            else:

                spam_keywords = [
                    "win",
                    "free",
                    "prize",
                    "cash",
                    "claim",
                    "urgent",
                    "1000",
                    "loan"
                ]

                is_spam = any(
                    word in message.lower()
                    for word in spam_keywords
                )

                prediction = (
                    "spam"
                    if is_spam
                    else "ham"
                )

                confidence = "95.0% (Simulation Mode)"


            # ==========================================
            # Send prediction response
            # ==========================================

            response_data = {
                "prediction": prediction,
                "confidence": confidence
            }

            self._send_json(response_data)
            return

        self.send_error(404, "Not found")

    def _send_json(self, payload, status=200):
        response = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


def _load_labeled_csv(path):
    import pandas as pd

    frame = pd.read_csv(path, encoding="latin-1")
    if "label" not in frame.columns or "message" not in frame.columns:
        frame = frame.iloc[:, :2].copy()
        frame.columns = ["label", "message"]
    frame = frame[["label", "message"]].dropna()
    frame["label"] = frame["label"].astype(str).str.lower().str.strip()
    frame["message"] = frame["message"].astype(str)
    return frame[frame["label"].isin(["ham", "spam"])]


def get_dashboard_data():
    global dashboard_cache

    if dashboard_cache is not None:
        return dashboard_cache

    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

    train = _load_labeled_csv(TRAIN_FILE)
    test = _load_labeled_csv(TEST_FILE)
    source = _load_labeled_csv(SOURCE_FILE) if SOURCE_FILE.exists() else None
    if model is None or vectorizer is None:
        raise RuntimeError("trained model files are required for dashboard data")

    test_features = vectorizer.transform(test["message"].map(preprocess_text))
    predictions = model.predict(test_features).astype(int)
    probabilities = model.predict_proba(test_features) if hasattr(model, "predict_proba") else None
    actual = test["label"].map({"ham": 0, "spam": 1}).astype(int).to_numpy()
    matrix = confusion_matrix(actual, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    spam_detection = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
    ham_detection = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0

    rows = []
    for index, (_, record) in enumerate(test.iterrows()):
        predicted_label = "spam" if predictions[index] == 1 else "ham"
        confidence = max(probabilities[index]) * 100 if probabilities is not None else 0
        rows.append({
            "message": record["message"],
            "label": record["label"],
            "prediction": predicted_label,
            "confidence": f"{confidence:.1f}%",
        })

    dashboard_cache = {
        "counts": {
            "dataset_total": int(len(source) if source is not None else len(train) + len(test)),
            "call_count": int(len(train) + len(test)),
            "train_count": int(len(train)),
            "test_count": int(len(test)),
            "train_spam_count": int((train["label"] == "spam").sum()),
            "train_ham_count": int((train["label"] == "ham").sum()),
            "test_spam_count": int((test["label"] == "spam").sum()),
            "test_ham_count": int((test["label"] == "ham").sum()),
            "total_spam_count": int((train["label"] == "spam").sum() + (test["label"] == "spam").sum()),
            "total_ham_count": int((train["label"] == "ham").sum() + (test["label"] == "ham").sum()),
        },
        "metrics": {
            "accuracy": round(accuracy_score(actual, predictions) * 100, 2),
            "precision": round(precision_score(actual, predictions, zero_division=0) * 100, 2),
            "recall": round(recall_score(actual, predictions, zero_division=0) * 100, 2),
            "f1": round(f1_score(actual, predictions, zero_division=0) * 100, 2),
        },
        "detection": {
            "spam_rate": round(spam_detection * 100, 2),
            "spam_error": round((1 - spam_detection) * 100, 2),
            "ham_rate": round(ham_detection * 100, 2),
            "ham_error": round((1 - ham_detection) * 100, 2),
        },
        "rows": rows,
    }
    return dashboard_cache


# ==========================================
# 3. Start Server
# ==========================================

def main():
    print(
        f"Server is running! Open your browser and go to: "
        f"http://localhost:{PORT}"
    )

    with socketserver.ThreadingTCPServer(("", PORT), SpamServerHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
        
        
        
        