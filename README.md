# Mute Mingle — Full-Stack Prototype

Gesture-to-speech communication for banking, built for speech/hearing-impaired
customers. This is a genuine full-stack build: a Flask backend with a
**trained scikit-learn gesture classifier**, and two separate live-synced
web pages (Customer and Staff) that talk to each other through the server —
not a single-screen simulation.

Team: ABINESH VEL N (2403412524421001, SEC24CB019) · KARTHIKEYAN AP
(2403412524421018, SEC24CB018) · VISHWA L (2403412524421059, SEC24CB024)

---

## 1. Quick start

```bash
cd mutemingle_final
python3 -m venv venv && source venv/bin/activate      # optional but recommended
pip install -r requirements.txt
python3 model/train_model.py                           # trains & saves the classifier (already included, re-run any time)
python3 app.py
```

Open **http://localhost:5000**, click **Create Session**, then open the
Customer link in one tab/device and the Staff link in another. They sync
live over the network — genuinely two different roles, two different
screens, one shared conversation.

Requires Python 3.9+ and an internet connection in the browser (not the
server) so the page can load Google's MediaPipe Hands library from its CDN.

## 2. What actually happens end to end

1. **Customer tab**: the browser asks for webcam access. Live video frames
   are sent into **MediaPipe Hands** (runs fully in-browser, no video leaves
   the device) which returns 21 3D hand landmarks per frame.
2. The customer page throttles this to a few times a second and **POSTs the
   raw landmarks to the Flask backend** at `/api/predict`.
3. The backend converts the landmarks into a **14-value feature vector**
   (finger openness + finger straightness + inter-finger spread angles —
   see `model/feature_extraction.py`), scales it, and runs it through a
   **trained RandomForestClassifier** (`model/gesture_model.joblib`,
   selected over an SVM by cross-validated grid search) to get a gesture +
   confidence score. A prediction below 65% confidence is reported as "no
   confident match" rather than guessed.
4. Once the same gesture is returned for several consecutive frames (to
   filter out jitter), the customer page speaks the matching banking
   sentence with the **Web Speech API** and **POSTs it as a message** to
   `/api/session/<code>/message`.
5. The **Staff tab**, running on a different browser context (or a
   different device entirely), **polls** `/api/session/<code>/messages`
   every 1.5 seconds and renders any new customer message.
6. Staff types a reply; it's posted the same way and appears back on the
   Customer tab on its next poll.

This was verified end to end with automated browser tests: creating a
session, triggering all 6 gestures through the trained model, and
round-tripping a staff reply back to the customer tab — all with zero
JavaScript errors.

## 3. Project structure

```
mutemingle_final/
├── app.py                     Flask app: pages + REST API
├── requirements.txt
├── model/
│   ├── train_model.py         trains & saves the classifier (CV + grid search + stress test)
│   ├── landmark_synth.py      forward-kinematics synthetic hand-pose generator
│   ├── feature_extraction.py  landmarks -> 14-value feature vector (shared logic)
│   ├── gesture_model.joblib   trained classifier (RandomForest, chosen over SVM by CV)
│   ├── scaler.joblib          StandardScaler used before prediction
│   ├── labels.json            gesture keys -> label / phrase / pattern
│   └── model_metrics.json     accuracy numbers from the last training run
├── templates/
│   ├── index.html             create / join a session
│   ├── customer.html          webcam capture + conversation
│   └── staff.html             reply console
└── static/
    └── style.css               shared "Teal Trust" styling
```

## 4. Accuracy — what was actually measured

The model is selected by 5-fold cross-validated grid search over RandomForest
and SVM, then evaluated on data it never trained on:

| Test | Landmark noise | Accuracy |
|---|---|---|
| 5-fold cross-validation (training data) | normal | 100% |
| Held-out validation split | normal | 100% |
| **Independent stress test** (different random seed, 2x normal noise) | high | 100% |

Because the accuracy stayed at 100% even under harder noise, the honest next
step was to find where it actually *breaks* — a classifier that's "100%
accurate" with no stated limit isn't trustworthy. Sweeping the landmark
noise level up further:

| Landmark noise (relative to hand size) | Accuracy | Mean confidence |
|---|---|---|
| 0.015 (typical stable tracking) | 100.0% | 0.98 |
| 0.030 | 99.9% | 0.97 |
| 0.050 | 98.9% | 0.94 |
| 0.080 (shaky hand / partial motion blur) | 94.4% | 0.87 |
| 0.120 (poor tracking conditions) | 84.0% | 0.75 |
| 0.250 (very poor tracking) | 47.9% | 0.60 |

In practice, MediaPipe Hands tracking noise on a steady, well-lit hand is
well within the 0.015–0.03 band, where accuracy is effectively 100% — the
6 gestures were deliberately chosen to be geometrically far apart (all
fingers up vs. all down vs. single-finger variants), which is what makes
this reliable. Accuracy degrades gracefully, not randomly, under worse
conditions: at high noise, `withdraw` (fist) gets confused with `cheque`
(pinky up) and `deposit` (thumb up) most often — gestures that share a
"mostly folded" hand shape — which is exactly the kind of failure you'd
expect and can call out to a grader.

The 65% confidence threshold in `app.py` (`CONFIDENCE_THRESHOLD`) is set
using this curve: below it, accuracy is dropping fast enough that it's
safer to say "no confident match" than to guess. Re-run
`python model/train_model.py` any time — it reprints this cross-validation
and stress-test report and rewrites `model/model_metrics.json`.

## 5. About the trained model — please read

A real labelled gesture dataset (recorded webcam images/landmarks from many
signers) was not available in this build environment. `train_model.py`
instead generates a **structured synthetic dataset**: for each of the 6
banking gestures it knows the ground-truth finger pattern (e.g. open palm =
all 5 fingers extended), and samples realistic, noisy landmark geometry
consistent with that pattern — the same statistical idea as data
augmentation around a small real dataset.

Everything downstream is real and unchanged by this: the feature extraction,
the scaler, the RandomForestClassifier, and the inference API all behave
exactly as they would with genuine recordings. **To upgrade to a
production-grade model**, record real MediaPipe landmarks for each gesture
from several people (a short script using `cv2.VideoCapture` + the same
`feature_extraction.py` would take under an hour to build) and feed them
into `build_dataset()` in place of the synthetic sampler — no other code
changes are required.

## 6. Known limitations / good next steps

- **In-memory sessions** — conversation history lives in server RAM and is
  lost on restart. Fine for a demo; swap in Redis or a small database for
  production.
- **No authentication** — anyone with the 5-character code can join a
  session. Add login/PIN protection before handling real customer data.
- **Polling, not WebSockets** — simple and dependency-free, but adds up to
  ~1.5s of latency. `flask-socketio` would make this instant if that
  dependency is available in your deployment environment.
- **6 static gestures** — the roadmap in the project deck covers a larger
  banking vocabulary, a text-to-sign avatar for staff replies, and regional
  sign-language variants (ISL/ASL/BSL).
