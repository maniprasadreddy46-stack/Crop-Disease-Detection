import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from PIL import Image

# Optional face detection helps reject obvious non-crop photos (for example, selfies).
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    cv2 = None
    OPENCV_AVAILABLE = False

try:
    from streamlit_mic_recorder import speech_to_text
    VOICE_INPUT_AVAILABLE = True
except ImportError:
    speech_to_text = None
    VOICE_INPUT_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(page_title="CropCare AI | Multi-Crop Disease Detection", page_icon="🌿", layout="wide", initial_sidebar_state="collapsed")

# -------------------- PROFESSIONAL UI THEME --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Noto+Sans+Telugu:wght@400;500;600;700&display=swap');
:root { --forest:#123d2b; --leaf:#23764b; --mint:#e9f5ed; --ink:#17251d; --muted:#64746a; --line:#dce8df; }
html, body, [class*="css"] { font-family:'DM Sans','Noto Sans Telugu',sans-serif; }
.stApp { background:linear-gradient(180deg,#f5faf6 0%,#f8faf8 42%,#f3f7f4 100%); color:var(--ink); }
[data-testid="stHeader"] { background:rgba(245,250,246,.85); }
.block-container { max-width:1180px; padding-top:1.6rem; padding-bottom:3rem; }
h1,h2,h3 { color:var(--forest)!important; letter-spacing:-.025em; }
h1 { font-weight:700!important; }
h2 { font-weight:700!important; margin-top:.4rem!important; }
p, label, .stCaption { color:var(--ink); }
.hero { padding:30px 32px; border-radius:24px; color:white; background:linear-gradient(120deg,#123d2b 0%,#1d6841 58%,#4a9461 100%); box-shadow:0 14px 35px rgba(18,61,43,.14); margin:0 0 22px 0; position:relative; overflow:hidden; }
.hero:after { content:'🌿'; position:absolute; right:34px; top:18px; font-size:78px; opacity:.18; }
.hero-kicker { font-size:12px; letter-spacing:.16em; text-transform:uppercase; font-weight:700; color:#c9ecd4; margin-bottom:9px; }
.hero h1 { color:white!important; font-size:clamp(28px,4vw,42px); margin:0 0 8px 0; line-height:1.15; }
.hero p { color:#e5f3e9; margin:0; max-width:680px; font-size:15px; line-height:1.6; }
.section-card { background:#fff; border:1px solid var(--line); border-radius:18px; padding:20px 22px; box-shadow:0 5px 18px rgba(22,54,34,.035); margin-bottom:16px; }
.section-eyebrow { font-size:11px; font-weight:700; color:var(--leaf); text-transform:uppercase; letter-spacing:.12em; margin-bottom:6px; }
div[data-testid="stFileUploader"] { background:#fbfdfb; border:1.5px dashed #a9cbb3; border-radius:14px; padding:12px; }
.stButton > button, .stFormSubmitButton > button { border-radius:11px; min-height:42px; font-weight:700; border:1px solid #c8dbce; transition:all .15s ease; }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] { background:var(--leaf); color:white; border:0; }
.stButton > button:hover, .stFormSubmitButton > button:hover { border-color:var(--leaf); transform:translateY(-1px); }
[data-testid="stMetric"] { background:#f7fbf8; border:1px solid var(--line); padding:15px 17px; border-radius:15px; }
[data-testid="stMetricLabel"] { color:var(--muted); font-weight:600; }
[data-testid="stMetricValue"] { color:var(--forest); font-weight:700; }
[data-testid="stChatMessage"] { border:1px solid #e1eae3; border-radius:15px; padding:12px 15px; margin-bottom:10px; }
[data-testid="stExpander"] { background:#fff; border:1px solid var(--line); border-radius:13px; }
[data-testid="stAlert"] { border-radius:13px; }
[data-testid="stSelectbox"] > div > div, [data-testid="stTextInput"] input { border-radius:10px; }
hr { border-color:var(--line); }
.small-note { color:var(--muted); font-size:12px; line-height:1.6; }
.status-pill { display:inline-block; background:#e1f3e6; color:#1d6841; padding:5px 10px; border-radius:999px; font-size:12px; font-weight:700; margin-top:12px; }
@media (max-width:700px) { .block-container { padding-left:1rem; padding-right:1rem; } .hero { padding:24px 20px; } .hero:after { right:12px; top:12px; font-size:52px; } }
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "three_crop_model_final.keras"
CLASS_NAMES_PATH = BASE_DIR / "model" / "three_crop_class_names.json"
IMAGE_SIZE = (224, 224)
CROP_PREFIXES = {"Paddy": "paddy_", "Cotton": "cotton_", "Chilli": "chilli_"}
LANGUAGES = {
    "English": {"speech_code": "en-IN", "system_language": "English"},
    "తెలుగు": {"speech_code": "te-IN", "system_language": "Telugu"},
}


def get_gemini_api_key():
    """Read the Gemini key from Streamlit secrets or the project .env file."""
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret_key = ""
    return (secret_key or os.getenv("GEMINI_API_KEY", "")).strip()


def get_gemini_client():
    """Create a Gemini client and preserve the real error for diagnostics."""
    api_key = get_gemini_api_key()
    if not api_key:
        return None, "API key not found. Check that .env is beside app.py and contains GEMINI_API_KEY=your_key."
    try:
        from google import genai
        return genai.Client(api_key=api_key), None
    except Exception as error:
        return None, f"Could not create Gemini client: {type(error).__name__}: {error}"


@st.cache_data
def load_class_names():
    if not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        names = json.load(file)
    if not isinstance(names, list) or not names:
        raise ValueError("The class names file is invalid.")
    return names


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


try:
    class_names = load_class_names()
    model = load_model()
    if model.output_shape[-1] != len(class_names):
        st.error("The model output count does not match the class names file. Check that both files came from the same training run.")
        st.stop()
except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()

crop_classes = {
    crop: [name for name in class_names if name.lower().startswith(prefix)]
    for crop, prefix in CROP_PREFIXES.items()
}
available_crops = [crop for crop in CROP_PREFIXES if crop_classes[crop]]
if not available_crops:
    st.error("No Paddy, Cotton, or Chilli classes were found.")
    st.stop()


def clean_disease_name(label):
    name = label
    for prefix in CROP_PREFIXES.values():
        if name.lower().startswith(prefix):
            name = name[len(prefix):]
            break
    return name.replace("_", " ").title()


def detect_human_face(image):
    """Return True when OpenCV's frontal-face detector finds an obvious face.

    This is a simple guard against selfies, not a general-purpose crop/leaf detector.
    If OpenCV is not installed, return None so the app can explain the limitation.
    """
    if not OPENCV_AVAILABLE:
        return None
    try:
        rgb = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            return None
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        return len(faces) > 0
    except Exception:
        return None


def get_recommendation(disease):
    name = disease.lower()
    if "healthy" in name or "normal" in name:
        return "The image is classified as healthy. Continue regular field monitoring, balanced irrigation, and good crop hygiene."
    if "nutritional" in name:
        return "Check soil and plant nutrient conditions. Confirm the crop's needs before changing fertilizer application."
    if any(word in name for word in ("virus", "tungro", "murda")):
        return "Inspect nearby plants and possible insect vectors. Consult an agricultural expert before choosing treatment."
    if any(word in name for word in ("bacterial", "blight", "streak")):
        return "Maintain field hygiene and avoid moving potentially contaminated plant material between fields. Seek expert advice."
    if any(word in name for word in ("blast", "spot", "mildew", "wilt", "hispa", "dead heart")):
        return "Inspect surrounding plants for similar symptoms. Confirm the suspected disease with an agricultural expert before treatment."
    return "This is a preliminary AI prediction. Ask a local agricultural expert to confirm the symptoms and suitable next steps."


def fallback_answer(question, language):
    q = question.lower()
    if language == "తెలుగు":
        if "blast" in q or "బ్లాస్ట్" in q:
            return "వరి బ్లాస్ట్ వ్యాధి అనుమానం ఉంటే ఆకులపై మచ్చలను పరిశీలించండి. ఫోటో ఆధారంగా మాత్రమే ఖచ్చితమైన నిర్ధారణ చేయలేం. దగ్గరలోని వ్యవసాయ నిపుణుడిని సంప్రదించి సరైన నియంత్రణ చర్యలను నిర్ధారించండి."
        if any(w in q for w in ["score", "accuracy", "శాతం", "స్కోర్"]):
            return "మోడల్ స్కోర్ అనేది ఆ తరగతికి మోడల్ ఇచ్చిన probability మాత్రమే; అది ఖచ్చితమైన వ్యాధి నిర్ధారణకు హామీ కాదు. గత టెస్ట్ సెట్‌లో నమోదైన మొత్తం accuracy 74.52%. ఒక్క ఫోటో ఫలితం వేరుగా ఉండవచ్చు."
        return "నేను వరి, పత్తి, మిరప పంటల వ్యాధుల గురించి ప్రాథమిక సమాచారం ఇవ్వగలను. పంట పేరు, కనిపిస్తున్న లక్షణాలు చెప్పండి లేదా లీఫ్ ఫోటోను అప్లోడ్ చేయండి. చికిత్సకు ముందు వ్యవసాయ నిపుణుడితో నిర్ధారించుకోండి. ChatGPT తరహా AI సమాధానాల కోసం Gemini API key సెటప్ చేయాలి."
    if any(w in q for w in ["hello", "hi", "hey"]):
        return "Hello! I am your Crop Disease Assistant. Ask about Paddy, Cotton, Chilli, symptoms, prevention, model scores, or your project."
    if "blast" in q:
        return "Paddy Blast is a possible rice disease, but an image model cannot confirm it with certainty. Check the leaf and surrounding plants, and ask an agricultural expert to confirm before choosing treatment."
    if any(w in q for w in ["score", "accuracy", "confidence", "probability"]):
        return "The model score is its output probability for a class, not a guarantee of correctness. The previously measured overall test accuracy was 74.52% on 2,076 test images."
    if any(w in q for w in ["prevent", "prevention", "control"]):
        return "Monitor plants regularly, maintain field hygiene, avoid moving potentially infected plant material between fields, and confirm the disease with an agricultural expert before applying treatment."
    return "Gemini API is not configured yet, so this is a basic fallback response. Add GEMINI_API_KEY to your local .env file to enable natural AI answers. I can help with Paddy, Cotton, and Chilli disease questions."


def ask_gemini(question, language, history, prediction):
    """Ask Gemini and return readable diagnostics instead of hiding errors."""
    client, client_error = get_gemini_client()
    if client is None:
        return None, client_error

    prediction_context = "No recent leaf prediction is available."
    if prediction:
        prediction_context = (
            f"Latest app prediction: crop={prediction['crop']}; "
            f"class={prediction['disease']}; score={prediction['score']:.2f}%. "
            "This score is not a guarantee of diagnosis."
        )

    instruction = (
        "You are a helpful agricultural education assistant for a student "
        "Multi-Crop Disease Detection prototype. Answer naturally and "
        f"conversationally in {LANGUAGES[language]['system_language']}. "
        "Supported crops are Paddy (rice), Cotton, and Chilli. Explain clearly, "
        "ask follow-up questions when needed, and be cautious. Never say an "
        "image prediction is a confirmed diagnosis. Do not invent pesticide "
        "dosages or prescribe chemicals. Recommend a qualified local agricultural "
        "expert before treatment. A model score is not calibrated certainty. "
        "Be transparent when uncertain."
    )

    lines = [f"Instructions: {instruction}", prediction_context]
    for item in history[-10:]:
        role = "User" if item.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {item.get('content', '')}")
    lines.append(f"User: {question}")

    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents="\n\n".join(lines),
        )
        answer = getattr(response, "text", None)
        if answer and answer.strip():
            return answer.strip(), None
        return None, "Gemini returned an empty answer. Check model access and API quota."
    except Exception as error:
        return None, f"Gemini error: {type(error).__name__}: {error}"


# ==================== MAIN PAGE ====================
st.markdown(f'''<div class="hero">
  <div class="hero-kicker">CropCare AI · Smart Agriculture</div>
  <h1>Multi-Crop Disease Detection</h1>
  <p>Screen leaf images for common Paddy, Cotton and Chilli disease classes with a deep-learning model, then explore practical guidance with the AI crop assistant.</p>
  <span class="status-pill">● Model ready &nbsp; · &nbsp; MobileNetV2 &nbsp; · &nbsp; {len(class_names)} trained classes</span>
</div>''', unsafe_allow_html=True)
st.markdown('<div class="section-card"><div class="section-eyebrow">01 / Leaf screening</div><h2 style="margin:0">Upload a crop image</h2><p class="small-note">Choose the crop first, then upload a clear JPG or PNG leaf photo. Results are preliminary and should be confirmed by an agricultural expert.</p></div>', unsafe_allow_html=True)

crop_col, upload_col = st.columns([0.9, 1.4], gap="large")
with crop_col:
    selected_crop = st.selectbox("Crop type", available_crops)
    selected_classes = crop_classes[selected_crop]
    st.caption(f"{len(selected_classes)} trained classes for {selected_crop}")
with upload_col:
    image_source_mode = st.radio(
        "Choose how to add a leaf image",
        ["📁 Upload image", "📷 Take photo with camera"],
        horizontal=True,
        key="image_source_mode",
    )
    uploaded_file = None
    camera_photo = None
    if image_source_mode == "📁 Upload image":
        uploaded_file = st.file_uploader(
            "Upload leaf image (JPG or PNG)",
            type=["jpg", "jpeg", "png"],
            help="Choose a clear photo of one crop leaf.",
        )
        image_input = uploaded_file
        image_caption = "Selected leaf image"
    else:
        st.caption("Allow camera permission when your browser asks. Hold the leaf in good light and keep it in focus.")
        camera_photo = st.camera_input("Take a leaf photo", key="crop_leaf_camera")
        image_input = camera_photo
        image_caption = "Captured leaf photo"

if image_input is not None:
    try:
        image = Image.open(image_input).convert("RGB")
    except Exception:
        st.error("Unable to read this image. Try another JPG or PNG, or take the photo again.")
        st.stop()
    st.image(image, caption=image_caption, width=350)
    face_check = detect_human_face(image)
    if face_check is True:
        st.error("🚫 A human face was detected in this photo. Please take or upload a clear photo of a Paddy, Cotton, or Chilli leaf instead. Disease prediction is disabled for this image.")
    elif face_check is None:
        st.info("Image check: automatic face screening is unavailable. The app will still warn you when the model score is low. For face-photo blocking, install OpenCV with: python -m pip install opencv-python")
    else:
        st.success("Image check passed: no obvious human face detected. This does not guarantee that the image contains a crop leaf.")

    if st.button("🔍 Detect Disease", type="primary", disabled=(face_check is True)):
        with st.spinner("Analyzing the image..."):
            resized_image = image.resize(IMAGE_SIZE)
            image_array = np.asarray(resized_image, dtype=np.float32)
            probabilities = model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0]
            if len(probabilities) != len(class_names):
                st.error("Prediction output does not match class names.")
                st.stop()
            crop_indices = [i for i, name in enumerate(class_names) if name in selected_classes]
            if not crop_indices:
                st.error("No classes are available for this crop.")
                st.stop()
            crop_probabilities = probabilities[crop_indices]
            best_position = int(np.argmax(crop_probabilities))
            predicted_index = crop_indices[best_position]
            disease = clean_disease_name(class_names[predicted_index])
            confidence = float(probabilities[predicted_index]) * 100
            top_positions = np.argsort(crop_probabilities)[::-1][:3]
            top_results = [
                (clean_disease_name(class_names[crop_indices[pos]]), float(crop_probabilities[pos]) * 100)
                for pos in top_positions
            ]

        # A low score should not be presented as a disease identification.
        LOW_CONFIDENCE_THRESHOLD = 35.0
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            st.session_state.pop("last_prediction", None)
            st.error(f"⚠️ Low model score ({confidence:.2f}%). The model cannot make a reliable prediction from this image.")
            st.warning("Please upload or capture a clear close-up of one Paddy, Cotton, or Chilli leaf in good lighting. Avoid selfies, people, soil-only photos, screenshots, and blurry images. No disease has been confirmed.")
            st.subheader("Top possible classes (not a diagnosis)")
            for label, score in top_results:
                st.write(f"**{label}** — {score:.2f}%")
            st.stop()

        st.session_state["last_prediction"] = {
            "crop": selected_crop, "disease": disease,
            "score": confidence, "top_results": top_results,
        }
        st.success("Prediction completed!")
        c1, c2 = st.columns(2)
        c1.metric("🌾 Selected Crop", selected_crop)
        c2.metric("🎯 Model Score", f"{confidence:.2f}%")
        st.subheader("🦠 Predicted Class")
        st.header(disease)
        if confidence >= 80:
            st.success(f"High model score: {confidence:.2f}%")
        else:
            st.warning(f"Moderate model score: {confidence:.2f}%. This is still only a preliminary prediction.")
        st.caption("The score is the model's output probability for this class, not a guarantee of diagnostic accuracy.")
        st.subheader("📊 Other Likely Classes")
        for label, score in top_results:
            st.write(f"**{label}** — {score:.2f}%")
        st.subheader("💡 Basic Guidance")
        st.write(get_recommendation(disease))
        st.caption("This is a screening prototype. Image quality, lighting, backgrounds, and differences from training data can affect predictions. Confirm results with an agricultural expert.")


# ==================== CHATBOT AND VOICE ====================
st.divider()
st.header("🤖 AI Crop Chatbot & Voice Assistant")
st.write("Choose Telugu or English, then type or speak your question. With a Gemini API key, this chatbot generates natural AI answers.")
language = st.selectbox("🌐 Response and voice language", ["English", "తెలుగు"], key="chat_language")
speech_language = LANGUAGES[language]["speech_code"]

if "crop_chat_history" not in st.session_state:
    st.session_state.crop_chat_history = []

last_prediction = st.session_state.get("last_prediction")
if last_prediction:
    with st.expander("🌿 Latest detection result"):
        st.write(f"**Crop:** {last_prediction['crop']}")
        st.write(f"**Predicted class:** {last_prediction['disease']}")
        st.write(f"**Model score:** {last_prediction['score']:.2f}%")
        st.caption("The model score is not a guarantee of diagnosis.")

for msg in st.session_state.crop_chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

st.markdown("### 🎙️ Voice input")
voice_question = None
if VOICE_INPUT_AVAILABLE:
    try:
        voice_question = speech_to_text(
            language=speech_language,
            start_prompt="🎙️ Start speaking / మాట్లాడటం ప్రారంభించండి",
            stop_prompt="⏹️ Stop / ఆపండి",
            just_once=True,
            use_container_width=True,
            key=f"crop_voice_{speech_language}",
        )
    except Exception:
        st.warning("Voice recognition failed. Check microphone permission or type your question. Telugu speech recognition depends on browser support.")
else:
    st.info("Voice input package is missing. Run: python -m pip install streamlit-mic-recorder")

with st.form("crop_chat_form", clear_on_submit=True):
    typed_question = st.text_input(
        "Your question / మీ ప్రశ్న",
        placeholder="Example: What is Paddy Blast?" if language == "English" else "ఉదాహరణ: వరి బ్లాస్ట్ వ్యాధి అంటే ఏమిటి?",
    )
    send_question = st.form_submit_button("💬 Ask AI / అడగండి")

question = None
if send_question and typed_question.strip():
    question = typed_question.strip()
elif isinstance(voice_question, str) and voice_question.strip():
    question = voice_question.strip()

if question:
    prior_history = list(st.session_state.crop_chat_history)
    with st.spinner("Generating answer... / సమాధానం తయారవుతోంది..."):
        answer, api_error = ask_gemini(question, language, prior_history, last_prediction)
        if answer is None:
            answer = fallback_answer(question, language)
            st.warning(api_error + " Showing a basic fallback answer for now.")
    st.session_state.crop_chat_history.append({"role": "user", "content": question})
    st.session_state.crop_chat_history.append({"role": "assistant", "content": answer})
    st.rerun()

latest_answer = next(
    (msg["content"] for msg in reversed(st.session_state.crop_chat_history) if msg["role"] == "assistant"),
    None,
)
if latest_answer:
    st.markdown("### 🔊 Voice output")
    st.caption("Telugu speech requires a Telugu voice installed or supported by your browser/device.")
    if st.button("🔊 Read Latest Answer Aloud / సమాధానం వినండి"):
        answer_json = json.dumps(latest_answer, ensure_ascii=False)
        lang_json = json.dumps(speech_language)
        components.html(
            f"""
            <div style="font-family:sans-serif">
              <button id="speak" style="padding:8px 12px;cursor:pointer">▶ Play / వినండి</button>
              <button id="stop" style="padding:8px 12px;cursor:pointer">■ Stop / ఆపండి</button>
              <p id="status" aria-live="polite"></p>
              <script>
                const answerText = {answer_json};
                const speechLang = {lang_json};
                const status = document.getElementById("status");
                document.getElementById("speak").onclick = () => {{
                  if (!("speechSynthesis" in window)) {{
                    status.textContent = "Speech output is not supported by this browser.";
                    return;
                  }}
                  window.speechSynthesis.cancel();
                  const u = new SpeechSynthesisUtterance(answerText);
                  u.lang = speechLang;
                  u.rate = 0.92;
                  u.onstart = () => status.textContent = "Playing...";
                  u.onend = () => status.textContent = "Finished.";
                  u.onerror = () => status.textContent = "No compatible voice found. Check installed system/browser voices.";
                  window.speechSynthesis.speak(u);
                }};
                document.getElementById("stop").onclick = () => {{
                  if ("speechSynthesis" in window) window.speechSynthesis.cancel();
                  status.textContent = "Stopped.";
                }};
              </script>
            </div>
            """,
            height=85,
        )

if st.session_state.crop_chat_history and st.button("🗑️ Clear Chat History"):
    st.session_state.crop_chat_history = []
    st.rerun()


# ==================== ABOUT AND MODEL INFO ====================
st.markdown('<div class="section-card"><div class="section-eyebrow">03 / Project details</div><h2 style="margin:0">About CropCare AI</h2><p class="small-note">Built as a student screening prototype for crop-leaf image classification and agricultural education.</p></div>', unsafe_allow_html=True)
with st.expander("🌾 About the Project"):
    st.write("""
    This project uses MobileNetV2 transfer learning to classify leaf images
    into classes present in the training dataset. Supported crops are Paddy,
    Cotton, and Chilli. It includes basic guidance, an optional Gemini-powered
    conversational chatbot, selectable Telugu/English responses, and browser
    voice input/output. Free-tier API availability and quotas depend on the
    current Google AI Studio account. This is a screening prototype, not a
    replacement for professional agricultural diagnosis.
    """)

with st.expander("📊 Model Information"):
    st.write("**Architecture:** MobileNetV2 with transfer learning")
    st.write(f"**Total trained classes:** {len(class_names)}")
    st.write(f"**Selected crop:** {selected_crop}")
    st.write(f"**Classes for selected crop:** {len(selected_classes)}")
    for name in selected_classes:
        st.write(f"• {clean_disease_name(name)}")
