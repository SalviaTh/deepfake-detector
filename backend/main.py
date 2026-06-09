import io, base64, cv2, torch, numpy as np, os, tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

from model import DeepFakeDetector
from gradcam import GradCAM
from preprocess import extract_face

app = FastAPI(title="DeepFake Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows frontend to connect from any URL
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model once at startup with memory optimizations ────────────────
import gc
torch.set_num_threads(1)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepFakeDetector()

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'efficientnet_b4_ff++.pth')

if os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH}")
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"ERROR loading model: {e}. Using untrained model.")
else:
    print(f"WARNING: Model file not found at {MODEL_PATH}. Using untrained model for inference.")

model.to(DEVICE).eval()
gc.collect()
grad_cam = GradCAM(model)

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Endpoints ──────────────────────────────────────────────────────────
@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    raw = await file.read()
    img_array = np.frombuffer(raw, np.uint8)
    bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "Could not decode image")

    # ── PASS 1: Global Analysis (Clothing, Background, Context) ──
    full_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    full_pil = Image.fromarray(full_rgb)
    full_tensor = TRANSFORM(full_pil).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        full_logits = model(full_tensor)
        full_fake_prob = torch.softmax(full_logits, 1)[0, 1].item()

    # ── PASS 2: Face-Level Analysis (Swaps, Expressions) ──
    face_bgr, bbox = extract_face(bgr)
    
    if face_bgr is not None:
        # Use Grad-CAM on the face for precise visualization
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        face_tensor = TRANSFORM(face_pil).unsqueeze(0).to(DEVICE)
        
        heatmap, pred_idx, probs = grad_cam.generate(face_tensor)
        face_fake_prob = probs[1].item()
        
        overlay_bgr = GradCAM.overlay(face_bgr, heatmap)
        _, buffer = cv2.imencode('.jpg', overlay_bgr)
        heatmap_b64 = base64.b64encode(buffer).decode()
        
        _, orig_buf = cv2.imencode('.jpg', face_bgr)
        face_b64 = base64.b64encode(orig_buf).decode()
    else:
        # If no face, run Grad-CAM on the full image to see what's suspicious
        heatmap, pred_idx, probs = grad_cam.generate(full_tensor)
        face_fake_prob = probs[1].item()
        
        overlay_bgr = GradCAM.overlay(bgr, heatmap)
        _, buffer = cv2.imencode('.jpg', overlay_bgr)
        heatmap_b64 = base64.b64encode(buffer).decode()
        face_b64 = None
        bbox = None

    # ── Final Decision Logic ──
    # We take the maximum suspiciousness (FAKE probability) from either check
    final_fake_prob = max(full_fake_prob, face_fake_prob)
    
    if final_fake_prob > 0.5:
        label = "FAKE"
        confidence = final_fake_prob
    else:
        label = "REAL"
        confidence = 1.0 - final_fake_prob

    return JSONResponse({
        "label": label,
        "confidence": round(confidence * 100, 2),
        "bbox": bbox,
        "face_image": face_b64,
        "heatmap": heatmap_b64,
        "analysis_details": {
            "global_score": round(full_fake_prob * 100, 2),
            "face_score": round(face_fake_prob * 100, 2),
            "face_detected": face_bgr is not None
        }
    })


@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...),
                       start_sec: float = 0.0,
                       end_sec: float = None):
    """
    Analyze trimmed video segment frame by frame,
    return per-frame scores and overall verdict.
    """
    # Use tempfile to handle temporary video file safely across OS
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(400, "Could not open video file")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0 # Fallback
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int(start_sec * fps)
        end_frame   = int(end_sec   * fps) if end_sec else total_frames
        sample_step = max(1, (end_frame - start_frame) // 30)  # max 30 frames

        frame_results = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for fi in range(start_frame, end_frame, sample_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                break

            face_bgr, _ = extract_face(frame)
            if face_bgr is None:
                continue

            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            tensor   = TRANSFORM(Image.fromarray(face_rgb)).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(tensor)
                prob_fake = torch.softmax(logits, 1)[0, 1].item()

            frame_results.append({
                "frame": fi,
                "time_sec": round(fi / fps, 2),
                "prob_fake": round(prob_fake * 100, 2),
            })

    finally:
        cap.release()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not frame_results:
        raise HTTPException(422, "No faces detected in video segment")

    avg_fake = sum(r['prob_fake'] for r in frame_results) / len(frame_results)
    overall  = "FAKE" if avg_fake > 50 else "REAL"

    return JSONResponse({
        "label":         overall,
        "confidence":    round(avg_fake if overall == "FAKE" else 100 - avg_fake, 2),
        "frame_scores":  frame_results,
    })


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)