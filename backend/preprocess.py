import cv2, os
import numpy as np

# Load Haar cascade once
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    # Fallback or alternative path if needed, but usually cv2.data.haarcascades works
    pass

face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    print(f"WARNING: Could not load face cascade from {cascade_path}")

def extract_face(bgr_image, margin_pct=0.2):
    """
    Detect the largest face using OpenCV Haar Cascade,
    return cropped BGR face and bounding box [x, y, w, h].
    """

    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) == 0:
        return None, None

    # Pick largest face (same logic as your MTCNN version)
    areas = [w * h for (x, y, w, h) in faces]
    x, y, w, h = faces[int(np.argmax(areas))]

    # Add margin (same behavior as before)
    mw = int(w * margin_pct)
    mh = int(h * margin_pct)

    x1 = max(0, x - mw)
    y1 = max(0, y - mh)
    x2 = min(bgr_image.shape[1], x + w + mw)
    y2 = min(bgr_image.shape[0], y + h + mh)

    face_crop = bgr_image[y1:y2, x1:x2]
    bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

    return face_crop, bbox