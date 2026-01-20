import cv2
import mediapipe as mp
import numpy as np
import torch
import onnxruntime as ort
import albumentations as A
from albumentations.pytorch import ToTensorV2
from mediapipe.framework.formats import landmark_pb2
from collections import deque
import time

# LOAD MODEL & METADATA
ONNX_MODEL_PATH = "asl_model.onnx"
METADATA_PATH = "dataset_metadata.pt"

print("Loading ONNX model...")
ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
metadata = torch.load(METADATA_PATH)
idx_to_class = metadata["idx_to_class"]
print(f"✓ Loaded {len(idx_to_class)} classes")

# MEDIAPIPE SETUP
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# HAND SMOOTHING
MOVEMENT_THRESHOLD = 0.007
SMOOTHING_ALPHA = 0.1
VELOCITY_SMOOTHING = 0.4
KEY_POINTS = [0, 4, 8, 12, 16, 20]

hand_state = {"prev": None, "smoothed": None, "velocity": None}

def landmarks_to_array(hand_landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

def average_movement(curr, prev, idxs):
    return np.mean([np.linalg.norm(curr[i] - prev[i]) for i in idxs])

def one_euro_filter(curr, prev, velocity, alpha_pos, alpha_vel):
    new_velocity = alpha_vel * velocity + (1 - alpha_vel) * (curr - prev)
    smoothed = alpha_pos * prev + (1 - alpha_pos) * (curr + new_velocity)
    return smoothed, new_velocity

def smooth_hand_landmarks(hand_landmarks):
    curr_landmarks = landmarks_to_array(hand_landmarks)
    
    if hand_state["prev"] is None:
        hand_state["prev"] = curr_landmarks
        hand_state["smoothed"] = curr_landmarks
        hand_state["velocity"] = np.zeros_like(curr_landmarks)
        return hand_landmarks
    
    movement = average_movement(curr_landmarks, hand_state["prev"], KEY_POINTS)
    
    if movement >= MOVEMENT_THRESHOLD:
        hand_state["smoothed"], hand_state["velocity"] = one_euro_filter(
            curr_landmarks, hand_state["smoothed"], hand_state["velocity"],
            SMOOTHING_ALPHA, VELOCITY_SMOOTHING
        )
        hand_state["prev"] = hand_state["smoothed"]
    
    smoothed_hand = landmark_pb2.NormalizedLandmarkList()
    for point in hand_state["smoothed"]:
        lm = smoothed_hand.landmark.add()
        lm.x, lm.y, lm.z = float(point[0]), float(point[1]), float(point[2])
    
    return smoothed_hand

def reset_hand_state():
    hand_state["prev"] = None
    hand_state["smoothed"] = None
    hand_state["velocity"] = None

# PREPROCESSING
def get_inference_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])

transform = get_inference_transform()

def create_landmark_image(hand_landmarks, img_size=400):
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    mp_draw.draw_landmarks(
        canvas, hand_landmarks, mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

def predict_sign(landmark_img):
    transformed = transform(image=landmark_img)
    img_tensor = transformed["image"].unsqueeze(0).numpy().astype(np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: img_tensor}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    logits = ort_outputs[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    return idx_to_class[pred_idx], confidence

# IMPROVED PREDICTION SMOOTHER
class PredictionSmoother:
    def __init__(self, window_size=15, min_confidence=0.7, stability_threshold=0.7):
        self.buffer = deque(maxlen=window_size)
        self.min_confidence = min_confidence
        self.stability_threshold = stability_threshold  # 70% of frames must agree
        
    def update(self, prediction, confidence):
        if confidence >= self.min_confidence:
            self.buffer.append(prediction)
    
    def get_stable(self):
        if len(self.buffer) < 10:  # Need at least 10 frames
            return None
        
        counts = {}
        for p in self.buffer:
            counts[p] = counts.get(p, 0) + 1
        
        best = max(counts, key=counts.get)
        # Require 70% agreement for stability
        if counts[best] >= len(self.buffer) * self.stability_threshold:
            return best
        return None
    
    def reset(self):
        self.buffer.clear()

smoother = PredictionSmoother(window_size=15, min_confidence=0.7, stability_threshold=0.7)

# LETTER COMMITMENT SYSTEM
class LetterCommitter:
    def __init__(self, hold_time=1.0, different_letter_cooldown=1.5):
        self.hold_time = hold_time  # How long to hold same sign
        self.different_letter_cooldown = different_letter_cooldown
        
        self.current_sign = None
        self.sign_start_time = None
        self.last_commit_time = 0
        self.last_committed = None
        
    def update(self, prediction, confidence, threshold=0.85):
        """Returns letter to commit, or None"""
        if prediction is None or confidence < threshold:
            self.current_sign = None
            self.sign_start_time = None
            return None
        
        now = time.time()
        
        # New sign detected
        if prediction != self.current_sign:
            self.current_sign = prediction
            self.sign_start_time = now
            return None
        
        # Same sign held
        if self.sign_start_time is None:
            self.sign_start_time = now
            return None
        
        hold_duration = now - self.sign_start_time
        
        # Check if we can commit
        if hold_duration >= self.hold_time:
            # Check cooldown
            time_since_last = now - self.last_commit_time
            
            # Same letter: need full cooldown
            if prediction == self.last_committed:
                if time_since_last >= self.hold_time:
                    self.last_commit_time = now
                    self.last_committed = prediction
                    self.sign_start_time = now  # Reset to avoid repeated commits
                    return prediction
            
            # Different letter: shorter cooldown
            else:
                if time_since_last >= self.different_letter_cooldown:
                    self.last_commit_time = now
                    self.last_committed = prediction
                    self.sign_start_time = now
                    return prediction
        
        return None
    
    def reset(self):
        self.current_sign = None
        self.sign_start_time = None

committer = LetterCommitter(hold_time=1.0, different_letter_cooldown=1.5)

# UI BUTTONS
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ret, frame = cap.read()
if not ret:
    print("Camera not accessible")
    exit()

frame = cv2.flip(frame, 1)
h, w = frame.shape[:2]

PANEL_WIDTH = 180
PANEL_X1 = w - PANEL_WIDTH
PANEL_X2 = w

class UIButton:
    def __init__(self, x1, y1, x2, y2, label, hold_time=1.2, cooldown=1.5):
        self.rect = (x1, y1, x2, y2)
        self.label = label
        self.hold_time = hold_time
        self.cooldown = cooldown
        self.hover_start = None
        self.last_trigger = 0
        self.active = False
    
    def inside(self, px, py):
        x1, y1, x2, y2 = self.rect
        return x1 <= px <= x2 and y1 <= py <= y2
    
    def update(self, px, py):
        now = time.time()
        if self.inside(px, py):
            if self.hover_start is None:
                self.hover_start = now
            elif (now - self.hover_start >= self.hold_time and
                  now - self.last_trigger >= self.cooldown):
                self.last_trigger = now
                self.hover_start = None
                return True
            self.active = True
        else:
            self.hover_start = None
            self.active = False
        return False
    
    def draw(self, frame):
        x1, y1, x2, y2 = self.rect
        
        # Visual feedback during hover
        if self.hover_start is not None:
            elapsed = time.time() - self.hover_start
            progress = min(1.0, elapsed / self.hold_time)
            
            # Fill button progressively
            fill_height = int((y2 - y1) * progress)
            cv2.rectangle(frame, (x1, y2 - fill_height), (x2, y2), (0, 200, 0), -1)
            color = (0, 255, 0)
        else:
            color = (200, 200, 200)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Adjust text size for different labels
        font_scale = 0.7 if len(self.label) > 5 else 1.0
        text_x = x1 + 12 if len(self.label) > 5 else x1 + 22
        
        cv2.putText(frame, self.label, (text_x, y1 + 38),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

# Create buttons
BTN_WIDTH = 160
BTN_HEIGHT = 70
BTN_GAP = 25

buttons = []
x = PANEL_X1 + (PANEL_WIDTH - BTN_WIDTH) // 2
y = h // 2 - 100

for label in ["SPACE", ".", "BACK", "CLEAR"]:
    buttons.append(UIButton(x, y, x + BTN_WIDTH, y + BTN_HEIGHT, label))
    y += BTN_HEIGHT + BTN_GAP

def is_inside_panel(px, py):
    return PANEL_X1 <= px <= PANEL_X2

def apply_button_action(label, typed_text):
    if label == "SPACE":
        if not typed_text.endswith(" "):
            typed_text += " "
    elif label == ".":
        typed_text += "."
    elif label == "BACK":
        typed_text = typed_text[:-1]
    elif label == "CLEAR":
        typed_text = ""
    return typed_text

# MAIN LOOP
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

CONFIDENCE_THRESHOLD = 0.85
fps_queue = deque(maxlen=30)
prev_time = time.time()
typed_text = ""

print("\nASL DETECTION STARTED")
print("Hold sign for 1 seconds to commit letter")
print("Press ESC to exit\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    # FPS
    now = time.time()
    fps = 1 / (now - prev_time) if now != prev_time else 0
    prev_time = now
    fps_queue.append(fps)
    avg_fps = sum(fps_queue) / len(fps_queue)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    hand_detected = False
    prediction = None
    confidence = 0.0
    landmark_img = None
    px, py = -1, -1
    finger_in_panel = False
    
    if results.multi_hand_landmarks:
        for raw_hand_landmarks in results.multi_hand_landmarks:
            hand_detected = True
            smoothed_hand_landmarks = smooth_hand_landmarks(raw_hand_landmarks)
            
            index_tip = smoothed_hand_landmarks.landmark[8]
            px = int(index_tip.x * frame.shape[1])
            py = int(index_tip.y * frame.shape[0])
            
            mp_draw.draw_landmarks(
                frame, smoothed_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            landmark_img = create_landmark_image(smoothed_hand_landmarks)
            finger_in_panel = is_inside_panel(px, py)
            
            if not finger_in_panel:
                pred, conf = predict_sign(landmark_img)
                smoother.update(pred, conf)
                stable = smoother.get_stable()
                
                if stable:
                    prediction = stable
                    confidence = conf
                    
                    # Try to commit letter
                    letter_to_add = committer.update(prediction, confidence, CONFIDENCE_THRESHOLD)
                    if letter_to_add:
                        typed_text += letter_to_add
            else:
                smoother.reset()
                committer.reset()
    else:
        reset_hand_state()
        smoother.reset()
        committer.reset()
    
    # UI - Right panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (PANEL_X1, 0), (PANEL_X2, h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw buttons
    for btn in buttons:
        btn.draw(frame)
    
    # Button interaction
    for btn in buttons:
        if btn.update(px, py):
            typed_text = apply_button_action(btn.label, typed_text)
            smoother.reset()
            committer.reset()
    
    # FPS display
    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (w - 150, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Typed text display
    text_overlay = frame.copy()
    cv2.rectangle(text_overlay, (0, h - 150), (w - 180, h), (40, 40, 40), -1)
    cv2.addWeighted(text_overlay, 0.6, frame, 0.4, 0, frame)
    
    cv2.putText(frame, typed_text[-50:], (30, h - 100),
               cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    
    # Prediction display
    if prediction and confidence >= CONFIDENCE_THRESHOLD and not finger_in_panel:
        pred_overlay = frame.copy()
        cv2.rectangle(pred_overlay, (0, 90), (w // 2 - 90, 200), (40, 40, 40), -1)
        cv2.addWeighted(pred_overlay, 0.6, frame, 0.4, 0, frame)
        
        color = (0, 255, 0) if confidence >= 0.9 else (0, 255, 255)
        cv2.putText(frame, f"Sign: {prediction}", (20, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f"Conf: {confidence*100:.1f}%", (20, 175),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Show hold progress
        if committer.sign_start_time:
            hold_duration = time.time() - committer.sign_start_time
            progress = min(100, int((hold_duration / committer.hold_time) * 100))
            cv2.putText(frame, f"Hold: {progress}%", (20, 220),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Landmark preview
    if landmark_img is not None:
        preview = cv2.resize(landmark_img, (160, 160))
        preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
        x, y = w - 180, 50
        frame[y:y+160, x:x+160] = preview
        cv2.rectangle(frame, (x, y), (x+160, y+160), (0, 255, 255), 2)
    
    cv2.imshow("ASL Real-Time Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Session ended.")