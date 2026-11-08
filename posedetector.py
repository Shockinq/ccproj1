"""Webcam pose -> meme. Press q to quit.

Both arms up + fists   -> hopejection.jpg
Both hands in an L    -> viberanium.jpg
One fist raised        -> ihatefunsign.jpg
Anything else          -> larpcopeamine.jpg
"""
import os
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = {n: cv2.imread(os.path.join(HERE, n + ".jpg"))
          for n in ("hopejection", "viberanium", "ihatefunsign", "larpcopeamine")}
L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 11, 12, 15, 16

pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=os.path.join(HERE, "pose_landmarker_lite.task"))))
hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=os.path.join(HERE, "hand_landmarker.task")),
    num_hands=2))


def xy(lm):
    return np.array([lm.x, lm.y])


def d(hand, a, b):
    return np.linalg.norm(xy(hand[a]) - xy(hand[b]))


def curled(hand, tip):  # fingertip closer to the wrist than its middle joint
    return d(hand, tip, 0) < d(hand, tip - 2, 0)


def is_fist(hand):
    return sum(curled(hand, t) for t in (8, 12, 16, 20)) >= 3


def is_L(hand):
    # index out, other three curled, thumb stuck out sideways
    return (not curled(hand, 8) and all(curled(hand, t) for t in (12, 16, 20))
            and d(hand, 4, 5) > 0.7 * d(hand, 0, 5))


def classify(frame_rgb):
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    p = pose.detect(img).pose_landmarks
    if not p:
        return "larpcopeamine"
    p = p[0]
    up = {side: p[w].y < p[s].y for side, w, s in (("L", L_WRIST, L_SHOULDER), ("R", R_WRIST, R_SHOULDER))}
    wrists = {"L": xy(p[L_WRIST]), "R": xy(p[R_WRIST])}
    fists = {"L": False, "R": False}
    ls = 0
    for h in hands.detect(img).hand_landmarks:
        # attach each detected hand to the nearest pose wrist
        side = min(wrists, key=lambda k: np.linalg.norm(wrists[k] - xy(h[0])))
        fists[side] = is_fist(h)
        ls += is_L(h)

    if ls == 2:
        return "viberanium"
    if up["L"] and up["R"] and fists["L"] and fists["R"]:
        return "hopejection"
    for s in "LR":
        if up[s] and fists[s]:
            return "ihatefunsign"
    return "larpcopeamine"


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam (check camera permission for your terminal).")
recent = deque(maxlen=5)  # majority vote so the image doesn't flicker
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    recent.append(classify(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    label = Counter(recent).most_common(1)[0][0]
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("webcam", frame)
    cv2.imshow("output", IMAGES[label])
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
