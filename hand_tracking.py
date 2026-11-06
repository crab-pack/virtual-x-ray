"""MediaPipe Hand Landmarker로 손을 읽고, 아바타 손가락 마디의 굽힘 각도로 바꾼다.

손 하나당 21개 랜드마크:
    0 손목
    엄지 1(CMC) 2(MCP) 3(IP) 4(끝)
    검지 5 6 7 8 / 중지 9 10 11 12 / 약지 13 14 15 16 / 새끼 17 18 19 20

아바타 쪽 손가락 본은 손가락마다 3마디(…1, …2, …3)이고, 각 마디의 굽힘은
'앞 마디 방향'과 '이 마디 방향' 사이 각도로 정한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("virtual_xray.hands")

HAND_MODEL = (
    "hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task",
)

FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
# 손가락별 랜드마크 체인: [마디1 시작 전 기준점, 마디1, 마디2, 마디3, 끝]
FINGER_CHAINS = {
    "Thumb": (0, 1, 2, 3, 4),
    "Index": (0, 5, 6, 7, 8),
    "Middle": (0, 9, 10, 11, 12),
    "Ring": (0, 13, 14, 15, 16),
    "Pinky": (0, 17, 18, 19, 20),
}
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)
MAX_FLEX = np.deg2rad(110.0)     # 한 마디가 굽을 수 있는 최대 각도
MATCH_DISTANCE = 0.12            # 손목 매칭 허용 거리 (정규화 화면 좌표)
FOLLOW = 0.5                     # 각도 스무딩 (프레임당 새 값 반영 비율)
HOLD_SECONDS = 0.4               # 손을 놓친 뒤 마지막 각도를 유지하는 시간
RELAX = 0.85                     # 유지 시간이 지나면 프레임마다 각도에 곱해 펴지게 함
BASELINE_DRIFT = np.deg2rad(2.0) # 편 상태 기준 각도가 초당 올라가는 양 (오래된 최소값을 잊게 함)
# 편 상태 기준 각도의 상한(마디별). 주먹을 쥔 채 시작해도 주먹이 '편 상태'가 되지 않게.
# 엄지 첫 마디는 손목-엄지 뿌리 선과 원래 크게 꺾여 있어서 상한을 높게 둔다.
BASELINE_MAX = {f: np.deg2rad([25.0, 25.0, 25.0]) for f in FINGERS}
BASELINE_MAX["Thumb"] = np.deg2rad([60.0, 35.0, 25.0])


@dataclass
class HandObservation:
    image: np.ndarray   # (21, 2) 정규화 화면 좌표
    world: np.ndarray   # (21, 3) 미터 단위 손 좌표
    score: float


class HandTracker:
    def __init__(self, model_path, num_hands: int = 2) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, image, stamp_ms: int) -> list[HandObservation]:
        result = self.landmarker.detect_for_video(image, stamp_ms)
        hands = []
        worlds = result.hand_world_landmarks or []
        scores = result.handedness or []
        for i, landmarks in enumerate(result.hand_landmarks or []):
            if i >= len(worlds):
                continue
            img = np.array([(lm.x, lm.y) for lm in landmarks[:21]], dtype=np.float32)
            world = np.array([(lm.x, lm.y, lm.z) for lm in worlds[i][:21]], dtype=np.float32)
            score = float(scores[i][0].score) if i < len(scores) and scores[i] else 1.0
            hands.append(HandObservation(img, world, score))
        return hands

    def close(self) -> None:
        self.landmarker.close()


def flex_angles(world: np.ndarray) -> dict[str, np.ndarray]:
    """손가락별 세 마디의 굽힘 각도(라디안, 0 = 일직선).

    네 손가락은 손바닥 가로 방향(검지 뿌리-새끼 뿌리) 성분을 빼고 잰다.
    손가락을 옆으로 벌리는 동작이 굽힘으로 잡히지 않게 하기 위해서다.
    """
    across = world[5].astype(np.float64) - world[17].astype(np.float64)
    across /= max(float(np.linalg.norm(across)), 1e-6)
    out = {}
    for finger, chain in FINGER_CHAINS.items():
        pts = world[list(chain)].astype(np.float64)
        seg = np.diff(pts, axis=0)
        if finger != "Thumb":
            seg = seg - np.outer(seg @ across, across)
        seg /= np.maximum(np.linalg.norm(seg, axis=1, keepdims=True), 1e-6)
        cos = np.clip(np.sum(seg[:-1] * seg[1:], axis=1), -1.0, 1.0)
        out[finger] = np.arccos(cos)
    return out


def match_hands(hands: list[HandObservation], pose: np.ndarray | None) -> dict[str, HandObservation]:
    """인식된 손을 자세 인식의 손목(15=Left, 16=Right)과 화면 거리로 짝지은다.

    아바타의 LeftHand 본은 자세 랜드마크 15번을 따르므로, 같은 기준으로 맞춘다.
    """
    if pose is None or not hands:
        return {}
    pairs = []
    for side, wrist in (("Left", 15), ("Right", 16)):
        target = pose[wrist, :2]
        for i, hand in enumerate(hands):
            dist = float(np.linalg.norm(hand.image[0] - target))
            pairs.append((dist, side, i))
    pairs.sort()
    out: dict[str, HandObservation] = {}
    used: set[int] = set()
    for dist, side, i in pairs:
        if dist > MATCH_DISTANCE or side in out or i in used:
            continue
        out[side] = hands[i]
        used.add(i)
    return out


class FingerState:
    """손마다 손가락 굽힘 각도를 부드럽게 유지하고, '편 상태' 기준값을 자동으로 맞춘다."""

    def __init__(self) -> None:
        self.angles = {side: {f: np.zeros(3) for f in FINGERS} for side in ("Left", "Right")}
        self.baseline = {side: {f: np.full(3, np.inf) for f in FINGERS} for side in ("Left", "Right")}
        self.seen_at = {"Left": -1e9, "Right": -1e9}
        self.last_update = None
        # 손 방향(손목→중지 뿌리)과 가로 방향(새끼 뿌리→검지 뿌리), 아바타 좌표
        self.frames: dict[str, tuple[np.ndarray, np.ndarray] | None] = {"Left": None, "Right": None}

    def update(self, matched: dict[str, HandObservation], now: float) -> None:
        dt = 0.0 if self.last_update is None else max(0.0, now - self.last_update)
        self.last_update = now
        for side in ("Left", "Right"):
            hand = matched.get(side)
            if hand is None:
                if now - self.seen_at[side] > HOLD_SECONDS:
                    for f in FINGERS:
                        self.angles[side][f] *= RELAX
                continue
            self.seen_at[side] = now
            self._update_frame(side, hand.world)
            raw = flex_angles(hand.world)
            for f in FINGERS:
                base = self.baseline[side][f]
                base = np.minimum(np.where(np.isfinite(base), base + BASELINE_DRIFT * dt, raw[f]), raw[f])
                base = np.minimum(base, BASELINE_MAX[f])
                self.baseline[side][f] = base
                target = np.clip(raw[f] - base, 0.0, MAX_FLEX)
                self.angles[side][f] = self.angles[side][f] * (1.0 - FOLLOW) + target * FOLLOW

    def _update_frame(self, side: str, world: np.ndarray) -> None:
        # MediaPipe 좌표(x 오른쪽, y 아래, z 카메라 반대쪽)를 아바타 좌표(x 오른쪽, y 위, z 카메라 쪽)로
        flip = np.array([1.0, -1.0, -1.0])
        direction = (world[9] - world[0]).astype(np.float64) * flip
        across = (world[5] - world[17]).astype(np.float64) * flip
        if np.linalg.norm(direction) < 1e-6 or np.linalg.norm(across) < 1e-6:
            return
        direction /= np.linalg.norm(direction)
        across /= np.linalg.norm(across)
        prev = self.frames[side]
        if prev is not None:
            direction = prev[0] * (1.0 - FOLLOW) + direction * FOLLOW
            across = prev[1] * (1.0 - FOLLOW) + across * FOLLOW
            direction /= max(np.linalg.norm(direction), 1e-6)
            across /= max(np.linalg.norm(across), 1e-6)
        self.frames[side] = (direction, across)

    def hand_frames(self, now: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """최근에 본 손의 방향만 돌려준다. 오래 놓친 손은 자세 인식 기준(기존 방식)으로 돌아간다."""
        return {
            side: frame
            for side, frame in self.frames.items()
            if frame is not None and now - self.seen_at[side] <= HOLD_SECONDS
        }

    def curls(self) -> dict[str, np.ndarray]:
        """아바타 본 이름 접두어("LeftHandIndex") -> 세 마디 각도."""
        return {f"{side}Hand{f}": self.angles[side][f].copy() for side in ("Left", "Right") for f in FINGERS}

    def summary(self) -> str:
        parts = []
        for side in ("Left", "Right"):
            cells = " ".join(f"{f[:2]}{np.rad2deg(self.angles[side][f]).round().astype(int).tolist()}" for f in FINGERS)
            parts.append(f"{side}: {cells}")
        return " | ".join(parts)


def draw_hands(frame: np.ndarray, hands: list[HandObservation], matched: dict[str, HandObservation]) -> None:
    """카메라 창에 손 랜드마크를 그린다. 아바타 손과 짝지어진 손은 밝게, 아닌 손은 흐리게."""
    import cv2

    height, width = frame.shape[:2]
    matched_ids = {id(h) for h in matched.values()}
    for hand in hands:
        bright = id(hand) in matched_ids
        line = (120, 230, 255) if bright else (90, 110, 120)
        pts = [(int(x * width), int(y * height)) for x, y in hand.image]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], line, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, p, 3, (255, 255, 255), -1, cv2.LINE_AA)
