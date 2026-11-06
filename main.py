"""웹캠으로 몸을 읽고, 두 번째 창의 뼈 아바타가 같은 움직임으로 따라갑니다.

실행:
    run.bat
    .venv\\Scripts\\python main.py
    .venv\\Scripts\\python main.py --camera 1 --model full
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("virtual_xray")

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
SHOT_DIR = ROOT / "screenshots"
WINDOW = "Skeleton Cam"
AVATAR_WINDOW = "Bone Avatar"
AVATAR_W = 760
AVATAR_H = 1100
CAM_DIST = 3.2
LOOK_Y = 0.06
FOCAL = (AVATAR_H * 0.5) / np.tan(np.deg2rad(36) * 0.5)
DEFAULT_YAW = 0.0
DEFAULT_PITCH = 0.08

MODELS = {
    "lite": (
        "pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    ),
    "full": (
        "pose_landmarker_full.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/1/pose_landmarker_full.task",
    ),
    "heavy": (
        "pose_landmarker_heavy.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
    ),
}

# MediaPipe Pose 33점. 얼굴 메시는 빼고 몸통·팔·다리·손·발을 기본으로 그린다.
BODY_CONNECTIONS = (
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
)
FACE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
)
LEFT_INDEX = {1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
RIGHT_INDEX = {4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}

# BGR
COLOR_LEFT = (255, 196, 64)
COLOR_RIGHT = (64, 164, 255)
COLOR_CENTER = (235, 245, 255)
COLOR_JOINT = (255, 255, 255)
VIS_MIN = 0.45
INFER_WIDTH = 640


def download_model(name: str) -> Path:
    filename, url = MODELS[name]
    return download_file(filename, url, f"포즈 모델({name})")


def download_file(filename: str, url: str, label: str) -> Path:
    dest = MODEL_DIR / filename
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    log.info("%s을 받는 중입니다.", label)
    next_step = [25]

    def hook(count: int, block: int, total: int) -> None:
        if total <= 0:
            return
        pct = min(100, int(count * block * 100 / total))
        # 로그는 줄을 덮어쓸 수 없으므로 25% 단위로 한 줄씩 남긴다.
        while next_step[0] <= 100 and pct >= next_step[0]:
            log.info("  다운로드 %d%%", next_step[0])
            next_step[0] += 25

    try:
        urllib.request.urlretrieve(url, tmp, hook)
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        log.error("모델 다운로드 실패: %s\n주소: %s", exc, url)
        raise SystemExit(1) from exc
    tmp.replace(dest)
    return dest


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\malgunbd.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


class Camera:
    def __init__(self, index: int, width: int, height: int) -> None:
        self.cap = self._open(index, width, height)
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.stopped = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    @staticmethod
    def _open(index: int, width: int, height: int) -> cv2.VideoCapture:
        errors = []
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                errors.append(f"backend {backend} open 실패")
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
            errors.append(f"backend {backend} 프레임 없음")
            cap.release()
        log.error(
            "카메라 %d번을 열지 못했습니다. 다른 앱이 카메라를 쓰는 중인지 확인하거나 "
            "--camera 번호를 바꿔 보세요.\n%s",
            index,
            "\n".join(errors),
        )
        raise SystemExit(1)

    def _loop(self) -> None:
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame

    def read(self) -> np.ndarray | None:
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def close(self) -> None:
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()


class PoseSmoother:
    """관절 떨림을 줄이되, 현재 프레임을 더 크게 반영해 따라붙는 느낌을 유지한다."""

    def __init__(self, alpha: float = 0.72) -> None:
        self.alpha = alpha
        self.tracks: list[np.ndarray] = []
        self.world_tracks: list[np.ndarray] = []

    def update(
        self,
        poses: list[np.ndarray],
        worlds: list[np.ndarray] | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if worlds is not None and len(worlds) != len(poses):
            worlds = None
        if not poses:
            self.tracks = []
            self.world_tracks = []
            return [], []
        if not self.tracks:
            self.tracks = [pose.copy() for pose in poses]
            self.world_tracks = [pose.copy() for pose in worlds] if worlds is not None else []
            return [pose.copy() for pose in self.tracks], [pose.copy() for pose in self.world_tracks]

        pairs: list[tuple[float, int, int]] = []
        for i, pose in enumerate(poses):
            for j, track in enumerate(self.tracks):
                pairs.append((self._distance(pose, track), i, j))
        pairs.sort()

        assigned_pose: set[int] = set()
        assigned_track: set[int] = set()
        mapping: dict[int, int] = {}
        for dist, i, j in pairs:
            if i in assigned_pose or j in assigned_track or dist > 0.28:
                continue
            assigned_pose.add(i)
            assigned_track.add(j)
            mapping[i] = j

        smoothed: list[np.ndarray] = []
        smoothed_worlds: list[np.ndarray] = []
        for i, pose in enumerate(poses):
            if i not in mapping:
                smoothed.append(pose.copy())
                if worlds is not None:
                    smoothed_worlds.append(worlds[i].copy())
                continue
            prev = self.tracks[mapping[i]]
            mixed = prev.copy()
            visible = pose[:, 3] >= VIS_MIN
            mixed[visible] = prev[visible] * (1.0 - self.alpha) + pose[visible] * self.alpha
            mixed[:, 3] = pose[:, 3]
            smoothed.append(mixed)
            if worlds is not None:
                prev_world = (
                    self.world_tracks[mapping[i]]
                    if mapping[i] < len(self.world_tracks)
                    else worlds[i]
                )
                mixed_world = prev_world.copy()
                # 깊이 값은 더 흔들려서 화면 관절보다 조금 더 부드럽게 붙인다.
                world_alpha = min(self.alpha, 0.62)
                mixed_world[visible] = (
                    prev_world[visible] * (1.0 - world_alpha) + worlds[i][visible] * world_alpha
                )
                mixed_world[:, 3] = pose[:, 3]
                smoothed_worlds.append(mixed_world)
        self.tracks = [pose.copy() for pose in smoothed]
        self.world_tracks = [pose.copy() for pose in smoothed_worlds]
        return smoothed, smoothed_worlds

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        ca = (a[23, :2] + a[24, :2]) * 0.5
        cb = (b[23, :2] + b[24, :2]) * 0.5
        if a[23, 3] < VIS_MIN or a[24, 3] < VIS_MIN:
            ca = a[0, :2]
        if b[23, 3] < VIS_MIN or b[24, 3] < VIS_MIN:
            cb = b[0, :2]
        return float(np.linalg.norm(ca - cb))


def create_landmarker(model_path: Path, num_poses: int):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options), mp


def _visibility(lm, fallback: float = 0.0) -> float:
    vis = getattr(lm, "visibility", None)
    if vis is None:
        vis = getattr(lm, "presence", fallback) or fallback
    return float(vis)


def poses_from_result(result) -> tuple[list[np.ndarray], list[np.ndarray]]:
    image_poses = []
    world_poses = []
    world_groups = result.pose_world_landmarks or []
    for index, landmarks in enumerate(result.pose_landmarks or []):
        arr = np.zeros((33, 4), dtype=np.float32)
        for i, lm in enumerate(landmarks[:33]):
            arr[i] = (lm.x, lm.y, lm.z, _visibility(lm))
        image_poses.append(arr)

        world = np.zeros((33, 4), dtype=np.float32)
        group = world_groups[index] if index < len(world_groups) else []
        for i, lm in enumerate(list(group)[:33]):
            world[i] = (lm.x, lm.y, lm.z, arr[i, 3])
        world_poses.append(world)
    return image_poses, world_poses


def infer_image(frame: np.ndarray, mp) -> object:
    height, width = frame.shape[:2]
    if width > INFER_WIDTH:
        scale = INFER_WIDTH / width
        small = cv2.resize(
            frame,
            (INFER_WIDTH, max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = frame
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def bone_color(a: int, b: int) -> tuple[int, int, int]:
    a_left, b_left = a in LEFT_INDEX, b in LEFT_INDEX
    a_right, b_right = a in RIGHT_INDEX, b in RIGHT_INDEX
    if a_left and b_left:
        return COLOR_LEFT
    if a_right and b_right:
        return COLOR_RIGHT
    return COLOR_CENTER


def to_xray(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.15, beta=8)
    inv = 255 - gray
    blue = inv
    green = np.clip(inv.astype(np.int16) * 9 // 10, 0, 255).astype(np.uint8)
    red = np.clip(inv.astype(np.int16) * 7 // 10, 0, 255).astype(np.uint8)
    return cv2.merge((blue, green, red))


def _px(point: np.ndarray, width: int, height: int) -> tuple[int, int]:
    x = int(np.clip(point[0] * width, -20, width + 20))
    y = int(np.clip(point[1] * height, -20, height + 20))
    return x, y


def _visible(pose: np.ndarray, index: int) -> bool:
    return float(pose[index, 3]) >= VIS_MIN


def seen_landmarks(pose: np.ndarray | None) -> np.ndarray | None:
    """웹캠 화면 안에 실제로 잡힌 랜드마크만 True."""
    if pose is None:
        return None
    x, y, vis = pose[:, 0], pose[:, 1], pose[:, 3]
    return (vis >= VIS_MIN) & (x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0)


def draw_connections(
    canvas: np.ndarray,
    pose: np.ndarray,
    connections: tuple[tuple[int, int], ...],
    thickness: int,
) -> None:
    height, width = canvas.shape[:2]
    for a, b in connections:
        if not (_visible(pose, a) and _visible(pose, b)):
            continue
        cv2.line(
            canvas,
            _px(pose[a], width, height),
            _px(pose[b], width, height),
            bone_color(a, b),
            thickness,
            cv2.LINE_AA,
        )


def draw_head(canvas: np.ndarray, pose: np.ndarray, thickness: int) -> None:
    height, width = canvas.shape[:2]
    if _visible(pose, 7) and _visible(pose, 8):
        left_ear = pose[7, :2]
        right_ear = pose[8, :2]
        center = (left_ear + right_ear) * 0.5
        radius = float(np.linalg.norm((left_ear - right_ear) * (width, height))) * 0.62
        center = center.copy()
        center[1] -= radius / height * 0.12
    elif _visible(pose, 0):
        center = pose[0, :2].copy()
        shoulder = 0.0
        if _visible(pose, 11) and _visible(pose, 12):
            shoulder = float(np.linalg.norm((pose[11, :2] - pose[12, :2]) * (width, height)))
        radius = max(18.0, shoulder * 0.38)
        center[1] -= radius / height * 0.55
    else:
        return
    if radius < 8:
        return
    origin = _px(np.array([center[0], center[1], 0, 1], dtype=np.float32), width, height)
    cv2.circle(canvas, origin, int(radius), COLOR_CENTER, thickness, cv2.LINE_AA)

    if _visible(pose, 11) and _visible(pose, 12):
        neck = (pose[11, :2] + pose[12, :2]) * 0.5
        cv2.line(canvas, origin, _px(neck, width, height), COLOR_CENTER, thickness, cv2.LINE_AA)


def draw_joints(canvas: np.ndarray, pose: np.ndarray, radius: int, detailed_face: bool) -> None:
    height, width = canvas.shape[:2]
    skip = set() if detailed_face else set(range(11))
    for index in range(33):
        if index in skip or not _visible(pose, index):
            continue
        point = _px(pose[index], width, height)
        color = bone_color(index, index)
        cv2.circle(canvas, point, radius + 1, (12, 18, 24), -1, cv2.LINE_AA)
        cv2.circle(canvas, point, radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, point, max(1, radius // 3), COLOR_JOINT, -1, cv2.LINE_AA)


def render_line_skeleton(frame: np.ndarray, poses: list[np.ndarray]) -> np.ndarray:
    height = frame.shape[0]
    thick = max(2, int(round(2 * height / 720)))
    joint = max(2, thick)
    for pose in poses:
        draw_connections(frame, pose, BODY_CONNECTIONS, thick)
        draw_head(frame, pose, thick)
        draw_joints(frame, pose, joint, False)
    return frame


def render_poses(
    frame: np.ndarray,
    poses: list[np.ndarray],
    detailed_face: bool,
    glow: bool,
    xray: bool,
) -> np.ndarray:
    canvas = to_xray(frame) if xray else frame
    if not poses:
        return canvas

    height = canvas.shape[0]
    scale = max(0.7, height / 720)
    bone = max(2, int(3 * scale))
    joint = max(3, int(5 * scale))
    connections = BODY_CONNECTIONS + (FACE_CONNECTIONS if detailed_face else ())

    if glow:
        bloom = np.zeros_like(canvas)
        for pose in poses:
            draw_connections(bloom, pose, connections, bone + 7)
            if not detailed_face:
                draw_head(bloom, pose, bone + 5)
            draw_joints(bloom, pose, joint + 2, detailed_face)
        bloom = cv2.GaussianBlur(bloom, (0, 0), 5)
        canvas = cv2.addWeighted(canvas, 1.0, bloom, 0.75, 0)

    for pose in poses:
        draw_connections(canvas, pose, connections, bone)
        if not detailed_face:
            draw_head(canvas, pose, bone)
        draw_joints(canvas, pose, joint, detailed_face)
    return canvas


# (관절 a, 관절 b, 뼈 두께 미터). 화면 위 골격과 같은 연결을 아바타 부피로 그린다.
AVATAR_BONES = (
    (11, 12, 0.046),
    (23, 24, 0.050),
    (11, 23, 0.036),
    (12, 24, 0.036),
    (11, 13, 0.032),
    (13, 15, 0.024),
    (12, 14, 0.032),
    (14, 16, 0.024),
    (15, 17, 0.012),
    (15, 19, 0.012),
    (15, 21, 0.012),
    (17, 19, 0.010),
    (16, 18, 0.012),
    (16, 20, 0.012),
    (16, 22, 0.012),
    (18, 20, 0.010),
    (23, 25, 0.044),
    (25, 27, 0.032),
    (27, 29, 0.020),
    (27, 31, 0.018),
    (29, 31, 0.016),
    (24, 26, 0.044),
    (26, 28, 0.032),
    (28, 30, 0.020),
    (28, 32, 0.018),
    (30, 32, 0.016),
)
AVATAR_JOINTS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
FINGER_INDEX = {17, 18, 19, 20, 21, 22}


def rest_avatar() -> tuple[np.ndarray, np.ndarray]:
    points = np.zeros((33, 3), dtype=np.float32)
    points[0] = (0.00, 0.62, 0.07)
    points[7] = (-0.07, 0.60, 0.00)
    points[8] = (0.07, 0.60, 0.00)
    points[11] = (-0.19, 0.42, 0.00)
    points[12] = (0.19, 0.42, 0.00)
    points[13] = (-0.22, 0.18, 0.03)
    points[14] = (0.22, 0.18, 0.03)
    points[15] = (-0.24, -0.04, 0.05)
    points[16] = (0.24, -0.04, 0.05)
    points[17] = (-0.27, -0.08, 0.03)
    points[19] = (-0.23, -0.10, 0.08)
    points[21] = (-0.20, -0.06, 0.08)
    points[18] = (0.27, -0.08, 0.03)
    points[20] = (0.23, -0.10, 0.08)
    points[22] = (0.20, -0.06, 0.08)
    points[23] = (-0.10, 0.00, 0.00)
    points[24] = (0.10, 0.00, 0.00)
    points[25] = (-0.11, -0.44, 0.03)
    points[26] = (0.11, -0.44, 0.03)
    points[27] = (-0.11, -0.88, 0.00)
    points[28] = (0.11, -0.88, 0.00)
    points[29] = (-0.13, -0.92, -0.05)
    points[30] = (0.13, -0.92, -0.05)
    points[31] = (-0.10, -0.93, 0.09)
    points[32] = (0.10, -0.93, 0.09)
    return points, np.ones(33, dtype=np.float32)


REST_POINTS, REST_VIS = rest_avatar()


# 뼈 메시의 팔 길이 비율 (어깨 관절 사이 거리 0.304m 기준: 위팔 0.286m, 아래팔 0.276~0.282m)
UPPER_ARM_RATIO = 0.94
FOREARM_RATIO = 0.92
# 어깨 기준 손목·팔꿈치 깊이(앞뒤) 허용 범위 (뒤쪽, 앞쪽). MediaPipe 깊이는 크기가 부정확해서 자른다.
ARM_DEPTH_RANGE = (0.12, 0.35)


def _arms_from_image(view: np.ndarray, pose: np.ndarray, aspect: float) -> np.ndarray:
    """팔의 화면상 위치는 웹캠 골격을 따른다.

    3D 월드 좌표의 팔은 화면에서는 정상인데도 골반 안으로 접힐 때가 있다.
    """
    rel = pose[:, :2].astype(np.float32).copy()
    hip = (rel[23] + rel[24]) * 0.5
    rel -= hip
    rel[:, 0] *= np.float32(aspect)
    rel[:, 1] *= np.float32(-1.0)
    img_span = float(np.linalg.norm(rel[11] - rel[12]))
    world_span = float(np.linalg.norm(view[11, :2] - view[12, :2]))
    if img_span < 1e-4 or world_span < 1e-4:
        return view
    rel *= np.float32(world_span / img_span)
    out = view.copy()
    for index in (11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22):
        if float(pose[index, 3]) < 0.2:
            continue
        out[index, 0] = rel[index, 0]
        out[index, 1] = rel[index, 1]
    for shoulder, elbow, wrist, hand in ((11, 13, 15, (17, 19, 21)), (12, 14, 16, (18, 20, 22))):
        base = float(np.clip(out[shoulder, 2], -0.2, 0.2))
        out[shoulder, 2] = base
        if min(float(pose[i, 3]) for i in (shoulder, elbow, wrist)) < 0.2:
            for joint in (elbow, wrist) + hand:
                depth = float(view[joint, 2] - view[shoulder, 2])
                out[joint, 2] = base + float(np.clip(depth, -0.12, 0.42))
            continue
        # 손목 위치를 목표로 두고, 아바타 자신의 팔 길이로 팔꿈치 위치를 푼다(2본 IK).
        # 사람과 아바타의 팔 비율이 달라도 손이 화면 속 손목 자리(예: 얼굴 앞)에 닿는다.
        def depth_of(joint: int) -> float:
            return base + float(np.clip(view[joint, 2] - view[shoulder, 2], -ARM_DEPTH_RANGE[0], ARM_DEPTH_RANGE[1]))

        old_wrist = out[wrist].copy()
        target = np.array([out[wrist, 0], out[wrist, 1], depth_of(wrist)], dtype=np.float32)
        pole = np.array([out[elbow, 0], out[elbow, 1], depth_of(elbow)], dtype=np.float32)
        elbow_pt, wrist_pt = _two_bone_ik(
            out[shoulder], target, pole, UPPER_ARM_RATIO * world_span, FOREARM_RATIO * world_span
        )
        out[elbow] = elbow_pt
        out[wrist] = wrist_pt
        for joint in hand:  # 손 점들은 손목이 옮겨진 만큼 함께 옮긴다
            out[joint, :2] += wrist_pt[:2] - old_wrist[:2]
            out[joint, 2] = wrist_pt[2] + float(np.clip(view[joint, 2] - view[wrist, 2], -0.1, 0.1))
    return out


def _two_bone_ik(
    shoulder: np.ndarray, target: np.ndarray, pole: np.ndarray, upper: float, fore: float
) -> tuple[np.ndarray, np.ndarray]:
    """어깨에서 길이 upper, fore인 두 뼈로 target에 닿는 (팔꿈치, 손목). 팔꿈치는 pole 쪽으로 굽힌다."""
    s = shoulder.astype(np.float64)
    to = target.astype(np.float64) - s
    dist = float(np.linalg.norm(to))
    if dist < 1e-6:
        return pole.astype(np.float32), target.astype(np.float32)
    u = to / dist
    dist = float(np.clip(dist, abs(upper - fore) + 1e-4, upper + fore - 1e-4))  # 닿지 않으면 그 방향으로 최대한
    along = (upper * upper - fore * fore + dist * dist) / (2.0 * dist)
    height = float(np.sqrt(max(upper * upper - along * along, 0.0)))
    bend = pole.astype(np.float64) - s
    bend -= (bend @ u) * u
    if float(np.linalg.norm(bend)) < 1e-6:
        bend = np.array([0.0, -1.0, 0.0]) - u[1] * u  # 팔꿈치 방향을 모르면 아래쪽으로
    bend /= max(float(np.linalg.norm(bend)), 1e-6)
    elbow = s + u * along + bend * height
    wrist = s + u * dist
    return elbow.astype(np.float32), wrist.astype(np.float32)


def world_to_avatar(world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MediaPipe 월드 좌표(y가 아래, 엉덩이 중심)를 아바타 공간으로 옮긴다.

    x는 화면 오른쪽이 양수, y는 위가 양수, z는 카메라 쪽이 양수다.
    """
    view = np.empty((33, 3), dtype=np.float32)
    view[:, 0] = world[:, 0]
    view[:, 1] = -world[:, 1]
    view[:, 2] = -world[:, 2]
    hip = (view[23] + view[24]) * 0.5
    if world[23, 3] < 0.05 and world[24, 3] < 0.05:
        hip = np.zeros(3, dtype=np.float32)
    view -= hip
    return view, world[:, 3].copy()


class AvatarState:
    def __init__(self) -> None:
        self.yaw = DEFAULT_YAW
        self.pitch = DEFAULT_PITCH
        self.scale = 0.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.zoom = 1.0
        self.track_x = 0.0
        self.points: np.ndarray | None = None
        self.vis: np.ndarray | None = None
        self.seen_at = 0.0

    def reset_view(self) -> None:
        self.yaw = DEFAULT_YAW
        self.pitch = DEFAULT_PITCH
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.zoom = 1.0

    def nudge(self, dx: float, dy: float) -> None:
        self.pan_x = float(np.clip(self.pan_x + dx, -1.2, 1.2))
        self.pan_y = float(np.clip(self.pan_y + dy, -1.2, 1.2))

    def zoom_by(self, factor: float) -> None:
        self.zoom = float(np.clip(self.zoom * factor, 0.4, 2.6))

    def follow_image(self, pose: np.ndarray | None) -> None:
        """화면에서 엉덩이가 좌우로 움직인 만큼 모델을 옮긴다."""
        if pose is None or (float(pose[23, 3]) < 0.2 and float(pose[24, 3]) < 0.2):
            return
        hip_x = float((pose[23, 0] + pose[24, 0]) * 0.5)
        target = (hip_x - 0.5) * 1.35
        self.track_x = self.track_x * 0.72 + target * 0.28

    def placement(self, live: bool) -> tuple[float, float, float]:
        x = self.pan_x + (self.track_x if live else 0.0)
        return x, self.pan_y, self.zoom

    def observe(self, world: np.ndarray | None, pose: np.ndarray | None, aspect: float, now: float) -> None:
        if world is None:
            return
        view, vis = world_to_avatar(world)
        if pose is not None:
            view = _arms_from_image(view, pose, aspect)
        shoulder = float(np.linalg.norm(view[11] - view[12]))
        if 0.08 < shoulder < 0.9:
            target = 0.38 / shoulder
            self.scale = target if self.scale == 0 else self.scale * 0.9 + target * 0.1
        if self.scale:
            view = view * np.float32(self.scale)
        if self.points is not None:
            kept = self._reject_spikes(view, vis)
            if kept is None:
                self.seen_at = now
                return
            view, vis = kept
        self.points = view
        self.vis = vis
        self.seen_at = now

    def _reject_spikes(self, view: np.ndarray, vis: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """한 프레임에 몸이 순간이동하면 그 샘플은 버리고 직전 자세를 유지한다."""
        assert self.points is not None and self.vis is not None
        delta = np.linalg.norm(view - self.points, axis=1)
        major = np.array([11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28])
        major_jump = delta[major] > 0.55
        if int(np.count_nonzero(major_jump)) >= 5:
            return None
        spiked = delta > 0.72
        if not np.any(spiked):
            return view, vis
        stable = view.copy()
        stable[spiked] = self.points[spiked]
        stable_vis = vis.copy()
        stable_vis[spiked] = self.vis[spiked]
        return stable, stable_vis

    def current(self, now: float) -> tuple[np.ndarray, np.ndarray, bool, float]:
        if self.points is None or now - self.seen_at > 0.8:
            return REST_POINTS, REST_VIS, False, 1.0
        return self.points, self.vis, True, self.scale or 1.0


def _rotate_view(points: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    x2 = cy * x + sy * z
    z2 = -sy * x + cy * z
    y2 = cp * y - sp * z2
    z3 = sp * y + cp * z2
    return np.stack((x2, y2, z3), axis=1)


def _project(
    points: np.ndarray,
    yaw: float,
    pitch: float,
    offset: tuple[float, float] = (0.0, 0.0),
    zoom: float = 1.0,
) -> np.ndarray:
    rotated = _rotate_view(points * np.float32(zoom), yaw, pitch)
    rotated = rotated.copy()
    rotated[:, 0] += offset[0] * zoom
    rotated[:, 1] += offset[1] * zoom
    depth = CAM_DIST - rotated[:, 2]
    depth_safe = np.maximum(depth, 0.25)
    screen = np.empty((len(points), 3), dtype=np.float32)
    screen[:, 0] = AVATAR_W * 0.5 + FOCAL * rotated[:, 0] / depth_safe
    screen[:, 1] = AVATAR_H * 0.44 - FOCAL * (rotated[:, 1] - LOOK_Y) / depth_safe
    screen[:, 2] = depth
    return screen


def _stage_background() -> np.ndarray:
    y = np.linspace(0.0, 1.0, AVATAR_H, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, AVATAR_W, dtype=np.float32)[None, :]
    radial = np.sqrt((y - 0.42) ** 2 + ((x - 0.5) * (AVATAR_H / AVATAR_W)) ** 2)
    shade = np.clip(1.0 - radial * 0.95, 0.0, 1.0) ** 1.4
    floor = np.clip((y - 0.62) / 0.38, 0.0, 1.0) ** 1.6
    color = np.empty((AVATAR_H, AVATAR_W, 3), dtype=np.float32)
    top = np.array([22, 16, 14], dtype=np.float32)
    mid = np.array([58, 40, 30], dtype=np.float32)
    ground = np.array([34, 30, 28], dtype=np.float32)
    body = top + (mid - top) * shade[..., None]
    color[:] = body * (1.0 - floor[..., None] * 0.35) + ground * (floor[..., None] * 0.35)
    vignette = np.clip(1.0 - radial * 0.55, 0.45, 1.0)
    color *= vignette[..., None]
    return np.clip(color, 0, 255).astype(np.uint8)


STAGE = _stage_background()


def _mix(color: tuple[int, int, int], toward: int, amount: float) -> tuple[int, int, int]:
    return tuple(int(channel * (1.0 - amount) + toward * amount) for channel in color)


def _draw_capsule(
    canvas: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
) -> None:
    radius = max(1, int(radius))
    dark = _mix(color, 0, 0.62)
    light = _mix(color, 255, 0.42)
    cv2.line(canvas, start, end, dark, radius * 2 + 2, cv2.LINE_AA)
    cv2.line(canvas, start, end, color, radius * 2, cv2.LINE_AA)
    cv2.circle(canvas, start, radius, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, end, radius, color, -1, cv2.LINE_AA)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = (dx * dx + dy * dy) ** 0.5 or 1.0
    shift = max(1, radius // 3)
    ox = int(round(-dy / length * shift))
    oy = int(round(dx / length * shift))
    cv2.line(
        canvas,
        (start[0] + ox, start[1] + oy),
        (end[0] + ox, end[1] + oy),
        light,
        max(1, radius // 2),
        cv2.LINE_AA,
    )


def _draw_sphere(
    canvas: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
) -> None:
    radius = max(2, int(radius))
    dark = _mix(color, 0, 0.55)
    light = _mix(color, 255, 0.5)
    cv2.circle(canvas, center, radius + 1, dark, -1, cv2.LINE_AA)
    cv2.circle(canvas, center, radius, color, -1, cv2.LINE_AA)
    highlight = (center[0] - radius // 3, center[1] - radius // 3)
    cv2.circle(canvas, highlight, max(2, radius // 3), light, -1, cv2.LINE_AA)


def _usable(index: int, vis: np.ndarray, live: bool) -> bool:
    if not live:
        return True
    if index in FINGER_INDEX:
        return float(vis[index]) >= 0.35
    return float(vis[index]) >= 0.08


def render_avatar(
    points: np.ndarray,
    vis: np.ndarray,
    yaw: float,
    pitch: float,
    live: bool,
    scale: float,
    offset: tuple[float, float] = (0.0, 0.0),
    zoom: float = 1.0,
) -> np.ndarray:
    canvas = STAGE.copy()
    energy = 1.0 if live else 0.62
    screen = _project(points, yaw, pitch, offset, zoom)
    ground = _project(np.array([[0.0, -1.02, 0.0]], dtype=np.float32), yaw, pitch, offset, zoom)[0]
    if ground[2] > 0.3:
        gx, gy = int(ground[0]), int(ground[1])
        rx = max(12, int(FOCAL * 0.62 * scale / ground[2]))
        ry = max(6, int(rx * 0.22))
        floor = canvas.copy()
        cv2.ellipse(floor, (gx, gy), (rx, ry), 0, 0, 360, (70, 62, 54), -1, cv2.LINE_AA)
        canvas = cv2.addWeighted(canvas, 0.72, floor, 0.28, 0)

        foot_ids = [i for i in (27, 28, 29, 30, 31, 32) if _usable(i, vis, live)]
        if foot_ids:
            fx = int(np.mean(screen[foot_ids, 0]))
            fy = gy
        else:
            fx, fy = gx, gy
        shadow = np.zeros_like(canvas)
        cv2.ellipse(shadow, (fx, min(AVATAR_H - 8, fy)), (max(18, rx // 2), max(8, ry // 2)), 0, 0, 360, (255, 255, 255), -1)
        shadow = cv2.GaussianBlur(shadow, (0, 0), 14)
        canvas = np.clip(canvas.astype(np.float32) - shadow.astype(np.float32) * 0.38, 0, 255).astype(np.uint8)

    bones: list[tuple[float, int, int, float]] = []
    for a, b, radius in AVATAR_BONES:
        if not (_usable(a, vis, live) and _usable(b, vis, live)):
            continue
        if screen[a, 2] < 0.3 or screen[b, 2] < 0.3:
            continue
        bones.append((float(screen[a, 2] + screen[b, 2]) * 0.5, a, b, radius))

    if _usable(11, vis, live) and _usable(12, vis, live) and (_usable(7, vis, live) or _usable(0, vis, live)):
        # 목은 어깨 중앙과 머리 사이를 잇는다. 별도 랜드마크가 없어 그릴 때만 계산한다.
        shoulder = (points[11] + points[12]) * 0.5
        if _usable(7, vis, live) and _usable(8, vis, live):
            crown = (points[7] + points[8]) * 0.5
            crown = crown.copy()
            crown[1] += np.linalg.norm(points[7] - points[8]) * 0.12
        else:
            crown = points[0].copy()
        neck_top = shoulder * 0.28 + crown * 0.72
        extra = np.vstack((points, shoulder, neck_top))
        screen_extra = _project(extra, yaw, pitch, offset, zoom)
        screen = screen_extra[:33]
        neck_screen = screen_extra[33:]
        depth = float(neck_screen[:, 2].mean())
        if depth > 0.3:
            bones.append((depth, -1, -2, 0.028))
            neck_pair = neck_screen
        else:
            neck_pair = None
    else:
        neck_pair = None

    bones.sort(key=lambda item: item[0], reverse=True)
    glow = np.zeros_like(canvas)

    def endpoint(index: int) -> tuple[int, int]:
        if index == -1:
            return int(neck_pair[0, 0]), int(neck_pair[0, 1])
        if index == -2:
            return int(neck_pair[1, 0]), int(neck_pair[1, 1])
        return int(screen[index, 0]), int(screen[index, 1])

    for depth, a, b, radius in bones:
        width = max(2, int(FOCAL * radius * scale / max(depth, 0.3)))
        color = bone_color(11 if a < 0 else a, 12 if b < 0 else b)
        color = tuple(int(np.clip(channel * energy, 0, 255)) for channel in color)
        pa, pb = endpoint(a), endpoint(b)
        _draw_capsule(glow, pa, pb, width + 4, color)

    glow = cv2.GaussianBlur(glow, (0, 0), 7)
    canvas = cv2.addWeighted(canvas, 1.0, glow, 0.55 if live else 0.28, 0)

    for depth, a, b, radius in bones:
        width = max(2, int(FOCAL * radius * scale / max(depth, 0.3)))
        color = bone_color(11 if a < 0 else a, 12 if b < 0 else b)
        color = tuple(int(np.clip(channel * energy, 0, 255)) for channel in color)
        _draw_capsule(canvas, endpoint(a), endpoint(b), width, color)

    joints = []
    for index in AVATAR_JOINTS:
        if not _usable(index, vis, live) or screen[index, 2] < 0.3:
            continue
        joints.append((float(screen[index, 2]), index))
    joints.sort(reverse=True)
    for depth, index in joints:
        radius = max(3, int(FOCAL * 0.02 * scale / max(depth, 0.3)))
        color = tuple(int(np.clip(channel * energy, 0, 255)) for channel in bone_color(index, index))
        _draw_sphere(canvas, endpoint(index), radius, color)

    if _usable(7, vis, live) and _usable(8, vis, live):
        center3 = (points[7] + points[8]) * 0.5
        center3 = center3.copy()
        ear = float(np.linalg.norm(points[7] - points[8]))
        center3[1] += ear * 0.12
        head_r = max(0.07, ear * 0.62)
    elif _usable(0, vis, live):
        center3 = points[0].copy()
        center3[1] += 0.06
        head_r = 0.09
    else:
        center3 = None
        head_r = 0.0

    if center3 is not None:
        head_screen = _project(center3.reshape(1, 3), yaw, pitch, offset, zoom)[0]
        if head_screen[2] > 0.3:
            head_px = max(8, int(FOCAL * head_r * scale / head_screen[2]))
            color = tuple(int(np.clip(channel * energy, 0, 255)) for channel in COLOR_CENTER)
            _draw_sphere(canvas, (int(head_screen[0]), int(head_screen[1])), head_px, color)
            if _usable(0, vis, live):
                nose = endpoint(0)
                cv2.line(canvas, (int(head_screen[0]), int(head_screen[1])), nose, color, max(2, head_px // 5), cv2.LINE_AA)
                cv2.circle(canvas, nose, max(3, head_px // 7), color, -1, cv2.LINE_AA)
    return canvas


COLLIMATOR_FILE = ROOT / "collimator.json"   # 맞춘 박스 위치 (기기마다 다르므로 git에 넣지 않음)
COLLIMATOR_DEFAULT = (173, 343, 589, 759)     # 기존 collimator_light.png의 투명한 구멍과 같은 위치
COLLIMATOR_DIM = 0.1           # 박스 바깥 밝기 (기존 PNG의 90% 검정 덮개와 같음)
COLLIMATOR_LINE = (255, 255, 255)
COLLIMATOR_THICK = 2
COLLIMATOR_TAIL = 60           # 십자선이 박스 밖으로 나오는 길이(px)
COLLIMATOR_MIN = 32            # 박스 최소 한 변(px)
COLLIMATOR_HANDLE = 22         # 꼭짓점을 잡을 수 있는 거리(px)
COLLIMATOR_EDIT_COLOR = (80, 200, 255)


class Collimator:
    """아바타 화면에 고정된 콜리메이터 조명. 편집 모드에서 꼭짓점을 드래그해 영역을 맞춘다.

    박스는 아바타 화면(이미지) 픽셀 좌표로 저장한다. 창 크기나 몸 위치와 상관없이 그대로 유지된다.
    """

    def __init__(self, width: int, height: int, path: Path = COLLIMATOR_FILE) -> None:
        self.width = width
        self.height = height
        self.path = path
        self.box = list(COLLIMATOR_DEFAULT)
        self.editing = False
        self._drag: tuple[str, int, int, list[int]] | None = None  # (종류, 시작 x, 시작 y, 시작 박스)
        self._load()

    # ---- 저장
    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            box = [int(v) for v in data["box"]]
            if len(box) == 4:
                self.box = self._clamped(box)
                log.info("콜리메이터 위치를 불러왔습니다: %s", self.box)
        except FileNotFoundError:
            pass
        except Exception as exc:
            log.warning("콜리메이터 설정을 읽지 못해 기본 위치를 씁니다 (%s): %s", self.path.name, exc)

    def _save(self) -> None:
        try:
            self.path.write_text(json.dumps({"box": self.box}, indent=1), encoding="utf-8")
            log.info("콜리메이터 위치를 저장했습니다: %s", self.box)
        except OSError as exc:
            log.warning("콜리메이터 위치를 저장하지 못했습니다: %s", exc)

    def toggle_edit(self) -> None:
        """편집 모드 전환. 편집을 끝내면(고정하면) 위치를 저장한다."""
        self.editing = not self.editing
        self._drag = None
        if self.editing:
            log.info("콜리메이터 편집: 꼭짓점을 드래그해 크기를, 안쪽을 드래그해 위치를 맞춘 뒤 L로 고정하세요.")
        else:
            self._save()

    # ---- 마우스
    def _clamped(self, box: list[int]) -> list[int]:
        x1, y1, x2, y2 = box
        x1, x2 = sorted((int(np.clip(x1, 0, self.width)), int(np.clip(x2, 0, self.width))))
        y1, y2 = sorted((int(np.clip(y1, 0, self.height)), int(np.clip(y2, 0, self.height))))
        if x2 - x1 < COLLIMATOR_MIN:
            x2 = min(self.width, x1 + COLLIMATOR_MIN)
            x1 = x2 - COLLIMATOR_MIN
        if y2 - y1 < COLLIMATOR_MIN:
            y2 = min(self.height, y1 + COLLIMATOR_MIN)
            y1 = y2 - COLLIMATOR_MIN
        return [x1, y1, x2, y2]

    def _corners(self) -> list[tuple[int, int]]:
        x1, y1, x2, y2 = self.box
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        """cv2.setMouseCallback용. 편집 모드가 아니면 아무것도 바꾸지 않는다."""
        if not self.editing:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            dist = [np.hypot(x - cx, y - cy) for cx, cy in self._corners()]
            nearest = int(np.argmin(dist))
            x1, y1, x2, y2 = self.box
            if dist[nearest] <= COLLIMATOR_HANDLE:
                self._drag = ("corner", nearest, 0, list(self.box))
            elif x1 <= x <= x2 and y1 <= y <= y2:
                self._drag = ("move", x, y, list(self.box))
        elif event == cv2.EVENT_MOUSEMOVE and self._drag is not None:
            kind, a, b, start = self._drag
            if kind == "corner":
                box = list(start)
                box[0 if a in (0, 3) else 2] = x   # 왼쪽 꼭짓점이면 x1, 오른쪽이면 x2
                box[1 if a in (0, 1) else 3] = y   # 위쪽 꼭짓점이면 y1, 아래쪽이면 y2
                self.box = self._clamped(box)
            else:
                w, h = start[2] - start[0], start[3] - start[1]
                nx = int(np.clip(start[0] + x - a, 0, self.width - w))
                ny = int(np.clip(start[1] + y - b, 0, self.height - h))
                self.box = [nx, ny, nx + w, ny + h]
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = None

    # ---- 그리기
    def draw(self, frame: np.ndarray) -> np.ndarray:
        out = cv2.convertScaleAbs(frame, alpha=COLLIMATOR_DIM)
        x1, y1, x2, y2 = self.box
        out[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        line = dict(color=COLLIMATOR_LINE, thickness=COLLIMATOR_THICK, lineType=cv2.LINE_AA)
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), **line)
        cv2.line(out, (cx, y1 - COLLIMATOR_TAIL), (cx, y2 + COLLIMATOR_TAIL), **line)
        cv2.line(out, (x1 - COLLIMATOR_TAIL, cy), (x2 + COLLIMATOR_TAIL, cy), **line)
        if self.editing:
            for px, py in self._corners():
                cv2.rectangle(out, (px - 7, py - 7), (px + 7, py + 7), COLLIMATOR_EDIT_COLOR, -1, cv2.LINE_AA)
                cv2.rectangle(out, (px - 7, py - 7), (px + 7, py + 7), (0, 0, 0), 1, cv2.LINE_AA)
        return out


def draw_collimator_hint(frame: np.ndarray, font_small: ImageFont.ImageFont) -> np.ndarray:
    """편집 모드 안내 문구 (콜리메이터 덮개 위에 그려서 어둡게 가려지지 않게)."""

    def paint(draw: ImageDraw.ImageDraw, width: int, _height: int) -> None:
        message = "콜리메이터 편집 중 · 꼭짓점 드래그: 크기 · 안쪽 드래그: 위치 · L: 고정"
        box = draw.textbbox((0, 0), message, font=font_small)
        text_w = box[2] - box[0]
        x = max(16, (width - text_w) // 2)
        y = 160  # 대기 중 안내 문구(y=118)와 겹치지 않게 그 아래에
        draw.rounded_rectangle((x - 16, y - 8, x + text_w + 16, y + 28), radius=12, fill=(10, 60, 90, 220))
        draw.text((x, y), message, font=font_small, fill=(236, 248, 255, 255))

    return HUD_CACHE.apply(frame, ("collimator-hint",), paint)


class OverlayCache:
    """PIL로 그린 반투명 안내 패널을 캐시해 두고, 매 프레임 패널이 있는 영역만 섞는다.

    전체 화면을 PIL로 바꿔 그리면 프레임마다 30ms 넘게 걸려서, 글자가 바뀔 때만 다시 그린다.
    """

    def __init__(self, limit: int = 48) -> None:
        self.limit = limit
        # 키 -> [(영역, 패널 색 uint8, 패널 가중치, 바탕 가중치)]
        self._items: dict[tuple, list[tuple[tuple[slice, slice], np.ndarray, np.ndarray, np.ndarray]]] = {}

    def apply(self, frame: np.ndarray, key: tuple, paint) -> np.ndarray:
        height, width = frame.shape[:2]
        full_key = (width, height) + tuple(key)
        item = self._items.get(full_key)
        if item is None:
            overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            paint(ImageDraw.Draw(overlay), width, height)
            rgba = np.array(overlay)
            bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            alpha = rgba[:, :, 3].astype(np.float32) / 255.0
            count, _labels, stats, _ = cv2.connectedComponentsWithStats((rgba[:, :, 3] > 0).astype(np.uint8))
            item = []
            for i in range(1, count):
                x, y, w, h = (int(v) for v in stats[i, :4])
                region = (slice(y, y + h), slice(x, x + w))
                a = np.ascontiguousarray(alpha[region])
                item.append((region, np.ascontiguousarray(bgr[region]), a, np.ascontiguousarray(1.0 - a)))
            if len(self._items) >= self.limit:
                self._items.pop(next(iter(self._items)))
            self._items[full_key] = item
        out = frame.copy()
        for region, color, weight, rest in item:
            out[region] = cv2.blendLinear(color, np.ascontiguousarray(out[region]), weight, rest)
        return out


HUD_CACHE = OverlayCache()


def draw_avatar_hud(
    frame: np.ndarray,
    live: bool,
    font: ImageFont.ImageFont,
    font_small: ImageFont.ImageFont,
) -> np.ndarray:
    def paint(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
        draw.rounded_rectangle((16, 16, 280, 96), radius=14, fill=(8, 12, 18, 170))
        draw.text((32, 26), "뼈 아바타", font=font, fill=(236, 244, 248, 255))
        status = "추적 중" if live else "대기"
        draw.text((32, 60), status, font=font_small, fill=(186, 214, 230, 255))

        help_text = "방향키 위치    + - 확대    A D 회전    C 조명    L 조명 맞추기    R 초기화"
        box = draw.textbbox((0, 0), help_text, font=font_small)
        text_w = box[2] - box[0]
        bar_x = max(16, (width - text_w) // 2 - 18)
        draw.rounded_rectangle((bar_x, height - 52, bar_x + text_w + 36, height - 16), radius=12, fill=(8, 12, 18, 170))
        draw.text((bar_x + 18, height - 44), help_text, font=font_small, fill=(226, 236, 244, 255))

        if not live:
            message = "카메라에 몸이 보이면 따라 움직입니다"
            box = draw.textbbox((0, 0), message, font=font_small)
            text_w = box[2] - box[0]
            x = (width - text_w) // 2
            y = 118
            draw.rounded_rectangle((x - 16, y - 8, x + text_w + 16, y + 28), radius=12, fill=(8, 12, 18, 160))
            draw.text((x, y), message, font=font_small, fill=(236, 244, 248, 255))

    return HUD_CACHE.apply(frame, ("avatar", live), paint)


def primary_index(poses: list[np.ndarray]) -> int | None:
    if not poses:
        return None
    scores = [float(np.mean(pose[:, 3])) for pose in poses]
    return int(np.argmax(scores))


def primary_pose(poses: list[np.ndarray]) -> np.ndarray | None:
    index = primary_index(poses)
    return None if index is None else poses[index]


def presence_from_result(result) -> list[np.ndarray]:
    """랜드마크별 presence(화면 안에 있을 가능성). 디버그 표시용."""
    out = []
    for landmarks in result.pose_landmarks or []:
        arr = np.full(33, np.nan, dtype=np.float32)
        for i, lm in enumerate(landmarks[:33]):
            value = getattr(lm, "presence", None)
            if value is not None:
                arr[i] = float(value)
        out.append(arr)
    return out


DIAG_POINTS = (0, 7, 8) + tuple(range(11, 33))
DIAG_ARM = (("어깨", 0), ("팔꿈치", 1), ("손목", 2))
DIAG_GREEN = (90, 230, 90)    # RGB
DIAG_RED = (255, 80, 80)


def _fmt(value: float) -> str:
    return " -- " if np.isnan(value) else f"{value:4.2f}"


def diag_rows(
    raw: np.ndarray,
    smooth: np.ndarray,
    presence: np.ndarray,
    seen: np.ndarray,
) -> list[tuple[str, list[tuple[str, int]]]]:
    """화면 왼쪽/오른쪽 팔별로 (제목, [(관절 이름, 랜드마크 번호)]) 목록."""
    left_series = (11, 13, 15)
    right_series = (12, 14, 16)
    # 화면에서 왼쪽에 있는 팔을 먼저 보여준다. (거울 모드면 본인 왼팔)
    if float(raw[11, 0]) > float(raw[12, 0]):
        left_series, right_series = right_series, left_series
    rows = []
    for title, series in (("화면 왼쪽 팔", left_series), ("화면 오른쪽 팔", right_series)):
        rows.append((title, [(name, series[k]) for name, k in DIAG_ARM]))
    return rows


def draw_seen_debug(
    frame: np.ndarray,
    raw: np.ndarray,
    smooth: np.ndarray,
    presence: np.ndarray,
    seen: np.ndarray,
    font_small: ImageFont.ImageFont,
) -> np.ndarray:
    """콜리메이터 박스가 '보인다'고 판정한 관절을 카메라 창에 표시한다.

    채운 점: 박스 판정에 쓰는 보정 후 좌표 (초록=보임, 빨강=안 보임)
    빈 원: MediaPipe 원본 좌표. 둘이 떨어져 있으면 보정 좌표가 굳어 있는 것.
    """
    height, width = frame.shape[:2]
    for index in DIAG_POINTS:
        color = DIAG_GREEN if bool(seen[index]) else DIAG_RED
        bgr = color[::-1]
        p_smooth = _px(smooth[index], width, height)
        p_raw = _px(raw[index], width, height)
        if abs(p_smooth[0] - p_raw[0]) + abs(p_smooth[1] - p_raw[1]) > 4:
            cv2.line(frame, p_smooth, p_raw, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.circle(frame, p_raw, 8, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.circle(frame, p_smooth, 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, p_smooth, 5, bgr, -1, cv2.LINE_AA)

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    panel_w, line_h = 470, 22
    rows = diag_rows(raw, smooth, presence, seen)
    lines = 2 + sum(1 + len(items) for _, items in rows)
    x0 = width - panel_w - 16
    y = 16
    draw.rounded_rectangle((x0, y, width - 16, y + 16 + lines * line_h), radius=12, fill=(6, 14, 24, 200))
    x = x0 + 14
    y += 8
    draw.text((x, y), "디버그: 보이는 부위 판정 (B 끄기)", font=font_small, fill=(236, 246, 255, 255))
    y += line_h
    draw.text((x, y), "관절      vis   pres   x     y    판정", font=font_small, fill=(160, 180, 196, 255))
    y += line_h
    for title, items in rows:
        draw.text((x, y), title, font=font_small, fill=(236, 246, 255, 255))
        y += line_h
        for name, index in items:
            ok = bool(seen[index])
            text = (
                f"  {name:<4} {_fmt(float(raw[index, 3]))}  {_fmt(float(presence[index]))}"
                f"  {float(raw[index, 0]):5.2f} {float(raw[index, 1]):5.2f}   {'보임' if ok else '없음'}"
            )
            draw.text((x, y), text, font=font_small, fill=(*(DIAG_GREEN if ok else DIAG_RED), 255))
            y += line_h
    merged = Image.alpha_composite(image, overlay).convert("RGB")
    return cv2.cvtColor(np.array(merged), cv2.COLOR_RGB2BGR)


def diag_log_line(raw: np.ndarray, smooth: np.ndarray, presence: np.ndarray, seen: np.ndarray) -> str:
    parts = []
    for title, items in diag_rows(raw, smooth, presence, seen):
        cells = []
        for name, index in items:
            drift = float(np.linalg.norm(smooth[index, :2] - raw[index, :2]))
            cells.append(
                f"{name} v{_fmt(float(raw[index, 3])).strip()} p{_fmt(float(presence[index])).strip()}"
                f" ({float(raw[index, 0]):.2f},{float(raw[index, 1]):.2f})"
                f" 굳음{drift:.2f} {'O' if bool(seen[index]) else 'X'}"
            )
        parts.append(f"{title}: " + ", ".join(cells))
    return " | ".join(parts)


def primary_world(worlds: list[np.ndarray], poses: list[np.ndarray]) -> np.ndarray | None:
    if not worlds or not poses:
        return None
    scores = [float(np.mean(pose[:, 3])) for pose in poses]
    return worlds[int(np.argmax(scores))]


def draw_hud(
    frame: np.ndarray,
    fps: float,
    people: int,
    mirror: bool,
    skeleton_on: bool,
    font: ImageFont.ImageFont,
    font_small: ImageFont.ImageFont,
) -> np.ndarray:
    mirror_text = "켜짐" if mirror else "꺼짐"
    status = f"FPS {fps:4.1f}    사람 {people}    반전 {mirror_text}"
    tracking = "추가기능 골격 표시 중" if skeleton_on else "라인 골격 표시 중"

    def paint(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
        draw.rounded_rectangle((16, 16, 430, 132), radius=14, fill=(6, 14, 24, 168))
        draw.text((32, 26), "카메라", font=font, fill=(236, 246, 255, 255))
        draw.text((32, 64), status, font=font_small, fill=(186, 214, 230, 255))
        draw.text((32, 92), tracking, font=font_small, fill=(186, 214, 230, 255))

        help_text = "Q 종료    M 반전    V 추가기능    H 손가락    B 디버그    S 저장"
        box = draw.textbbox((0, 0), help_text, font=font_small)
        text_w = box[2] - box[0]
        bar_x = max(16, (width - text_w) // 2 - 18)
        draw.rounded_rectangle(
            (bar_x, height - 52, bar_x + text_w + 36, height - 16),
            radius=12,
            fill=(6, 14, 24, 168),
        )
        draw.text((bar_x + 18, height - 44), help_text, font=font_small, fill=(226, 238, 246, 255))

        if people == 0:
            message = "상반신이 보이게 서면 아바타가 따라합니다"
            box = draw.textbbox((0, 0), message, font=font)
            text_w = box[2] - box[0]
            x = (width - text_w) // 2
            y = height // 2 - 18
            draw.rounded_rectangle((x - 18, y - 10, x + text_w + 18, y + 40), radius=12, fill=(6, 14, 24, 150))
            draw.text((x, y), message, font=font, fill=(236, 246, 255, 255))

    return HUD_CACHE.apply(frame, ("camera", status, tracking, people == 0), paint)


def self_test(model_name: str) -> int:
    model_path = download_model(model_name)
    landmarker, mp = create_landmarker(model_path, 1)
    blank = np.zeros((360, 640, 3), dtype=np.uint8)
    image = infer_image(blank, mp)
    try:
        result = landmarker.detect_for_video(image, 0)
    finally:
        landmarker.close()
    count = len(result.pose_landmarks or [])
    log.info("self-test ok  model=%s  poses=%d", model_path.name, count)

    from hand_tracking import HAND_MODEL, HandTracker

    hand_path = download_file(*HAND_MODEL, "손 모델")
    tracker = HandTracker(hand_path)
    try:
        hands = tracker.detect(image, 0)
    finally:
        tracker.close()
    log.info("self-test ok  model=%s  hands=%d", hand_path.name, len(hands))
    return 0


def open_hand_tracker():
    """손 인식기를 연다. 실패하면 경고만 남기고 None (손가락 연동 없이 계속 실행)."""
    from hand_tracking import HAND_MODEL, HandTracker

    try:
        tracker = HandTracker(download_file(*HAND_MODEL, "손 모델"))
    except (Exception, SystemExit) as exc:
        log.warning("손 인식기를 열지 못해 손가락 연동 없이 실행합니다: %s", exc)
        return None
    log.info("손 인식기를 열었습니다. H 키로 손가락 연동을 켜고 끕니다.")
    return tracker


def run(args: argparse.Namespace) -> int:
    model_path = download_model(args.model)
    log.info("포즈 모델을 불러오는 중입니다.")
    landmarker, mp = create_landmarker(model_path, args.poses)
    camera = Camera(args.camera, args.width, args.height)
    smoother = PoseSmoother()
    avatar = AvatarState()
    from avatar_mesh import load_bone_avatar

    bone_avatar = load_bone_avatar(bone_map=ROOT / "assets" / "bone_map.json")
    collimator = Collimator(AVATAR_W, AVATAR_H)
    font = load_font(22)
    font_small = load_font(16)
    from hand_tracking import FingerState, draw_hands, match_hands

    hands_on = not args.no_hands
    hand_tracker = open_hand_tracker() if hands_on else None
    hands_on = hand_tracker is not None
    fingers = FingerState()
    hand_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hands")

    mirror = not args.no_mirror
    show_collimator = True
    show_skeleton = False
    xray = False
    glow = True
    detailed_face = False
    debug_mode = False
    debug_logged = 0.0
    fps = 0.0
    fps_shown, fps_shown_at = 0.0, 0.0
    last = time.perf_counter()
    clock_origin = last
    last_stamp = [-1]  # 인식 스레드에서 갱신하므로 리스트로 감싼다
    infer_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")

    def capture(mirror_on: bool, hands_enabled: bool) -> dict | None:
        """인식 스레드: 최신 카메라 프레임을 받아 자세·손 인식까지 마친다. 프레임이 없으면 None."""
        deadline = time.perf_counter() + 0.5
        frame = camera.read()
        while frame is None and time.perf_counter() < deadline:
            time.sleep(0.01)
            frame = camera.read()
        if frame is None:
            return None
        if mirror_on:
            frame = cv2.flip(frame, 1)
        now = time.perf_counter()
        stamp = max(int((now - clock_origin) * 1000), last_stamp[0] + 1)
        last_stamp[0] = stamp
        image = infer_image(frame, mp)
        # 손 인식은 또 다른 스레드에서 자세 인식과 동시에 돌린다 (MediaPipe가 추론 중 GIL을 풀어 줌)
        hand_job = (
            hand_pool.submit(hand_tracker.detect, image, stamp)
            if hands_enabled and hand_tracker is not None
            else None
        )
        result = landmarker.detect_for_video(image, stamp)
        hands = hand_job.result() if hand_job is not None else []
        return {"frame": frame, "now": now, "result": result, "hands": hands}
    placed = False

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(AVATAR_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 800, 450)
    cv2.resizeWindow(AVATAR_WINDOW, 480, 700)
    # 마우스 좌표는 창 크기와 상관없이 아바타 화면(이미지) 픽셀 좌표로 들어온다
    cv2.setMouseCallback(AVATAR_WINDOW, collimator.on_mouse)
    log.info("카메라 창과 뼈 아바타 창을 열었습니다. Q 또는 Esc 로 종료합니다.")

    try:
        pending = infer_pool.submit(capture, mirror, hands_on)
        while True:
            # 이번 프레임의 인식 결과를 받자마자 다음 프레임 인식을 시작해 두고, 그동안 이번 프레임을 그린다.
            job = pending.result()
            pending = infer_pool.submit(capture, mirror, hands_on)
            if job is None:
                if time.perf_counter() - clock_origin > 5 and last_stamp[0] < 0:
                    log.error("카메라 프레임을 받지 못했습니다.")
                    return 1
                continue
            frame, now, result, hands = job["frame"], job["now"], job["result"], job["hands"]
            raw_poses, raw_worlds = poses_from_result(result)
            presences = presence_from_result(result) if debug_mode else []
            poses, worlds = smoother.update(raw_poses, raw_worlds)
            aspect = frame.shape[1] / max(frame.shape[0], 1)
            avatar.observe(primary_world(worlds, poses), primary_pose(poses), aspect, now)
            avatar.follow_image(primary_pose(poses))
            points, vis, live, scale = avatar.current(now)
            place_x, place_y, zoom = avatar.placement(live)

            matched = match_hands(hands, primary_pose(poses)) if hands else {}
            fingers.update(matched, now)

            view = frame
            if poses:
                if show_skeleton:
                    view = render_poses(frame, poses, detailed_face, glow, xray)
                else:
                    view = render_line_skeleton(frame, poses)
            if hands:
                draw_hands(view, hands, matched)
            if bone_avatar is not None:
                avatar_view = bone_avatar.render(
                    points, vis, avatar.yaw, avatar.pitch, live, STAGE, (place_x, place_y), zoom,
                    curls=fingers.curls() if hands_on else None,
                    hand_frames=fingers.hand_frames(now) if hands_on else None,
                )
            else:
                avatar_view = render_avatar(
                    points, vis, avatar.yaw, avatar.pitch, live, scale, (place_x, place_y), zoom
                )

            dt = now - last
            last = now
            if dt > 0:
                fps = fps * 0.9 + (1.0 / dt) * 0.1 if fps else 1.0 / dt

            if debug_mode and poses:
                index = primary_index(poses)
                if index is not None and index < len(raw_poses) and index < len(presences):
                    raw, smooth, presence = raw_poses[index], poses[index], presences[index]
                    seen = seen_landmarks(smooth)
                    view = draw_seen_debug(view, raw, smooth, presence, seen, font_small)
                    if now - debug_logged >= 1.0:
                        log.debug(diag_log_line(raw, smooth, presence, seen))
                        if hands_on:
                            log.debug("손가락(도) 손 %d개 인식, 짝 %s | %s", len(hands), sorted(matched), fingers.summary())
                        debug_logged = now
            if now - fps_shown_at >= 0.5:  # 화면 FPS 숫자는 0.5초마다만 바꾼다 (안내 패널 다시 그리기 줄이기)
                fps_shown, fps_shown_at = fps, now
            view = draw_hud(view, fps_shown, len(poses), mirror, show_skeleton, font, font_small)
            if show_collimator or collimator.editing:
                avatar_view = collimator.draw(avatar_view)
            # 안내 문구는 콜리메이터 덮개 위에 그려야 박스 바깥에서도 보인다
            avatar_view = draw_avatar_hud(avatar_view, live, font, font_small)
            if collimator.editing:
                avatar_view = draw_collimator_hint(avatar_view, font_small)
            cv2.imshow(WINDOW, view)
            cv2.imshow(AVATAR_WINDOW, avatar_view)
            if not placed:
                cv2.moveWindow(WINDOW, 20, 40)
                cv2.moveWindow(AVATAR_WINDOW, 840, 30)
                placed = True

            key = cv2.waitKeyEx(1)
            if key == -1:
                pass
            elif key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("l"), ord("L")):
                collimator.toggle_edit()
            elif key in (ord("c"), ord("C")):
                show_collimator = not show_collimator
            elif key in (ord("m"), ord("M")):
                mirror = not mirror
            elif key in (ord("v"), ord("V")):
                show_skeleton = not show_skeleton
            elif key in (ord("x"), ord("X")):
                xray = not xray
            elif key in (ord("g"), ord("G")):
                glow = not glow
            elif key in (ord("f"), ord("F")):
                detailed_face = not detailed_face
            elif key in (ord("h"), ord("H")):
                if hand_tracker is None:
                    hand_tracker = open_hand_tracker()
                hands_on = hand_tracker is not None and not hands_on
                log.info("손가락 연동 %s", "켜짐" if hands_on else "꺼짐")
            elif key in (ord("b"), ord("B")):
                debug_mode = not debug_mode
                # 디버그 중에는 콘솔 출력 레벨을 DEBUG로 낮추고, 끄면 INFO로 되돌린다.
                log.setLevel(logging.DEBUG if debug_mode else logging.INFO)
                log.info("디버그 %s (콘솔 레벨 %s)", "켜짐" if debug_mode else "꺼짐", logging.getLevelName(log.level))
            elif key in (ord("a"), ord("A")):
                avatar.yaw -= 0.08
            elif key in (ord("d"), ord("D")):
                avatar.yaw += 0.08
            elif key in (2424832, 65361):
                avatar.nudge(-0.05, 0.0)
            elif key in (2555904, 65363):
                avatar.nudge(0.05, 0.0)
            elif key in (2490368, 65362):
                avatar.nudge(0.0, 0.05)
            elif key in (2621440, 65364):
                avatar.nudge(0.0, -0.05)
            elif key in (ord("+"), ord("="), 187, 65451):
                avatar.zoom_by(1.08)
            elif key in (ord("-"), ord("_"), 189, 65453):
                avatar.zoom_by(1.0 / 1.08)
            elif key in (ord("r"), ord("R")):
                avatar.reset_view()
            elif key in (ord("s"), ord("S")):
                SHOT_DIR.mkdir(parents=True, exist_ok=True)
                stamp_name = time.strftime("%Y%m%d_%H%M%S")
                camera_path = SHOT_DIR / f"camera_{stamp_name}.png"
                avatar_path = SHOT_DIR / f"avatar_{stamp_name}.png"
                cv2.imwrite(str(camera_path), view)
                cv2.imwrite(str(avatar_path), avatar_view)
                log.info("저장: %s", camera_path)
                log.info("저장: %s", avatar_path)

            if now - clock_origin > 0.8:
                for name in (WINDOW, AVATAR_WINDOW):
                    if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) == 0:
                        return 0
    finally:
        infer_pool.shutdown(wait=True)   # 진행 중인 인식이 끝난 뒤에 인식기를 닫는다
        hand_pool.shutdown(wait=True)
        camera.close()
        landmarker.close()
        if hand_tracker is not None:
            hand_tracker.close()
        if collimator.editing:  # 편집 중에 종료하면 그때까지 맞춘 위치를 저장
            collimator.toggle_edit()
        cv2.destroyAllWindows()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="카메라로 읽은 몸 움직임을 두 번째 창의 뼈 아바타가 따라합니다.")
    parser.add_argument("--camera", type=int, default=0, help="카메라 번호 (기본 0)")
    parser.add_argument("--width", type=int, default=1280, help="요청할 카메라 가로 해상도")
    parser.add_argument("--height", type=int, default=720, help="요청할 카메라 세로 해상도")
    parser.add_argument("--model", choices=tuple(MODELS), default="lite", help="lite가 가장 빠르고 full/heavy가 더 정확합니다")
    parser.add_argument("--poses", type=int, default=1, help="동시에 추적할 최대 사람 수 (1-4)")
    parser.add_argument("--no-mirror", action="store_true", help="좌우 반전 없이 카메라 원본을 봅니다")
    parser.add_argument("--self-test", action="store_true", help="모델 로드와 빈 프레임 추론만 확인합니다")
    parser.add_argument("--inspect-glb", action="store_true", help="뼈 메시의 본 이름과 크기를 출력합니다")
    parser.add_argument("--no-hands", action="store_true", help="손가락 연동(손 인식) 없이 시작합니다. 실행 중 H 키로 켤 수 있습니다")
    args = parser.parse_args()
    args.poses = min(4, max(1, args.poses))
    return args


def setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def main() -> int:
    setup_logging()
    args = parse_args()
    if args.inspect_glb:
        from avatar_mesh import DEFAULT_GLB, describe_glb

        log.info("%s", describe_glb(DEFAULT_GLB))
        return 0
    if args.self_test:
        return self_test(args.model)
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
