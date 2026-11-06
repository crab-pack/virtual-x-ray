"""리그 GLB 뼈 메시를 포즈에 맞춰 아바타 창에 그린다.

뼈 길이는 모델 바인드 포즈를 유지하고, 각 본의 방향만 랜드마크에 맞춘다.
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path

import numpy as np

# main.py의 "virtual_xray" 로거 아래에 두어 같은 출력 형식과 레벨을 따른다.
log = logging.getLogger("virtual_xray.avatar")

# 손목이 아래팔 방향에서 꺾일 수 있는 최대 각도 (손 인식이 튀어도 손이 뒤집히지 않게)
_WRIST_LIMIT = float(np.deg2rad(80.0))

# 골반부터 머리까지는 카메라 앞뒤(z)로 숙이지 않고 수직을 유지한다.
_UPRIGHT_BONES = {"Hips", "Spine02", "Spine01", "Spine", "neck", "Head"}

ROOT = Path(__file__).resolve().parent
BASE_GLB = ROOT / "assets" / "skeleton.glb"
# 손가락 본을 추가한 버전 (tools/add_finger_bones.py로 생성). 있으면 이것을 쓴다.
FINGER_GLB = ROOT / "assets" / "skeleton_fingers.glb"
DEFAULT_GLB = FINGER_GLB if FINGER_GLB.exists() else BASE_GLB
DEFAULT_MAP = ROOT / "assets" / "bone_map.json"

COMPONENT = {
    5120: ("b", 1),
    5121: ("B", 1),
    5122: ("h", 2),
    5123: ("H", 2),
    5125: ("I", 4),
    5126: ("f", 4),
}
TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def describe_glb(path: Path) -> str:
    gltf, _blob = _read_glb(path)
    nodes = gltf.get("nodes", [])
    lines = [f"file {path}", f"nodes {len(nodes)}", f"meshes {len(gltf.get('meshes', []))}"]
    for skin in gltf.get("skins", []):
        lines.append("joints:")
        for joint in skin.get("joints", []):
            lines.append(f"  {nodes[joint].get('name', joint)}")
    for mesh in gltf.get("meshes", []):
        acc = gltf["accessors"]
        tris = 0
        verts = 0
        for prim in mesh.get("primitives", []):
            tris += acc[prim["indices"]]["count"] // 3
            verts += acc[prim["attributes"]["POSITION"]]["count"]
        lines.append(f"triangles {tris}  vertices {verts}")
    return "\n".join(lines)


def load_bone_avatar(path: Path | None = None, bone_map: Path | None = None):
    path = path or DEFAULT_GLB
    bone_map = bone_map or DEFAULT_MAP
    if not path.exists():
        log.warning("뼈 메시 파일이 없어 캡슐 아바타를 사용합니다.")
        return None
    try:
        return BoneAvatar(path, bone_map)
    except Exception as exc:
        log.warning("뼈 메시를 열지 못해 캡슐 아바타를 사용합니다: %s", exc)
        return None


class BoneAvatar:
    def __init__(self, path: Path, bone_map: Path) -> None:
        import moderngl

        gltf, blob = _read_glb(path)
        mapping = json.loads(bone_map.read_text(encoding="utf-8"))
        self.unit_scale = float(mapping.get("unit_scale", 0.01))
        self.aims: list[tuple[str, str]] = [tuple(pair) for pair in mapping["aims"]]
        self.nodes = _build_nodes(gltf)
        self.skin_joints, self.inverse_bind = _load_skin(gltf, blob)
        self.positions, self.normals, self.joints, self.weights, self.indices = _load_mesh(gltf, blob)
        self.bind_world = _bind_world(self.nodes)
        self.by_name = {node.name: node.index for node in self.nodes if node.name}
        hips = self.bind_world[self.by_name["Hips"]]
        self.hip_pos = hips[:3, 3].copy()
        self._check_aims()
        self.finger_axes = self._finger_axes()
        self.hand_rest = self._hand_rest()

        self.ctx = moderngl.create_context(standalone=True)
        self._init_gl()
        log.info("뼈 메시를 불러왔습니다. 삼각형 %d, 본 %d.", len(self.indices) // 3, len(self.skin_joints))

    def _finger_axes(self) -> dict[str, tuple[int, np.ndarray]]:
        """손가락 마디 본마다 굽힘 축(본 로컬 좌표). 손가락 본이 없는 모델이면 빈 dict.

        바인드 자세에서 손바닥은 정면(+z)을 향하므로, 굽힘 축은 '마디 방향 x 정면'이다.
        키는 "LeftHandIndex2" 같은 본 이름, 값은 (노드 번호, 로컬 축).
        """
        out: dict[str, tuple[int, np.ndarray]] = {}
        palm = np.array([0.0, 0.0, 1.0])
        for name, index in self.by_name.items():
            if not name or name[-1] not in "123" or "Hand" not in name:
                continue
            nxt = self.by_name.get(name[:-1] + str(int(name[-1]) + 1))
            if nxt is None:
                continue
            bind = self.bind_world[index]
            direction = self.bind_world[nxt][:3, 3] - bind[:3, 3]
            axis = np.cross(direction, palm)
            if float(np.linalg.norm(axis)) < 1e-6:
                continue
            local = np.linalg.inv(bind[:3, :3]) @ (axis / np.linalg.norm(axis))
            out[name] = (index, (local / np.linalg.norm(local)).astype(np.float32))
        return out

    def _hand_rest(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """바인드 자세의 손 방향(손목→중지 뿌리)과 가로 방향(새끼 뿌리→검지 뿌리)."""
        out = {}
        for side in ("Left", "Right"):
            names = [f"{side}Hand", f"{side}HandMiddle1", f"{side}HandIndex1", f"{side}HandPinky1"]
            if not all(n in self.by_name for n in names):
                continue
            hand, middle, index, pinky = (self.bind_world[self.by_name[n]][:3, 3] for n in names)
            out[side] = ((middle - hand).astype(np.float32), (index - pinky).astype(np.float32))
        return out

    def _check_aims(self) -> None:
        for bone, child in self.aims:
            if bone not in self.by_name:
                raise ValueError(f"본 이름 없음: {bone}")
            if not child.startswith("_") and child not in self.by_name:
                raise ValueError(f"자식 본 이름 없음: {child}")

    def _init_gl(self) -> None:
        ctx = self.ctx
        vertex = """
            #version 330
            uniform mat4 u_mvp;
            uniform mat4 u_model;
            uniform mat4 u_view;
            uniform mat4 u_bones[BONE_COUNT];
            in vec3 in_pos;
            in vec3 in_nrm;
            in vec4 in_joints;
            in vec4 in_weights;
            out vec3 v_view_nrm;
            void main() {
                ivec4 j = ivec4(in_joints);
                mat4 skin = u_bones[j.x] * in_weights.x
                          + u_bones[j.y] * in_weights.y
                          + u_bones[j.z] * in_weights.z
                          + u_bones[j.w] * in_weights.w;
                vec4 posed = skin * vec4(in_pos, 1.0);
                vec3 world_n = mat3(u_model) * mat3(skin) * in_nrm;
                v_view_nrm = mat3(u_view) * world_n;
                gl_Position = u_mvp * posed;
            }
        """
        fragment = """
            #version 330
            uniform float u_energy;
            in vec3 v_view_nrm;
            out vec4 frag;
            void main() {
                vec3 n = normalize(v_view_nrm);
                float facing = abs(dot(n, vec3(0.0, 0.0, 1.0)));
                float rim = pow(1.0 - facing, 1.05);
                float density = (0.16 + 1.05 * rim) * u_energy;
                vec3 tint = vec3(0.82, 0.91, 1.0);
                frag = vec4(tint * density, 1.0);
            }
        """
        bone_count = len(self.skin_joints)
        vertex = vertex.replace("BONE_COUNT", str(bone_count))
        self.program = ctx.program(vertex_shader=vertex, fragment_shader=fragment)
        packed = np.hstack(
            [
                self.positions,
                self.normals,
                self.joints.astype(np.float32),
                self.weights,
            ]
        ).astype(np.float32)
        vbo = ctx.buffer(packed.tobytes())
        ibo = ctx.buffer(self.indices.astype(np.uint32).tobytes())
        self.vao = ctx.vertex_array(self.program, [(vbo, "3f 3f 4f 4f", "in_pos", "in_nrm", "in_joints", "in_weights")], ibo)
        self.width = 760
        self.height = 1100
        self.color = ctx.texture((self.width, self.height), 4)
        self.depth = ctx.depth_renderbuffer((self.width, self.height))
        self.fbo = ctx.framebuffer(color_attachments=[self.color], depth_attachment=self.depth)
        self._bone_buf = np.zeros((bone_count, 4, 4), dtype=np.float32)
        self._aim_dir: dict[int, np.ndarray] = {}

    def render(
        self,
        points: np.ndarray,
        vis: np.ndarray,
        yaw: float,
        pitch: float,
        live: bool,
        background: np.ndarray,
        offset: tuple[float, float] = (0.0, 0.0),
        zoom: float = 1.0,
        curls: dict[str, np.ndarray] | None = None,
        hand_frames: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> np.ndarray:
        """curls: 손가락 굽힘 각도 {"LeftHandIndex": [마디1, 마디2, 마디3] 라디안}.
        hand_frames: 손 방향 {"Left": (손목→중지 뿌리, 새끼 뿌리→검지 뿌리)} 아바타 좌표.
        둘 다 추적 중일 때만 쓴다."""
        if not live:
            self._aim_dir.clear()
        worlds = self.bind_world if not live else self._posed_worlds(points, vis, curls, hand_frames)
        self._write_bones(worlds)
        view, mvp = self._camera(yaw, pitch, offset, zoom)
        ctx = self.ctx
        ctx.disable(ctx.DEPTH_TEST)
        ctx.enable(ctx.BLEND)
        ctx.blend_func = ctx.ONE, ctx.ONE
        ctx.enable(ctx.CULL_FACE)
        ctx.cull_face = "back"
        self.fbo.use()
        self.fbo.clear(0.0, 0.0, 0.0, 0.0)
        self.program["u_mvp"].write(mvp.T.astype(np.float32).tobytes())
        self.program["u_model"].write(self._model_matrix(zoom).T.astype(np.float32).tobytes())
        self.program["u_view"].write(view.T.astype(np.float32).tobytes())
        self.program["u_energy"].value = 1.0 if live else 0.62
        self.program["u_bones"].write(np.transpose(self._bone_buf, (0, 2, 1)).astype(np.float32).tobytes())
        self.vao.render()
        raw = self.fbo.read(components=4, alignment=1)
        image = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 4)
        image = np.flipud(image)
        return _xray_plate(image)

    def _write_bones(self, worlds: list[np.ndarray]) -> None:
        for slot, node_index in enumerate(self.skin_joints):
            self._bone_buf[slot] = worlds[node_index] @ self.inverse_bind[slot]

    def _model_matrix(self, zoom: float = 1.0) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float32)
        scale = self.unit_scale * float(zoom)
        matrix[0, 0] = matrix[1, 1] = matrix[2, 2] = scale
        matrix[:3, 3] = -self.hip_pos * scale
        return matrix

    def _camera(
        self,
        yaw: float,
        pitch: float,
        offset: tuple[float, float] = (0.0, 0.0),
        zoom: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        target = np.array([0.0, 0.02, 0.0], dtype=np.float32)
        dist = 3.55
        eye = target + np.array(
            [
                dist * np.sin(yaw) * np.cos(pitch),
                dist * np.sin(pitch) + 0.15,
                dist * np.cos(yaw) * np.cos(pitch),
            ],
            dtype=np.float32,
        )
        view = _look_at(eye, target)
        # 카메라 오른쪽·위 방향으로 옮겨, 회전과 상관없이 화면에서 상하좌우로 움직인다.
        view[0, 3] += float(offset[0]) * float(zoom)
        view[1, 3] += float(offset[1]) * float(zoom)
        proj = _perspective(np.deg2rad(34.0), self.width / self.height, 0.05, 30.0)
        return view, proj @ view @ self._model_matrix(zoom)

    def _posed_worlds(
        self,
        points: np.ndarray,
        vis: np.ndarray,
        curls: dict[str, np.ndarray] | None = None,
        hand_frames: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> list[np.ndarray]:
        targets = _pose_targets(points, vis)
        rotations: dict[int, np.ndarray] = {}
        # 손가락 굽힘: 노드 번호 -> 로컬 회전 (부모 기준)
        flex: dict[int, np.ndarray] = {}
        for prefix, angles in (curls or {}).items():
            for k in range(3):
                entry = self.finger_axes.get(f"{prefix}{k + 1}")
                if entry is not None and abs(float(angles[k])) > 1e-4:
                    flex[entry[0]] = _axis_angle(entry[1], float(angles[k]))
        torso = self._torso_delta(targets)
        for bone, child in self.aims:
            if bone in _UPRIGHT_BONES or bone not in targets or child not in targets:
                continue
            joint = self.by_name[bone]
            desired = self._limit_direction(joint, targets[child] - targets[bone])
            if desired is None:
                continue
            bind = self.bind_world[joint][:3, :3]
            if child.startswith("_"):
                rest = bind @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
            else:
                rest = bind @ self.nodes[self.by_name[child]].local_t
            if float(np.linalg.norm(rest)) < 1e-5:
                continue
            rotations[joint] = _rot_from_to(rest, desired) @ bind
        if torso is not None:
            for name in _UPRIGHT_BONES:
                joint = self.by_name[name]
                rotations[joint] = torso @ self.bind_world[joint][:3, :3]
        # 손 인식이 있으면 손목 방향과 손바닥 방향(비틀림)까지 손 모델 기준으로 맞춘다.
        for side, (direction, across) in (hand_frames or {}).items():
            hand_name, fore_name = f"{side}Hand", f"{side}ForeArm"
            if side not in self.hand_rest or hand_name not in targets or fore_name not in targets:
                continue
            fore = targets[hand_name] - targets[fore_name]
            if float(np.linalg.norm(fore)) < 1e-5 or float(np.linalg.norm(direction)) < 1e-5:
                continue
            fore = fore / np.linalg.norm(fore)
            direction = _step_toward(fore.astype(np.float32), (direction / np.linalg.norm(direction)).astype(np.float32), _WRIST_LIMIT)
            rest_dir, rest_across = self.hand_rest[side]
            turn = _align_frames(rest_dir, rest_across, direction, across)
            if turn is not None:
                joint = self.by_name[hand_name]
                rotations[joint] = turn @ self.bind_world[joint][:3, :3]

        worlds: list[np.ndarray | None] = [None] * len(self.nodes)

        def visit(index: int, parent: np.ndarray | None) -> None:
            node = self.nodes[index]
            local_t = node.local_t
            if parent is None:
                pos = self.bind_world[index][:3, 3]
            else:
                pos = parent[:3, 3] + parent[:3, :3] @ local_t
            if index in rotations:
                rot = rotations[index]
            elif parent is None:
                rot = self.bind_world[index][:3, :3]
            else:
                rot = parent[:3, :3] @ node.local_r
            if index in flex:
                rot = rot @ flex[index]
            matrix = np.eye(4, dtype=np.float32)
            matrix[:3, :3] = rot
            matrix[:3, 3] = pos
            worlds[index] = matrix
            for child in node.children:
                visit(child, matrix)

        visit(self.by_name["Hips"], None)
        for index, world in enumerate(worlds):
            if world is None:
                worlds[index] = self.bind_world[index]
        return worlds

    def _torso_delta(self, targets: dict[str, np.ndarray]) -> np.ndarray | None:
        """갈비뼈가 붙은 척추·목·머리를 같은 회전으로 기울여 정중선이 벌어지지 않게 한다."""
        if "Hips" not in targets or "Spine" not in targets:
            return None
        desired = targets["Spine"] - targets["Hips"]
        desired = desired.copy()
        desired[2] = 0.0
        if float(np.linalg.norm(desired)) < 1e-4:
            desired = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        desired = self._limit_direction(self.by_name["Hips"], desired)
        if desired is None:
            return None
        rest = self.bind_world[self.by_name["Hips"]][:3, :3] @ self.nodes[self.by_name["Spine02"]].local_t
        if float(np.linalg.norm(rest)) < 1e-5:
            return None
        return _rot_from_to(rest, desired)

    def _limit_direction(self, joint: int, desired: np.ndarray) -> np.ndarray | None:
        """한 프레임에 너무 크게 꺾이지 않게 하되, 뒤집힌 팔도 목표 쪽으로 되돌린다."""
        length = float(np.linalg.norm(desired))
        if length < 1e-5:
            return None
        desired = desired.astype(np.float32) / length
        previous = self._aim_dir.get(joint)
        if previous is None:
            self._aim_dir[joint] = desired
            return desired
        desired = _step_toward(previous, desired, np.deg2rad(55.0))
        self._aim_dir[joint] = desired
        return desired


class _Node:
    def __init__(self, index: int, name: str) -> None:
        self.index = index
        self.name = name
        self.children: list[int] = []
        self.local_t = np.zeros(3, dtype=np.float32)
        self.local_r = np.eye(3, dtype=np.float32)
        self.local_s = np.ones(3, dtype=np.float32)


def _build_nodes(gltf: dict) -> list[_Node]:
    nodes = []
    for index, raw in enumerate(gltf.get("nodes", [])):
        node = _Node(index, raw.get("name", ""))
        node.children = list(raw.get("children", []))
        if "matrix" in raw:
            matrix = np.array(raw["matrix"], dtype=np.float32).reshape(4, 4).T
            node.local_t = matrix[:3, 3].copy()
            node.local_r = matrix[:3, :3].copy()
        else:
            if "translation" in raw:
                node.local_t = np.array(raw["translation"], dtype=np.float32)
            if "scale" in raw:
                node.local_s = np.array(raw["scale"], dtype=np.float32)
            if "rotation" in raw:
                node.local_r = _quat_to_mat(np.array(raw["rotation"], dtype=np.float32))
        nodes.append(node)
    return nodes


def _bind_world(nodes: list[_Node]) -> list[np.ndarray]:
    parent = {child: index for index, node in enumerate(nodes) for child in node.children}
    worlds: list[np.ndarray | None] = [None] * len(nodes)

    def visit(index: int, parent_world: np.ndarray | None) -> None:
        local = _local_matrix(nodes[index])
        worlds[index] = local if parent_world is None else parent_world @ local
        for child in nodes[index].children:
            visit(child, worlds[index])

    for index in range(len(nodes)):
        if index not in parent:
            visit(index, None)
    return [world if world is not None else np.eye(4, dtype=np.float32) for world in worlds]


def _local_matrix(node: _Node) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = node.local_r * node.local_s
    matrix[:3, 3] = node.local_t
    return matrix


def _load_skin(gltf: dict, blob: bytes) -> tuple[list[int], np.ndarray]:
    skin = gltf["skins"][0]
    joints = list(skin["joints"])
    mats = _read_accessor(gltf, blob, skin["inverseBindMatrices"]).reshape(len(joints), 4, 4)
    mats = np.transpose(mats, (0, 2, 1)).astype(np.float32)
    return joints, mats


def _load_mesh(gltf: dict, blob: bytes):
    prim = gltf["meshes"][0]["primitives"][0]
    attrs = prim["attributes"]
    positions = _read_accessor(gltf, blob, attrs["POSITION"]).reshape(-1, 3).astype(np.float32)
    normals = _read_accessor(gltf, blob, attrs["NORMAL"]).reshape(-1, 3).astype(np.float32)
    joints = _read_accessor(gltf, blob, attrs["JOINTS_0"]).reshape(-1, 4).astype(np.float32)
    weights = _read_accessor(gltf, blob, attrs["WEIGHTS_0"]).reshape(-1, 4).astype(np.float32)
    weight_sum = weights.sum(axis=1, keepdims=True)
    weight_sum[weight_sum < 1e-6] = 1.0
    weights = weights / weight_sum
    indices = _read_accessor(gltf, blob, prim["indices"]).astype(np.uint32).reshape(-1)
    return positions, normals, joints, weights, indices


def _hand_target(
    points: np.ndarray,
    vis: np.ndarray,
    wrist: int,
    elbow: int,
    index: int,
    pinky: int,
) -> np.ndarray | None:
    """손 메시는 아래팔의 연장선에 두고, 손가락이 팔과 반대면 그 방향은 쓰지 않는다."""
    if float(vis[wrist]) < 0.12 or float(vis[elbow]) < 0.12:
        return None
    forearm = points[wrist] - points[elbow]
    length = float(np.linalg.norm(forearm)) or 1.0
    straight = points[wrist] + forearm / length * 0.09
    tips = []
    if float(vis[index]) >= 0.35:
        tips.append(points[index])
    if float(vis[pinky]) >= 0.35:
        tips.append(points[pinky])
    if not tips:
        return straight.astype(np.float32)
    mid = np.mean(tips, axis=0)
    delta = mid - points[wrist]
    span = float(np.linalg.norm(delta))
    if span < 0.03:
        return straight.astype(np.float32)
    align = float(np.dot(delta, forearm) / (span * length))
    if align < 0.25:
        return straight.astype(np.float32)
    return mid.astype(np.float32)


def _pose_targets(points: np.ndarray, vis: np.ndarray) -> dict[str, np.ndarray]:
    def ok(*ids: int, thresh: float = 0.12) -> bool:
        return all(float(vis[i]) >= thresh for i in ids)

    def vec(*values: float) -> np.ndarray:
        return np.array(values, dtype=np.float32)

    hip = (points[23] + points[24]) * 0.5
    shoulder = (points[11] + points[12]) * 0.5
    if ok(7, 8):
        ear = (points[7] + points[8]) * 0.5
    else:
        ear = shoulder + vec(0, 0.22, 0)
    nose = points[0] if ok(0) else ear + vec(0, 0.02, 0.06)
    out: dict[str, np.ndarray] = {
        "Hips": hip.astype(np.float32),
        "Spine02": (hip * 0.72 + shoulder * 0.28).astype(np.float32),
        "Spine01": (hip * 0.42 + shoulder * 0.58).astype(np.float32),
        "Spine": (hip * 0.14 + shoulder * 0.86).astype(np.float32),
        "neck": (shoulder * 0.4 + ear * 0.6).astype(np.float32),
        "Head": ear.astype(np.float32),
        "head_end": (ear + vec(0, 0.16, 0) + (nose - ear) * 0.5).astype(np.float32),
    }
    pairs = {
        "LeftUpLeg": 23,
        "LeftLeg": 25,
        "LeftFoot": 27,
        "LeftToeBase": 31,
        "RightUpLeg": 24,
        "RightLeg": 26,
        "RightFoot": 28,
        "RightToeBase": 32,
        "LeftArm": 11,
        "LeftForeArm": 13,
        "LeftHand": 15,
        "RightArm": 12,
        "RightForeArm": 14,
        "RightHand": 16,
    }
    for name, index in pairs.items():
        if ok(index):
            out[name] = points[index].astype(np.float32)
    if ok(11, 12):
        out["LeftShoulder"] = (shoulder * 0.35 + points[11] * 0.65).astype(np.float32)
        out["RightShoulder"] = (shoulder * 0.35 + points[12] * 0.65).astype(np.float32)
    left_hand = _hand_target(points, vis, 15, 13, 19, 17)
    if left_hand is not None:
        out["_LeftFingers"] = left_hand
    right_hand = _hand_target(points, vis, 16, 14, 20, 18)
    if right_hand is not None:
        out["_RightFingers"] = right_hand
    if ok(27, 31):
        direction = points[31] - points[27]
        norm = float(np.linalg.norm(direction)) or 1.0
        out["_LeftToeEnd"] = (points[31] + direction / norm * 0.06).astype(np.float32)
    if ok(28, 32):
        direction = points[32] - points[28]
        norm = float(np.linalg.norm(direction)) or 1.0
        out["_RightToeEnd"] = (points[32] + direction / norm * 0.06).astype(np.float32)
    return out


def _read_glb(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    magic, _version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValueError("GLB 파일이 아닙니다.")
    offset = 12
    gltf = None
    blob = b""
    while offset + 8 <= length:
        chunk_len, chunk_type = struct.unpack_from("<I4s", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_len]
        offset += chunk_len
        if chunk_type == b"JSON":
            gltf = json.loads(chunk)
        elif chunk_type == b"BIN\x00":
            blob = chunk
    if gltf is None:
        raise ValueError("GLB에 JSON 청크가 없습니다.")
    return gltf, blob


def _read_accessor(gltf: dict, blob: bytes, index: int) -> np.ndarray:
    accessor = gltf["accessors"][index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    code, size = COMPONENT[accessor["componentType"]]
    count = TYPE_COUNT[accessor["type"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride", size * count)
    tight = stride == size * count
    if tight:
        raw = np.frombuffer(blob, dtype=np.dtype("<" + code), count=accessor["count"] * count, offset=start)
        values = np.array(raw, dtype=np.float32, copy=True)
    else:
        values = np.empty(accessor["count"] * count, dtype=np.float32)
        for i in range(accessor["count"]):
            part = blob[start + i * stride : start + i * stride + size * count]
            raw = np.array(struct.unpack_from("<" + code * count, part), dtype=np.float32)
            values[i * count : (i + 1) * count] = raw
    if accessor.get("normalized"):
        limit = float(np.iinfo(np.dtype(code)).max) if code != "f" else 1.0
        if code in ("b", "h"):
            limit = float(np.iinfo(np.dtype(code)).max)
        values /= limit or 1.0
    return values


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _align_frames(
    rest_dir: np.ndarray, rest_across: np.ndarray, direction: np.ndarray, across: np.ndarray
) -> np.ndarray | None:
    """(방향, 가로) 두 벡터로 정한 바인드 자세 좌표계를 목표 좌표계로 돌리는 회전."""

    def basis(d: np.ndarray, a: np.ndarray) -> np.ndarray | None:
        d = np.asarray(d, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64)
        if np.linalg.norm(d) < 1e-6:
            return None
        e1 = d / np.linalg.norm(d)
        a = a - (a @ e1) * e1
        if np.linalg.norm(a) < 1e-6:
            return None
        e2 = a / np.linalg.norm(a)
        return np.stack([e1, e2, np.cross(e1, e2)], axis=1)

    rest = basis(rest_dir, rest_across)
    target = basis(direction, across)
    if rest is None or target is None:
        return None
    return (target @ rest.T).astype(np.float32)


def _step_toward(src: np.ndarray, dst: np.ndarray, max_angle: float) -> np.ndarray:
    """반대 방향이어도 목표 쪽으로 일정 각도만 회전한다."""
    cosine = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle <= max_angle or angle < 1e-4:
        return dst.astype(np.float32)
    axis = np.cross(src, dst)
    if float(np.linalg.norm(axis)) < 1e-4:
        axis = np.cross(src, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        if float(np.linalg.norm(axis)) < 1e-4:
            axis = np.cross(src, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    axis = axis / np.linalg.norm(axis)
    turned = _axis_angle(axis.astype(np.float32), max_angle) @ src
    turned = turned / (np.linalg.norm(turned) or 1.0)
    return turned.astype(np.float32)


def _rot_from_to(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    a = src.astype(np.float32)
    b = dst.astype(np.float32)
    a /= np.linalg.norm(a) or 1.0
    b /= np.linalg.norm(b) or 1.0
    cross = np.cross(a, b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 0.9999:
        return np.eye(3, dtype=np.float32)
    if dot < -0.9999:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        if np.linalg.norm(axis) < 1e-4:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        axis = axis / np.linalg.norm(axis)
        return _axis_angle(axis, np.pi)
    skew = np.array(
        [
            [0, -cross[2], cross[1]],
            [cross[2], 0, -cross[0]],
            [-cross[1], cross[0], 0],
        ],
        dtype=np.float32,
    )
    return np.eye(3, dtype=np.float32) + skew + skew @ skew * (1.0 / (1.0 + dot))


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float32,
    )


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) or 1.0)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-5:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    zaxis = -forward
    view = np.eye(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = true_up
    view[2, :3] = zaxis
    view[0, 3] = -float(np.dot(right, eye))
    view[1, 3] = -float(np.dot(true_up, eye))
    view[2, 3] = -float(np.dot(zaxis, eye))
    return view


def _perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(fov_y * 0.5)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = f / aspect
    matrix[1, 1] = f
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


_PLATE_GRAIN_FRAMES = 6   # 미리 만들어 두고 돌려 쓰는 입자 노이즈 장수
_plate_cache: dict[tuple[int, int], dict] = {}


def _plate_constants(height: int, width: int) -> dict:
    """크기별로 한 번만 만드는 비네팅·배경·입자 노이즈."""
    cached = _plate_cache.get((height, width))
    if cached is not None:
        return cached
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    vignette = np.clip(1.0 - (x * x * 0.28 + y * y * 0.42), 0.0, 1.0)
    background = np.empty((height, width, 3), dtype=np.float32)
    background[:, :, 0] = 7.0
    background[:, :, 1] = 12.0
    background[:, :, 2] = 18.0
    background *= (0.45 + 0.55 * vignette)[:, :, None]
    rng = np.random.default_rng()
    vignette3 = np.repeat(vignette[:, :, None], 3, axis=2)
    # 결과 = 배경 + (판 + 입자) x 비네팅 = 배경 + 판 x 비네팅 + 입자 x 비네팅.
    # 입자 x 비네팅은 미리 계산해 양수/음수 부분으로 나눠 둔다 (uint8 포화 덧셈·뺄셈으로 처리).
    grain = []
    for _ in range(_PLATE_GRAIN_FRAMES):
        g = rng.normal(0.0, 5.0, (height, width, 1)).astype(np.float32) * vignette3
        grain.append((
            np.clip(np.rint(g), 0, 255).astype(np.uint8),
            np.clip(np.rint(-g), 0, 255).astype(np.uint8),
        ))
    cached = {
        "vignette": np.clip(np.rint(vignette3 * 255.0), 0, 255).astype(np.uint8),
        "background": np.clip(np.rint(background), 0, 255).astype(np.uint8),
        "grain": grain,
        "rng": rng,
    }
    _plate_cache[(height, width)] = cached
    return cached


def _xray_plate(rgba: np.ndarray) -> np.ndarray:
    """겹쳐 쌓인 뼈 밀도를 흐리고 입자를 더해 방사선 사진처럼 만든다.

    매 프레임 도는 부분이라 OpenCV 연산 위주로 짰다. 넓은 산란 블러는 1/4 크기에서 계산하고,
    비네팅·배경·입자 노이즈는 미리 만들어 둔다.
    """
    import cv2

    height, width = rgba.shape[:2]
    const = _plate_constants(height, width)
    exposure = np.ascontiguousarray(rgba[:, :, :3])
    sharp = cv2.GaussianBlur(exposure, (0, 0), 1.15)
    small = cv2.resize(exposure, (max(1, width // 4), max(1, height // 4)), interpolation=cv2.INTER_AREA)
    scatter = cv2.resize(cv2.GaussianBlur(small, (0, 0), 7.5 / 4), (width, height), interpolation=cv2.INTER_LINEAR)
    # (sharp*0.82 + scatter*0.48 - 12) * 1.28, 0~255로 잘림 (uint8 포화 연산)
    plate = cv2.addWeighted(sharp, 0.82 * 1.28, scatter, 0.48 * 1.28, -12.0 * 1.28)
    merged = cv2.multiply(plate, const["vignette"], scale=1.0 / 255.0)
    cv2.add(merged, const["background"], merged)
    grain_up, grain_down = const["grain"][int(const["rng"].integers(len(const["grain"])))]
    cv2.add(merged, grain_up, merged)
    cv2.subtract(merged, grain_down, merged)
    return cv2.cvtColor(merged, cv2.COLOR_RGB2BGR)


def _composite(rgba: np.ndarray, background: np.ndarray) -> np.ndarray:
    import cv2

    if background.shape[:2] != rgba.shape[:2]:
        background = cv2.resize(background, (rgba.shape[1], rgba.shape[0]), interpolation=cv2.INTER_LINEAR)
    color = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    bgr = cv2.cvtColor(background, cv2.COLOR_BGR2RGB).astype(np.float32)
    merged = color * alpha + bgr * (1.0 - alpha)
    return cv2.cvtColor(np.clip(merged, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
