"""skeleton.glb에 손가락 본을 추가해 skeleton_fingers.glb를 만든다.

기존 24개 본과 메시, 애니메이션은 그대로 두고, 손마다 손가락 본 15개(5개 손가락 x 3마디)와
손끝 끝 노드 5개를 덧붙인다. 손가락 정점은 해당 마디 본에 100% 연결한다.

메시 조각은 뼈 경계와 맞지 않아서, 관절 위치를 직접 정해 둔 JSON(finger_joints_*.json)을 쓴다.
각 JSON에는 Blender 정면 직교 카메라 렌더에서 읽은 관절의 픽셀 좌표가 있다.
손가락마다 [첫 관절, 둘째 관절, 셋째 관절, 손끝] 4개 점이다.
  - 엄지: 손목-손바닥 관절(CMC), 손바닥-첫마디(MCP), 마디 사이(IP), 손끝
  - 나머지: 손허리-첫마디(MCP), 첫마디-중간(PIP), 중간-끝(DIP), 손끝
깊이(z)는 그 위치 근처 메시 정점의 평균으로 정한다.

정점은 가장 가까운 손가락 마디 구간에 붙인다. 손허리뼈(엄지 제외)와 손목뼈는 손 본에 남는다.

실행:
    .venv\\Scripts\\python tools\\add_finger_bones.py
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import avatar_mesh as am  # noqa: E402

SRC = ROOT / "assets" / "skeleton.glb"
DST = ROOT / "assets" / "skeleton_fingers.glb"
JOINT_FILES = {"Left": Path(__file__).with_name("finger_joints_left.json"),
               "Right": Path(__file__).with_name("finger_joints_right.json")}
FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
DEPTH_RADIUS = 0.004   # 관절 깊이를 구할 때 볼 정점 반경(m, 정면 평면 기준)
MAX_DISTANCE = 0.016   # 마디 중심선에서 이보다 먼 정점은 손 본에 남긴다(m). 손가락 사이 떠 있는 조각도 포함
MAX_DISTANCE_THUMB_BASE = 0.011  # 엄지 첫 마디(손바닥뼈)는 옆 손목뼈가 따라오지 않게 더 좁게
HAND_WEIGHT_MIN = 0.99 # 손 본 가중치가 이 이상인 정점만 옮긴다(손목 쪽 섞인 정점은 그대로)


def joint_points(cfg: dict, verts: np.ndarray) -> dict[str, np.ndarray]:
    cam = cfg["camera"]
    out = {}
    for finger in FINGERS:
        points = []
        for px, py in cfg[finger]:
            x = cam["cx"] + (px / cam["pixels"] - 0.5) * cam["scale"]
            y = cam["cy"] - (py / cam["pixels"] - 0.5) * cam["scale"]
            d = np.hypot(verts[:, 0] - x, verts[:, 1] - y)
            near = verts[d < DEPTH_RADIUS]
            z = float(near[:, 2].mean()) if len(near) else float(verts[np.argmin(d), 2])
            points.append(np.array([x, y, z], dtype=np.float64))
        out[finger] = np.array(points)
    return out


def segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """점들과 선분 ab 사이 거리, 그리고 a→b 위 위치(0~1, 범위 밖 허용)."""
    ab = b - a
    t = ((p - a) @ ab) / float(ab @ ab)
    closest = a + np.clip(t, 0.0, 1.0)[:, None] * ab
    return np.linalg.norm(p - closest, axis=1), t


def write_glb(path: Path, gltf: dict, blob: bytes) -> None:
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * (-len(js) % 4)
    blob += b"\0" * (-len(blob) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    with path.open("wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, total))
        fh.write(struct.pack("<I4s", len(js), b"JSON"))
        fh.write(js)
        fh.write(struct.pack("<I4s", len(blob), b"BIN\0"))
        fh.write(blob)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    gltf, blob = am._read_glb(SRC)
    blob = bytearray(blob)
    nodes = am._build_nodes(gltf)
    skin_joints, ibm = am._load_skin(gltf, bytes(blob))
    pos, _nrm, joints, weights, _indices = am._load_mesh(gltf, bytes(blob))
    bind = am._bind_world(nodes)
    by_name = {n.name: n.index for n in nodes}
    slot_of = {node: slot for slot, node in enumerate(skin_joints)}
    if any("Thumb" in name for name in by_name):
        raise SystemExit("이미 손가락 본이 있는 파일입니다.")

    new_nodes: list[dict] = []
    new_slots: list[tuple[int, np.ndarray]] = []   # (노드 번호, 바인드 월드 행렬)
    assign = np.full(len(pos), -1, dtype=np.int64)  # 정점 -> 새 skin 슬롯
    dump: dict[str, list[float]] = {}

    for side in ("Left", "Right"):
        hand_node = by_name[f"{side}Hand"]
        hand_slot = slot_of[hand_node]
        hand_world = bind[hand_node]
        hand_weight = np.where(joints == hand_slot, weights, 0.0).sum(1)
        hand_ids = np.where(hand_weight > 0.5)[0]
        cfg = json.loads(JOINT_FILES[side].read_text(encoding="utf-8"))
        points = joint_points(cfg, pos[hand_ids].astype(np.float64))

        movable = hand_ids[hand_weight[hand_ids] >= HAND_WEIGHT_MIN]
        p = pos[movable].astype(np.float64)
        best_dist = np.full(len(movable), np.inf)
        best_slot = np.full(len(movable), -1, dtype=np.int64)
        for finger in FINGERS:
            chain = points[finger]
            parent_index, parent_world = hand_node, hand_world
            for k, point in enumerate(chain):
                name = f"{side}Hand{finger}{k + 1}"
                local = np.linalg.inv(parent_world) @ np.append(point, 1.0)
                node_index = len(gltf["nodes"]) + len(new_nodes)
                new_nodes.append({"name": name, "translation": [float(v) for v in local[:3]]})
                parent = (gltf["nodes"][parent_index] if parent_index < len(gltf["nodes"])
                          else new_nodes[parent_index - len(gltf["nodes"])])
                parent.setdefault("children", []).append(node_index)
                world = parent_world.copy()
                world[:3, 3] = point
                dump[name] = [float(v) for v in point]
                if k < 3:  # 손끝 노드는 정점이 없으므로 skin에 넣지 않는다
                    slot = len(skin_joints) + len(new_slots)
                    new_slots.append((node_index, world))
                    dist, t = segment_distance(p, chain[k], chain[k + 1])
                    seg_len = float(np.linalg.norm(chain[k + 1] - chain[k]))
                    # 관절보다 손목 쪽(뒤)에 있는 정점은 이 마디가 아니다. 손끝 마디는 끝 너머도 허용.
                    ok = t >= -0.5 / 1000 / seg_len
                    if k < 2:
                        ok &= t <= 1.0 + 3.0 / 1000 / seg_len
                    limit = MAX_DISTANCE_THUMB_BASE if (finger == "Thumb" and k == 0) else MAX_DISTANCE
                    ok &= dist <= limit
                    dist = np.where(ok, dist, np.inf)
                    better = dist < best_dist
                    best_dist[better] = dist[better]
                    best_slot[better] = slot
                parent_index, parent_world = node_index, world

        # 손허리뼈(엄지 제외): 첫 관절보다 손목 쪽 중심선. 여기에 더 가까우면 손 본에 남긴다.
        for finger in FINGERS[1:]:
            chain = points[finger]
            back = chain[0] - (chain[1] - chain[0]) / np.linalg.norm(chain[1] - chain[0]) * 0.07
            dist, _t = segment_distance(p, back, chain[0])
            hand_better = dist < best_dist
            best_dist[hand_better] = dist[hand_better]
            best_slot[hand_better] = -1
        keep = (best_slot >= 0) & np.isfinite(best_dist)
        assign[movable[keep]] = best_slot[keep]
        counts = np.bincount(best_slot[keep] - len(skin_joints), minlength=len(new_slots))
        print(f"{side}: 손가락 정점 {int(keep.sum())}개 / 손 정점 {len(hand_ids)}개, "
              f"마디별 최소 {int(counts[counts > 0].min())}개")

    moved = np.where(assign >= 0)[0]
    joints[moved] = 0
    weights[moved] = 0.0
    joints[moved, 0] = assign[moved]
    weights[moved, 0] = 1.0
    prim = gltf["meshes"][0]["primitives"][0]
    for attr, data, comp in (("JOINTS_0", joints.astype(np.uint8), 5121), ("WEIGHTS_0", weights.astype(np.float32), 5126)):
        acc = gltf["accessors"][prim["attributes"][attr]]
        assert acc["componentType"] == comp, attr
        view = gltf["bufferViews"][acc["bufferView"]]
        stride = view["byteStride"]
        base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        size = data.itemsize * 4
        span = stride * (len(pos) - 1) + size
        rows = np.frombuffer(bytes(blob[base: base + span]) + b"\0" * (stride - size), dtype=np.uint8)
        rows = rows.reshape(len(pos), stride).copy()
        rows[moved, :size] = data[moved].view(np.uint8).reshape(len(moved), size)
        blob[base: base + span] = rows.tobytes()[:span]

    gltf["nodes"].extend(new_nodes)
    skin = gltf["skins"][0]
    skin["joints"] = list(skin["joints"]) + [index for index, _ in new_slots]
    new_ibm = np.concatenate([ibm, np.stack([np.linalg.inv(world) for _, world in new_slots])]).astype(np.float32)
    blob += b"\0" * (-len(blob) % 16)
    offset = len(blob)
    blob += np.transpose(new_ibm, (0, 2, 1)).tobytes()  # glTF는 열 우선
    gltf["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": new_ibm.nbytes})
    gltf["accessors"].append({"bufferView": len(gltf["bufferViews"]) - 1, "componentType": 5126,
                              "count": len(new_ibm), "type": "MAT4"})
    skin["inverseBindMatrices"] = len(gltf["accessors"]) - 1
    gltf["buffers"][0]["byteLength"] = len(blob)

    write_glb(DST, gltf, bytes(blob))
    print(f"저장: {DST}  (본 {len(skin['joints'])}개, 손가락 정점 {len(moved)}개)")
    if "--dump-joints" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--dump-joints") + 1])
        out.write_text(json.dumps(dump, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
