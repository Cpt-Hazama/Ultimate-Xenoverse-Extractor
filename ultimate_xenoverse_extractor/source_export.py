from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import math
import re
import shutil
from io import BytesIO
from typing import Callable, Iterable

import numpy as np
from PIL import Image

from .models import AssetKind, LibraryEntry
from .parsers import load_ean, load_emd, load_esk, load_bac, animation_events
from .parsers.ean import EanAnimation, EanNode
from .parsers.esk import EskBone, EskFile, EskFormatError

# Xenoverse assets are authored at roughly meter scale. Source character assets
# are conventionally authored in inches/Hammer units.
EXPORTER_VERSION = "0.2.0"
SOURCE_SCALE = 42.41732283464567
OUTLINE_WIDTH = 0.001

# XV2 character space: +X forward, +Y up, +Z right
# Source/3ds Max SMD space: +X forward, +Y right, +Z up
# Swap XV2 Y/Z while preserving forward. This is a reflection, so triangle
# winding is reversed exactly once below. The previous mapping assumed XV2 +Z
# was forward, which rotated exported characters 90 degrees around the up axis.
_C = np.array(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)), dtype=np.float64)
_C_INV = np.linalg.inv(_C)

# When an unused EXP_* container root is removed, its local transform still
# has to be baked into b_C_Base. The registry is keyed by the stripped EskFile
# instance so animation sampling and bind-pose calculations use the same true
# global root orientation.
_ROOT_CONTAINER_BAKE: dict[int, np.ndarray] = {}


@dataclass(slots=True)
class ExportOptions:
    meshes: bool = True
    animations: bool = True
    textures: bool = True
    events: bool = True
    source_scale: float = SOURCE_SCALE
    fps: float = 60.0


@dataclass(slots=True)
class ExportResult:
    output_dir: Path
    mesh_files: list[Path] = field(default_factory=list)
    animation_files: list[Path] = field(default_factory=list)
    texture_files: list[Path] = field(default_factory=list)
    metadata_files: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _safe_name(value: str, fallback: str = "unnamed") -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    return value or fallback


def _vec_source(v: Iterable[float], scale: float = 1.0) -> tuple[float, float, float]:
    out = _C @ np.asarray(tuple(v)[:3], dtype=np.float64)
    return tuple(float(x * scale) for x in out)


def _quat_matrix(q: Iterable[float]) -> np.ndarray:
    x, y, z, w = (float(v) for v in tuple(q)[:4])
    length = math.sqrt(x*x + y*y + z*z + w*w)
    if length < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x/length, y/length, z/length, w/length
    return np.array((
        (1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w),
        (2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w),
        (2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y),
    ), dtype=np.float64)


def _quat_source_matrix(q: Iterable[float]) -> np.ndarray:
    return _C @ _quat_matrix(q) @ _C_INV


def _matrix_to_euler_xyz(m: np.ndarray) -> tuple[float, float, float]:
    # R = Rz * Ry * Rx, matching the XYZ Euler fields used by SMD skeleton rows.
    sy = math.sqrt(float(m[0, 0] * m[0, 0] + m[1, 0] * m[1, 0]))
    singular = sy < 1e-8
    if not singular:
        x = math.atan2(float(m[2, 1]), float(m[2, 2]))
        y = math.atan2(float(-m[2, 0]), sy)
        z = math.atan2(float(m[1, 0]), float(m[0, 0]))
    else:
        x = math.atan2(float(-m[1, 2]), float(m[1, 1]))
        y = math.atan2(float(-m[2, 0]), sy)
        z = 0.0
    return x, y, z


def _sample_node(node: EanNode | None, frame: float, bind_pos, bind_rot, bind_scale):
    pos = tuple(float(v) for v in bind_pos[:3])
    rot = tuple(float(v) for v in bind_rot[:4])
    scale = tuple(float(v) for v in bind_scale[:3])
    if node is None:
        return pos, rot, scale
    track = node.tracks.get(0)
    if track:
        value = track.sample(frame)
        if value is not None:
            w = value[3]
            pos = tuple(value[i] / w if w not in (0.0, 1.0) else value[i] for i in range(3))
    track = node.tracks.get(1)
    if track:
        value = track.sample(frame, normalize_quaternion=True)
        if value is not None:
            rot = value
    track = node.tracks.get(2)
    if track:
        value = track.sample(frame)
        if value is not None:
            w = value[3]
            scale = tuple(value[i] / w if w not in (0.0, 1.0) else value[i] for i in range(3))
    return pos, rot, scale


def _nodes_block(esk: EskFile) -> list[str]:
    lines = ["nodes"]
    for bone in esk.bones:
        parent = -1 if bone.parent is None else bone.parent
        name = bone.name.replace('"', "'")
        lines.append(f'{bone.index} "{name}" {parent}')
    lines.append("end")
    return lines


def _skeleton_frame_lines(esk: EskFile, animation: EanAnimation | None, frame: float, scale: float) -> list[str]:
    by_name = animation.node_name_map if animation is not None else {}
    by_index = animation.node_map if animation is not None and not by_name else {}
    root_bake = _ROOT_CONTAINER_BAKE.get(id(esk))
    lines: list[str] = []
    for bone in esk.bones:
        node = by_name.get(bone.name) if by_name else by_index.get(bone.index)
        pos, rot, sampled_scale = _sample_node(node, frame, bone.position, bone.rotation, bone.scale)

        # Removing EXP_<character>_000 without preserving its transform makes
        # b_C_Base inherit the wrong heading. Source only exposes that mistake
        # once walkframe motion extraction is enabled, producing the observed
        # 90-degree clockwise character rotation. Bake the removed container's
        # complete local transform into the new true root for every frame.
        if bone.index == 0 and root_bake is not None:
            local = np.eye(4, dtype=np.float64)
            local[:3, :3] = _quat_matrix(rot) @ np.diag([
                float(sampled_scale[0]),
                float(sampled_scale[1]),
                float(sampled_scale[2]),
            ])
            local[:3, 3] = np.asarray(pos[:3], dtype=np.float64)
            baked = root_bake @ local
            px, py, pz = _vec_source(baked[:3, 3], scale)
            source_rotation = _C @ baked[:3, :3] @ _C_INV
            # Strip scale before converting the baked basis to Euler angles.
            for column in range(3):
                length = float(np.linalg.norm(source_rotation[:, column]))
                if length > 1e-12:
                    source_rotation[:, column] /= length
            rx, ry, rz = _matrix_to_euler_xyz(source_rotation)
        else:
            px, py, pz = _vec_source(pos, scale)
            rx, ry, rz = _matrix_to_euler_xyz(_quat_source_matrix(rot))
        lines.append(f"{bone.index} {px:.6f} {py:.6f} {pz:.6f} {rx:.6f} {ry:.6f} {rz:.6f}")
    return lines


def _normalise3(v: Iterable[float]) -> tuple[float, float, float]:
    values = tuple(float(x) for x in tuple(v)[:3])
    length = math.sqrt(sum(x * x for x in values))
    if length < 1e-8:
        return (0.0, 0.0, 1.0)
    return tuple(x / length for x in values)


def _smd_vertex_line(parent: int, position, normal, uv, links: list[tuple[int, float]]) -> str:
    link_text = " ".join(f"{idx} {weight:.6f}" for idx, weight in links)
    return (
        f"{parent} {position[0]:.6f} {position[1]:.6f} {position[2]:.6f} "
        f"{normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f} {uv[0]:.6f} {1.0-uv[1]:.6f} "
        f"{len(links)} {link_text}"
    )


def write_reference_smd(
    path: Path,
    esk: EskFile,
    emd,
    scale: float,
    material_prefix: str = "",
) -> None:
    bone_by_name = {bone.name: bone.index for bone in esk.bones}
    lines = ["version 1", *_nodes_block(esk), "skeleton", "time 0"]
    lines.extend(_skeleton_frame_lines(esk, None, 0.0, scale))
    lines.extend(("end", "triangles"))

    for model in emd.models:
        for mesh in model.meshes:
            for submesh in mesh.submeshes:
                base_material = _safe_name(submesh.name, "material")
                material = f"{material_prefix}_{base_material}" if material_prefix else base_material
                positions = submesh.positions
                normals = submesh.normals
                uvs = submesh.uvs
                for group in submesh.triangle_groups:
                    palette = [bone_by_name.get(name, 0) for name in group.bone_names]
                    indices = group.indices
                    for tri_start in range(0, len(indices) - 2, 3):
                        source_tri = (indices[tri_start], indices[tri_start + 1], indices[tri_start + 2])
                        converted_tri = (source_tri[0], source_tri[2], source_tri[1])
                        records = {}
                        for vi in source_tri:
                            if vi >= len(positions):
                                continue
                            normal_xv = _normalise3(normals[vi] or (0.0, 0.0, 1.0))
                            p = _vec_source(positions[vi], scale)
                            n = _normalise3(_vec_source(normal_xv, 1.0))
                            uv = uvs[vi] if vi < len(uvs) and uvs[vi] is not None else (0.0, 0.0)
                            blend_i = submesh.blend_indices[vi] if vi < len(submesh.blend_indices) else None
                            blend_w = submesh.blend_weights[vi] if vi < len(submesh.blend_weights) else None
                            links: list[tuple[int, float]] = []
                            if blend_i is not None and blend_w is not None:
                                merged: dict[int, float] = {}
                                for slot in range(4):
                                    weight = float(blend_w[slot])
                                    local_index = int(blend_i[slot])
                                    if weight <= 1e-6:
                                        continue
                                    bone_index = palette[local_index] if local_index < len(palette) else 0
                                    merged[bone_index] = merged.get(bone_index, 0.0) + weight
                                total = sum(merged.values())
                                if total > 1e-8:
                                    links = [(idx, weight / total) for idx, weight in merged.items()]
                            if not links:
                                links = [(0, 1.0)]
                            parent = max(links, key=lambda pair: pair[1])[0]
                            outline_pos_xv = tuple(positions[vi][axis] + normal_xv[axis] * OUTLINE_WIDTH for axis in range(3))
                            outline_p = _vec_source(outline_pos_xv, scale)
                            outline_n = tuple(-value for value in n)
                            records[vi] = (
                                _smd_vertex_line(parent, p, n, uv, links),
                                _smd_vertex_line(parent, outline_p, outline_n, uv, links),
                            )
                        if len(records) != 3:
                            continue
                        lines.append(material)
                        lines.extend(records[vi][0] for vi in converted_tri)
                        # Reverse the converted winding again so only the shell's
                        # back faces are visible from outside in Source.
                        lines.append("outline")
                        lines.extend(records[vi][1] for vi in source_tri)
    lines.append("end")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

def write_animation_smd(
    path: Path,
    esk: EskFile,
    animation: EanAnimation,
    scale: float,
    root_motion: tuple[float, float, float] | None = None,
    root_offset: tuple[float, float, float] | None = None,
) -> None:
    """Write an animation SMD with optional bone-0 translation adjustments.

    ``root_offset`` is a constant Source-space offset applied to every frame.
    ``root_motion`` is a linear displacement across the complete cycle. Neither
    option mutates the parsed EAN animation.
    """
    lines = ["version 1", *_nodes_block(esk), "skeleton"]
    duration = max(1, int(animation.duration))
    denominator = max(1, duration - 1)
    for frame in range(duration):
        lines.append(f"time {frame}")
        frame_lines = _skeleton_frame_lines(esk, animation, float(frame), scale)
        if (root_offset or root_motion) and frame_lines:
            parts = frame_lines[0].split()
            if len(parts) >= 7:
                amount = frame / denominator
                offset = root_offset or (0.0, 0.0, 0.0)
                motion = root_motion or (0.0, 0.0, 0.0)
                parts[1] = f"{float(parts[1]) + offset[0] + motion[0] * amount:.6f}"
                parts[2] = f"{float(parts[2]) + offset[1] + motion[1] * amount:.6f}"
                parts[3] = f"{float(parts[3]) + offset[2] + motion[2] * amount:.6f}"
                frame_lines[0] = " ".join(parts)
        lines.extend(frame_lines)
    lines.extend(("end",))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _ground_stride_scale(entry: LibraryEntry, character_esk: EskFile) -> float:
    """Compare world-space bind-height to Vegeta's VGT_000 rig.

    If the install doesn't contain Vegeta, retain the legacy motion instead of
    guessing a physical scale and corrupting exports. Ignore non-body attachment
    bones, cameras and auxiliary physics helpers when measuring height.
    """
    def height(esk: EskFile) -> float:
        matrices = _global_bind_matrices(esk)
        valid = [float(matrices[b.index][1, 3]) for b in esk.bones
                 if b.name.lower().startswith(("b_", "a_")) and "scd" not in b.name.lower()]
        if len(valid) < 4:
            valid = [float(m[1, 3]) for m in matrices]
        return max(valid) - min(valid) if valid else 0.0

    own_height = height(character_esk)
    if own_height <= 1e-4:
        return 1.0
    # Installations may point at data/, data/chara, or the entire _Files root.
    for ancestor in (entry.path, *entry.path.parents):
        if ancestor.name.casefold() == "chara":
            reference = ancestor / "VGT" / "VGT_000.esk"
            if reference.is_file():
                try:
                    ref_height = height(load_esk(reference))
                    if ref_height > 1e-4:
                        return max(0.65, min(3.0, own_height / ref_height))
                except (OSError, EskFormatError):
                    pass
            break
    return 1.0


def _write_root_motion_copy(
    source: Path,
    target: Path,
    root_motion: tuple[float, float, float],
) -> None:
    """Create a _WF SMD by changing only bone 0 translation.

    The source animation is copied verbatim apart from the XYZ position fields
    on bone 0. This guarantees the generated walkframe version keeps exactly
    the same root rotation as the untouched animation.
    """
    lines = source.read_text(encoding="utf-8", errors="ignore").splitlines()
    times = [index for index, line in enumerate(lines) if line.lstrip().startswith("time ")]
    frame_count = max(1, len(times))
    denominator = max(1, frame_count - 1)
    current_frame = -1
    in_skeleton = False

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "skeleton":
            in_skeleton = True
            continue
        if in_skeleton and stripped == "end":
            in_skeleton = False
            continue
        if not in_skeleton:
            continue
        if stripped.startswith("time "):
            current_frame += 1
            continue
        parts = stripped.split()
        if current_frame < 0 or len(parts) < 7 or parts[0] != "0":
            continue
        amount = current_frame / denominator
        parts[1] = f"{float(parts[1]) + root_motion[0] * amount:.6f}"
        parts[2] = f"{float(parts[2]) + root_motion[1] * amount:.6f}"
        parts[3] = f"{float(parts[3]) + root_motion[2] * amount:.6f}"
        # Rotation fields parts[4:7] are intentionally untouched.
        prefix = line[: len(line) - len(line.lstrip())]
        lines[index] = prefix + " ".join(parts)

    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _body_root_motion(name: str, stride_scale: float = 1.0) -> tuple[float, float, float] | None:
    """Ground locomotion: stride length scales with the character's rig height.

    Vegeta is the reference for the original hardcoded distances. Keeping the
    exact prior numbers at scale=1 preserves his behavior, while taller rigs
    translate farther per animation cycle. Flight/Bukuu movement is unchanged.
    """
    upper = name.upper()
    # _S variants are air poses; keep their existing generic motion values.
    ground_multiplier = stride_scale if not upper.endswith("_S") else 1.0
    if "BAS_WALK" in upper:
        return (0.0, -50.0 * ground_multiplier, 0.0)
    if "BAS_RUN" in upper and "BUKUU" not in upper:
        return (0.0, -200.0 * ground_multiplier, 0.0)
    return None


def _bukuu_root_motion(name: str) -> tuple[float, float, float] | None:
    upper = name.upper()
    if "BAS_BUKUU_F" in upper:
        return (0.0, -300.0, 0.0)
    if "BAS_BUKUU_B" in upper:
        return (0.0, 200.0, 0.0)
    if "BAS_BUKUU_L" in upper:
        return (300.0, 0.0, 0.0)
    if "BAS_BUKUU_R" in upper:
        return (-300.0, 0.0, 0.0)
    return None


def _keyframe_markers(animation: EanAnimation) -> list[dict]:
    frames: dict[int, set[str]] = {}
    for node in animation.nodes:
        bone = node.bone_name or f"bone_{node.bone_index}"
        for flag, track in node.tracks.items():
            channel = {0: "position", 1: "rotation", 2: "scale"}.get(flag, f"track_{flag}")
            for key in track.keyframes:
                frames.setdefault(int(key.frame), set()).add(f"{bone}:{channel}")
    return [
        {"frame": frame, "channels": sorted(channels)}
        for frame, channels in sorted(frames.items())
    ]


def _ascii_tokens(path: Path) -> list[str]:
    try:
        data = path.read_bytes()
    except OSError:
        return []
    tokens = re.findall(rb"[A-Za-z_][A-Za-z0-9_./-]{3,63}", data)
    out: list[str] = []
    seen = set()
    for token in tokens:
        text = token.decode("ascii", errors="ignore")
        if text not in seen:
            seen.add(text)
            out.append(text)
        if len(out) >= 200:
            break
    return out


def write_event_manifest(path: Path, bac_files: list[Path]) -> None:
    """Write only a BAC-source inventory.

    EAN transform keys are deliberately not reported as gameplay events. Until
    BAC records are decoded, this file clearly marks the sources as undecoded
    instead of fabricating bone-keyframe events.
    """
    payload = {
        "format": "xv2_bac_event_sources_v1",
        "decoded": False,
        "note": (
            "Gameplay/action events are stored in BAC. The BAC files listed here "
            "were found, but their timeline records are not decoded yet."
        ),
        "bac_sources": [
            {
                "file": source.name,
                "size": source.stat().st_size if source.exists() else 0,
                "magic": source.read_bytes()[:4].decode("ascii", errors="replace") if source.exists() else "",
                "string_hints": _ascii_tokens(source),
            }
            for source in bac_files
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _matching_asset(entry: LibraryEntry, filename: str) -> Path | None:
    wanted = filename.casefold()
    return next((asset.path for asset in entry.assets if asset.path.name.casefold() == wanted), None)


def _is_physics_model(asset) -> bool:
    """Detect secondary/physics EMDs even when the scanner flag is missing."""
    if asset.kind != AssetKind.MODEL:
        return False
    stem = asset.path.stem.casefold()
    return bool(getattr(asset, "is_physics", False) or "_scd" in stem)


def _reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _save_tga(image: Image.Image, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGBA").save(target, format="TGA")


def export_textures(entry: LibraryEntry, output: Path, result: ExportResult) -> None:
    """Export Source-ready diffuse masks and material DYT data.

    Base EMB textures use the same alpha/levels/invert conversion as the
    viewport and diffuse_ui.py. Matching DYT/EMM files additionally produce a
    composed preview texture, the original 4-pixel DYT band, and a $color2 text
    file for every material.
    """
    from .parsers.emb import load_emb
    from .parsers.emm import load_emm
    from .texture_processing import (
        choose_dyt_image,
        compose_dyt_texture,
        entry_to_pil,
        find_dyt_bands,
        process_diffuse_alpha,
    )

    _reset_directory(output)
    base_assets = [
        asset.path for asset in entry.assets
        if asset.path.suffix.casefold() == ".emb" and not asset.path.name.casefold().endswith(".dyt.emb")
    ]

    for base_path in base_assets:
        try:
            base_emb = load_emb(base_path)
        except Exception as exc:
            result.warnings.append(f"Texture container {base_path.name}: {exc}")
            continue

        part_stem = base_path.name[:-4]  # remove only the final .emb
        part_dir = output / _safe_name(part_stem)
        diffuse_dir = part_dir / "diffuse"
        material_dir = part_dir / "materials"
        dyt_dir = part_dir / "dyt"
        diffuse_dir.mkdir(parents=True, exist_ok=True)

        decoded_base: list[Image.Image | None] = []
        for item in base_emb.entries:
            image = entry_to_pil(item)
            decoded_base.append(image)
            if image is None:
                result.warnings.append(f"Texture {base_path.name}:{item.name} could not be decoded")
                continue
            name = _safe_name(Path(item.name).stem if item.name else f"DATA{item.index:03d}")
            # Exact diffuse_ui.py path: alpha -> levels 100/1/155 -> invert.
            mask = process_diffuse_alpha(image)
            # Save the exact single-channel result produced by diffuse_ui.py.
            target = diffuse_dir / f"{name}.tga"
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                mask.save(target, format="TGA")
                result.texture_files.append(target)
            except Exception as exc:
                result.warnings.append(f"Diffuse {base_path.name}:{item.name}: {exc}")

        dyt_path = _matching_asset(entry, f"{part_stem}.dyt.emb")
        emm_path = _matching_asset(entry, f"{part_stem}.emm")
        emd_path = _matching_asset(entry, f"{part_stem}.emd")
        if dyt_path is None or emm_path is None:
            continue

        try:
            dyt_emb = load_emb(dyt_path)
            dyt_image = choose_dyt_image(dyt_emb)
            emm = load_emm(emm_path)
        except Exception as exc:
            result.warnings.append(f"DYT/material set {part_stem}: {exc}")
            continue
        if dyt_image is None:
            result.warnings.append(f"DYT {dyt_path.name}: no valid 128x128 image")
            continue

        bands = find_dyt_bands(dyt_image)
        if not bands:
            result.warnings.append(f"DYT {dyt_path.name}: no lightmap bands found")
            continue

        # Resolve the base texture index used by each EMD material/submesh.
        material_texture_indices: dict[str, int] = {}
        if emd_path is not None:
            try:
                emd = load_emd(emd_path)
                for submesh in emd.submeshes:
                    texture_index = submesh.texture_definitions[0].texture_index if submesh.texture_definitions else 0
                    material_texture_indices.setdefault(submesh.name.casefold(), int(texture_index))
            except Exception as exc:
                result.warnings.append(f"Material/texture mapping {emd_path.name}: {exc}")

        for material in emm.materials:
            requested = max(0, int(round(float(material.value("MatScale1X", 0.0)))))
            actual = min(requested, len(bands) - 1)
            band = bands[actual]
            material_name = _safe_name(material.name, f"material_{actual:02d}")

            # Mirror dyt_ui.py: export the 128x4 band and its sampled $color2.
            band_target = dyt_dir / f"{material_name}_dyt.tga"
            color_target = dyt_dir / f"{material_name}.txt"
            try:
                _save_tga(band.image, band_target)
                color_target.parent.mkdir(parents=True, exist_ok=True)
                r, g, b = band.color2
                color_target.write_text(f'\"$color2\" \"{{{r} {g} {b}}}\"\n', encoding="utf-8")
                result.texture_files.append(band_target)
                result.metadata_files.append(color_target)
            except Exception as exc:
                result.warnings.append(f"DYT material {material.name}: {exc}")

            texture_index = material_texture_indices.get(material.name.casefold(), 0)
            if not (0 <= texture_index < len(decoded_base)) or decoded_base[texture_index] is None:
                continue
            try:
                composed, _requested, _actual = compose_dyt_texture(
                    decoded_base[texture_index], dyt_image, material
                )
                composed_target = material_dir / f"{material_name}.tga"
                _save_tga(composed, composed_target)
                result.texture_files.append(composed_target)
            except Exception as exc:
                result.warnings.append(f"Composed material {material.name}: {exc}")


def write_camera_animation_json(path: Path, ean_file, animation: EanAnimation, fps: float) -> None:
    payload = {
        "source": ean_file.path.name,
        "index": animation.index,
        "name": animation.name,
        "frame_count": animation.duration,
        "fps": fps,
        "animation_type": ean_file.animation_type,
        "nodes": [],
    }
    for node in animation.nodes:
        node_data = {
            "bone_index": node.bone_index,
            "bone_name": node.bone_name,
            "tracks": {},
        }
        for flag, track in sorted(node.tracks.items()):
            channel = {0: "position", 1: "rotation", 2: "scale_or_camera_properties"}.get(flag, f"track_{flag}")
            node_data["tracks"][channel] = [
                {"frame": key.frame, "value": list(key.value)}
                for key in track.keyframes
            ]
        payload["nodes"].append(node_data)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _find_ean(entry: LibraryEntry, suffix: str) -> Path | None:
    suffix = suffix.casefold()
    return next((a.path for a in entry.assets if a.path.name.casefold().endswith(suffix)), None)


def _export_ean_smd_set(
    ean_file,
    esk: EskFile,
    target_dir: Path,
    result: ExportResult,
    scale: float,
    prefix: str = "",
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for animation in ean_file.animations:
        target = target_dir / f"{animation.index:03d}_{prefix}{_safe_name(animation.name)}.smd"
        try:
            write_animation_smd(target, esk, animation, scale)
            result.animation_files.append(target)
        except Exception as exc:
            result.warnings.append(f"Animation {ean_file.path.name}:{animation.name}: {exc}")


def _local_bind_matrix(bone: EskBone) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    rotation = _quat_matrix(bone.rotation)
    scale = np.diag([float(bone.scale[0]), float(bone.scale[1]), float(bone.scale[2])])
    matrix[:3, :3] = rotation @ scale
    matrix[:3, 3] = np.asarray(bone.position[:3], dtype=np.float64)
    return matrix


def _global_bind_matrices(esk: EskFile) -> list[np.ndarray]:
    output: list[np.ndarray] = [np.eye(4, dtype=np.float64) for _ in esk.bones]
    root_bake = _ROOT_CONTAINER_BAKE.get(id(esk))
    for bone in esk.bones:
        local = _local_bind_matrix(bone)
        if bone.index == 0 and root_bake is not None:
            local = root_bake @ local
        output[bone.index] = local if bone.parent is None else output[bone.parent] @ local
    return output


def _build_combined_skeleton(
    main_esk: EskFile,
    secondary_esks: list[EskFile],
) -> tuple[EskFile, dict[Path, list[int]]]:
    """Merge physics skeletons into the character skeleton by bone name.

    Shared anchor bones such as b_C_Chest/b_C_Pelvis map to the main rig. Unique
    SCD bones are appended, preserving their local transforms and parenting.
    """
    bones: list[EskBone] = [
        EskBone(
            index=bone.index,
            name=bone.name,
            parent_index=bone.parent_index,
            child_index=bone.child_index,
            sibling_index=bone.sibling_index,
            ik_flag=bone.ik_flag,
            position=bone.position,
            rotation=bone.rotation,
            scale=bone.scale,
            absolute_matrix=bone.absolute_matrix,
        )
        for bone in main_esk.bones
    ]
    name_to_index = {bone.name.casefold(): bone.index for bone in bones}
    mappings: dict[Path, list[int]] = {}

    for secondary in secondary_esks:
        mapping = [-1] * len(secondary.bones)
        for bone in secondary.bones:
            key = bone.name.casefold()
            # Secondary EXP roots are containers, not useful independent bones.
            if bone.index == 0 and bone.name.upper().startswith("EXP_"):
                mapping[bone.index] = 0
                continue
            existing = name_to_index.get(key)
            if existing is not None:
                mapping[bone.index] = existing
                continue
            parent = 0 if bone.parent is None else mapping[bone.parent]
            if parent < 0:
                parent = 0
            index = len(bones)
            bones.append(EskBone(
                index=index,
                name=bone.name,
                parent_index=parent,
                child_index=0xFFFF,
                sibling_index=0xFFFF,
                ik_flag=bone.ik_flag,
                position=bone.position,
                rotation=bone.rotation,
                scale=bone.scale,
                absolute_matrix=bone.absolute_matrix,
            ))
            name_to_index[key] = index
            mapping[bone.index] = index
        mappings[secondary.path] = mapping

    combined = EskFile(
        path=main_esk.path,
        version=main_esk.version,
        endian=main_esk.endian,
        flags=main_esk.flags,
        skeleton_id=main_esk.skeleton_id,
        bones=bones,
    )
    return combined, mappings


def _strip_unused_exp_root(
    esk: EskFile,
    secondary_maps: dict[Path, list[int]],
) -> tuple[EskFile, dict[Path, list[int]]]:
    """Remove the unused character EXP_* container root from Source exports.

    XV2 character skeletons commonly place an unused ``EXP_<character>_000``
    bone above ``b_C_Base``. Source locomotion must operate on the real root,
    so the container is removed, every remaining bone is renumbered, and
    children of the removed container become true roots.
    """
    if not esk.bones:
        return esk, secondary_maps

    container = esk.bones[0]
    if not container.name.upper().startswith("EXP_"):
        return esk, secondary_maps
    if not any(bone.name.casefold() == "b_c_base" for bone in esk.bones[1:]):
        return esk, secondary_maps

    old_to_new = {old.index: new for new, old in enumerate(esk.bones[1:])}
    bones: list[EskBone] = []
    for old in esk.bones[1:]:
        old_parent = old.parent_index
        parent = 0xFFFF if old_parent in (0, 0xFFFF) else old_to_new.get(old_parent, 0xFFFF)
        child = old_to_new.get(old.child_index, 0xFFFF)
        sibling = old_to_new.get(old.sibling_index, 0xFFFF)
        bones.append(EskBone(
            index=old_to_new[old.index],
            name=old.name,
            parent_index=parent,
            child_index=child,
            sibling_index=sibling,
            ik_flag=old.ik_flag,
            position=old.position,
            rotation=old.rotation,
            scale=old.scale,
            absolute_matrix=old.absolute_matrix,
        ))

    remapped_secondary: dict[Path, list[int]] = {}
    for path, mapping in secondary_maps.items():
        remapped_secondary[path] = [
            old_to_new.get(index, 0) for index in mapping
        ]

    stripped = EskFile(
        path=esk.path,
        version=esk.version,
        endian=esk.endian,
        flags=esk.flags,
        skeleton_id=esk.skeleton_id,
        bones=bones,
    )
    # Preserve the removed container's transform instead of silently dropping
    # it. b_C_Base remains bone 0, but now has the same global orientation it
    # had beneath EXP_* in Xenoverse.
    _ROOT_CONTAINER_BAKE[id(stripped)] = _local_bind_matrix(container)
    return stripped, remapped_secondary


def _find_secondary_esk(entry: LibraryEntry, model_path: Path) -> Path | None:
    skeletons = [asset.path for asset in entry.assets if asset.kind == AssetKind.SKELETON]
    wanted = model_path.stem.casefold()
    exact = next((path for path in skeletons if path.stem.casefold() == wanted), None)
    if exact is not None:
        return exact
    # LOD physics meshes generally share the non-LOD secondary ESK.
    base = re.sub(r"_lod\d+$", "", wanted, flags=re.IGNORECASE)
    return next((path for path in skeletons if path.stem.casefold() == base), None)


def write_physics_reference_smd(
    path: Path,
    combined_esk: EskFile,
    secondary_esk: EskFile,
    secondary_to_combined: list[int],
    emd,
    scale: float,
    material_prefix: str = "",
) -> None:
    """Export a secondary mesh attached to the main bind hierarchy, plus outline shell."""
    secondary_by_name = {bone.name.casefold(): bone.index for bone in secondary_esk.bones}
    old_globals = _global_bind_matrices(secondary_esk)
    new_globals = _global_bind_matrices(combined_esk)
    corrections: list[np.ndarray] = []
    for old_index, new_index in enumerate(secondary_to_combined):
        try:
            corrections.append(new_globals[new_index] @ np.linalg.inv(old_globals[old_index]))
        except (IndexError, np.linalg.LinAlgError):
            corrections.append(np.eye(4, dtype=np.float64))

    lines = ["version 1", *_nodes_block(combined_esk), "skeleton", "time 0"]
    lines.extend(_skeleton_frame_lines(combined_esk, None, 0.0, scale))
    lines.extend(("end", "triangles"))

    for model in emd.models:
        for mesh in model.meshes:
            for submesh in mesh.submeshes:
                base_material = _safe_name(submesh.name, "material")
                material = f"{material_prefix}_{base_material}" if material_prefix else base_material
                positions = submesh.positions
                normals = submesh.normals
                uvs = submesh.uvs
                for group in submesh.triangle_groups:
                    palette_old = [secondary_by_name.get(name.casefold(), 0) for name in group.bone_names]
                    palette_new = [secondary_to_combined[index] if 0 <= index < len(secondary_to_combined) else 0 for index in palette_old]
                    indices = group.indices
                    for tri_start in range(0, len(indices) - 2, 3):
                        source_tri = (indices[tri_start], indices[tri_start + 1], indices[tri_start + 2])
                        converted_tri = (source_tri[0], source_tri[2], source_tri[1])
                        records = {}
                        for vi in source_tri:
                            if vi >= len(positions):
                                continue
                            blend_i = submesh.blend_indices[vi] if vi < len(submesh.blend_indices) else None
                            blend_w = submesh.blend_weights[vi] if vi < len(submesh.blend_weights) else None
                            influences: list[tuple[int, int, float]] = []
                            if blend_i is not None and blend_w is not None:
                                for slot in range(4):
                                    weight = float(blend_w[slot])
                                    local_index = int(blend_i[slot])
                                    if weight <= 1e-6:
                                        continue
                                    old_index = palette_old[local_index] if local_index < len(palette_old) else 0
                                    new_index = palette_new[local_index] if local_index < len(palette_new) else 0
                                    influences.append((old_index, new_index, weight))
                            if not influences:
                                influences = [(0, secondary_to_combined[0] if secondary_to_combined else 0, 1.0)]
                            total = sum(weight for _old, _new, weight in influences) or 1.0
                            influences = [(old, new, weight / total) for old, new, weight in influences]

                            source_position = np.array((*positions[vi], 1.0), dtype=np.float64)
                            source_normal = np.asarray(_normalise3(normals[vi] or (0.0, 0.0, 1.0)), dtype=np.float64)
                            moved_position = np.zeros(4, dtype=np.float64)
                            moved_normal = np.zeros(3, dtype=np.float64)
                            merged_links: dict[int, float] = {}
                            for old_index, new_index, weight in influences:
                                correction = corrections[old_index] if old_index < len(corrections) else np.eye(4)
                                moved_position += weight * (correction @ source_position)
                                try:
                                    normal_matrix = np.linalg.inv(correction[:3, :3]).T
                                except np.linalg.LinAlgError:
                                    normal_matrix = correction[:3, :3]
                                moved_normal += weight * (normal_matrix @ source_normal)
                                merged_links[new_index] = merged_links.get(new_index, 0.0) + weight

                            moved_normal = np.asarray(_normalise3(moved_normal), dtype=np.float64)
                            p = _vec_source(moved_position[:3], scale)
                            n = _normalise3(_vec_source(moved_normal, 1.0))
                            outline_position = moved_position[:3] + moved_normal * OUTLINE_WIDTH
                            outline_p = _vec_source(outline_position, scale)
                            outline_n = tuple(-value for value in n)
                            uv = uvs[vi] if vi < len(uvs) and uvs[vi] is not None else (0.0, 0.0)
                            links = sorted(merged_links.items())
                            parent = max(links, key=lambda pair: pair[1])[0]
                            records[vi] = (
                                _smd_vertex_line(parent, p, n, uv, links),
                                _smd_vertex_line(parent, outline_p, outline_n, uv, links),
                            )
                        if len(records) != 3:
                            continue
                        lines.append(material)
                        lines.extend(records[vi][0] for vi in converted_tri)
                        lines.append("outline")
                        lines.extend(records[vi][1] for vi in source_tri)
    lines.append("end")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _model_set_and_part(path: Path, character_code: str) -> tuple[str, str]:
    stem = path.stem
    match = re.match(rf"(?i)^{re.escape(character_code)}_(\d{{3}})_(.+)$", stem)
    if match:
        return match.group(1), match.group(2).casefold()
    match = re.search(r"(?i)(?:^|_)(\d{3})(?:_|$)", stem)
    subset = match.group(1) if match else "000"
    part = stem[match.end():] if match and match.end() < len(stem) else stem
    return subset, part.casefold()


def _mesh_ground_offset(model_paths: list[Path], scale: float) -> float:
    minimum_z: float | None = None
    for model_path in model_paths:
        try:
            emd = load_emd(model_path)
            for model in emd.models:
                for mesh in model.meshes:
                    for submesh in mesh.submeshes:
                        for position in submesh.positions:
                            z = _vec_source(position, scale)[2]
                            minimum_z = z if minimum_z is None else min(minimum_z, z)
        except Exception:
            continue
    return -minimum_z if minimum_z is not None else 0.0


def write_reference_animation_smd(path: Path, esk: EskFile, scale: float, ground_offset: float) -> None:
    lines = ["version 1", *_nodes_block(esk), "skeleton", "time 0"]
    frame_lines = _skeleton_frame_lines(esk, None, 0.0, scale)
    if frame_lines:
        parts = frame_lines[0].split()
        if len(parts) >= 7:
            parts[3] = f"{float(parts[3]) + ground_offset:.6f}"
            frame_lines[0] = " ".join(parts)
    lines.extend(frame_lines)
    lines.append("end")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _relative_smd(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def _find_animation_by_token(animation_files: list[Path], *tokens: str) -> Path | None:
    upper_tokens = tuple(token.upper() for token in tokens)
    for path in animation_files:
        if path.suffix.casefold() != ".smd":
            continue
        upper = path.stem.upper()
        if all(token in upper for token in upper_tokens):
            return path
    return None


def _strip_character_prefix(name: str, character_code: str) -> str:
    """Remove only the active character's ID prefix from an animation name."""
    code = _safe_name(character_code).upper()
    return re.sub(rf"^{re.escape(code)}_", "", name, flags=re.IGNORECASE)


def _sequence_name_from_smd(path: Path, character_code: str | None = None) -> str:
    name = re.sub(r"^\d{3}_", "", path.stem)
    if character_code:
        name = _strip_character_prefix(name, character_code)
    return _safe_name(name).upper()


def _animation_root_translation(
    esk: EskFile,
    animation: EanAnimation,
    scale: float,
    frame: float = 0.0,
) -> tuple[float, float, float]:
    """Return bone 0's exported Source-space translation for one animation frame."""
    frame_lines = _skeleton_frame_lines(esk, animation, frame, scale)
    if not frame_lines:
        return (0.0, 0.0, 0.0)
    parts = frame_lines[0].split()
    if len(parts) < 4:
        return (0.0, 0.0, 0.0)
    return (float(parts[1]), float(parts[2]), float(parts[3]))


def _walk_ground_offset(
    esk: EskFile,
    ean_file,
    scale: float,
) -> float:
    """Align GOK_BAS_WALK's root height to the normal grounded BAS_STAND pose.

    The previous implementation reused the mesh/reference ground offset. That
    value describes model-space mesh placement and is far too large when added
    directly to an animation root. Here we compare the exported root height of
    the shared walk against the character's grounded stand animation instead.
    """
    if ean_file is None:
        return 0.0

    walk = next(
        (anim for anim in ean_file.animations if "GOK_BAS_WALK" in anim.name.upper()),
        None,
    )
    if walk is None:
        return 0.0

    # Prefer the character's main grounded idle, not the two air-blend poses.
    stand = next(
        (
            anim for anim in ean_file.animations
            if re.search(r"(?:^|_)BAS_STAND_S$", anim.name.upper())
        ),
        None,
    )
    if stand is None:
        stand = next(
            (
                anim for anim in ean_file.animations
                if "BAS_STAND" in anim.name.upper() and not anim.name.upper().endswith(("_00_S", "_01_S"))
            ),
            None,
        )
    if stand is None:
        return 0.0

    walk_z = _animation_root_translation(esk, walk, scale)[2]
    stand_z = _animation_root_translation(esk, stand, scale)[2]
    return stand_z - walk_z


def _smd_time_count(path: Path) -> int:
    """Count exported SMD time records, matching qcgen.py's frame handling."""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return sum(1 for line in handle if line.lstrip().startswith("time "))
    except OSError:
        return 0


def _write_walkframes(handle, path: Path, axes: str = "LX LY") -> None:
    """Write the 32 evenly-spaced walkframe records used by qcgen.py."""
    total_frames = _smd_time_count(path)
    if total_frames <= 0:
        return
    increase = total_frames / 32.0
    handle.write("\n")
    for index in range(1, 33):
        frame = round(increase * index)
        if frame > total_frames:
            frame = total_frames
        if frame == 0:
            frame = 1
        handle.write(f"    walkframe {frame} {axes}\n")


def _path_key(path: Path, output_dir: Path) -> str:
    try:
        return _relative_smd(path, output_dir).casefold()
    except ValueError:
        return path.resolve().as_posix().casefold()


def _is_additive_sequence_name(name: str) -> bool:
    lower = name.casefold()
    return (
        lower.startswith("exp_")
        or lower.startswith("face")
        or lower.startswith("facialparts_")
        or lower.endswith("_a")
        or lower.endswith("_p")
    )


def _is_loop_sequence_name(name: str) -> bool:
    lower = name.casefold()
    loop = any(
        keyword in lower
        for keyword in (
            "loop", "cycle", "idle", "glide", "walk", "run", "sprint",
            "standby", "pose", "fall", "lp", "mrm_00", "dwn", "snt", "trm",
        )
    )
    if any(keyword in lower for keyword in ("melee", "attack", "_a", "turn")):
        loop = False
    return loop


def _animation_identity(path: Path) -> str:
    """Animation name without the numeric file ID prefix."""
    return re.sub(r"^\d{3}_", "", path.stem, flags=re.IGNORECASE)


def _air_pair_key(path: Path) -> tuple[str, bool]:
    identity = _animation_identity(path)
    is_air = identity.upper().endswith("_S")
    base = identity[:-2] if is_air else identity
    return base.casefold(), is_air


def _animation_index_from_smd(path: Path) -> int | None:
    match = re.match(r"^(\d{3})_", path.stem)
    return int(match.group(1)) if match else None


def _events_for_paths(
    paths: Iterable[Path],
    events_by_animation: dict[int, list[tuple[int, str]]] | None,
    events_by_path: dict[str, list[tuple[int, str]]] | None = None,
    output_dir: Path | None = None,
) -> list[tuple[int, str]]:
    """Resolve events for main and additional animation SMDs.

    Main character events use the EAN animation-index map. Additional EANs have
    their own index namespaces, so their events are keyed by the exported SMD
    path instead of being mixed into the main character map.
    """
    values: set[tuple[int, str]] = set()
    event_map = events_by_animation or {}
    path_map = events_by_path or {}
    for path in paths:
        path_key = _path_key(path, output_dir) if output_dir is not None else path.resolve().as_posix().casefold()
        path_events = path_map.get(path_key)
        if path_events is not None:
            values.update(path_events)
            continue
        index = _animation_index_from_smd(path)
        if index is not None:
            values.update(event_map.get(index, ()))
    return sorted(values, key=lambda item: (item[0], item[1]))


def _write_sequence_events(handle, events: Iterable[tuple[int, str]]) -> None:
    values = list(events)
    if not values:
        return
    handle.write("\n")
    for frame, text in values:
        safe = str(text).replace('"', "'")
        handle.write(f'    {{ event 1100 {int(frame)} "{safe}" }}\n')


def _write_generic_sequence(
    handle,
    paths: list[Path],
    output_dir: Path,
    character_code: str,
    events_by_animation: dict[int, list[tuple[int, str]]] | None = None,
    events_by_path: dict[str, list[tuple[int, str]]] | None = None,
    sequence_name: str | None = None,
) -> None:
    primary = next((path for path in paths if not _air_pair_key(path)[1]), paths[0])
    name = sequence_name or _safe_name(_strip_character_prefix(_air_pair_key(primary)[0], character_code)).upper()
    is_additive = _is_additive_sequence_name(name)
    is_loop = _is_loop_sequence_name(name)

    handle.write(f'\n$Sequence "{name}" {{\n')
    ordered = sorted(paths, key=lambda path: _air_pair_key(path)[1])
    for path in ordered:
        handle.write(f'    "{_relative_smd(path, output_dir)}"\n')
    if not is_additive:
        handle.write(f'    activity "ACT_{name}" 1\n')
    if is_loop:
        handle.write("    loop\n")
    if len(ordered) == 2 and any(_air_pair_key(path)[1] for path in ordered):
        handle.write("    blendwidth 2 blend air 0 1\n")

    if is_additive:
        handle.write(f'    subtract "{name}" 0\n')
        handle.write("    delta\n")
    elif not is_loop:
        _write_walkframes(handle, primary)
    elif any(keyword in name.casefold() for keyword in ("walk", "run", "sprint")):
        handle.write("    LX LY\n")
    _write_sequence_events(handle, _events_for_paths(ordered, events_by_animation or {}, events_by_path, output_dir))
    handle.write("}\n")


def _atc_shot_direction(path: Path) -> str | None:
    """Return F/L/B/R for directional ATC_SHOT and ATC_SHOT_*_LOOP animations.

    Neutral ``ATC_SHOT_N`` variants are intentionally excluded and continue
    through ordinary sequence generation. Optional trailing suffixes such as
    ``_LOOP`` and ``_S`` do not change the detected movement direction.
    """
    identity = _animation_identity(path).upper()
    match = re.search(
        r"(?:^|_)ATC_SHOT_(F|L|B|R)(?=$|_(?:LOOP(?:_S)?|S)$)",
        identity,
    )
    return match.group(1) if match else None


def _atc_shot_is_loop(path: Path) -> bool:
    """Return True when a directional ATC_SHOT animation is a LOOP variant."""
    return bool(re.search(r"(?:^|_)ATC_SHOT_[FLBR]_LOOP(?:_S)?$", _animation_identity(path), re.IGNORECASE))


def _atc_shot_group_name(path: Path, character_code: str) -> str:
    """Build the shared ATC_SHOT sequence name without any character prefix.

    Some animation sets use a character ID that differs from ``entry.code``
    (shared/common animation sources are the usual cause), so stripping only
    the active library-entry code is not reliable. Anchor on ``ATC_SHOT``
    itself and discard everything before it.
    """
    identity = _safe_name(_animation_identity(path)).upper()
    match = re.search(r"(?:^|_)(ATC_SHOT_(?:F|L|B|R)(?:_LOOP)?(?:_S)?)$", identity)
    if match:
        identity = match.group(1)
    else:
        identity = _strip_character_prefix(identity, character_code)
    identity = re.sub(r"_(?:F|L|B|R)(?:_LOOP)?(?:_S)?$", "", identity)
    return f"{identity}_LOOP" if _atc_shot_is_loop(path) else identity


def _write_atc_shot_blend_sequence(
    handle,
    paths_by_direction: dict[str, Path],
    output_dir: Path,
    sequence_name: str,
    events_by_animation: dict[int, list[tuple[int, str]]] | None = None,
) -> None:
    """Write one B/R/F/L/B directional blend sequence, matching BAS_BUKUU."""
    rotations = {"F": 0, "B": -180, "L": 90, "R": -90}
    animation_names: dict[str, str] = {}
    for direction in ("F", "B", "L", "R"):
        path = paths_by_direction[direction]
        animation_name = f"@{sequence_name}_{direction}"
        animation_names[direction] = animation_name
        handle.write(
            f'\n$Animation "{animation_name}" "{_relative_smd(path, output_dir)}" '
            f'loop LX LY rotateto {rotations[direction]} alignto BAS_STAND\n'
        )

    handle.write(f'$Sequence "{sequence_name}" {{\n')
    handle.write(f'    "{animation_names["B"]}"\n')
    handle.write(f'    "{animation_names["R"]}"\n')
    handle.write(f'    "{animation_names["F"]}"\n')
    handle.write(f'    "{animation_names["L"]}"\n')
    handle.write(f'    "{animation_names["B"]}"\n')
    handle.write(f'    activity "ACT_{sequence_name}" 1\n')
    handle.write('    blendwidth 5 blend move_yaw -180 180\n')
    _write_sequence_events(handle, _events_for_paths([paths_by_direction['F']], events_by_animation or {}))
    handle.write('}\n')


def write_shared_qci(
    output_dir: Path,
    character_code: str,
    animation_files: list[Path],
    events_by_animation: dict[int, list[tuple[int, str]]] | None = None,
    additional_events_by_path: dict[str, list[tuple[int, str]]] | None = None,
) -> Path:
    target = output_dir / "shared.qci"
    code = _safe_name(character_code).upper()
    body_files = sorted(
        (
            path for path in animation_files
            if path.suffix.casefold() == ".smd"
            and "/body/" in f"/{path.as_posix()}"
        ),
        key=lambda path: path.name.casefold(),
    )
    original_body_files = [path for path in body_files if not path.stem.upper().endswith("_WF")]
    wf_by_original = {
        path.stem[:-3].casefold(): path
        for path in body_files
        if path.stem.upper().endswith("_WF")
    }
    face_files = sorted(
        (path for path in animation_files if path.suffix.casefold() == ".smd" and "/face/" in f"/{path.as_posix()}"),
        key=lambda path: path.name.casefold(),
    )
    additional_root = output_dir / "animations" / "additional"
    additional_files = {
        path.resolve(): path
        for path in animation_files
        if path.suffix.casefold() == ".smd"
        and "/animations/additional/" in f"/{path.as_posix()}".casefold()
    }
    if additional_root.exists():
        for path in additional_root.rglob("*.smd"):
            additional_files.setdefault(path.resolve(), path)
    additional_body_files = sorted(additional_files.values(), key=lambda path: path.as_posix().casefold())
    used_body_paths: set[str] = set()

    def mark(path: Path | None) -> None:
        if path is not None:
            used_body_paths.add(_path_key(path, output_dir))

    def find_exact(token: str, *, air: bool | None = None) -> Path | None:
        token = token.upper()
        for path in original_body_files:
            identity = _animation_identity(path).upper()
            if identity != token:
                continue
            if air is None or identity.endswith("_S") == air:
                return path
        return None

    stand = find_exact(f"{code}_BAS_STAND") or _find_animation_by_token(original_body_files, "BAS_STAND")
    walk = _find_animation_by_token(original_body_files, "BAS_WALK")
    run = next((p for p in original_body_files if "BAS_RUN" in p.stem.upper() and "BUKUU" not in p.stem.upper()), None)
    stand_air = [
        _find_animation_by_token(original_body_files, "BAS_STAND_00_S"),
        next((p for p in original_body_files if _animation_identity(p).upper().endswith("BAS_STAND_S")), None),
        _find_animation_by_token(original_body_files, "BAS_STAND_01_S"),
    ]
    stand_air = [path for path in stand_air if path is not None]
    bukuu = {
        direction: _find_animation_by_token(original_body_files, f"BAS_BUKUU_{direction}")
        for direction in ("F", "B", "L", "R")
    }

    with target.open("w", encoding="utf-8", newline="\n") as f:
        f.write('$SurfaceProp "flesh"\n')
        f.write('$Contents "solid"\n')
        f.write(f'$CDMaterials "models/cpthazama/dbxv2/{code}/"\n')
        f.write('$CDMaterials "models/cpthazama/dbxv2/"\n\n')
        f.write('$fps 60\n')
        f.write('$JiggleBoneRealign -90 0 -90\n\n')
        f.write('$poseparameter "move_yaw" -180 180 wrap\n\n')
        f.write('$sequence "ref" "animations/ref.smd"\n\n')

        if face_files:
            blink = next((path for path in face_files if "FCE_BHV_BLINK" in path.stem.upper()), None)
            f.write('$weightlist "EMO" {\n    "f_C_FaceRoot" 1\n}\n\n')
            f.write('$Animation "EMO_NONE" "animations/ref.smd" frame 0 0 subtract "ref" 0 weightlist EMO\n')
            if blink is not None:
                f.write(
                    f'$Animation "EMO_BLINK" "{_relative_smd(blink, output_dir)}" '
                    'frame 0 0 subtract "ref" 0 weightlist EMO\n'
                )
                f.write('$sequence "FCE_BLINK" {\n')
                f.write('    "EMO_NONE"\n    "EMO_BLINK"\n')
                f.write('    blendcenter ref\n')
                f.write('    blendwidth 2 blend "blink" 0 1\n')
                f.write('    subtract FCE_BLINK 0\n    delta\n    autoplay\n    hidden\n}\n\n')

            emotion_paths = [path for path in face_files if path is not blink]
            emotion_names: list[str] = []
            for path in emotion_paths:
                seq = _sequence_name_from_smd(path, code)
                short = re.sub(rf"^{re.escape(code)}_FCE_(?:EMO_)?", "", seq, flags=re.IGNORECASE)
                anim = f"EMO_{short}"
                emotion_names.append(anim)
                f.write(
                    f'$animation "{anim}" "{_relative_smd(path, output_dir)}" '
                    'subtract ref 0 weightlist EMO\n'
                )
            if emotion_names:
                f.write('\n$Sequence "FCE" {\n')
                for anim in emotion_names:
                    f.write(f'    "{anim}"\n')
                f.write('    delta\n    autoplay\n    hidden\n')
                f.write(f'    blendwidth {len(emotion_names)} blend FCE 0 {len(emotion_names)-1}\n')
                f.write('}\n')

        if stand:
            mark(stand)
            f.write('\n$Sequence "BAS_STAND" {\n')
            f.write(f'    "{_relative_smd(stand, output_dir)}"\n')
            f.write('    activity "ACT_IDLE" 1\n    loop\n')
            _write_sequence_events(f, _events_for_paths([stand], events_by_animation))
            f.write('}\n')

        if len(stand_air) == 3:
            for path in stand_air:
                mark(path)
            f.write('\n$Sequence "BAS_STAND_AIR" {\n')
            for path in stand_air:
                f.write(f'    "{_relative_smd(path, output_dir)}"\n')
            f.write('    activity "ACT_IDLE_STIMULATED" 1\n')
            f.write('    loop\n    blendwidth 3 blend aim_pitch -90 90\n')
            _write_sequence_events(f, _events_for_paths([stand_air[1]], events_by_animation))
            f.write('}\n')

        if walk:
            mark(walk)
            wf = wf_by_original.get(walk.stem.casefold())
            if wf:
                mark(wf)
            used = wf or walk
            f.write(f'\n$Animation "@BAS_WALK" "{_relative_smd(used, output_dir)}" loop LX LY rotateto 0 alignto BAS_STAND\n')
            f.write('$Sequence "BAS_WALK" {\n    "@BAS_WALK"\n    activity "ACT_WALK" 1\n')
            _write_sequence_events(f, _events_for_paths([walk], events_by_animation))
            f.write('}\n')

        if run:
            mark(run)
            wf = wf_by_original.get(run.stem.casefold())
            if wf:
                mark(wf)
            used = wf or run
            f.write(f'\n$Animation "@BAS_RUN" "{_relative_smd(used, output_dir)}" loop LX LY rotateto 0 alignto BAS_STAND\n')
            f.write('$Sequence "BAS_RUN" {\n    "@BAS_RUN"\n    activity "ACT_RUN" 1\n')
            _write_sequence_events(f, _events_for_paths([run], events_by_animation))
            f.write('}\n')

        if all(bukuu.values()):
            rotations = {"F": 0, "B": -180, "L": 90, "R": -90}
            for direction in ("F", "B", "L", "R"):
                path = bukuu[direction]
                mark(path)
                f.write(
                    f'$Animation "@BAS_BUKUU_{direction}" "{_relative_smd(path, output_dir)}" '
                    f'loop LX LY rotateto {rotations[direction]} alignto BAS_STAND origin 0 0 -10\n'
                )
            f.write('$Sequence "BAS_BUKUU" {\n')
            f.write('    "@BAS_BUKUU_B"\n    "@BAS_BUKUU_R"\n    "@BAS_BUKUU_F"\n')
            f.write('    "@BAS_BUKUU_L"\n    "@BAS_BUKUU_B"\n')
            f.write('    activity "ACT_WALK_STIMULATED" 1\n')
            f.write('    blendwidth 5 blend move_yaw -180 180\n}\n')

            if all(wf_by_original.get(bukuu[d].stem.casefold()) for d in ("F", "B", "L", "R")):
                for direction in ("F", "B", "L", "R"):
                    wf = wf_by_original[bukuu[direction].stem.casefold()]
                    mark(wf)
                    f.write(
                        f'$Animation "@BAS_BUKUU_WF_{direction}" "{_relative_smd(wf, output_dir)}" '
                        f'loop LX LY rotateto {rotations[direction]} alignto BAS_STAND\n'
                    )
                f.write('$Sequence "BAS_BUKUU_WF" {\n')
                f.write('    "@BAS_BUKUU_WF_B"\n    "@BAS_BUKUU_WF_R"\n    "@BAS_BUKUU_WF_F"\n')
                f.write('    "@BAS_BUKUU_WF_L"\n    "@BAS_BUKUU_WF_B"\n')
                f.write('    activity "ACT_RUN_STIMULATED" 1\n')
                f.write('    blendwidth 5 blend move_yaw -180 180\n}\n')

        # Directional ATC_SHOT animations are emitted as shared B/R/F/L/B
        # move_yaw blend sequences, matching the BAS_BUKUU setup. The normal
        # and _LOOP sets are grouped separately. Neutral (N) shots remain
        # ordinary singular sequence blocks.
        atc_shot_groups: dict[str, dict[str, Path]] = {}
        for path in original_body_files:
            if _path_key(path, output_dir) in used_body_paths:
                continue
            direction = _atc_shot_direction(path)
            if direction is None:
                continue
            group_name = _atc_shot_group_name(path, code)
            atc_shot_groups.setdefault(group_name, {}).setdefault(direction, path)

        for group_name in sorted(atc_shot_groups):
            directions = atc_shot_groups[group_name]
            # Only consume the set when all four directions exist; incomplete
            # sets fall through to generic generation rather than disappearing.
            if not all(direction in directions for direction in ("F", "B", "L", "R")):
                continue
            for path in directions.values():
                mark(path)
            _write_atc_shot_blend_sequence(f, directions, output_dir, group_name, events_by_animation)

        # Pair matching ground/_S animations into one air blend and consume both.
        grouped: dict[str, list[Path]] = {}
        for path in original_body_files:
            if _path_key(path, output_dir) in used_body_paths:
                continue
            key, _is_air = _air_pair_key(path)
            grouped.setdefault(key, []).append(path)

        for key in sorted(grouped):
            paths = grouped[key]
            ground = [path for path in paths if not _air_pair_key(path)[1]]
            air = [path for path in paths if _air_pair_key(path)[1]]
            selected = [ground[0], air[0]] if ground and air else [paths[0]]
            for path in selected:
                mark(path)
            _write_generic_sequence(f, selected, output_dir, code, events_by_animation)

        # Additional loaded EAN animations already live under
        # animations/additional/<entry>/<ean>/. Include those exact exported
        # SMDs in shared.qci and attach events from their own BAC namespace.
        if additional_body_files:
            f.write("\n// Additional loaded animations\n")
            used_sequence_names = {
                _safe_name(_strip_character_prefix(_air_pair_key(path)[0], code)).upper()
                for path in original_body_files
            }
            by_folder: dict[Path, list[Path]] = {}
            for path in additional_body_files:
                by_folder.setdefault(path.parent, []).append(path)

            for folder in sorted(by_folder, key=lambda value: value.as_posix().casefold()):
                grouped_additional: dict[str, list[Path]] = {}
                for path in by_folder[folder]:
                    key, _is_air = _air_pair_key(path)
                    grouped_additional.setdefault(key, []).append(path)

                for key in sorted(grouped_additional):
                    paths = grouped_additional[key]
                    ground = [path for path in paths if not _air_pair_key(path)[1]]
                    air = [path for path in paths if _air_pair_key(path)[1]]
                    selected = [ground[0], air[0]] if ground and air else [paths[0]]
                    base_name = _safe_name(_strip_character_prefix(_air_pair_key(selected[0])[0], code)).upper()
                    sequence_name = base_name
                    if sequence_name in used_sequence_names:
                        folder_tag = _safe_name(folder.name).upper()
                        sequence_name = f"{folder_tag}_{base_name}"
                        suffix = 2
                        while sequence_name in used_sequence_names:
                            sequence_name = f"{folder_tag}_{base_name}_{suffix}"
                            suffix += 1
                    used_sequence_names.add(sequence_name)
                    _write_generic_sequence(
                        f, selected, output_dir, code, {}, additional_events_by_path, sequence_name
                    )
    return target

def _smd_scd_bones(paths: Iterable[Path]) -> list[str]:
    """Collect unique SCD_* bone names from exported mesh/physics SMD node blocks."""
    bones: dict[str, str] = {}
    node_pattern = re.compile(r'^\s*\d+\s+"([^"]+)"\s+-?\d+\s*$')

    for path in paths:
        try:
            in_nodes = False
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not in_nodes:
                        if line.casefold() == "nodes":
                            in_nodes = True
                        continue
                    if line.casefold() == "end":
                        break
                    match = node_pattern.match(raw_line)
                    if not match:
                        continue
                    name = match.group(1)
                    if name.upper().startswith("SCD_"):
                        bones.setdefault(name.casefold(), name)
        except OSError:
            continue

    return sorted(bones.values(), key=str.casefold)


def _write_jigglebones(handle, bones: Iterable[str]) -> None:
    for bone in bones:
        safe_bone = bone.replace('"', "'")
        handle.write(
            f'$jigglebone "{safe_bone}" {{ is_flexible {{ length 30 tip_mass 40 '
            'pitch_stiffness 180 pitch_damping 15 yaw_stiffness 180 yaw_damping 15 '
            'along_stiffness 180 along_damping 0 angle_constraint 39.999999 } }\n'
        )


def write_subset_qcs(output_dir: Path, character_code: str, mesh_files: list[Path]) -> list[Path]:
    code = _safe_name(character_code).upper()
    entries: dict[str, dict[str, Path]] = {}
    for path in mesh_files:
        # Source models do not need Xenoverse's alternate LOD mesh exports.
        if re.search(r"_lod\d+(?:$|_)", path.stem, flags=re.IGNORECASE):
            continue
        subset, part = _model_set_and_part(path, code)
        entries.setdefault(subset, {})[part] = path
    if not entries:
        return []
    base_subset = "000" if "000" in entries else sorted(entries)[0]
    base_parts = entries[base_subset]
    outputs: list[Path] = []
    for subset in sorted(entries):
        resolved = dict(base_parts)
        resolved.update(entries[subset])
        target = output_dir / f"mdl_{subset}.qc"
        with target.open("w", encoding="utf-8", newline="\n") as f:
            f.write(f'$modelname "cpthazama/xv2/{code}/{subset}.mdl"\n\n')
            f.write('$maxverts 65536\n$maxbones 300\n\n')
            resolved_paths = sorted(resolved.values(), key=lambda p: p.name.casefold())
            for path in resolved_paths:
                f.write(f'$model "model" "{_relative_smd(path, output_dir)}"\n')

            scd_bones = _smd_scd_bones(resolved_paths)
            if scd_bones:
                f.write('\n')
                _write_jigglebones(f, scd_bones)
            f.write('\n$include "shared.qci"\n')
        outputs.append(target)
    return outputs


def write_patch_vmts(output_dir: Path, character_code: str) -> list[Path]:
    """Flatten converted textures/VMTs without costume-name collisions.

    Xenoverse commonly reuses material names across costume sets. Prefix every
    flattened material with its three-digit costume subset so the generated
    Source material name matches the prefixed material written into each SMD.
    """
    code = _safe_name(character_code).upper()
    outputs: list[Path] = []
    texture_root = output_dir / "textures"
    material_tgas = sorted(texture_root.glob("**/materials/*.tga"))
    seen: set[str] = set()

    for source_tga in material_tgas:
        # Layout produced by export_textures:
        # textures/<character_part_stem>/materials/<material>.tga
        try:
            part_stem = source_tga.parent.parent.name
        except IndexError:
            part_stem = ""
        subset, _part = _model_set_and_part(Path(part_stem), code)
        base_material = _safe_name(source_tga.stem)
        material = f"{subset}_{base_material}"
        key = material.casefold()
        if key in seen:
            continue
        seen.add(key)

        flat_tga = texture_root / f"{material}.tga"
        if source_tga != flat_tga:
            shutil.copy2(source_tga, flat_tga)
            outputs.append(flat_tga)

        target = texture_root / f"{material}.vmt"
        texture_path = f"models/cpthazama/dbxv2/{code}/{material}"
        target.write_text(
            '"Patch"\n{\n'
            '    include "materials/models/cpthazama/dbxv2/patch_common.vmt"\n'
            '    replace\n    {\n'
            f'        "$basetexture"             "{texture_path}"\n'
            f'        "$detail"                  "{texture_path}"\n'
            '    }\n}\n',
            encoding="utf-8",
            newline="\n",
        )
        outputs.append(target)
    return outputs


def generate_source_compile_files(
    output_dir: Path,
    character_code: str,
    esk: EskFile,
    source_models: list[Path],
    result: ExportResult,
    scale: float,
    events_by_animation: dict[int, list[tuple[int, str]]] | None = None,
    additional_events_by_path: dict[str, list[tuple[int, str]]] | None = None,
) -> None:
    """Generate Source compile metadata without one stage blocking the others."""
    if source_models:
        try:
            ground_offset = _mesh_ground_offset(source_models, scale)
            ref_target = output_dir / "animations" / "ref.smd"
            write_reference_animation_smd(ref_target, esk, scale, ground_offset)
            result.animation_files.append(ref_target)
        except Exception as exc:
            result.warnings.append(f"Reference SMD generation: {exc}")

    try:
        qcs = write_subset_qcs(output_dir, character_code, result.mesh_files)
        result.metadata_files.extend(qcs)
    except Exception as exc:
        result.warnings.append(f"QC generation: {exc}")

    try:
        qci = write_shared_qci(
            output_dir, character_code, result.animation_files, events_by_animation or {}, additional_events_by_path or {}
        )
        result.metadata_files.append(qci)
    except Exception as exc:
        result.warnings.append(f"QCI generation: {exc}")

    try:
        generated_materials = write_patch_vmts(output_dir, character_code)
        result.texture_files.extend(
            path for path in generated_materials if path.suffix.casefold() == ".tga"
        )
        result.metadata_files.extend(
            path for path in generated_materials if path.suffix.casefold() == ".vmt"
        )
    except Exception as exc:
        result.warnings.append(f"VMT generation: {exc}")

def export_character(
    entry: LibraryEntry,
    output_root: str | Path,
    options: ExportOptions,
    progress: Callable[[str, int, int], None] | None = None,
    additional_entries: list[LibraryEntry] | None = None,
) -> ExportResult:
    additional_entries = additional_entries or []
    export_name = str(entry.metadata.get("export_name", entry.code))
    output_dir = Path(output_root) / _safe_name(export_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = ExportResult(output_dir)

    explicit_skeleton = str(entry.metadata.get("main_skeleton", ""))
    skeleton_asset = Path(explicit_skeleton) if explicit_skeleton and Path(explicit_skeleton).is_file() else None
    race_code = str(entry.metadata.get("race", entry.code))
    if skeleton_asset is None:
        skeleton_asset = next((a.path for a in entry.assets if a.kind == AssetKind.SKELETON and a.path.stem.casefold() == f"{race_code}_000".casefold()), None)
    if skeleton_asset is None:
        skeleton_asset = next((a.path for a in entry.assets if a.kind == AssetKind.SKELETON), None)
    if skeleton_asset is None:
        raise ValueError("No ESK skeleton was found for this character.")
    main_esk = load_esk(skeleton_asset)
    ground_stride_scale = _ground_stride_scale(entry, main_esk)

    physics_models = [asset.path for asset in entry.assets if _is_physics_model(asset)]
    secondary_by_model: dict[Path, EskFile] = {}
    secondary_unique: dict[Path, EskFile] = {}
    for model_path in physics_models:
        secondary_path = _find_secondary_esk(entry, model_path)
        if secondary_path is None:
            result.warnings.append(f"Physics mesh {model_path.name}: no matching secondary ESK")
            continue
        try:
            secondary = secondary_unique.get(secondary_path) or load_esk(secondary_path)
            secondary_unique[secondary_path] = secondary
            secondary_by_model[model_path] = secondary
        except Exception as exc:
            result.warnings.append(f"Physics skeleton {secondary_path.name}: {exc}")

    esk, secondary_maps = _build_combined_skeleton(main_esk, list(secondary_unique.values()))
    esk, secondary_maps = _strip_unused_exp_root(esk, secondary_maps)

    explicit_animation = str(entry.metadata.get("main_animation", ""))
    main_ean_path = Path(explicit_animation) if explicit_animation and Path(explicit_animation).is_file() else None
    if main_ean_path is None:
        main_ean_path = next((a.path for a in entry.assets if a.kind == AssetKind.ANIMATION and a.path.name.casefold() == f"{race_code}.ean".casefold()), None)
    if main_ean_path is None:
        main_ean_path = next((a.path for a in entry.assets if a.kind == AssetKind.ANIMATION and ".fce." not in a.path.name.casefold() and ".cam." not in a.path.name.casefold()), None)
    ean = load_ean(main_ean_path) if main_ean_path else None

    models = [a.path for a in entry.assets if a.kind == AssetKind.MODEL and not _is_physics_model(a)]
    # GOK_BAS_WALK has a small vertical root offset. Align it against the
    # character's regular grounded stand pose rather than using the much larger
    # mesh/reference offset.
    # walk_ground_offset = _walk_ground_offset(esk, ean, options.source_scale)
    walk_ground_offset = 0
    additional_animation_count = 0
    additional_events_by_path: dict[str, list[tuple[int, str]]] = {}
    if options.animations:
        for additional_entry in additional_entries:
            for asset in additional_entry.assets:
                if asset.kind not in (AssetKind.ANIMATION, AssetKind.CAMERA):
                    continue
                try:
                    additional_animation_count += len(load_ean(asset.path).animations)
                except Exception:
                    pass
    total = (((len(models) + len(physics_models)) if options.meshes else 0)
             + (len(ean.animations) if options.animations and ean else 0)
             + additional_animation_count)
    done = 0

    if options.meshes:
        mesh_dir = output_dir / "meshes"
        _reset_directory(mesh_dir)
        for model_path in models:
            if progress:
                progress(f"Exporting mesh {model_path.name}", done, max(total, 1))
            try:
                emd = load_emd(model_path)
                target = mesh_dir / f"{_safe_name(model_path.stem)}.smd"
                subset, _part = _model_set_and_part(model_path, entry.code)
                write_reference_smd(
                    target, esk, emd, options.source_scale, material_prefix=subset
                )
                result.mesh_files.append(target)
            except Exception as exc:
                result.warnings.append(f"Mesh {model_path.name}: {exc}")
            done += 1

        physics_dir = mesh_dir / "physics"
        physics_dir.mkdir(exist_ok=True)
        for model_path in physics_models:
            if progress:
                progress(f"Exporting physics mesh {model_path.name}", done, max(total, 1))
            secondary = secondary_by_model.get(model_path)
            if secondary is None:
                done += 1
                continue
            try:
                emd = load_emd(model_path)
                target = physics_dir / f"{_safe_name(model_path.stem)}.smd"
                write_physics_reference_smd(
                    target, esk, secondary, secondary_maps[secondary.path], emd,
                    options.source_scale, material_prefix=_model_set_and_part(model_path, entry.code)[0]
                )
                result.mesh_files.append(target)
            except Exception as exc:
                result.warnings.append(f"Physics mesh {model_path.name}: {exc}")
            done += 1

    if options.animations:
        anim_root = output_dir / "animations"

        if ean is not None:
            body_dir = anim_root / "body"
            body_dir.mkdir(parents=True, exist_ok=True)
            for animation in ean.animations:
                if progress:
                    progress(f"Exporting body animation {animation.name}", done, max(total, 1))
                try:
                    target = body_dir / f"{animation.index:03d}_{_safe_name(animation.name)}.smd"
                    # Always preserve the untouched XV2 animation. Locomotion
                    # root motion is written to a separate _WF copy so its root
                    # rotation remains byte-for-byte identical to the original.
                    root_offset = (
                        (0.0, 0.0, walk_ground_offset)
                        if "GOK_BAS_WALK" in animation.name.upper()
                        else None
                    )
                    write_animation_smd(
                        target,
                        esk,
                        animation,
                        options.source_scale,
                        root_offset=root_offset,
                    )
                    result.animation_files.append(target)
                    root_motion = _body_root_motion(animation.name, ground_stride_scale) or _bukuu_root_motion(animation.name)
                    if root_motion is not None:
                        wf_target = target.with_name(f"{target.stem}_WF.smd")
                        _write_root_motion_copy(target, wf_target, root_motion)
                        result.animation_files.append(wf_target)
                except Exception as exc:
                    result.warnings.append(f"Body animation {animation.name}: {exc}")
                done += 1

        face_path = _find_ean(entry, ".fce.ean")
        lip_path = _find_ean(entry, ".fce.lip.ean")
        camera_path = _find_ean(entry, ".cam.ean")

        if face_path is not None:
            try:
                _export_ean_smd_set(load_ean(face_path), esk, anim_root / "face", result, options.source_scale)
            except Exception as exc:
                result.warnings.append(f"Face animation file {face_path.name}: {exc}")

        if lip_path is not None:
            try:
                _export_ean_smd_set(load_ean(lip_path), esk, anim_root / "lip", result, options.source_scale)
            except Exception as exc:
                result.warnings.append(f"Lip animation file {lip_path.name}: {exc}")

        if camera_path is not None:
            try:
                camera_ean = load_ean(camera_path)
                camera_dir = anim_root / "camera"
                camera_dir.mkdir(parents=True, exist_ok=True)
                for animation in camera_ean.animations:
                    target = camera_dir / f"{animation.index:03d}_{_safe_name(animation.name)}.json"
                    write_camera_animation_json(target, camera_ean, animation, options.fps)
                    result.animation_files.append(target)
            except Exception as exc:
                result.warnings.append(f"Camera animation file {camera_path.name}: {exc}")

        # Skill/additional EANs are exported alongside the character without
        # replacing or merging the base EAN. Their folder and source EAN names
        # are preserved to prevent collisions between common animation names.
        additional_root = anim_root / "additional"
        for additional_entry in additional_entries:
            for asset in additional_entry.assets:
                if asset.kind not in (AssetKind.ANIMATION, AssetKind.CAMERA):
                    continue
                try:
                    additional_ean = load_ean(asset.path)
                    is_camera = (
                        getattr(additional_ean, "is_camera", False)
                        or asset.kind == AssetKind.CAMERA
                        or ".cam." in asset.path.name.casefold()
                    )
                    target_dir = additional_root / _safe_name(additional_entry.code) / _safe_name(asset.path.stem)
                    target_dir.mkdir(parents=True, exist_ok=True)

                    additional_event_map: dict[int, list[tuple[int, str]]] = {}
                    if not is_camera and options.events:
                        frame_counts = {anim.index: max(1, int(anim.duration)) for anim in additional_ean.animations}
                        for bac_asset in additional_entry.assets:
                            if bac_asset.path.suffix.casefold() != ".bac":
                                continue
                            try:
                                parsed = animation_events(
                                    load_bac(bac_asset.path),
                                    animation_frame_counts=frame_counts,
                                )
                                for anim_index, values in parsed.items():
                                    existing = additional_event_map.setdefault(anim_index, [])
                                    existing.extend(value for value in values if value not in existing)
                            except Exception as exc:
                                result.warnings.append(
                                    f"Additional BAC {additional_entry.code}/{bac_asset.path.name}: {exc}"
                                )

                    for animation in additional_ean.animations:
                        if progress:
                            progress(
                                f"Exporting additional {'camera' if is_camera else 'body'} animation "
                                f"{additional_entry.code}: {animation.name}",
                                done,
                                max(total, 1),
                            )
                        if is_camera:
                            target = target_dir / f"{animation.index:03d}_{_safe_name(animation.name)}.json"
                            write_camera_animation_json(target, additional_ean, animation, options.fps)
                        else:
                            target = target_dir / f"{animation.index:03d}_{_safe_name(animation.name)}.smd"
                            write_animation_smd(target, esk, animation, options.source_scale)
                        result.animation_files.append(target)
                        if not is_camera:
                            values = additional_event_map.get(animation.index)
                            if values:
                                additional_events_by_path[_path_key(target, output_dir)] = list(values)
                        done += 1
                except Exception as exc:
                    result.warnings.append(
                        f"Additional animation file {additional_entry.code}/{asset.path.name}: {exc}"
                    )

    if options.textures:
        if progress:
            progress("Converting EMB textures", done, max(total, 1))
        export_textures(entry, output_dir / "textures", result)

    bac_files = [asset.path for asset in entry.assets if asset.path.suffix.casefold() == ".bac"]
    player_bac = next((
        path for path in bac_files
        if path.name.casefold().endswith("_player.bac")
    ), None)
    events_by_animation: dict[int, list[tuple[int, str]]] = {}
    if options.events and player_bac is not None and ean is not None:
        try:
            frame_counts = {animation.index: max(1, int(animation.duration)) for animation in ean.animations}
            events_by_animation = animation_events(
                load_bac(player_bac),
                animation_frame_counts=frame_counts,
            )
        except Exception as exc:
            result.warnings.append(f"BAC event parsing {player_bac.name}: {exc}")

    for additional_entry in additional_entries:
        bac_files.extend(
            asset.path for asset in additional_entry.assets
            if asset.path.suffix.casefold() == ".bac"
        )
    events_dir = output_dir / "events"
    event_target = events_dir / "animation_events.json"
    if options.events and bac_files:
        events_dir.mkdir(exist_ok=True)
        write_event_manifest(event_target, bac_files)
        result.metadata_files.append(event_target)
    else:
        if events_dir.exists():
            shutil.rmtree(events_dir)

    try:
        generate_source_compile_files(
            output_dir, entry.code, esk, models, result, options.source_scale,
            events_by_animation, additional_events_by_path
        )
    except Exception as exc:
        result.warnings.append(f"QC/VMT generation: {exc}")

    summary = {
        "exporter_version": EXPORTER_VERSION,
        "character": entry.code,
        "display_name": entry.display_name,
        "is_cac": bool(entry.metadata.get("cac")),
        "cac_profile": entry.metadata.get("cac_profile") if entry.metadata.get("cac") else None,
        "additional_animation_sets": [
            {"code": extra.code, "path": str(extra.path)} for extra in additional_entries
        ],
        "skeleton": skeleton_asset.name,
        "coordinate_conversion": "XV2 (X forward, Y up, Z right) -> Source (X forward, Y right, Z up)",
        "source_scale": options.source_scale,
        "mesh_files": [str(p.relative_to(output_dir)) for p in result.mesh_files],
        "animation_files": [str(p.relative_to(output_dir)) for p in result.animation_files],
        "texture_files": [str(p.relative_to(output_dir)) for p in result.texture_files],
        "metadata_files": [str(p.relative_to(output_dir)) for p in result.metadata_files],
        "warnings": result.warnings,
    }
    summary_path = output_dir / "export_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    result.metadata_files.append(summary_path)
    if progress:
        progress("Export complete", max(total, 1), max(total, 1))
    return result
