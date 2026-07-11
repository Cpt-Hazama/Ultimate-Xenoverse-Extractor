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
from .parsers import load_ean, load_emd, load_esk
from .parsers.ean import EanAnimation, EanNode
from .parsers.esk import EskBone, EskFile

# Xenoverse assets are authored at roughly meter scale. Source character assets
# are conventionally authored in inches/Hammer units.
EXPORTER_VERSION = "0.1.0"
SOURCE_SCALE = 42.41732283464567

# XV2 character space: +X forward, +Y up, +Z right
# Source/3ds Max SMD space: +X forward, +Y right, +Z up
# Swap XV2 Y/Z while preserving forward. This is a reflection, so triangle
# winding is reversed exactly once below. The previous mapping assumed XV2 +Z
# was forward, which rotated exported characters 90 degrees around the up axis.
_C = np.array(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)), dtype=np.float64)
_C_INV = np.linalg.inv(_C)


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
    lines: list[str] = []
    for bone in esk.bones:
        node = by_name.get(bone.name) if by_name else by_index.get(bone.index)
        pos, rot, _scale = _sample_node(node, frame, bone.position, bone.rotation, bone.scale)
        px, py, pz = _vec_source(pos, scale)
        rx, ry, rz = _matrix_to_euler_xyz(_quat_source_matrix(rot))
        lines.append(f"{bone.index} {px:.6f} {py:.6f} {pz:.6f} {rx:.6f} {ry:.6f} {rz:.6f}")
    return lines


def write_reference_smd(path: Path, esk: EskFile, emd, scale: float) -> None:
    bone_by_name = {bone.name: bone.index for bone in esk.bones}
    lines = ["version 1", *_nodes_block(esk), "skeleton", "time 0"]
    lines.extend(_skeleton_frame_lines(esk, None, 0.0, scale))
    lines.extend(("end", "triangles"))

    for model in emd.models:
        for mesh in model.meshes:
            for submesh in mesh.submeshes:
                material = _safe_name(submesh.name, "material")
                positions = submesh.positions
                normals = submesh.normals
                uvs = submesh.uvs
                for group in submesh.triangle_groups:
                    palette = [bone_by_name.get(name, 0) for name in group.bone_names]
                    indices = group.indices
                    for tri_start in range(0, len(indices) - 2, 3):
                        # Coordinate conversion changes handedness, so reverse once.
                        tri = (indices[tri_start], indices[tri_start + 2], indices[tri_start + 1])
                        lines.append(material)
                        for vi in tri:
                            if vi >= len(positions):
                                continue
                            p = _vec_source(positions[vi], scale)
                            n = _vec_source(normals[vi] or (0.0, 0.0, 1.0), 1.0)
                            nl = math.sqrt(sum(v*v for v in n)) or 1.0
                            n = tuple(v / nl for v in n)
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
                            link_text = " ".join(f"{idx} {weight:.6f}" for idx, weight in links)
                            lines.append(
                                f"{parent} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                                f"{n[0]:.6f} {n[1]:.6f} {n[2]:.6f} {uv[0]:.6f} {1.0-uv[1]:.6f} "
                                f"{len(links)} {link_text}"
                            )
    lines.append("end")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_animation_smd(path: Path, esk: EskFile, animation: EanAnimation, scale: float) -> None:
    lines = ["version 1", *_nodes_block(esk), "skeleton"]
    duration = max(1, int(animation.duration))
    for frame in range(duration):
        lines.append(f"time {frame}")
        lines.extend(_skeleton_frame_lines(esk, animation, float(frame), scale))
    lines.extend(("end",))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


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
    for bone in esk.bones:
        local = _local_bind_matrix(bone)
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
) -> None:
    """Export a secondary mesh attached to the main bind hierarchy.

    Vertices are rebased from the secondary ESK bind space into the merged main
    skeleton bind space. This is the attachment correction missing from the old
    converter for hair, skirts, aprons, capes, and similar SCD meshes.
    """
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
                material = _safe_name(submesh.name, "material")
                positions = submesh.positions
                normals = submesh.normals
                uvs = submesh.uvs
                for group in submesh.triangle_groups:
                    palette_old = [secondary_by_name.get(name.casefold(), 0) for name in group.bone_names]
                    palette_new = [
                        secondary_to_combined[index] if 0 <= index < len(secondary_to_combined) else 0
                        for index in palette_old
                    ]
                    indices = group.indices
                    for tri_start in range(0, len(indices) - 2, 3):
                        tri = (indices[tri_start], indices[tri_start + 2], indices[tri_start + 1])
                        lines.append(material)
                        for vi in tri:
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
                            source_normal = np.asarray(normals[vi] or (0.0, 0.0, 1.0), dtype=np.float64)
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

                            p = _vec_source(moved_position[:3], scale)
                            n = _vec_source(moved_normal, 1.0)
                            nl = math.sqrt(sum(value * value for value in n)) or 1.0
                            n = tuple(value / nl for value in n)
                            uv = uvs[vi] if vi < len(uvs) and uvs[vi] is not None else (0.0, 0.0)
                            links = sorted(merged_links.items())
                            parent = max(links, key=lambda pair: pair[1])[0]
                            link_text = " ".join(f"{index} {weight:.6f}" for index, weight in links)
                            lines.append(
                                f"{parent} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                                f"{n[0]:.6f} {n[1]:.6f} {n[2]:.6f} {uv[0]:.6f} {1.0-uv[1]:.6f} "
                                f"{len(links)} {link_text}"
                            )
    lines.append("end")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

def export_character(
    entry: LibraryEntry,
    output_root: str | Path,
    options: ExportOptions,
    progress: Callable[[str, int, int], None] | None = None,
    additional_entries: list[LibraryEntry] | None = None,
) -> ExportResult:
    additional_entries = additional_entries or []
    output_dir = Path(output_root) / _safe_name(entry.code)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = ExportResult(output_dir)

    skeleton_asset = next((a.path for a in entry.assets if a.kind == AssetKind.SKELETON and a.path.stem.casefold() == f"{entry.code}_000".casefold()), None)
    if skeleton_asset is None:
        skeleton_asset = next((a.path for a in entry.assets if a.kind == AssetKind.SKELETON), None)
    if skeleton_asset is None:
        raise ValueError("No ESK skeleton was found for this character.")
    main_esk = load_esk(skeleton_asset)

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

    main_ean_path = next((a.path for a in entry.assets if a.kind == AssetKind.ANIMATION and a.path.name.casefold() == f"{entry.code}.ean".casefold()), None)
    if main_ean_path is None:
        main_ean_path = next((a.path for a in entry.assets if a.kind == AssetKind.ANIMATION and ".fce." not in a.path.name.casefold() and ".cam." not in a.path.name.casefold()), None)
    ean = load_ean(main_ean_path) if main_ean_path else None

    models = [a.path for a in entry.assets if a.kind == AssetKind.MODEL and not _is_physics_model(a)]
    additional_animation_count = 0
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
                write_reference_smd(target, esk, emd, options.source_scale)
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
                    target, esk, secondary, secondary_maps[secondary.path], emd, options.source_scale
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
                    write_animation_smd(target, esk, animation, options.source_scale)
                    result.animation_files.append(target)
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
        # No BAC means no event output at all. Also remove stale output from older builds.
        if events_dir.exists():
            shutil.rmtree(events_dir)

    summary = {
        "exporter_version": EXPORTER_VERSION,
        "character": entry.code,
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
