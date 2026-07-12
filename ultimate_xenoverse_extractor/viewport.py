from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
import moderngl
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from .parsers.emd import EmdFile
from .parsers.emm import EmmFile
from .parsers.emb import EmbFile
from .parsers.esk import EskFile
from .parsers.ean import EanAnimation
from .texture_processing import compose_surface_texture

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


@dataclass(slots=True)
class GpuSurface:
    vao: Any = None
    vbo: Any = None
    ibo: Any = None
    texture_obj: Any = None
    index_count: int = 0
    texture_key: int | None = None


@dataclass(slots=True)
class RenderSurface:
    name: str
    vertices: list[Vec3]
    triangles: list[tuple[int, int, int]]
    color: tuple[int, int, int, int] = (110, 125, 145, 235)
    uvs: list[tuple[float, float] | None] | None = None
    texture: QImage | None = None
    texture_scale: tuple[float, float] = (1.0, 1.0)
    bind_vertices: list[Vec3] | None = None
    blend_indices: list[tuple[int, int, int, int] | None] | None = None
    blend_weights: list[tuple[float, float, float, float] | None] | None = None
    bone_palette_names: list[str] | None = None
    resolved_bones: list[int | None] | None = None
    gpu: GpuSurface = field(default_factory=GpuSurface)
    gpu_dirty: bool = True


@dataclass(slots=True)
class RenderPart:
    name: str
    surfaces: list[RenderSurface]
    visible: bool = True

    @property
    def vertices(self) -> list[Vec3]:
        return [vertex for surface in self.surfaces for vertex in surface.vertices]

    @property
    def triangle_count(self) -> int:
        return sum(len(surface.triangles) for surface in self.surfaces)


def _vadd(a: Vec3, b: Vec3) -> Vec3:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def _vsub(a: Vec3, b: Vec3) -> Vec3:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _vmul(a: Vec3, scalar: float) -> Vec3:
    return a[0] * scalar, a[1] * scalar, a[2] * scalar


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _length(v: Vec3) -> float:
    return math.sqrt(_dot(v, v))


def _normalize(v: Vec3, fallback: Vec3 = (0.0, 1.0, 0.0)) -> Vec3:
    length = _length(v)
    return fallback if length < 1e-8 else _vmul(v, 1.0 / length)


def _quat_mul(a: Quat, b: Quat) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _quat_rotate(q: Quat, v: Vec3) -> Vec3:
    qv = (v[0], v[1], v[2], 0.0)
    qi = (-q[0], -q[1], -q[2], q[3])
    out = _quat_mul(_quat_mul(q, qv), qi)
    return out[0], out[1], out[2]


def _perspective_positive_z(fov_degrees: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_degrees) * 0.5)
    a = (far + near) / (far - near)
    b = (-2.0 * far * near) / (far - near)
    return np.array(
        [
            [f / max(aspect, 1e-6), 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, a, b],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )


def _view_from_basis(camera: Vec3, right: Vec3, up: Vec3, forward: Vec3) -> np.ndarray:
    return np.array(
        [
            [right[0], right[1], right[2], -_dot(camera, right)],
            [up[0], up[1], up[2], -_dot(camera, up)],
            [forward[0], forward[1], forward[2], -_dot(camera, forward)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


class ModelViewport(QOpenGLWidget):
    """GPU-backed XV2 viewport.

    Geometry, textures, depth testing, alpha blending, wireframe rendering,
    and skeletal skinning are handled by OpenGL. Python only evaluates the
    relatively small bone pose each frame; mesh vertices remain in static GPU
    buffers and are deformed in the vertex shader.
    """

    statusChanged = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(240, 180)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

        self.parts: dict[str, RenderPart] = {}
        self.current_esk: EskFile | None = None
        self.current_animation: EanAnimation | None = None
        self.facial_animation: EanAnimation | None = None
        self.lip_animation: EanAnimation | None = None
        self.camera_animation: EanAnimation | None = None
        self.follow_camera_animation = False
        self.animation_frame = 0.0

        self.skeleton_points: list[Vec3] = []
        self.skeleton_links: list[tuple[int, int]] = []
        self._bind_positions: list[Vec3] = []
        self._bind_rotations: list[Quat] = []
        self._bind_scales: list[Vec3] = []
        self._pose_positions: list[Vec3] = []
        self._pose_rotations: list[Quat] = []
        self._pose_scales: list[Vec3] = []

        self.yaw = -35.0
        self.pitch = -12.0
        self.distance = 4.0
        self.target: Vec3 = (0.0, 1.0, 0.0)
        self.pan = (0.0, 0.0)
        self.orbit_fov = 45.0
        self.camera_fov = 45.0
        self._scene_radius = 1.0

        self.show_grid = True
        self.show_wireframe = False
        self.show_solid = True
        self.show_skeleton = True
        self._last_mouse = None
        self._drag_button = Qt.NoButton

        self.ctx: moderngl.Context | None = None
        self.mesh_program = None
        self.line_program = None
        self.line_vbo = None
        self.line_vao = None
        self.white_texture = None
        self.bone_buffer = None
        self._bone_buffer_capacity = 256
        self._bone_texture_data: np.ndarray | None = None
        self._bone_texture_dirty = True
        self._inverse_bind_matrices: np.ndarray | None = None
        self._skin_matrix_buffer: np.ndarray | None = None
        self._bone_upload_buffer: np.ndarray | None = None
        self._gl_ready = False
        self._gpu_error: str | None = None
        self._qt_framebuffer = None
        self._qt_fbo_id: int | None = None

    # ------------------------------------------------------------------ OpenGL
    def initializeGL(self) -> None:
        try:
            self.ctx = moderngl.create_context(require=330)
            self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            self.ctx.gc_mode = "auto"

            self.mesh_program = self.ctx.program(
                vertex_shader="""
                    #version 330
                    in vec3 in_position;
                    in vec2 in_uv;
                    in vec4 in_bone_indices;
                    in vec4 in_bone_weights;
                    uniform mat4 u_mvp;
                    uniform vec2 u_uv_scale;
                    uniform bool u_use_skinning;
                    layout(std140) uniform BoneMatrices {
                        mat4 u_bones[256];
                    };
                    out vec2 v_uv;

                    mat4 fetch_bone_matrix(int bone_index) {
                        return u_bones[clamp(bone_index, 0, 255)];
                    }

                    void main() {
                        vec4 position = vec4(in_position, 1.0);
                        if (u_use_skinning) {
                            mat4 skin = mat4(0.0);
                            float total_weight = 0.0;
                            for (int slot = 0; slot < 4; ++slot) {
                                float weight = in_bone_weights[slot];
                                int bone_index = int(in_bone_indices[slot] + 0.5);
                                if (weight > 0.000001 && bone_index >= 0) {
                                    skin += fetch_bone_matrix(bone_index) * weight;
                                    total_weight += weight;
                                }
                            }
                            if (total_weight > 0.000001) {
                                position = (skin / total_weight) * position;
                            }
                        }
                        gl_Position = u_mvp * position;
                        v_uv = vec2(in_uv.x * u_uv_scale.x,
                                    in_uv.y * u_uv_scale.y);
                    }
                """,
                fragment_shader="""
                    #version 330
                    uniform sampler2D u_texture;
                    uniform vec4 u_color;
                    uniform bool u_use_texture;
                    in vec2 v_uv;
                    out vec4 fragColor;
                    void main() {
                        vec4 base = u_use_texture ? texture(u_texture, fract(v_uv)) : u_color;
                        if (base.a < 0.015) discard;
                        fragColor = base * u_color;
                    }
                """,
            )
            self.line_program = self.ctx.program(
                vertex_shader="""
                    #version 330
                    in vec3 in_position;
                    uniform mat4 u_mvp;
                    void main() { gl_Position = u_mvp * vec4(in_position, 1.0); }
                """,
                fragment_shader="""
                    #version 330
                    uniform vec4 u_color;
                    out vec4 fragColor;
                    void main() { fragColor = u_color; }
                """,
            )
            self.line_vbo = self.ctx.buffer(reserve=4 * 1024 * 1024, dynamic=True)
            self.line_vao = self.ctx.simple_vertex_array(self.line_program, self.line_vbo, "in_position")
            self.white_texture = self.ctx.texture((1, 1), 4, bytes((255, 255, 255, 255)))
            self.white_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._ensure_bone_buffer()
            # QOpenGLWidget renders into an internal framebuffer rather than
            # OpenGL framebuffer 0. ModernGL must explicitly target that FBO.
            self._bind_qt_framebuffer(force=True)
            self._gl_ready = True
            self._upload_all_surfaces()
            self.statusChanged.emit(f"GPU viewport active: OpenGL {self.ctx.version_code}")
        except Exception as exc:  # pragma: no cover - depends on local GPU/driver
            self._gpu_error = f"OpenGL initialization failed: {exc}"
            self.statusChanged.emit(self._gpu_error)

    def _bind_qt_framebuffer(self, force: bool = False) -> bool:
        """Bind the internal framebuffer owned by QOpenGLWidget.

        QOpenGLWidget does not render directly to framebuffer 0. Qt may also
        recreate its framebuffer after a resize or screen/DPI change, so the
        ModernGL wrapper has to be refreshed whenever the FBO id changes.
        """
        if self.ctx is None:
            return False
        try:
            fbo_id = int(self.defaultFramebufferObject())
            if force or self._qt_framebuffer is None or fbo_id != self._qt_fbo_id:
                # Qt has already bound its internal FBO before initializeGL /
                # paintGL are called, allowing ModernGL to detect it safely.
                self._qt_framebuffer = self.ctx.detect_framebuffer()
                self._qt_fbo_id = fbo_id
            self._qt_framebuffer.use()
            return True
        except Exception as exc:
            self._gpu_error = f"Unable to bind Qt OpenGL framebuffer: {exc}"
            self.statusChanged.emit(self._gpu_error)
            return False

    def resizeGL(self, width: int, height: int) -> None:
        if self.ctx is not None:
            # A resize can cause QOpenGLWidget to replace its internal FBO.
            self._bind_qt_framebuffer(force=True)
            ratio = self.devicePixelRatioF()
            self.ctx.viewport = (0, 0, max(1, int(width * ratio)), max(1, int(height * ratio)))

    def paintGL(self) -> None:
        if not self._gl_ready or self.ctx is None:
            return
        if not self._bind_qt_framebuffer():
            return
        ratio = self.devicePixelRatioF()
        self.ctx.viewport = (0, 0, max(1, int(self.width() * ratio)), max(1, int(self.height() * ratio)))
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.ctx.clear(0.035, 0.045, 0.058, 1.0, depth=1.0)
        self._upload_bone_buffer_if_needed()
        mvp = self._mvp_matrix()

        if self.show_grid:
            self._render_grid(mvp)
        self._render_axes(mvp)
        self._render_models(mvp)
        if self.show_skeleton:
            self._render_skeleton(mvp)
        if self.camera_animation is not None and not self.follow_camera_animation:
            self._render_camera_gizmo(mvp)

    def _release_surface_gpu(self, surface: RenderSurface) -> None:
        for obj in (surface.gpu.vao, surface.gpu.vbo, surface.gpu.ibo, surface.gpu.texture_obj):
            try:
                if obj is not None:
                    obj.release()
            except Exception:
                pass
        surface.gpu = GpuSurface()

    def _dispose_part(self, part: RenderPart) -> None:
        """Release both OpenGL objects and large CPU-side mesh/image buffers."""
        for surface in part.surfaces:
            self._release_surface_gpu(surface)
            surface.texture = None
            surface.vertices.clear()
            surface.triangles.clear()
            if surface.uvs is not None:
                surface.uvs.clear()
            if surface.bind_vertices is not None:
                surface.bind_vertices.clear()
            if surface.blend_indices is not None:
                surface.blend_indices.clear()
            if surface.blend_weights is not None:
                surface.blend_weights.clear()
            if surface.bone_palette_names is not None:
                surface.bone_palette_names.clear()
            if surface.resolved_bones is not None:
                surface.resolved_bones.clear()
        part.surfaces.clear()

    def _flush_gpu_releases(self) -> None:
        if self.ctx is None:
            return
        try:
            self.ctx.finish()
        except Exception:
            pass
        try:
            collect = getattr(self.ctx, "gc", None)
            if callable(collect):
                collect()
        except Exception:
            pass

    def _upload_all_surfaces(self) -> None:
        if not self._gl_ready:
            return
        for part in self.parts.values():
            for surface in part.surfaces:
                self._upload_surface(surface, rebuild=True)

    def _qimage_rgba_bytes(self, image: QImage) -> tuple[bytes, int, int]:
        converted = image.convertToFormat(QImage.Format_RGBA8888)
        width, height = converted.width(), converted.height()
        ptr = converted.constBits()
        data = bytes(ptr[: converted.sizeInBytes()])
        return data, width, height

    def _upload_surface(self, surface: RenderSurface, rebuild: bool = False) -> None:
        if not self._gl_ready or self.ctx is None or self.mesh_program is None:
            surface.gpu_dirty = True
            return

        bind_vertices = surface.bind_vertices or surface.vertices
        vertex_count = len(bind_vertices)
        if vertex_count == 0:
            return
        positions = np.asarray(bind_vertices, dtype=np.float32)
        uv_data = np.zeros((vertex_count, 2), dtype=np.float32)
        if surface.uvs:
            for index, uv in enumerate(surface.uvs[:vertex_count]):
                if uv is not None:
                    uv_data[index] = uv

        bone_indices = np.full((vertex_count, 4), -1.0, dtype=np.float32)
        bone_weights = np.zeros((vertex_count, 4), dtype=np.float32)
        palette = surface.resolved_bones or []
        local_indices = surface.blend_indices or []
        local_weights = surface.blend_weights or []
        for vertex_index in range(vertex_count):
            indices4 = local_indices[vertex_index] if vertex_index < len(local_indices) else None
            weights4 = local_weights[vertex_index] if vertex_index < len(local_weights) else None
            if indices4 is None or weights4 is None:
                continue
            for slot in range(4):
                palette_index = indices4[slot]
                if palette_index < len(palette):
                    resolved = palette[palette_index]
                    if resolved is not None:
                        bone_indices[vertex_index, slot] = float(resolved)
                        bone_weights[vertex_index, slot] = float(weights4[slot])

        interleaved = np.concatenate((positions, uv_data, bone_indices, bone_weights), axis=1).astype(np.float32, copy=False)
        indices = np.asarray(surface.triangles, dtype=np.uint32).reshape(-1)

        if rebuild or surface.gpu.vbo is None or surface.gpu.index_count != len(indices):
            self._release_surface_gpu(surface)
            # Static mesh data: animation no longer rewrites this buffer.
            surface.gpu.vbo = self.ctx.buffer(interleaved.tobytes())
            surface.gpu.ibo = self.ctx.buffer(indices.tobytes())
            surface.gpu.vao = self.ctx.vertex_array(
                self.mesh_program,
                [(surface.gpu.vbo, "3f 2f 4f 4f", "in_position", "in_uv", "in_bone_indices", "in_bone_weights")],
                surface.gpu.ibo,
            )
            surface.gpu.index_count = len(indices)

        texture_key = surface.texture.cacheKey() if surface.texture is not None else None
        if texture_key != surface.gpu.texture_key:
            if surface.gpu.texture_obj is not None:
                surface.gpu.texture_obj.release()
                surface.gpu.texture_obj = None
            if surface.texture is not None:
                data, width, height = self._qimage_rgba_bytes(surface.texture)
                surface.gpu.texture_obj = self.ctx.texture((width, height), 4, data)
                surface.gpu.texture_obj.filter = (moderngl.LINEAR, moderngl.LINEAR)
                surface.gpu.texture_obj.repeat_x = True
                surface.gpu.texture_obj.repeat_y = True
            surface.gpu.texture_key = texture_key
        surface.gpu_dirty = False

    # ------------------------------------------------------------- model setup
    def clear(self) -> None:
        if self._gl_ready:
            self.makeCurrent()
            for part in self.parts.values():
                self._dispose_part(part)
            self._flush_gpu_releases()
            self.doneCurrent()
        self.parts.clear()
        self.current_esk = None
        self.current_animation = None
        self.facial_animation = None
        self.lip_animation = None
        self.camera_animation = None
        self.follow_camera_animation = False
        self.animation_frame = 0.0
        self.skeleton_points.clear()
        self.skeleton_links.clear()
        self._bind_positions.clear()
        self._bind_rotations.clear()
        self._bind_scales.clear()
        self._pose_positions.clear()
        self._pose_rotations.clear()
        self._pose_scales.clear()
        self._bone_texture_data = None
        self._bone_texture_dirty = True
        if self.bone_buffer is not None:
            try:
                self.bone_buffer.release()
            except Exception:
                pass
            self.bone_buffer = None
        self._inverse_bind_matrices = None
        self._skin_matrix_buffer = None
        self._bone_upload_buffer = None
        self.update()
        self.statusChanged.emit("Viewport cleared")

    def set_emd(self, emd: EmdFile, key: str | None = None, emm: EmmFile | None = None,
                emb: EmbFile | None = None, dyt: EmbFile | None = None) -> None:
        key = key or emd.path.name
        surfaces: list[RenderSurface] = []
        vertex_count = triangle_count = weighted_vertices = 0
        texture_notes: set[str] = set()

        for submesh in emd.submeshes:
            rgba = (110, 125, 145, 235)
            material = emm.material(submesh.name) if emm else None
            if material is not None:
                color = material.preview_color()
                raw = tuple(int(max(0.0, min(1.0, value)) * 255) for value in color)
                rgba = (raw[0], raw[1], raw[2], max(90, raw[3]))

            texture = None
            scale = (1.0, 1.0)
            if submesh.texture_definitions:
                definition = submesh.texture_definitions[0]
                texture, note = compose_surface_texture(emb, dyt, definition.texture_index, material)
                texture_notes.add(note)
                scale = (definition.scale_u, definition.scale_v)

            groups = submesh.triangle_groups or []
            if not groups:
                groups = [type("_EmptyGroup", (), {"indices": [], "bone_names": []})()]

            for group in groups:
                triangles = []
                for offset in range(0, len(group.indices) - 2, 3):
                    a, b, c = group.indices[offset:offset + 3]
                    if a < len(submesh.positions) and b < len(submesh.positions) and c < len(submesh.positions):
                        triangles.append((a, b, c))
                bind_vertices = list(submesh.positions)
                weights = list(submesh.blend_weights)
                weighted_vertices += sum(value is not None for value in weights)
                surfaces.append(RenderSurface(
                    name=submesh.name,
                    vertices=list(bind_vertices),
                    triangles=triangles,
                    color=rgba,
                    uvs=list(submesh.uvs),
                    texture=texture,
                    texture_scale=scale,
                    bind_vertices=bind_vertices,
                    blend_indices=list(submesh.blend_indices),
                    blend_weights=weights,
                    bone_palette_names=list(group.bone_names),
                ))
                triangle_count += len(triangles)
            vertex_count += len(submesh.positions)

        old = self.parts.pop(key, None)
        if old is not None and self._gl_ready:
            self.makeCurrent()
            self._dispose_part(old)
            self._flush_gpu_releases()
            self.doneCurrent()
        self.parts[key] = RenderPart(key, surfaces)
        self._resolve_surface_bones()
        self._skin_meshes()
        if self._gl_ready:
            self.makeCurrent()
            for surface in surfaces:
                self._upload_surface(surface, rebuild=True)
            self.doneCurrent()
        self.fit_scene()
        notes = "; ".join(sorted(texture_notes)) if texture_notes else "material fallback"
        self.statusChanged.emit(
            f"Loaded {emd.path.name}: {vertex_count:,} vertices, {triangle_count:,} triangles, "
            f"{weighted_vertices:,} weighted records, GPU textures: {notes}"
        )

    def apply_emm(self, key: str, emm: EmmFile) -> None:
        part = self.parts.get(key)
        if part is None:
            return
        matched = 0
        for surface in part.surfaces:
            material = emm.material(surface.name)
            if material is None:
                continue
            color = material.preview_color()
            raw = tuple(int(max(0.0, min(1.0, value)) * 255) for value in color)
            surface.color = (raw[0], raw[1], raw[2], max(90, raw[3]))
            matched += 1
        self.update()
        self.statusChanged.emit(f"Applied {emm.path.name}: matched {matched}/{len(part.surfaces)} surfaces")

    def remove_part(self, key: str) -> None:
        part = self.parts.pop(key, None)
        if part is not None:
            if self._gl_ready:
                self.makeCurrent()
                self._dispose_part(part)
                self._flush_gpu_releases()
                self.doneCurrent()
            else:
                self._dispose_part(part)
        self.fit_scene()

    def set_part_visible(self, key: str, visible: bool) -> None:
        if key in self.parts:
            self.parts[key].visible = visible
            self.update()

    # ------------------------------------------------------------ animation rig
    def set_esk(self, esk: EskFile) -> None:
        self.current_esk = esk
        self.current_animation = None
        self.animation_frame = 0.0
        self._build_bind_pose()
        self._resolve_surface_bones()
        if self._gl_ready:
            self.makeCurrent()
            for part in self.parts.values():
                for surface in part.surfaces:
                    self._upload_surface(surface, rebuild=True)
            self.doneCurrent()
        self._rebuild_skeleton_pose()
        self.fit_scene()
        self.statusChanged.emit(f"Loaded {esk.path.name}: {len(esk.bones)} bones")

    def set_animation(self, animation: EanAnimation | None) -> None:
        self.current_animation = animation
        self.animation_frame = 0.0
        self._rebuild_skeleton_pose()
        if animation is None:
            self.statusChanged.emit("Animation cleared; showing bind pose")
        else:
            self.statusChanged.emit(f"Loaded animation {animation.name}: {animation.duration} frames")

    def set_overlay_animations(self, facial: EanAnimation | None = None,
                               lip: EanAnimation | None = None) -> None:
        self.facial_animation = facial
        self.lip_animation = lip
        self._rebuild_skeleton_pose()

    def set_camera_animation(self, animation: EanAnimation | None, enabled: bool | None = None) -> None:
        self.camera_animation = animation
        if enabled is not None:
            self.set_follow_camera_animation(bool(enabled and animation is not None))
        else:
            self.update()

    def set_follow_camera_animation(self, enabled: bool) -> None:
        self.follow_camera_animation = bool(enabled and self.camera_animation is not None)
        if self.follow_camera_animation:
            self.pan = (0.0, 0.0)
        self.update()

    def set_animation_frame(self, frame: float) -> None:
        self.animation_frame = max(0.0, frame)
        self._rebuild_skeleton_pose()

    def _build_bind_pose(self) -> None:
        esk = self.current_esk
        if esk is None:
            self._bind_positions = []
            self._bind_rotations = []
            self._bind_scales = []
            return
        count = len(esk.bones)
        positions: list[Vec3] = [(0.0, 0.0, 0.0)] * count
        rotations: list[Quat] = [(0.0, 0.0, 0.0, 1.0)] * count
        scales: list[Vec3] = [(1.0, 1.0, 1.0)] * count
        for bone in esk.bones:
            local_pos = bone.position[:3]
            local_rot = bone.rotation
            local_scale = bone.scale[:3]
            parent = bone.parent
            if parent is None or parent >= count:
                positions[bone.index], rotations[bone.index], scales[bone.index] = local_pos, local_rot, local_scale
            else:
                parent_scale = scales[parent]
                scaled = tuple(local_pos[i] * parent_scale[i] for i in range(3))
                positions[bone.index] = _vadd(positions[parent], _quat_rotate(rotations[parent], scaled))
                rotations[bone.index] = _quat_mul(rotations[parent], local_rot)
                scales[bone.index] = tuple(parent_scale[i] * local_scale[i] for i in range(3))
        self._bind_positions, self._bind_rotations, self._bind_scales = positions, rotations, scales

        # Bind transforms never change for a loaded skeleton. Inverting every
        # bind matrix on every animation frame was both expensive and extremely
        # allocation-heavy. Build and retain the inverse matrices exactly once.
        self._inverse_bind_matrices = np.empty((count, 4, 4), dtype=np.float32)
        for index in range(count):
            bind = self._transform_matrix(positions[index], rotations[index], scales[index])
            try:
                self._inverse_bind_matrices[index] = np.linalg.inv(bind)
            except np.linalg.LinAlgError:
                self._inverse_bind_matrices[index] = np.eye(4, dtype=np.float32)
        self._skin_matrix_buffer = np.empty((count, 4, 4), dtype=np.float32)
        self._bone_upload_buffer = np.empty((count, 4, 4), dtype=np.float32)

    def _resolve_surface_bones(self) -> None:
        if self.current_esk is None:
            return
        by_name = {bone.name: bone.index for bone in self.current_esk.bones}
        for part in self.parts.values():
            for surface in part.surfaces:
                surface.resolved_bones = [by_name.get(name) for name in (surface.bone_palette_names or [])]
                surface.gpu_dirty = True

    @staticmethod
    def _inverse_bind_point(point: Vec3, position: Vec3, rotation: Quat, scale: Vec3) -> Vec3:
        translated = _vsub(point, position)
        inverse_rotation = (-rotation[0], -rotation[1], -rotation[2], rotation[3])
        rotated = _quat_rotate(inverse_rotation, translated)
        return tuple(rotated[i] / scale[i] if abs(scale[i]) > 1e-8 else rotated[i] for i in range(3))

    @staticmethod
    def _pose_point(point: Vec3, position: Vec3, rotation: Quat, scale: Vec3) -> Vec3:
        scaled = tuple(point[i] * scale[i] for i in range(3))
        return _vadd(position, _quat_rotate(rotation, scaled))

    @staticmethod
    def _transform_matrix(position: Vec3, rotation: Quat, scale: Vec3) -> np.ndarray:
        x, y, z, w = rotation
        length = math.sqrt(x * x + y * y + z * z + w * w)
        if length > 1e-8:
            x, y, z, w = x / length, y / length, z / length, w / length
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        rotation_matrix = np.array((
            (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
            (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
            (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
        ), dtype=np.float32)
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = rotation_matrix @ np.diag(np.asarray(scale, dtype=np.float32))
        matrix[:3, 3] = np.asarray(position, dtype=np.float32)
        return matrix

    def _build_skin_matrices(self) -> np.ndarray:
        count = min(len(self._bind_positions), len(self._pose_positions))
        if count <= 0:
            if self._skin_matrix_buffer is None or self._skin_matrix_buffer.shape[0] != 1:
                self._skin_matrix_buffer = np.eye(4, dtype=np.float32)[None, ...]
            return self._skin_matrix_buffer

        if self._skin_matrix_buffer is None or self._skin_matrix_buffer.shape != (count, 4, 4):
            self._skin_matrix_buffer = np.empty((count, 4, 4), dtype=np.float32)
        if self._inverse_bind_matrices is None or self._inverse_bind_matrices.shape[0] < count:
            # Defensive fallback for a skeleton supplied by an older code path.
            self._inverse_bind_matrices = np.empty((count, 4, 4), dtype=np.float32)
            for index in range(count):
                bind = self._transform_matrix(
                    self._bind_positions[index], self._bind_rotations[index], self._bind_scales[index]
                )
                try:
                    self._inverse_bind_matrices[index] = np.linalg.inv(bind)
                except np.linalg.LinAlgError:
                    self._inverse_bind_matrices[index] = np.eye(4, dtype=np.float32)

        out = self._skin_matrix_buffer
        for index in range(count):
            pose = self._transform_matrix(
                self._pose_positions[index], self._pose_rotations[index], self._pose_scales[index]
            )
            # matmul's out parameter avoids creating a new 4x4 result per bone.
            np.matmul(pose, self._inverse_bind_matrices[index], out=out[index])
        return out

    def _ensure_bone_buffer(self) -> None:
        """Create one persistent UBO for all skin matrices.

        A texture upload every frame caused the Windows OpenGL driver to retain
        staging allocations in the Python process working set.  A persistently
        allocated uniform buffer avoids that path entirely.
        """
        if self.ctx is None or self.bone_buffer is not None:
            return
        self.bone_buffer = self.ctx.buffer(
            reserve=self._bone_buffer_capacity * 16 * 4,
            dynamic=True,
        )
        self.bone_buffer.bind_to_uniform_block(0)
        try:
            self.mesh_program["BoneMatrices"].binding = 0
        except Exception:
            pass

    def _upload_bone_buffer_if_needed(self) -> None:
        if not self._bone_texture_dirty or self.ctx is None:
            return
        matrices = self._bone_texture_data
        if matrices is None:
            matrices = self._build_skin_matrices()
        self._ensure_bone_buffer()
        if self.bone_buffer is None:
            return
        count = min(len(matrices), self._bone_buffer_capacity)
        # GLSL mat4/std140 expects column-major matrices. Keep a reusable,
        # contiguous array and write directly without creating a bytes object.
        if self._bone_upload_buffer is None or self._bone_upload_buffer.shape != (self._bone_buffer_capacity, 4, 4):
            self._bone_upload_buffer = np.zeros(
                (self._bone_buffer_capacity, 4, 4), dtype=np.float32
            )
            for i in range(self._bone_buffer_capacity):
                self._bone_upload_buffer[i] = np.eye(4, dtype=np.float32)
        self._bone_upload_buffer[:count] = np.transpose(matrices[:count], (0, 2, 1))
        self.bone_buffer.write(memoryview(self._bone_upload_buffer).cast("B"))
        self._bone_texture_dirty = False

    def _skin_meshes(self) -> None:
        """Queue GPU skinning data instead of deforming every vertex in Python."""
        self._bone_texture_data = self._build_skin_matrices()
        self._bone_texture_dirty = True

    def _rebuild_skeleton_pose(self) -> None:
        esk = self.current_esk
        if esk is None:
            self.skeleton_points = []
            self.skeleton_links = []
            self.update()
            return
        # EAN node indices refer to the skeleton embedded in that EAN. Several
        # characters use a different numeric bone order from their model ESK,
        # so applying tracks by raw index can turn the mesh into spaghetti.
        # Resolve by bone name first, with index fallback only for old/parserless
        # animation objects that do not carry embedded skeleton names.
        node_name_map = {}
        node_index_map = {}
        for animation in (self.current_animation, self.facial_animation, self.lip_animation):
            if animation is None:
                continue
            name_map = getattr(animation, "node_name_map", {})
            if name_map:
                node_name_map.update(name_map)
            else:
                node_index_map.update(animation.node_map)

        points: list[Vec3] = [(0.0, 0.0, 0.0)] * len(esk.bones)
        rotations: list[Quat] = [(0.0, 0.0, 0.0, 1.0)] * len(esk.bones)
        scales: list[Vec3] = [(1.0, 1.0, 1.0)] * len(esk.bones)
        links: list[tuple[int, int]] = []

        for bone in esk.bones:
            local_pos = bone.position[:3]
            local_rot = bone.rotation
            local_scale = bone.scale[:3]
            node = node_name_map.get(bone.name)
            if node is None:
                node = node_index_map.get(bone.index)
            if node is not None:
                pos_track, rot_track, scale_track = node.tracks.get(0), node.tracks.get(1), node.tracks.get(2)
                if pos_track is not None:
                    sampled = pos_track.sample(self.animation_frame)
                    if sampled is not None:
                        w = sampled[3]
                        local_pos = tuple(sampled[i] / w if w not in (0.0, 1.0) else sampled[i] for i in range(3))
                if rot_track is not None:
                    sampled = rot_track.sample(self.animation_frame, normalize_quaternion=True)
                    if sampled is not None:
                        local_rot = sampled
                if scale_track is not None:
                    sampled = scale_track.sample(self.animation_frame)
                    if sampled is not None:
                        w = sampled[3]
                        local_scale = tuple(sampled[i] / w if w not in (0.0, 1.0) else sampled[i] for i in range(3))

            parent = bone.parent
            if parent is None or parent >= len(esk.bones):
                points[bone.index], rotations[bone.index], scales[bone.index] = local_pos, local_rot, local_scale
            else:
                parent_scale = scales[parent]
                scaled = tuple(local_pos[i] * parent_scale[i] for i in range(3))
                points[bone.index] = _vadd(points[parent], _quat_rotate(rotations[parent], scaled))
                rotations[bone.index] = _quat_mul(rotations[parent], local_rot)
                scales[bone.index] = tuple(parent_scale[i] * local_scale[i] for i in range(3))
                links.append((parent, bone.index))

        self.skeleton_points, self.skeleton_links = points, links
        self._pose_positions, self._pose_rotations, self._pose_scales = points, rotations, scales
        self._skin_meshes()
        self.update()

    # ------------------------------------------------------------------ camera
    def fit_scene(self) -> None:
        points = [vertex for part in self.parts.values() if part.visible for vertex in part.vertices]
        points.extend(self.skeleton_points)
        if not points:
            self.target, self.distance, self._scene_radius = (0.0, 1.0, 0.0), 4.0, 1.0
            self.update()
            return
        mins = [min(point[i] for point in points) for i in range(3)]
        maxs = [max(point[i] for point in points) for i in range(3)]
        self.target = tuple((mins[i] + maxs[i]) * 0.5 for i in range(3))
        self._scene_radius = max(0.01, max(maxs[i] - mins[i] for i in range(3)) * 0.5)
        self.distance = self._scene_radius * 3.0
        self.pan = (0.0, 0.0)
        self.update()

    def reset_camera(self) -> None:
        self.yaw, self.pitch = -35.0, -12.0
        self.fit_scene()

    def _sample_camera_animation(self):
        if self.camera_animation is None or not self.camera_animation.nodes:
            return None
        node = self.camera_animation.nodes[0]
        pos_track, target_track, props_track = node.tracks.get(0), node.tracks.get(1), node.tracks.get(2)
        pos = pos_track.sample(self.animation_frame) if pos_track else None
        target = target_track.sample(self.animation_frame) if target_track else None
        props = props_track.sample(self.animation_frame) if props_track else None
        if pos is None or target is None:
            return None
        pw, tw = pos[3], target[3]
        camera = tuple(pos[i] / pw if pw not in (0.0, 1.0) else pos[i] for i in range(3))
        look = tuple(target[i] / tw if tw not in (0.0, 1.0) else target[i] for i in range(3))
        forward = _normalize(_vsub(look, camera), (0.0, 0.0, 1.0))
        right = _normalize(_cross((0.0, 1.0, 0.0), forward), (1.0, 0.0, 0.0))
        up = _normalize(_cross(forward, right), (0.0, 1.0, 0.0))
        fov = 45.0
        if props is not None:
            candidate = math.degrees(props[1])
            if 5.0 <= candidate <= 170.0:
                fov = candidate
            roll = props[0]
            if abs(roll) > 1e-6:
                cr, sr = math.cos(roll), math.sin(roll)
                old_right, old_up = right, up
                right = _vadd(_vmul(old_right, cr), _vmul(old_up, sr))
                up = _vadd(_vmul(old_up, cr), _vmul(old_right, -sr))
        return camera, look, right, up, forward, fov

    def _camera_basis(self):
        if self.follow_camera_animation:
            sampled = self._sample_camera_animation()
            if sampled is not None:
                camera, _look, right, up, forward, fov = sampled
                self.camera_fov = fov
                return camera, right, up, forward
        self.camera_fov = self.orbit_fov
        yaw, pitch = math.radians(self.yaw), math.radians(self.pitch)
        forward = _normalize((math.cos(pitch) * math.sin(yaw), math.sin(pitch), math.cos(pitch) * math.cos(yaw)))
        right = _normalize(_cross((0.0, 1.0, 0.0), forward), (1.0, 0.0, 0.0))
        up = _normalize(_cross(forward, right), (0.0, 1.0, 0.0))
        camera = _vsub(self.target, _vmul(forward, self.distance))
        if not self.follow_camera_animation:
            world_per_pixel = max(self.distance, 0.01) / max(self.height(), 1)
            camera = _vadd(camera, _vadd(_vmul(right, -self.pan[0] * world_per_pixel * 1.8),
                                         _vmul(up, self.pan[1] * world_per_pixel * 1.8)))
        return camera, right, up, forward

    def _mvp_matrix(self) -> np.ndarray:
        camera, right, up, forward = self._camera_basis()
        view = _view_from_basis(camera, right, up, forward)
        radius = max(self._scene_radius, 1.0)
        near = max(0.001, min(0.1, radius * 0.002))
        far = max(100.0, self.distance + radius * 20.0)
        projection = _perspective_positive_z(self.camera_fov, max(self.width(), 1) / max(self.height(), 1), near, far)
        return projection @ view

    # ---------------------------------------------------------------- rendering
    @staticmethod
    def _write_matrix(program, name: str, matrix: np.ndarray) -> None:
        program[name].write(matrix.astype(np.float32).T.tobytes())

    def _render_models(self, mvp: np.ndarray) -> None:
        if self.ctx is None or self.mesh_program is None:
            return
        self._write_matrix(self.mesh_program, "u_mvp", mvp)
        if self.bone_buffer is not None:
            self.bone_buffer.bind_to_uniform_block(0)
        for part in self.parts.values():
            if not part.visible:
                continue
            for surface in part.surfaces:
                if surface.gpu_dirty or surface.gpu.vao is None:
                    self._upload_surface(surface, rebuild=surface.gpu.vao is None)
                if surface.gpu.vao is None:
                    continue
                rgba = tuple(channel / 255.0 for channel in surface.color)
                has_texture = surface.gpu.texture_obj is not None
                has_skinning = bool(self.current_esk is not None and surface.resolved_bones and surface.blend_weights)
                self.mesh_program["u_use_skinning"].value = has_skinning
                if self.show_solid:
                    self.ctx.wireframe = False
                    self.mesh_program["u_color"].value = (1.0, 1.0, 1.0, rgba[3]) if has_texture else rgba
                    self.mesh_program["u_use_texture"].value = has_texture
                    self.mesh_program["u_uv_scale"].value = surface.texture_scale
                    (surface.gpu.texture_obj or self.white_texture).use(0)
                    self.mesh_program["u_texture"].value = 0
                    surface.gpu.vao.render(moderngl.TRIANGLES)
                if self.show_wireframe:
                    self.ctx.wireframe = True
                    self.mesh_program["u_color"].value = (0.52, 0.68, 0.82, 0.75)
                    self.mesh_program["u_use_texture"].value = False
                    self.mesh_program["u_uv_scale"].value = (1.0, 1.0)
                    self.white_texture.use(0)
                    surface.gpu.vao.render(moderngl.TRIANGLES)
                    self.ctx.wireframe = False

    def _draw_lines(self, vertices: list[Vec3], color: tuple[float, float, float, float],
                    mvp: np.ndarray, width: float = 1.0, mode=moderngl.LINES) -> None:
        if not vertices or self.ctx is None or self.line_program is None or self.line_vbo is None:
            return
        data = np.asarray(vertices, dtype=np.float32).tobytes()
        if len(data) > self.line_vbo.size:
            self.line_vbo.orphan(len(data))
        self.line_vbo.write(data)
        self._write_matrix(self.line_program, "u_mvp", mvp)
        self.line_program["u_color"].value = color
        self.ctx.line_width = width
        self.line_vao.render(mode, vertices=len(vertices))

    def _render_grid(self, mvp: np.ndarray) -> None:
        radius = max(5, int(math.ceil(self._scene_radius * 2.0)))
        step = max(0.25, 10 ** math.floor(math.log10(max(self._scene_radius / 5.0, 0.25))))
        extent = math.ceil(radius / step) * step
        vertices: list[Vec3] = []
        count = int(extent / step)
        for index in range(-count, count + 1):
            value = index * step
            vertices.extend([(-extent, 0.0, value), (extent, 0.0, value), (value, 0.0, -extent), (value, 0.0, extent)])
        self._draw_lines(vertices, (0.18, 0.23, 0.30, 0.72), mvp, 1.0)

    def _render_axes(self, mvp: np.ndarray) -> None:
        size = max(0.35, self._scene_radius * 0.45)
        self._draw_lines([(0, 0, 0), (size, 0, 0)], (1.0, 0.32, 0.42, 1.0), mvp, 2.0)
        self._draw_lines([(0, 0, 0), (0, size, 0)], (0.32, 1.0, 0.63, 1.0), mvp, 2.0)
        self._draw_lines([(0, 0, 0), (0, 0, size)], (0.30, 0.58, 1.0, 1.0), mvp, 2.0)

    def _render_skeleton(self, mvp: np.ndarray) -> None:
        vertices = []
        for parent, child in self.skeleton_links:
            if parent < len(self.skeleton_points) and child < len(self.skeleton_points):
                vertices.extend([self.skeleton_points[parent], self.skeleton_points[child]])
        self._draw_lines(vertices, (1.0, 0.82, 0.35, 1.0), mvp, 2.0)
        if self.skeleton_points:
            self._draw_lines(self.skeleton_points, (1.0, 0.95, 0.65, 1.0), mvp, 4.0, moderngl.POINTS)

    def _render_camera_gizmo(self, mvp: np.ndarray) -> None:
        sampled = self._sample_camera_animation()
        if sampled is None:
            return
        camera, look, right, up, forward, fov = sampled
        distance = max(0.25, min(_length(_vsub(look, camera)), max(0.5, self._scene_radius)))
        half_h = math.tan(math.radians(fov) * 0.5) * distance
        half_w = half_h * max(self.width(), 1) / max(self.height(), 1)
        center = _vadd(camera, _vmul(forward, distance))
        corners = [
            _vadd(center, _vadd(_vmul(right, sx * half_w), _vmul(up, sy * half_h)))
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
        lines = [camera, look]
        for corner in corners:
            lines.extend([camera, corner])
        for index in range(4):
            lines.extend([corners[index], corners[(index + 1) % 4]])
        self._draw_lines(lines, (0.25, 0.92, 1.0, 1.0), mvp, 2.0)
        self._draw_lines([camera, look], (1.0, 0.82, 0.25, 1.0), mvp, 3.0, moderngl.POINTS)

    # -------------------------------------------------------------- interaction
    def _invalidate_frame(self, interactive: bool = False) -> None:
        del interactive
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._last_mouse = event.position()
        self._drag_button = event.button()
        self.setFocus()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last_mouse is None or self.follow_camera_animation:
            return
        delta = event.position() - self._last_mouse
        self._last_mouse = event.position()
        if self._drag_button == Qt.LeftButton:
            self.yaw += delta.x() * 0.45
            self.pitch = max(-89.0, min(89.0, self.pitch + delta.y() * 0.45))
        elif self._drag_button in (Qt.MiddleButton, Qt.RightButton):
            self.pan = (self.pan[0] + delta.x(), self.pan[1] + delta.y())
        self.update()

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:
        self._last_mouse = None
        self._drag_button = Qt.NoButton

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.follow_camera_animation:
            return
        factor = math.pow(0.88, event.angleDelta().y() / 120.0)
        self.distance = max(0.02, self.distance * factor)
        self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_F:
            self.fit_scene()
        elif event.key() == Qt.Key_R:
            self.reset_camera()
        else:
            super().keyPressEvent(event)
