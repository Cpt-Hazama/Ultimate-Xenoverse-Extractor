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
from .animation_cache import DensePoseCache
from .texture_processing import compose_surface_texture, compose_dyt_lightwarp

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]

# Inverted-hull outline thickness in Xenoverse model-space units.
OUTLINE_WIDTH = 0.001


@dataclass(slots=True)
class GpuSurface:
    vao: Any = None
    position_vbo: Any = None
    static_vbo: Any = None
    ibo: Any = None
    texture_obj: Any = None
    lightwarp_obj: Any = None
    lightwarp_key: int | None = None
    palette_ubo: Any = None
    palette_revision: int = -1
    index_count: int = 0
    texture_key: int | None = None


@dataclass(slots=True)
class RenderSurface:
    name: str
    vertices: list[Vec3]
    triangles: list[tuple[int, int, int]]
    normals: list[Vec3 | None] | None = None
    color: tuple[int, int, int, int] = (110, 125, 145, 235)
    uvs: list[tuple[float, float] | None] | None = None
    texture: QImage | None = None
    lightwarp: QImage | None = None
    has_skinning: bool = False
    skin_ready: bool = False
    texture_scale: tuple[float, float] = (1.0, 1.0)
    bind_vertices: list[Vec3] | None = None
    blend_indices: list[tuple[int, int, int, int] | None] | None = None
    blend_weights: list[tuple[float, float, float, float] | None] | None = None
    bone_palette_names: list[str] | None = None
    resolved_bones: list[int | None] | None = None
    bind_positions_np: np.ndarray | None = None
    bind_homogeneous_np: np.ndarray | None = None
    skin_bone_indices_np: np.ndarray | None = None
    skin_weights_np: np.ndarray | None = None
    gpu_palette_np: np.ndarray | None = None
    gpu_local_indices_np: np.ndarray | None = None
    animated_positions_np: np.ndarray | None = None
    skin_accumulator_np: np.ndarray | None = None
    skin_temp_np: np.ndarray | None = None
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
        self._pose_cache: DensePoseCache | None = None
        self._pose_cache_hits = 0
        self._pose_cache_misses = 0

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
        self._drag_light = False
        # World-space, rotatable directional light (original: 0.36, 0.78, 0.53).
        self.light_yaw = math.degrees(math.atan2(0.36, 0.53))
        self.light_pitch = math.degrees(math.asin(0.78 / math.sqrt(0.36**2 + 0.78**2 + 0.53**2)))
        self.setToolTip("Left drag: orbit camera | Middle/right drag: pan | Ctrl + left drag: move cel-shading light | Wheel: zoom")

        self.ctx: moderngl.Context | None = None
        self.mesh_program = None
        self.line_program = None
        self.line_vbo = None
        self.line_vao = None
        self.white_texture = None
        self.bone_buffer = None
        self._bone_buffer_capacity = 256
        self.use_gpu_skinning = True
        # Diagnostic isolation switches. Both default OFF; toggling them does
        # not delete GPU resources or alter animation/cache state.
        self.freeze_gpu_uploads = False
        self.skip_model_draw = False
        self._bone_texture_data: np.ndarray | None = None
        self._bone_texture_dirty = True
        self._pose_revision = 0
        self._pending_cpu_uploads: dict[int, RenderSurface] = {}
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
            # The vertex buffers and bone UBOs are long-lived; never allocate
            # transient OpenGL objects in the animation/paint hot paths.
            try:
                self._bone_buffer_capacity = max(32, min(512, int(self.ctx.info.get("GL_MAX_UNIFORM_BLOCK_SIZE", 16384)) // 64))
            except (TypeError, ValueError):
                self._bone_buffer_capacity = 256
            # XV2's original pipeline recomputed/uploaded every animated vertex
            # from numpy on each frame. Skin in the vertex shader via a single
            # persistent uniform block (as done in the newer Uma extractor).
            self.mesh_program = self.ctx.program(
                vertex_shader="""
                    #version 330
                    in vec3 in_position;
                    in vec2 in_uv;
                    in vec3 in_normal;
                    in vec4 in_bone_indices;
                    in vec4 in_bone_weights;
                    uniform mat4 u_mvp;
                    uniform vec2 u_uv_scale;
                    uniform float u_outline_width;
                    uniform bool u_use_skinning;
                    layout(std140) uniform BoneMatrices {
                        mat4 u_bones[__MAX_BONES__];
                    };
                    out vec2 v_uv;
                    out vec3 v_normal;
                    void main() {
                        vec3 p = in_position;
                        vec3 n = in_normal;
                        if (u_use_skinning) {
                            vec3 p_sum = vec3(0.0);
                            vec3 n_sum = vec3(0.0);
                            float total = 0.0;
                            for (int i = 0; i < 4; ++i) {
                                float w = in_bone_weights[i];
                                if (w < 0.000001) continue;
                                int bone = int(in_bone_indices[i] + 0.5);
                                mat4 matrix = u_bones[bone];
                                p_sum += (matrix * vec4(in_position, 1.0)).xyz * w;
                                n_sum += mat3(matrix) * in_normal * w;
                                total += w;
                            }
                            if (total > 0.000001) {
                                p = p_sum / total;
                                n = normalize(n_sum / total);
                            }
                        }
                        n = normalize(n);
                        gl_Position = u_mvp * vec4(p + n * u_outline_width, 1.0);
                        v_uv = vec2(in_uv.x * u_uv_scale.x, in_uv.y * u_uv_scale.y);
                        v_normal = n;
                    }
                """.replace("__MAX_BONES__", str(self._bone_buffer_capacity)),
                fragment_shader="""
                    #version 330
                    uniform sampler2D u_texture;
                    uniform sampler2D u_lightwarp;
                    uniform vec4 u_color;
                    uniform bool u_use_texture;
                    uniform bool u_use_lightwarp;
                    uniform bool u_outline;
                    uniform vec3 u_light_dir;
                    in vec2 v_uv;
                    in vec3 v_normal;
                    out vec4 fragColor;
                    void main() {
                        if (u_outline) {
                            fragColor = vec4(0.025, 0.021, 0.035, 1.0);
                            return;
                        }
                        vec4 base = u_use_texture ? texture(u_texture, fract(v_uv)) : vec4(1.0);
                        if (base.a < 0.015) discard;
                        // Quantized light bands to recover cel shading from the
                        // static fullbright appearance. DYT's selected color
                        // strip supplies each material's shading gradient.
                        float lambert = dot(normalize(v_normal), normalize(u_light_dir));
                        float intensity = clamp(lambert * 0.48 + 0.53, 0.0, 1.0);
                        intensity = intensity < 0.28 ? 0.19 : (intensity < 0.63 ? 0.51 : 0.9);
                        if (u_use_lightwarp) {
                            vec3 shaded = texture(u_lightwarp, vec2(intensity, 0.5)).rgb;
                            // Normalize to the light end of the same DYT band.
                            // The RGB diffuse was already DYT-colored at export.
                            vec3 reference = max(texture(u_lightwarp, vec2(0.90, 0.5)).rgb, vec3(0.12));
                            base.rgb *= clamp(shaded / reference, vec3(0.25), vec3(1.12));
                        } else {
                            base.rgb *= intensity < 0.3 ? 0.47 : (intensity < 0.7 ? 0.77 : 1.0);
                        }
                        fragColor = base * u_color;
                    }
                """,
            )
            try:
                self.mesh_program["BoneMatrices"].binding = 0
            except Exception:
                pass
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
            # Helpers usually use only a few KB. A 4 MiB buffer rewritten
            # several times in every frame can cause driver-side renaming /
            # staging pressure on some Windows OpenGL implementations.
            self.line_vbo = self.ctx.buffer(reserve=256 * 1024, dynamic=True)
            self.line_vao = self.ctx.simple_vertex_array(self.line_program, self.line_vbo, "in_position")
            self.white_texture = self.ctx.texture((1, 1), 4, bytes((255, 255, 255, 255)))
            self.white_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
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
        # QOpenGLWidget owns the GL context here. Queue CPU fallback VBO
        # updates instead of writing GL buffers from the Qt animation timer.
        for surface in self._pending_cpu_uploads.values():
            if self.freeze_gpu_uploads:
                continue
            if surface.gpu.position_vbo is not None and surface.animated_positions_np is not None:
                surface.gpu.position_vbo.write(memoryview(np.ascontiguousarray(surface.animated_positions_np)).cast("B"))
        self._pending_cpu_uploads.clear()
        mvp = self._mvp_matrix()

        if self.show_grid:
            self._render_grid(mvp)
        self._render_axes(mvp)
        if not self.skip_model_draw:
            self._render_models(mvp)
        if self.show_skeleton:
            self._render_skeleton(mvp)
        if self.camera_animation is not None and not self.follow_camera_animation:
            self._render_camera_gizmo(mvp)
        if self._drag_light:
            self._render_light_gizmo(mvp)

    def _release_surface_gpu(self, surface: RenderSurface) -> None:
        for obj in (surface.gpu.vao, surface.gpu.position_vbo, surface.gpu.static_vbo, surface.gpu.ibo, surface.gpu.texture_obj, surface.gpu.lightwarp_obj, surface.gpu.palette_ubo):
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
            surface.lightwarp = None
            surface.vertices.clear()
            surface.triangles.clear()
            if surface.uvs is not None:
                surface.uvs.clear()
            if surface.normals is not None:
                surface.normals.clear()
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
            surface.bind_positions_np = None
            surface.bind_homogeneous_np = None
            surface.skin_bone_indices_np = None
            surface.skin_weights_np = None
            surface.gpu_palette_np = None
            surface.gpu_local_indices_np = None
            surface.animated_positions_np = None
            surface.skin_accumulator_np = None
            surface.skin_temp_np = None
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

        if surface.bind_positions_np is None or surface.bind_positions_np.shape != (vertex_count, 3):
            surface.bind_positions_np = np.ascontiguousarray(bind_vertices, dtype=np.float32)
            # CPU deformation scratch space is allocated on demand only for
            # oversized rigs that cannot fit in the GPU bone uniform buffer.
            surface.animated_positions_np = surface.bind_positions_np

        # Static UV, normal and up-to-four bone influences live on the GPU.
        # Resolving skinning once is critical; rebuilding per frame was costly.
        if not surface.skin_ready:
            self._prepare_surface_skinning(surface)
        static_data = np.zeros((vertex_count, 13), dtype=np.float32)
        if surface.uvs:
            for index, uv in enumerate(surface.uvs[:vertex_count]):
                if uv is not None:
                    static_data[index, :2] = uv
        if surface.normals:
            for index, normal in enumerate(surface.normals[:vertex_count]):
                if normal is not None:
                    static_data[index, 2:5] = normal
        # Degenerate/missing normals should not produce NaNs in the shader.
        lengths = np.linalg.norm(static_data[:, 2:5], axis=1)
        static_data[lengths < 1e-8, 2:5] = (0.0, 0.0, 1.0)
        if surface.skin_bone_indices_np is not None:
            # Oversized character rigs use a compact *per-surface* palette,
            # just as Uma does. A 350-bone character may have only 8-20
            # influences in each mesh; there is no reason to CPU-skin it.
            oversized = self.current_esk is not None and len(self.current_esk.bones) > self._bone_buffer_capacity
            if oversized and surface.gpu_local_indices_np is not None and self._surface_gpu_eligible(surface):
                static_data[:, 5:9] = surface.gpu_local_indices_np
            else:
                static_data[:, 5:9] = surface.skin_bone_indices_np
        if surface.skin_weights_np is not None:
            static_data[:, 9:13] = surface.skin_weights_np
        static_data = np.ascontiguousarray(static_data)
        indices = np.asarray(surface.triangles, dtype=np.uint32).reshape(-1)

        needs_rebuild = (
            rebuild
            or surface.gpu.position_vbo is None
            or surface.gpu.static_vbo is None
            or surface.gpu.index_count != len(indices)
        )
        if needs_rebuild:
            self._release_surface_gpu(surface)
            gpu_eligible = self._surface_gpu_eligible(surface)
            position_data = np.ascontiguousarray(
                surface.bind_positions_np if gpu_eligible else surface.animated_positions_np,
                dtype=np.float32,
            )
            surface.gpu.position_vbo = self.ctx.buffer(position_data.tobytes(), dynamic=not gpu_eligible)
            surface.gpu.static_vbo = self.ctx.buffer(static_data.tobytes())
            surface.gpu.ibo = self.ctx.buffer(indices.tobytes())
            surface.gpu.vao = self.ctx.vertex_array(
                self.mesh_program,
                [
                    (surface.gpu.position_vbo, "3f", "in_position"),
                    (surface.gpu.static_vbo, "2f 3f 4f 4f", "in_uv", "in_normal", "in_bone_indices", "in_bone_weights"),
                ],
                surface.gpu.ibo,
            )
            surface.gpu.index_count = len(indices)
            if gpu_eligible and self.current_esk is not None and len(self.current_esk.bones) > self._bone_buffer_capacity and surface.has_skinning:
                # Fixed-size buffer, allocated once at mesh creation and reused
                # at every pose revision. No per-frame GL object creation.
                surface.gpu.palette_ubo = self.ctx.buffer(
                    reserve=self._bone_buffer_capacity * 64, dynamic=True
                )

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
        lightwarp_key = surface.lightwarp.cacheKey() if surface.lightwarp is not None else None
        if lightwarp_key != surface.gpu.lightwarp_key:
            if surface.gpu.lightwarp_obj is not None:
                surface.gpu.lightwarp_obj.release()
                surface.gpu.lightwarp_obj = None
            if surface.lightwarp is not None:
                data, width, height = self._qimage_rgba_bytes(surface.lightwarp)
                surface.gpu.lightwarp_obj = self.ctx.texture((width, height), 4, data)
                surface.gpu.lightwarp_obj.filter = (moderngl.LINEAR, moderngl.LINEAR)
                surface.gpu.lightwarp_obj.repeat_x = False
                surface.gpu.lightwarp_obj.repeat_y = False
            surface.gpu.lightwarp_key = lightwarp_key
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
        self._pending_cpu_uploads.clear()
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
        self._pose_revision = 0
        self._pose_cache = None
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
                emb: EmbFile | None = None, dyt: EmbFile | None = None,
                secondary_esk: EskFile | None = None) -> None:
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
            lightwarp = None
            scale = (1.0, 1.0)
            if submesh.texture_definitions:
                definition = submesh.texture_definitions[0]
                texture, note = compose_surface_texture(emb, dyt, definition.texture_index, material)
                lightwarp = compose_dyt_lightwarp(dyt, material)
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
                bind_normals = list(submesh.normals)
                weights = list(submesh.blend_weights)
                # SCD secondary rigs use their own bind coordinate space. Apply
                # the same per-influence bind correction as SMD physics export.
                if secondary_esk is not None and self.current_esk is not None:
                    bind_vertices, bind_normals = self._attach_physics_bind(
                        bind_vertices, bind_normals, submesh.blend_indices,
                        weights, group.bone_names, secondary_esk,
                    )
                weighted_vertices += sum(value is not None for value in weights)
                surfaces.append(RenderSurface(
                    name=submesh.name,
                    vertices=list(bind_vertices),
                    triangles=triangles,
                    normals=bind_normals,
                    color=rgba,
                    uvs=list(submesh.uvs),
                    texture=texture,
                    lightwarp=lightwarp,
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
        self.update()
        notes = "; ".join(sorted(texture_notes)) if texture_notes else "material fallback"
        self.statusChanged.emit(
            f"Loaded {emd.path.name}: {vertex_count:,} vertices, {triangle_count:,} triangles, "
            f"{weighted_vertices:,} weighted records, GPU textures: {notes}"
        )

    def _attach_physics_bind(self, vertices, normals, blend_indices, blend_weights,
                             palette_names, secondary_esk):
        """Relocate a secondary SCD mesh into the combined character rig.

        A secondary ESK's EXP root is not in the main character's origin; using
        its mesh coordinates uncorrected puts cloth/hair at (0,0,0) or pelvis.
        """
        from .source_export import _global_bind_matrices
        if self.current_esk is None:
            return vertices, normals
        old_global = _global_bind_matrices(secondary_esk)
        new_global = _global_bind_matrices(self.current_esk)
        old_by_name = {bone.name.casefold(): bone.index for bone in secondary_esk.bones}
        new_by_name = {bone.name.casefold(): bone.index for bone in self.current_esk.bones}
        corrections = []
        for name in palette_names:
            old = old_by_name.get(name.casefold())
            new = new_by_name.get(name.casefold())
            if old is None or new is None:
                corrections.append(np.eye(4, dtype=np.float32))
                continue
            try:
                corrections.append((new_global[new] @ np.linalg.inv(old_global[old])).astype(np.float32))
            except np.linalg.LinAlgError:
                corrections.append(np.eye(4, dtype=np.float32))
        if not corrections:
            return vertices, normals
        points = np.asarray(vertices, dtype=np.float32)
        n = np.asarray([(v or (0.0, 0.0, 1.0)) for v in normals], dtype=np.float32)
        adjusted_p = points.copy()
        adjusted_n = n.copy()
        for i in range(len(points)):
            ids = blend_indices[i] if i < len(blend_indices) else None
            ws = blend_weights[i] if i < len(blend_weights) else None
            if ids is None or ws is None:
                continue
            acc_p = np.zeros(3, dtype=np.float32)
            acc_n = np.zeros(3, dtype=np.float32)
            total = 0.0
            for slot in range(4):
                j = ids[slot]
                w = float(ws[slot])
                if not (0 <= j < len(corrections)) or w <= 0:
                    continue
                c = corrections[j]
                acc_p += (c[:3, :3] @ points[i] + c[:3, 3]) * w
                acc_n += (c[:3, :3] @ n[i]) * w
                total += w
            if total > 1e-8:
                adjusted_p[i] = acc_p / total
                length = np.linalg.norm(acc_n)
                if length > 1e-8:
                    adjusted_n[i] = acc_n / length
        return [tuple(v) for v in adjusted_p], [tuple(v) for v in adjusted_n]

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
        self._pose_cache = None
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
        if self.current_animation is not animation:
            self._pose_cache = None  # Free previous clip immediately.
        self.current_animation = animation
        self.animation_frame = 0.0
        self._rebuild_skeleton_pose()
        if animation is None:
            self.statusChanged.emit("Animation cleared; showing bind pose")
        else:
            self.statusChanged.emit(f"Loaded animation {animation.name}: {animation.duration} frames")

    def set_overlay_animations(self, facial: EanAnimation | None = None,
                               lip: EanAnimation | None = None) -> None:
        if self.facial_animation is not facial or self.lip_animation is not lip:
            self._pose_cache = None
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
        by_name = {bone.name.casefold(): bone.index for bone in self.current_esk.bones}
        for part in self.parts.values():
            for surface in part.surfaces:
                surface.resolved_bones = [by_name.get(name.casefold()) for name in (surface.bone_palette_names or [])]
                surface.skin_ready = False
                self._prepare_surface_skinning(surface)
                surface.gpu_dirty = True

    def _prepare_surface_skinning(self, surface: RenderSurface) -> None:
        bind_vertices = surface.bind_vertices or surface.vertices
        vertex_count = len(bind_vertices)
        if vertex_count <= 0:
            return

        if surface.bind_positions_np is None or surface.bind_positions_np.shape != (vertex_count, 3):
            surface.bind_positions_np = np.ascontiguousarray(bind_vertices, dtype=np.float32)
            # CPU deformation scratch space is allocated on demand only for
            # oversized rigs that cannot fit in the GPU bone uniform buffer.
            surface.animated_positions_np = surface.bind_positions_np

        indices = np.full((vertex_count, 4), -1, dtype=np.int32)
        weights = np.zeros((vertex_count, 4), dtype=np.float32)
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
                if 0 <= palette_index < len(palette):
                    resolved = palette[palette_index]
                    if resolved is not None:
                        indices[vertex_index, slot] = int(resolved)
                        weights[vertex_index, slot] = float(weights4[slot])

        totals = weights.sum(axis=1)
        valid = totals > 1e-8
        if np.any(valid):
            weights[valid] /= totals[valid, None]
        surface.skin_bone_indices_np = indices
        surface.skin_weights_np = weights
        surface.has_skinning = bool(np.any(valid))
        # Build palette/index remap once, not once per animation frame.
        used = np.unique(indices[weights > 1e-8])
        used = used[used >= 0]
        surface.gpu_palette_np = used
        if len(used):
            # searchsorted on an already sorted palette; protect -1 indices.
            local = np.searchsorted(used, np.maximum(indices, 0)).astype(np.float32)
            local[weights <= 1e-8] = 0.0
            surface.gpu_local_indices_np = local
        else:
            surface.gpu_local_indices_np = None
        surface.skin_ready = True

    def _surface_gpu_eligible(self, surface: RenderSurface) -> bool:
        if not self.use_gpu_skinning or self.current_esk is None or not self._gl_ready:
            return False
        if not surface.has_skinning:
            return True  # static geometry never needs CPU deformation
        if len(self.current_esk.bones) <= self._bone_buffer_capacity:
            return True
        palette = surface.gpu_palette_np
        return palette is not None and 0 < len(palette) <= self._bone_buffer_capacity

    def _upload_surface_palette(self, surface: RenderSurface, matrices: np.ndarray) -> bool:
        """Reuse an independent per-surface UBO for skeletons over GL's UBO limit."""
        gpu = surface.gpu
        if gpu.palette_ubo is None or surface.gpu_palette_np is None:
            return False
        if self.freeze_gpu_uploads:
            gpu.palette_ubo.bind_to_uniform_block(0)
            return True
        if gpu.palette_revision != self._pose_revision:
            palette = surface.gpu_palette_np
            # Copy only the bones actually referenced by this surface.
            compact = np.ascontiguousarray(matrices[palette].transpose(0, 2, 1))
            gpu.palette_ubo.write(memoryview(compact).cast("B"))
            gpu.palette_revision = self._pose_revision
        gpu.palette_ubo.bind_to_uniform_block(0)
        return True

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
        if self.freeze_gpu_uploads:
            return
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
            self._bone_upload_buffer[:] = np.eye(4, dtype=np.float32)
        self._bone_upload_buffer[:count] = np.transpose(matrices[:count], (0, 2, 1))
        self.bone_buffer.write(memoryview(np.ascontiguousarray(self._bone_upload_buffer[:count])).cast("B"))
        self._bone_texture_dirty = False

    def _skin_meshes(self, precomputed_matrices: np.ndarray | None = None) -> None:
        """GPU skin when possible; allocate bounded scratch for rare CPU fallbacks.

        Uma's mesh-local palettes prevent a large full skeleton from pushing
        otherwise-small meshes onto the CPU. For an exceptional single mesh
        exceeding the UBO limit, work in fixed 2048-vertex chunks rather than
        constructing multi-megabyte boolean-indexed matrices per tick.
        """
        matrices = (precomputed_matrices if precomputed_matrices is not None
                    else self._build_skin_matrices())
        bone_count = len(matrices)
        self._pose_revision += 1
        self._bone_texture_data = matrices
        self._bone_texture_dirty = True

        # The pose advances without any CPU vertex calculation for surfaces
        # supported by the GPU (including mesh-local bone palettes).
        for part in self.parts.values():
            if not part.visible:
                continue
            for surface in part.surfaces:
                if not surface.skin_ready:
                    self._prepare_surface_skinning(surface)
                if self._surface_gpu_eligible(surface):
                    continue
                if surface.bind_positions_np is None:
                    continue
                if surface.bind_homogeneous_np is None:
                    vertex_count = len(surface.bind_positions_np)
                    surface.bind_homogeneous_np = np.empty((vertex_count, 4), dtype=np.float32)
                    surface.bind_homogeneous_np[:, :3] = surface.bind_positions_np
                    surface.bind_homogeneous_np[:, 3] = 1.0
                    surface.animated_positions_np = surface.bind_positions_np.copy()
                    # Persistent O(chunk-size) scratch instead of O(mesh-size).
                    surface.skin_accumulator_np = np.empty((2048, 4), dtype=np.float32)
                    surface.skin_temp_np = np.empty((2048, 4), dtype=np.float32)
                bind_h = surface.bind_homogeneous_np
                indices = surface.skin_bone_indices_np
                weights = surface.skin_weights_np
                output = surface.animated_positions_np
                accum = surface.skin_accumulator_np
                temp = surface.skin_temp_np
                if bind_h is None or output is None or accum is None or temp is None:
                    continue
                if indices is None or weights is None or not surface.has_skinning:
                    np.copyto(output, surface.bind_positions_np)
                else:
                    for offset in range(0, len(bind_h), 2048):
                        end = min(offset + 2048, len(bind_h))
                        n = end - offset
                        out_chunk = output[offset:end]
                        bind_chunk = bind_h[offset:end]
                        inds_chunk = indices[offset:end]
                        weights_chunk = weights[offset:end]
                        acc = accum[:n]
                        acc.fill(0.0)
                        # No full-mesh mask/gather arrays, no growing frame cache.
                        for slot in range(4):
                            bones = inds_chunk[:, slot]
                            ws = weights_chunk[:, slot]
                            valid = (bones >= 0) & (bones < bone_count) & (ws > 1e-8)
                            if not np.any(valid):
                                continue
                            where = np.flatnonzero(valid)
                            # Bounded gather; never more than 2048 matrices.
                            np.einsum("nij,nj->ni", matrices[bones[valid]],
                                      bind_chunk[valid], out=temp[:len(where)], optimize=False)
                            acc[where] += temp[:len(where)] * ws[valid, None]
                        total = weights_chunk.sum(axis=1)
                        good = total > 1e-8
                        out_chunk[~good] = surface.bind_positions_np[offset:end][~good]
                        out_chunk[good] = acc[good, :3] / total[good, None]
                if self._gl_ready and surface.gpu.position_vbo is not None:
                    self._pending_cpu_uploads[id(surface)] = surface

        self.update()

    def _rebuild_skeleton_pose(self) -> None:
        esk = self.current_esk
        if esk is None:
            self.skeleton_points = []
            self.skeleton_links = []
            self.update()
            return
        # Cache the fully evaluated *viewport pose*, not another copy of the
        # decoded EAN file. This is bounded to 96 MiB and evicted when the
        # animation or overlay changes. Only integral authored frames cache;
        # fractional scrubbing still uses the exact original interpolator.
        if self.current_animation is not None and self._pose_cache is None:
            self._pose_cache = DensePoseCache.create(
                self.current_animation.duration, len(esk.bones)
            )
        cached = self._pose_cache.get(self.animation_frame) if self._pose_cache else None
        if cached is not None:
            pose, skin = cached
            self._pose_cache_hits += 1
            # Use separate tuple lists: normal viewport helpers expect Vec3/Quat
            # sequences, while GPU uploads consume the cached float32 matrices.
            self.skeleton_points = [tuple(v) for v in pose[:, :3]]
            self._pose_positions = self.skeleton_points
            self._pose_rotations = [tuple(v) for v in pose[:, 3:7]]
            self._pose_scales = [tuple(v) for v in pose[:, 7:10]]
            self.skeleton_links = [
                (bone.parent, bone.index) for bone in esk.bones
                if bone.parent is not None and 0 <= bone.parent < len(esk.bones)
            ]
            self._skin_meshes(precomputed_matrices=skin)
            return
        self._pose_cache_misses += 1
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
        if self._pose_cache is not None:
            self._pose_cache.store(
                self.animation_frame, points, rotations, scales,
                self._bone_texture_data,
            )
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
        # One uniform update per rendered frame, not per submesh/draw pass.
        y = math.radians(self.light_yaw)
        p = math.radians(self.light_pitch)
        self.mesh_program["u_light_dir"].value = (
            math.sin(y) * math.cos(p), math.sin(p), math.cos(y) * math.cos(p)
        )
        gpu_skin = (self.use_gpu_skinning and self._gl_ready and self.current_esk is not None)
        global_palette = gpu_skin and len(self.current_esk.bones) <= self._bone_buffer_capacity
        if global_palette:
            self._upload_bone_buffer_if_needed()
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
                has_skinning = bool(gpu_skin and surface.has_skinning and self._surface_gpu_eligible(surface))
                if has_skinning and not global_palette:
                    has_skinning = self._upload_surface_palette(surface, self._bone_texture_data)
                # The shared UBO was bound once above for a normal-size rig;
                # avoid rebinding the identical buffer for every surface.
                self.mesh_program["u_use_skinning"].value = has_skinning
                self.mesh_program["u_use_lightwarp"].value = surface.gpu.lightwarp_obj is not None
                if surface.gpu.lightwarp_obj is not None:
                    surface.gpu.lightwarp_obj.use(1)
                self.mesh_program["u_lightwarp"].value = 1
                if self.show_solid:
                    self.ctx.wireframe = False

                    # Anime-style inverted hull: expand along vertex normals,
                    # cull the normal front faces, and render the remaining
                    # back faces as a flat black shell behind the model.
                    self.ctx.enable(moderngl.CULL_FACE)
                    # XV2 surface winding is opposite ModernGL's default
                    # front-face convention. Cull the winding that represents
                    # the visible exterior so only the inverted hull's inner
                    # faces remain around the silhouette.
                    self.ctx.cull_face = "back"
                    self.mesh_program["u_outline"].value = True
                    self.mesh_program["u_outline_width"].value = OUTLINE_WIDTH
                    self.mesh_program["u_color"].value = (0.0, 0.0, 0.0, 1.0)
                    self.mesh_program["u_use_texture"].value = False
                    self.mesh_program["u_uv_scale"].value = (1.0, 1.0)
                    self.white_texture.use(0)
                    self.mesh_program["u_texture"].value = 0
                    surface.gpu.vao.render(moderngl.TRIANGLES)

                    # The original viewer rendered meshes double-sided.
                    # Restore that behavior for the actual textured surface.
                    self.ctx.disable(moderngl.CULL_FACE)
                    self.mesh_program["u_outline"].value = False
                    self.mesh_program["u_outline_width"].value = 0.0
                    self.mesh_program["u_color"].value = (1.0, 1.0, 1.0, rgba[3]) if has_texture else rgba
                    self.mesh_program["u_use_texture"].value = has_texture
                    self.mesh_program["u_uv_scale"].value = surface.texture_scale
                    (surface.gpu.texture_obj or self.white_texture).use(0)
                    self.mesh_program["u_texture"].value = 0
                    surface.gpu.vao.render(moderngl.TRIANGLES)
                    self.ctx.disable(moderngl.CULL_FACE)
                if self.show_wireframe:
                    self.mesh_program["u_outline"].value = False
                    self.mesh_program["u_use_lightwarp"].value = False
                    self.ctx.wireframe = True
                    self.mesh_program["u_outline_width"].value = 0.0
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
        # Send the NumPy memoryview directly: no per-draw bytes duplication.
        data = memoryview(np.ascontiguousarray(vertices, dtype=np.float32)).cast("B")
        if len(data) > self.line_vbo.size:
            # Grow geometrically only when a helper genuinely needs more space.
            self.line_vbo.orphan(1 << (len(data) - 1).bit_length())
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
        if count > 200:
            # Avoid huge helper arrays for unusually scaled imported meshes.
            count = 200
            step = extent / count
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

    def _render_light_gizmo(self, mvp: np.ndarray) -> None:
        """Show a small direction handle while Ctrl-dragging the toon light."""
        yaw, pitch = math.radians(self.light_yaw), math.radians(self.light_pitch)
        direction = (math.sin(yaw) * math.cos(pitch), math.sin(pitch),
                     math.cos(yaw) * math.cos(pitch))
        size = max(0.5, self._scene_radius * 1.25)
        marker = _vadd(self.target, _vmul(direction, size))
        self._draw_lines([self.target, marker], (1.0, 0.92, 0.4, 0.85), mvp, 1.5)
        self._draw_lines([marker], (1.0, 0.98, 0.65, 1.0), mvp, 8.0, moderngl.POINTS)

    # -------------------------------------------------------------- interaction
    def _invalidate_frame(self, interactive: bool = False) -> None:
        del interactive
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._last_mouse = event.position()
        self._drag_button = event.button()
        self._drag_light = bool(
            event.button() == Qt.LeftButton and event.modifiers() & Qt.ControlModifier
        )
        self.setFocus()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last_mouse is None:
            return
        delta = event.position() - self._last_mouse
        self._last_mouse = event.position()
        if self._drag_light and self._drag_button == Qt.LeftButton:
            # Rotate the directional light independently of camera orbit.
            # Also works when following an animated camera.
            self.light_yaw = (self.light_yaw + delta.x() * 0.6) % 360.0
            self.light_pitch = max(-89.0, min(89.0, self.light_pitch - delta.y() * 0.6))
        elif self.follow_camera_animation:
            return
        elif self._drag_button == Qt.LeftButton:
            self.yaw += delta.x() * 0.45
            self.pitch = max(-89.0, min(89.0, self.pitch + delta.y() * 0.45))
        elif self._drag_button in (Qt.MiddleButton, Qt.RightButton):
            self.pan = (self.pan[0] + delta.x(), self.pan[1] + delta.y())
        self.update()

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:
        if self._drag_light:
            self.statusChanged.emit(
                f"Cel light moved: yaw {self.light_yaw:.1f}°, elevation {self.light_pitch:.1f}°"
            )
        self._last_mouse = None
        self._drag_button = Qt.NoButton
        self._drag_light = False
        self.update()

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
