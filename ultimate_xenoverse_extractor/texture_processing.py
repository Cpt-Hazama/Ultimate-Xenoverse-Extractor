from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Iterable

from PIL import Image, ImageOps
from PySide6.QtGui import QImage

from .parsers.emb import EmbEntry, EmbFile
from .parsers.emm import EmmMaterial

DEFAULT_LEVELS = (100.0, 1.0, 155.0)
DEFAULT_OUTPUT = (0.0, 255.0)
DYT_WIDTH = 128
DYT_HEIGHT = 128
DYT_BAND_HEIGHT = 4
DYT_STEP_Y = 16
DYT_END_GAP = 5


@dataclass(slots=True)
class DytBand:
    index: int
    y: int
    image: Image.Image
    color2: tuple[int, int, int]


def entry_to_pil(entry: EmbEntry) -> Image.Image | None:
    try:
        with Image.open(BytesIO(entry.data)) as image:
            return image.convert("RGBA")
    except Exception:
        return None


def pil_to_qimage(image: Image.Image) -> QImage:
    rgba = image.convert("RGBA")
    raw = rgba.tobytes("raw", "RGBA")
    return QImage(raw, rgba.width, rgba.height, rgba.width * 4, QImage.Format_RGBA8888).copy()


def extract_alpha(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA"):
        return image.getchannel("A").convert("L")
    return image.convert("L")


def apply_levels(
    channel: Image.Image,
    black: float = DEFAULT_LEVELS[0],
    gamma: float = DEFAULT_LEVELS[1],
    white: float = DEFAULT_LEVELS[2],
    out_low: float = DEFAULT_OUTPUT[0],
    out_high: float = DEFAULT_OUTPUT[1],
) -> Image.Image:
    channel = channel.convert("L")
    denominator = max(1e-6, white - black)
    inverse_gamma = 1.0 / max(1e-6, gamma)
    lut: list[int] = []
    for value in range(256):
        normalized = max(0.0, min(1.0, (value - black) / denominator))
        normalized = pow(normalized, inverse_gamma)
        output = out_low + normalized * (out_high - out_low)
        lut.append(int(round(max(0.0, min(255.0, output)))))
    return channel.point(lut)


def process_diffuse_alpha(
    image: Image.Image,
    levels: tuple[float, float, float] = DEFAULT_LEVELS,
    output_levels: tuple[float, float] = DEFAULT_OUTPUT,
    invert: bool = True,
) -> Image.Image:
    result = apply_levels(extract_alpha(image), levels[0], levels[1], levels[2], output_levels[0], output_levels[1])
    return ImageOps.invert(result) if invert else result


def _is_void(image: Image.Image) -> bool:
    rgba = image.convert("RGBA")
    return all(r == 0 and g == 0 and b == 0 for r, g, b, _a in rgba.getdata())


def find_dyt_bands(image: Image.Image) -> list[DytBand]:
    image = image.convert("RGBA")
    if image.size != (DYT_WIDTH, DYT_HEIGHT):
        return []
    bands: list[DytBand] = []
    y = 0
    while y < DYT_HEIGHT:
        end_test = image.crop((0, y, DYT_WIDTH, min(y + DYT_END_GAP, DYT_HEIGHT)))
        if _is_void(end_test):
            break
        band = image.crop((0, y, DYT_WIDTH, min(y + DYT_BAND_HEIGHT, DYT_HEIGHT)))
        sample = band.getpixel((min(120, band.width - 1), min(1, band.height - 1)))
        bands.append(DytBand(len(bands), y, band, tuple(sample[:3])))
        y += DYT_STEP_Y
    return bands


def _material_dyt_index(material: EmmMaterial | None) -> int:
    if material is None:
        return 0
    # Xenoverse character materials commonly store the DYT row selector in
    # MatScale1X. Values in stock character EMMs are normally whole numbers.
    try:
        return max(0, int(round(float(material.value("MatScale1X", 0.0)))))
    except (TypeError, ValueError):
        return 0


def compose_dyt_texture(
    base_image: Image.Image,
    dyt_image: Image.Image,
    material: EmmMaterial | None = None,
    band_index: int | None = None,
) -> tuple[Image.Image, int, int]:
    """Compose an XV2 viewport diffuse from the base alpha and a DYT band.

    Returns ``(image, requested_band, actual_band)`` so the UI can report when
    an EMM row points past the available DYT bands.
    """
    mask = process_diffuse_alpha(base_image)
    bands = find_dyt_bands(dyt_image)
    if not bands:
        grayscale = Image.merge("RGBA", (mask, mask, mask, Image.new("L", mask.size, 255)))
        return grayscale, 0, 0

    requested = _material_dyt_index(material) if band_index is None else max(0, int(band_index))
    actual = min(requested, len(bands) - 1)
    band = bands[actual].image.convert("RGBA")

    # The four lines in one DYT lightmap are intentionally collapsed to a
    # single 128-entry palette for the software viewport. Averaging them avoids
    # choosing a single edge row and follows the 4px extraction used by dyt_ui.
    palette: list[tuple[int, int, int, int]] = []
    for x in range(band.width):
        samples = [band.getpixel((x, y)) for y in range(band.height)]
        palette.append(tuple(sum(pixel[channel] for pixel in samples) // len(samples) for channel in range(4)))

    pixels = bytearray(mask.width * mask.height * 4)
    alpha_pixels = mask.tobytes()
    for index, shade in enumerate(alpha_pixels):
        palette_x = int(round((shade / 255.0) * (len(palette) - 1)))
        r, g, b, _a = palette[palette_x]
        offset = index * 4
        pixels[offset:offset + 4] = bytes((r, g, b, 255))
    return Image.frombytes("RGBA", mask.size, bytes(pixels)), requested, actual


def choose_dyt_image(dyt: EmbFile | None, preferred_index: int = 0) -> Image.Image | None:
    if dyt is None or not dyt.entries:
        return None
    ordered = list(dyt.entries)
    if 0 <= preferred_index < len(ordered):
        ordered.insert(0, ordered.pop(preferred_index))
    for entry in ordered:
        image = entry_to_pil(entry)
        if image is not None and image.size == (DYT_WIDTH, DYT_HEIGHT):
            return image
    return None


def compose_surface_texture(
    base: EmbFile | None,
    dyt: EmbFile | None,
    texture_index: int,
    material: EmmMaterial | None,
) -> tuple[QImage | None, str]:
    if base is None or not 0 <= texture_index < len(base.entries):
        return None, "no base texture"
    base_image = entry_to_pil(base.entries[texture_index])
    if base_image is None:
        return None, "base texture could not be decoded"

    dyt_image = choose_dyt_image(dyt)
    if dyt_image is None:
        # Useful fallback for assets which genuinely use an ordinary RGBA map.
        return pil_to_qimage(base_image), "raw EMB"

    composed, requested, actual = compose_dyt_texture(base_image, dyt_image, material)
    note = f"DYT row {actual}"
    if requested != actual:
        note += f" (requested {requested}, clamped)"
    return pil_to_qimage(composed), note


def compose_dyt_lightwarp(dyt: EmbFile | None, material: EmmMaterial | None) -> QImage | None:
    """Extract the very same material-selected DYT strip as a GPU lightwarp.

    Keep the original 128 horizontal palette samples (averaging the 4-pixel
    strip vertically), rather than baking a single constant into the shader.
    Missing/nonstandard DYT files fall back to neutral cel lighting.
    """
    image = choose_dyt_image(dyt)
    if image is None:
        return None
    bands = find_dyt_bands(image)
    if not bands:
        return None
    index = min(_material_dyt_index(material), len(bands) - 1)
    band = bands[index].image.convert("RGBA")
    pixels = []
    for x in range(band.width):
        colors = [band.getpixel((x, y)) for y in range(band.height)]
        pixels.append(tuple(sum(color[c] for color in colors) // len(colors) for c in range(3)) + (255,))
    strip = Image.new("RGBA", (band.width, 1))
    strip.putdata(pixels)
    return pil_to_qimage(strip)
