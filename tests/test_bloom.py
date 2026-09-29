"""Testes da Fase 5b: bloom (glare e clássico) na GPU, por propriedades (matriz I/O).

Não há réplica do bloom na CPU: os testes conferem as propriedades da PSF (média 1,
simetria, decaimento, suporte finito) e a cadeia de saída contra ``scene.encode_pixel``.
A validação da ``BloomConfig`` e os tamanhos da cadeia rodam sem GL; o resto usa um
contexto standalone e dá ``skip`` sem OpenGL 3.3.
"""

import math
from dataclasses import replace

import moderngl
import numpy as np
import pytest

from blackhole import scene
from blackhole.bloom import (
    DOWN_CENTER_WEIGHT,
    DOWN_CORNER_WEIGHT,
    DOWN_EDGE_WEIGHT,
    DOWN_INNER_WEIGHT,
    MODES,
    TENT_CENTER_WEIGHT,
    TENT_CORNER_WEIGHT,
    TENT_EDGE_WEIGHT,
    BloomPass,
    hdr_texture,
    mip_sizes,
    validate_bloom,
)
from blackhole.camera import Camera
from blackhole.config import BloomConfig, CameraConfig, GLCheckConfig
from blackhole.render_gpu import GpuRenderer

BLOOM = BloomConfig()
BOTH_MODES = tuple(MODES)
# Tolerância da saída codificada (8 bits) contra a cadeia de saída da CPU.
ONE_LSB = 1

INVALID = [
    {"glare_strength": -0.01},
    {"glare_strength": 1.01},
    {"classic_strength": -0.5},
    {"classic_strength": 2.0},
    {"glare_strength": math.nan},
    {"glare_strength": True},
    {"classic_strength": True},
    {"threshold": True},
    {"threshold": False},
    {"threshold": -1.0e-6},
    {"threshold": math.inf},
    {"threshold": math.nan},
    {"levels": 0},
    {"levels": -3},
    {"levels": 2.5},
    {"levels": True},
    {"mode": "lens_flare"},
]


@pytest.fixture(scope="module")
def ctx():
    try:
        context = moderngl.create_standalone_context(require=GLCheckConfig().min_version_code)
    except Exception as exc:  # moderngl/glcontext não expõem um tipo de erro específico
        pytest.skip(f"sem contexto OpenGL 3.3: {exc}")
    yield context
    context.release()


@pytest.fixture(scope="module")
def bloom(ctx):
    bloom_pass = BloomPass(ctx)
    yield bloom_pass
    bloom_pass.release()


def upload(ctx, image):
    """Textura HDR RGBA16F com a radiância ``image`` (H, W, 3), linha 0 embaixo."""
    height, width, _ = image.shape
    rgba = np.dstack([image, np.ones((height, width))]).astype(np.float16)
    texture = hdr_texture(ctx, (width, height))
    texture.write(rgba.tobytes())
    return texture


def as_half(image):
    """A radiância como ela fica na textura RGBA16F."""
    return np.asarray(image, dtype=np.float16).astype(np.float64)


def apply(ctx, bloom_pass, image, mode, encode):
    """Bloom de ``image``: uint8 codificado (``encode``) ou radiância linear float32."""
    height, width, _ = image.shape
    source = upload(ctx, image)
    dtype = "f1" if encode else "f4"
    target_texture = ctx.texture((width, height), 4, dtype=dtype)
    target = ctx.framebuffer(color_attachments=[target_texture])
    before = ctx.fbo
    try:
        bloom_pass.render(source, target, mode=mode, encode=encode)
        data = target.read(components=3, alignment=1, dtype=dtype)
    finally:
        ctx.fbo = before
        target.release()
        target_texture.release()
        source.release()
    numpy_dtype = np.uint8 if encode else np.float32
    return np.frombuffer(data, dtype=numpy_dtype).reshape(height, width, 3)


def expected_encoded(image):
    """Cadeia de saída da CPU (ACES + sRGB, exposição 1) de cada pixel, em uint8."""
    half = as_half(image)
    out = np.empty(half.shape, dtype=np.uint8)
    for j in range(half.shape[0]):
        for i in range(half.shape[1]):
            rgb = scene.encode_pixel(half[j, i, 0], half[j, i, 1], half[j, i, 2], 1.0)
            out[j, i] = np.round(np.clip(rgb, 0.0, 1.0) * 255.0)
    return out


def assert_within_one_lsb(actual, expected):
    diff = np.abs(actual.astype(int) - expected.astype(int))
    assert diff.max() <= ONE_LSB, f"diferença máxima {diff.max()}/255"


def point_image(size, position, value):
    width, height = size
    image = np.zeros((height, width, 3))
    image[position[1], position[0]] = value
    return image


# --- configuração (sem GL) -----------------------------------------------------


def test_defaults_are_valid_and_match_plan():
    assert validate_bloom(BLOOM) is BLOOM
    assert (BLOOM.enabled, BLOOM.mode) == (True, "glare")
    assert 0.0 <= BLOOM.glare_strength <= 1.0 and 0.0 <= BLOOM.classic_strength <= 1.0
    assert BLOOM.threshold >= 0.0 and BLOOM.levels >= 1
    assert set(MODES) == {"glare", "classic"}


@pytest.mark.parametrize("change", INVALID)
def test_invalid_config_raises(change):
    with pytest.raises(ValueError):
        validate_bloom(replace(BLOOM, **change))


@pytest.mark.parametrize("change", INVALID[::4])
def test_invalid_config_raises_on_construction(ctx, change):
    with pytest.raises(ValueError):
        BloomPass(ctx, replace(BLOOM, **change))


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_strength_limits_accepted(value):
    validate_bloom(replace(BLOOM, glare_strength=value, classic_strength=value, threshold=0.0))


def test_filter_weights_sum_to_one():
    """PSF de média 1: os pesos de cada filtro somam 1."""
    down = DOWN_CENTER_WEIGHT + 4 * (DOWN_INNER_WEIGHT + DOWN_EDGE_WEIGHT + DOWN_CORNER_WEIGHT)
    tent = TENT_CENTER_WEIGHT + 4 * (TENT_EDGE_WEIGHT + TENT_CORNER_WEIGHT)
    assert down == 1.0 and tent == 1.0


def test_luminance_weights_are_rec709_y_row():
    """Pesos do limiar do clássico = linha Y da inversa de XYZ → sRGB linear."""
    y_row = np.linalg.inv(scene.XYZ_TO_SRGB)[1]
    np.testing.assert_allclose(scene.LUMINANCE_WEIGHTS, y_row, atol=1e-3)
    assert sum(scene.LUMINANCE_WEIGHTS) == pytest.approx(1.0, abs=1e-12)


def test_mip_sizes_halve_up_to_levels():
    assert mip_sizes((1920, 1080), 6) == [
        (960, 540),
        (480, 270),
        (240, 135),
        (120, 67),
        (60, 33),
        (30, 16),
    ]
    assert mip_sizes((1920, 1080), 1) == [(960, 540)]


@pytest.mark.parametrize(
    "size, expected",
    [
        ((5, 3), [(2, 1), (1, 1)]),
        ((1, 1), [(1, 1)]),
        ((40, 1), [(20, 1), (10, 1), (5, 1), (2, 1), (1, 1)]),
    ],
)
def test_mip_sizes_stop_at_one_pixel(size, expected):
    """Imagem pequena: a cadeia para no nível de 1×1 px, antes dos ``levels`` níveis."""
    assert mip_sizes(size, 8) == expected


# --- matriz I/O (GL) ------------------------------------------------------------


@pytest.mark.parametrize("value", [0.0, 0.02, 0.4, 1.0, 7.5])
def test_glare_uniform_image_is_unchanged(ctx, bloom, value):
    image = np.full((45, 80, 3), value) * np.array([1.0, 0.8, 0.5])
    out = apply(ctx, bloom, image, "glare", encode=True)
    assert_within_one_lsb(out, expected_encoded(image))


def test_output_chain_matches_cpu_encode_pixel(ctx):
    """Sem bloom (força 0), a composição é a cadeia de saída da CPU (ACES + sRGB)."""
    rng = np.random.default_rng(7)
    image = rng.exponential(0.5, (24, 40, 3)) ** 2
    local = BloomPass(ctx, replace(BLOOM, glare_strength=0.0, classic_strength=0.0))
    try:
        for mode in BOTH_MODES:
            assert_within_one_lsb(apply(ctx, local, image, mode, True), expected_encoded(image))
    finally:
        local.release()


@pytest.mark.parametrize(
    "strength, size",
    [
        (BLOOM.glare_strength, (320, 180)),
        (BLOOM.glare_strength, (97, 61)),
        # Só a PSF (k = 1): a cadeia sozinha conserva a média. Em tamanhos ímpares muito
        # pequenos (97×61 termina em 3×1 → 1×1) o clamp-to-edge desvia alguns %.
        (1.0, (320, 180)),
    ],
)
def test_glare_conserves_mean_energy(ctx, strength, size):
    """Glare: média linear da saída = média da entrada (±1%), na cena do buraco negro."""
    width, height = size
    renderer = GpuRenderer(ctx)
    source = hdr_texture(ctx, size)
    fbo = ctx.framebuffer(color_attachments=[source])
    local = BloomPass(ctx, replace(BLOOM, glare_strength=strength))
    target_texture = ctx.texture(size, 4, dtype="f4")
    target = ctx.framebuffer(color_attachments=[target_texture])
    before = ctx.fbo
    try:
        camera = Camera.from_config(CameraConfig())
        renderer.set_uniforms(camera, width, height, texture=True, time=30.0, hdr=True)
        fbo.use()
        fbo.viewport = (0, 0, width, height)
        renderer.draw()
        data = fbo.read(components=3, alignment=1, dtype="f2")
        scene_hdr = np.frombuffer(data, dtype=np.float16).astype(np.float64)
        local.render(source, target, mode="glare", encode=False)
        out = np.frombuffer(target.read(components=3, alignment=1, dtype="f4"), dtype=np.float32)
    finally:
        ctx.fbo = before
        for resource in (target, target_texture, local, fbo, source, renderer):
            resource.release()
    assert scene_hdr.mean() > 0.0
    assert out.mean() == pytest.approx(scene_hdr.mean(), rel=0.01)


@pytest.mark.parametrize("mode", BOTH_MODES)
def test_bright_point_halo_symmetric_and_decreasing(ctx, bloom, mode):
    size, center = (129, 129), (64, 64)
    out = apply(ctx, bloom, point_image(size, center, 1000.0), mode, encode=False)[..., 1]
    assert np.unravel_index(out.argmax(), out.shape) == (center[1], center[0])
    cx, cy = center
    distances = np.arange(1, 48)
    right = out[cy, cx + distances]
    left = out[cy, cx - distances]
    up = out[cy + distances, cx]
    down = out[cy - distances, cx]
    for profile in (right, left, up, down):
        assert (profile > 0.0).all()
        assert (np.diff(profile) < 0.0).all(), "o halo precisa decrescer com a distância"
    np.testing.assert_allclose(left, right, rtol=0.05)
    np.testing.assert_allclose(up, right, rtol=0.05)
    np.testing.assert_allclose(down, left, rtol=0.05)
    # Diagonais: a mesma distância radial dá quase o mesmo valor em qualquer direção.
    diag = distances[:32]
    ne = out[cy + diag, cx + diag]
    sw = out[cy - diag, cx - diag]
    np.testing.assert_allclose(ne, sw, rtol=0.05)


@pytest.mark.parametrize("mode", BOTH_MODES)
def test_far_pixel_stays_black(ctx, bloom, mode):
    """Sombra: um pixel longe das fontes continua preto (±1/255) com uma fonte forte."""
    size = (512, 512)
    image = point_image(size, (40, 40), 1000.0)
    out = apply(ctx, bloom, image, mode, encode=True)
    assert out[480:, 480:].max() <= ONE_LSB
    # A fonte ganhou halo (não é um teste vazio).
    assert out[40, 60].max() > 0


@pytest.mark.parametrize("value", [0.0, 0.3, 0.99])
def test_classic_below_threshold_is_unchanged(ctx, bloom, value):
    """Clássico: com toda luminância abaixo do limiar, a saída é a de sem bloom."""
    rng = np.random.default_rng(3)
    image = value * BLOOM.threshold * rng.random((30, 50, 3))
    luminance = image @ np.array(scene.LUMINANCE_WEIGHTS)
    assert luminance.max() < BLOOM.threshold
    out = apply(ctx, bloom, image, "classic", encode=True)
    assert_within_one_lsb(out, expected_encoded(image))


def test_classic_adds_energy_only_above_threshold(ctx, bloom):
    """Clássico: o halo soma energia (glare conserva) e vem só do excesso sobre o limiar."""
    # A fonte num canto e o fundo medido no oposto, fora do suporte da PSF.
    image = np.full((512, 512, 3), 0.5 * BLOOM.threshold)
    image[20:24, 20:24] = 200.0
    classic = apply(ctx, bloom, image, "classic", encode=False).astype(np.float64)
    glare = apply(ctx, bloom, image, "glare", encode=False).astype(np.float64)
    half = as_half(image)
    assert classic.mean() > half.mean() * 1.001
    assert glare.mean() == pytest.approx(half.mean(), rel=0.01)
    # Longe da fonte, o clássico não muda o fundo abaixo do limiar.
    np.testing.assert_allclose(classic[-1, -1], half[-1, -1], rtol=1e-3)


@pytest.mark.parametrize("size", [(5, 3), (1, 1), (40, 2)])
@pytest.mark.parametrize("mode", BOTH_MODES)
def test_small_image_chain_stops_at_one_pixel(ctx, mode, size):
    """Imagem menor que 2^níveis: a cadeia para em 1×1 px, sem erro, e o uniforme fica."""
    local = BloomPass(ctx, replace(BLOOM, glare_strength=1.0))
    try:
        width, height = size
        image = np.full((height, width, 3), 0.25)
        out = apply(ctx, local, image, mode, encode=False)
        assert local.level_sizes[-1] == (1, 1)
        assert len(local.level_sizes) <= BLOOM.levels
        assert np.isfinite(out).all()
        if mode == "glare":
            np.testing.assert_allclose(out, 0.25, rtol=1e-3)
    finally:
        local.release()


def test_levels_reused_until_size_changes(ctx):
    local = BloomPass(ctx)
    try:
        image = np.zeros((36, 64, 3))
        apply(ctx, local, image, "glare", encode=True)
        first = [texture for texture, _ in local._levels]
        apply(ctx, local, image, "classic", encode=True)
        assert [texture for texture, _ in local._levels] == first
        apply(ctx, local, np.zeros((18, 32, 3)), "glare", encode=True)
        assert local.level_sizes[0] == (16, 9)
    finally:
        local.release()


def test_non_finite_pixel_does_not_spread(ctx, bloom):
    """Um NaN na cena não contamina o bloom dos outros pixels nem apaga os vizinhos: o
    resultado é o de um 0 no lugar dele (fora o próprio pixel, que segue a cena)."""
    image = point_image((64, 64), (48, 48), 50.0)
    image[10, 11] = 30.0  # vizinho do NaN, que precisa continuar contribuindo
    clean = image.copy()
    image[10, 10] = math.nan
    for mode in BOTH_MODES:
        out = apply(ctx, bloom, image, mode, encode=False)
        reference = apply(ctx, bloom, clean, mode, encode=False)
        finite = np.isfinite(out).all(axis=2)
        finite[10, 10] = True  # o próprio pixel segue a cena (como sem bloom)
        assert finite.all()
        mask = np.ones(out.shape[:2], dtype=bool)
        mask[10, 10] = False
        np.testing.assert_allclose(out[mask], reference[mask], rtol=1e-3, atol=1e-6)
        assert out[10, 14].max() > 0.0  # halo do vizinho


def test_classic_threshold_is_per_pixel(ctx, bloom):
    """Clássico: um pixel isolado com limiar < L < 4·limiar ganha halo, com energia extra
    k_c·(L − limiar)/L·I somada (o filtro vale por pixel, não na média 2×2)."""
    value = 2.0 * BLOOM.threshold  # cinza: L = valor
    image = point_image((512, 512), (256, 256), value)
    out = apply(ctx, bloom, image, "classic", encode=False).astype(np.float64)
    extra = out.sum(axis=(0, 1)) - as_half(image).sum(axis=(0, 1))
    expected = BLOOM.classic_strength * (value - BLOOM.threshold) / value * value
    np.testing.assert_allclose(extra, expected, rtol=0.05)
    assert out[256, 262].max() > 0.0


def test_invalid_mode_or_viewport_raises(ctx, bloom):
    source = upload(ctx, np.zeros((4, 4, 3)))
    target_texture = ctx.texture((4, 4), 4)
    target = ctx.framebuffer(color_attachments=[target_texture])
    before = ctx.fbo
    try:
        with pytest.raises(ValueError):
            bloom.render(source, target, mode="flare")
        with pytest.raises(ValueError):
            bloom.render(source, target, viewport=(0, 0, 8, 8))
    finally:
        ctx.fbo = before
        target.release()
        target_texture.release()
        source.release()
