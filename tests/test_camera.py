"""Testes da Fase 2: câmera de observador estático, fundo em xadrez e render CPU.

A medida da sombra renderiza só a linha central (altura 1, linha em q = 0) com um FOV
estreito escolhido por r_cam, para que o raio da sombra tenha ~200 px e o erro de
quantização (±0,5 px) fique bem abaixo da tolerância de 2%.

Desde a Fase 3 o render liga o disco e usa estrelas por padrão; os testes que dependem de
uma sombra limpa ou da paleta do xadrez pedem ``disk=NO_DISK`` e ``background="checker"``.
"""

import math
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from blackhole.camera import Camera, pixel_offset
from blackhole.config import (
    CameraConfig,
    CheckerConfig,
    ColorConfig,
    DiskConfig,
    IntegratorConfig,
    RenderConfig,
)
from blackhole.physics import (
    B_CRIT,
    CAPTURED,
    ESCAPED,
    R_PHOTON,
    angular_momentum,
    escape_radius,
    impact_parameter,
    shadow_angular_radius,
    static_observer_velocity,
    trace_ray,
)
from blackhole.render_cpu import render, save_png, to_uint8
from blackhole.scene import checker_color, encode_pixel

R_CAMS = (10.0, 30.0, 100.0)
NO_DISK = DiskConfig(enabled=False)
# Raio da sombra desejado na linha central, em pixels; largura ímpar com p = 0 no centro.
# Não inteiro, para a borda exata não cair em cima do centro de um pixel.
SHADOW_PX = 200.3
SHADOW_ROW_WIDTH = 601


def camera_at(r_cam, **kwargs):
    return Camera.from_config(replace(CameraConfig(), r_cam=r_cam, **kwargs))


# --- tétrade estática e fórmula de Synge (seção 3.3) ------------------------


def test_shadow_angular_radius_limits():
    assert shadow_angular_radius(R_PHOTON) == pytest.approx(math.pi / 2.0, abs=1e-6)
    r = 1.0e6
    assert shadow_angular_radius(r) == pytest.approx(B_CRIT / r, rel=1e-5)
    # Dentro da esfera de fótons a sombra cobre mais de meio céu.
    assert shadow_angular_radius(1.2) > math.pi / 2.0


def test_escape_radius():
    cfg = RenderConfig()
    assert escape_radius(10.0, cfg.r_max_min, cfg.r_max_factor) == 50.0
    assert escape_radius(30.0, cfg.r_max_min, cfg.r_max_factor) == 60.0
    assert escape_radius(100.0, cfg.r_max_min, cfg.r_max_factor) == 200.0


@pytest.mark.parametrize("r_cam", R_CAMS)
def test_static_observer_velocity_reproduces_synge(r_cam):
    # Direção arbitrária a um ângulo α de −r̂: b = r sin α / √(1 − 1/r).
    x = r_cam * np.array([0.3, 0.8, -0.52]) / np.linalg.norm([0.3, 0.8, -0.52])
    rhat = x / r_cam
    tangent = np.cross(rhat, [0.0, 0.0, 1.0])
    tangent /= np.linalg.norm(tangent)
    alpha = 0.37
    n = -math.cos(alpha) * rhat + math.sin(alpha) * tangent
    v0 = static_observer_velocity(x, n)
    expected = r_cam * math.sin(alpha) / math.sqrt(1.0 - 1.0 / r_cam)
    assert impact_parameter(x, v0) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("r_cam", R_CAMS)
@pytest.mark.parametrize("phi", [0.0, 0.9, 2.5, 4.4])
def test_ray_at_synge_angle_has_critical_impact_parameter(r_cam, phi):
    """Caminho real da câmera: (p, q) → n → tétrade → v₀, com n a α_Synge do centro."""
    camera = camera_at(r_cam)
    height = RenderConfig().height
    alpha = shadow_angular_radius(r_cam)
    t = math.tan(alpha) / camera.pixel_scale(height)
    p, q = t * math.cos(phi), t * math.sin(phi)

    n = camera.direction(p, q, height)
    to_center = -camera.position / camera.r_cam
    assert math.acos(np.clip(np.dot(n, to_center), -1.0, 1.0)) == pytest.approx(alpha, rel=1e-9)

    v0 = camera.initial_velocity(p, q, height)
    b = impact_parameter(camera.position, v0)
    assert abs(b / B_CRIT - 1.0) < 1e-6


# --- câmera: base, pixel central e erros (matriz I/O) -----------------------


def test_camera_basis_is_orthonormal_and_right_handed():
    camera = camera_at(30.0)
    basis = np.array([camera.right, camera.up, -camera.forward])
    np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-14)
    assert np.linalg.det(basis) == pytest.approx(1.0, abs=1e-14)
    np.testing.assert_allclose(camera.forward, -camera.position / 30.0, atol=1e-14)
    # Inclinação de 80° a partir de +y: câmera levemente acima do plano do disco.
    assert camera.position[1] == pytest.approx(30.0 * math.cos(math.radians(80.0)))
    assert camera.up[1] > 0.0


def test_pixel_offset_convention():
    # Canto superior esquerdo: p negativo (esquerda), q positivo (cima).
    assert pixel_offset(0, 0, 0.0, 0.0, 800, 450) == (-399.5, 224.5)
    # Largura/altura ímpares: o pixel do meio tem p = q = 0.
    assert pixel_offset(300, 0, 0.0, 0.0, 601, 1) == (0.0, 0.0)


def test_central_pixel_is_radial_and_captured():
    camera = camera_at(30.0)
    height = RenderConfig().height
    n = camera.direction(0.0, 0.0, height)
    np.testing.assert_allclose(n, -camera.position / camera.r_cam, atol=1e-15)
    v0 = camera.initial_velocity(0.0, 0.0, height)
    assert angular_momentum(camera.position, v0) < 1e-12
    status, *_ = trace_ray(camera.position, v0, 60.0, IntegratorConfig(), NO_DISK)
    assert status == CAPTURED

    image = render(
        camera,
        replace(RenderConfig(), width=9, height=9),
        disk=NO_DISK,
        background="checker",
    )
    np.testing.assert_array_equal(image[4, 4], [0, 0, 0])


@pytest.mark.parametrize("r", [0.5, 1.0])
def test_camera_inside_horizon_raises(r):
    # Posição exata no eixo x: r = 1,0 testa a fronteira sem depender de arredondamento.
    with pytest.raises(ValueError):
        Camera((r, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 60.0)


def test_orbital_camera_inside_horizon_raises():
    with pytest.raises(ValueError):
        camera_at(0.5)


@pytest.mark.parametrize("inclination", [0.0, 180.0])
def test_up_parallel_to_view_raises(inclination):
    with pytest.raises(ValueError):
        camera_at(30.0, inclination_deg=inclination)


def test_invalid_fov_raises():
    with pytest.raises(ValueError):
        camera_at(30.0, fov_v_deg=0.0)
    with pytest.raises(ValueError):
        camera_at(30.0, fov_v_deg=180.0)


# --- fundo em xadrez (seção 3.7) --------------------------------------------


def test_checker_alternates_across_cells():
    cfg = CheckerConfig()
    cell = math.radians(cfg.cell_deg)
    a, b = cfg.color_a, cfg.color_b

    def color(lon, lat):
        d = np.array([math.cos(lat) * math.cos(lon), math.sin(lat), math.cos(lat) * math.sin(lon)])
        return checker_color(d * 7.0, cell, a, b)  # não precisa ser unitária

    half = 0.5 * cell
    assert color(half, half) == a
    assert color(cell + half, half) == b
    assert color(half, -half) == b
    assert color(-half, -half) == a
    # 24 casas em longitude: a 24ª volta à primeira sem quebrar a paridade.
    assert color(-math.pi + half, half) == a


# --- render -----------------------------------------------------------------


def shadow_radius_on_central_row(r_cam):
    """Renderiza a linha q = 0 e devolve (raio angular medido, raio de Synge)."""
    alpha = shadow_angular_radius(r_cam)
    scale = math.tan(alpha) / SHADOW_PX
    height = 1
    fov_v = math.degrees(2.0 * math.atan(scale * 0.5 * height))
    camera = camera_at(r_cam, fov_v_deg=fov_v)
    assert camera.pixel_scale(height) == pytest.approx(scale, rel=1e-12)

    cfg = replace(RenderConfig(), width=SHADOW_ROW_WIDTH, height=height)
    row = render(camera, cfg, disk=NO_DISK, background="checker")[0]
    black = np.all(row == 0, axis=1)
    center = SHADOW_ROW_WIDTH // 2
    assert black[center]
    assert not black[0] and not black[-1]

    right = center
    while black[right + 1]:
        right += 1
    left = center
    while black[left - 1]:
        left -= 1
    # A borda fica entre o último pixel preto e o primeiro colorido: ±0,5 px.
    radius_px = 0.5 * ((right - center + 0.5) + (center - left + 0.5))
    return math.atan(radius_px * scale), alpha


@pytest.mark.parametrize("r_cam", R_CAMS)
def test_rendered_shadow_matches_synge(r_cam):
    measured, expected = shadow_radius_on_central_row(r_cam)
    assert abs(measured / expected - 1.0) < 0.02


def test_render_output_shape_and_background():
    cfg = replace(RenderConfig(), width=32, height=18)
    image = render(camera_at(30.0), cfg, disk=NO_DISK, background="checker")
    assert image.shape == (18, 32, 3)
    assert image.dtype == np.uint8
    # Cantos longe da sombra: fundo (não preto).
    assert image[0, 0].any() and image[-1, -1].any()


def test_step_limit_pixels_are_black():
    cfg = replace(RenderConfig(), width=32, height=18)
    integrator = replace(IntegratorConfig(), max_steps=1)
    image = render(
        camera_at(30.0), cfg, integrator=integrator, disk=NO_DISK, background="checker"
    )
    assert not image.any()


def test_supersampling_averages_edges():
    cfg = replace(RenderConfig(), width=64, height=36)
    checker = CheckerConfig()
    exposure = ColorConfig().exposure
    # Paleta pela cadeia exposição → ACES → sRGB, a mesma do render (seção 3.6).
    palette = {(0, 0, 0)} | {
        tuple(to_uint8(np.array(encode_pixel(*col, exposure))))
        for col in (checker.color_a, checker.color_b)
    }
    plain = render(camera_at(30.0), cfg, disk=NO_DISK, background="checker")
    smooth = render(
        camera_at(30.0), replace(cfg, supersample=2), disk=NO_DISK, background="checker"
    )
    plain_colors = {tuple(px) for px in plain.reshape(-1, 3)}
    smooth_colors = {tuple(px) for px in smooth.reshape(-1, 3)}
    assert plain_colors <= palette
    assert len(smooth_colors - palette) > 0


def test_outward_radial_ray_escapes():
    """Um raio disparado da câmera na direção +r̂ termina com status ESCAPED."""
    camera = camera_at(30.0)
    v0 = static_observer_velocity(camera.position, camera.position / camera.r_cam)
    status, *_ = trace_ray(camera.position, v0, 60.0, IntegratorConfig(), NO_DISK)
    assert status == ESCAPED


def test_save_png_roundtrip(tmp_path):
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[1, 2] = (10, 20, 30)
    path = save_png(image, tmp_path / "sub" / "x.png")
    np.testing.assert_array_equal(np.asarray(Image.open(path)), image)
