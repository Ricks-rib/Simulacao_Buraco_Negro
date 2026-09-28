"""Testes da Fase 3: estrutura da imagem do disco na câmera padrão (r = 30, 80°).

Usam ``trace_fields`` (um raio por pixel) e classificam os acertos pelo ângulo φ
percorrido no plano orbital: imagem primária |φ| < π, secundária π ≤ |φ| < 2π e anel de
fótons |φ| ≥ 2π (PLANO, Fase 3).
"""

import math

import numpy as np
import pytest

from blackhole.camera import Camera
from blackhole.config import (
    CameraConfig,
    CheckerConfig,
    ColorConfig,
    DiskConfig,
    RenderConfig,
    StarfieldConfig,
)
from blackhole.physics import (
    B_CRIT,
    CAPTURED,
    DISK,
    ESCAPED,
    MAX_STEPS,
    impact_parameter,
)
from blackhole.render_cpu import BACKGROUNDS, render, shade_sample, trace_fields
from blackhole.scene import blackbody_table

TWO_PI = 2.0 * math.pi
# Faixa do anel de fótons: b_c < b < (1 + RING_WIDTH)·b_c.
RING_WIDTH = 0.05


@pytest.fixture(scope="module")
def camera():
    return Camera.from_config(CameraConfig())


@pytest.fixture(scope="module")
def fields(camera):
    return trace_fields(camera, RenderConfig())


@pytest.fixture(scope="module")
def central_column(fields):
    status, x, v, phi = fields
    c = status.shape[1] // 2
    return status[:, c], np.abs(phi[:, c])


def test_default_camera_is_at_80_degrees(camera):
    cfg = CameraConfig()
    assert cfg.inclination_deg == 80.0
    assert camera.r_cam == pytest.approx(30.0)


def test_trace_fields_shapes(fields):
    status, x, v, phi = fields
    cfg = RenderConfig()
    assert status.shape == (cfg.height, cfg.width)
    assert x.shape == v.shape == (cfg.height, cfg.width, 3)
    assert phi.shape == (cfg.height, cfg.width)
    # Impactos no disco ficam no plano y = 0, dentro de [r_in, r_out].
    hits = status == DISK
    assert hits.any()
    np.testing.assert_array_equal(x[hits][:, 1], 0.0)
    radius = np.hypot(x[hits][:, 0], x[hits][:, 2])
    disk = DiskConfig()
    assert radius.min() >= disk.r_in and radius.max() <= disk.r_out


def test_primary_image_arches_over_shadow(central_column):
    status, phi = central_column
    shadow = np.nonzero((status == CAPTURED) | (status == MAX_STEPS))[0]
    assert shadow.size > 0
    top = shadow.min()  # linha 0 é o topo da imagem
    primary = np.nonzero((status == DISK) & (phi < math.pi))[0]
    assert np.any(primary < top)


def test_secondary_image_below_center(central_column):
    status, phi = central_column
    height = status.shape[0]
    secondary = np.nonzero((status == DISK) & (phi >= math.pi) & (phi < TWO_PI))[0]
    # q < 0 ⇔ j + 0,5 > H/2.
    assert np.any(secondary + 0.5 > 0.5 * height)


def test_photon_ring_is_thin_band_at_shadow_edge(fields):
    status, x, v, phi = fields
    ring = ((status == ESCAPED) | (status == DISK)) & (np.abs(phi) >= TWO_PI)
    assert ring.any()
    b = np.array([impact_parameter(x[j, i], v[j, i]) for j, i in zip(*np.nonzero(ring))])
    assert np.all(b > B_CRIT)
    assert np.all(b < (1.0 + RING_WIDTH) * B_CRIT)


def test_both_disk_faces_hit_and_emit_equally(fields):
    """Face de cima (v_y < 0) e de baixo (v_y > 0) aparecem e emitem igual no mesmo raio."""
    status, x, v, _ = fields
    hits = status == DISK
    assert np.any(v[hits][:, 1] < 0.0)
    assert np.any(v[hits][:, 1] > 0.0)

    color = ColorConfig()
    disk = DiskConfig()
    checker = CheckerConfig()
    stars = StarfieldConfig()
    table = blackbody_table(color)
    j, i = np.argwhere(hits)[0]
    flipped = v[j, i] * np.array([1.0, -1.0, 1.0])

    def shade(vel):
        return shade_sample(
            DISK,
            x[j, i],
            vel,
            disk.r_in,
            disk.t_peak,
            table,
            color.bb_t_min,
            color.bb_t_max,
            BACKGROUNDS["stars"],
            math.radians(checker.cell_deg),
            checker.color_a,
            checker.color_b,
            float(stars.grid_size),
            stars.density,
            stars.jitter,
            stars.seed,
            stars.sigma,
            stars.intensity_min,
            stars.intensity_max,
            stars.t_min,
            stars.t_max,
        )

    assert shade(v[j, i]) == shade(flipped)
    assert any(c > 0.0 for c in shade(v[j, i]))


def test_render_default_has_bright_disk_and_black_shadow(camera, fields):
    cfg = RenderConfig()
    image = render(camera, cfg)
    assert image.shape == (cfg.height, cfg.width, 3)
    # Centro da imagem: sombra preta.
    np.testing.assert_array_equal(image[cfg.height // 2, cfg.width // 2], [0, 0, 0])
    # O disco da frente, abaixo da sombra na coluna central, é brilhante.
    status = fields[0]
    c = cfg.width // 2
    front = np.nonzero(status[:, c] == DISK)[0]
    front = front[front > cfg.height // 2]
    assert front.size > 0
    assert image[front, c].max() > 128
