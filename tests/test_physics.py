"""Testes da Fase 1: a física do fóton contra resultados analíticos.

Raios de teste (PLANO, Fase 1): partem de x = (−R, b, 0) com v = (1, 0, 0), R = 10⁴ e
r_max = 2·10⁴, com o disco desligado. Esses parâmetros são próprios dos testes e não
dependem dos padrões de renderização.
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from blackhole.config import DiskConfig, IntegratorConfig
from blackhole.physics import (
    B_CRIT,
    CAPTURED,
    DISK,
    ESCAPED,
    M,
    MAX_STEPS,
    R_ISCO,
    R_PHOTON,
    R_S,
    angular_momentum,
    axial_angular_momentum,
    deflection_angle,
    energy_at_infinity,
    energy_eps,
    impact_parameter,
    rk4_step,
    step_size,
    trace_ray,
)

R_START = 1.0e4
R_MAX = 2.0e4
NO_DISK = DiskConfig(enabled=False)
TEST_DISK = DiskConfig(r_in=3.0, r_out=12.0, enabled=True)
DISK_R_MAX = 1.0e3
# Folga de passos para os raios quase críticos, que dão várias voltas na esfera de fótons.
TEST_INTEGRATOR = replace(IntegratorConfig(), max_steps=200_000)


def shoot(b, disk=NO_DISK, integrator=TEST_INTEGRATOR, trace=None):
    """Integra o raio de teste com parâmetro de impacto b."""
    x0 = np.array([-R_START, b, 0.0])
    v0 = np.array([1.0, 0.0, 0.0])
    return x0, v0, trace_ray(x0, v0, R_MAX, integrator, disk, trace)


def deflection(b):
    x0, v0, (status, x, v, phi_swept, _) = shoot(b)
    assert status == ESCAPED
    return deflection_angle(x0, v0, x, v, phi_swept)


def strong_field_deflection(delta):
    """Limite de deflexão forte: α = −ln δ + ln[216(7 − 4√3)] − π."""
    return -math.log(delta) + math.log(216.0 * (7.0 - 4.0 * math.sqrt(3.0))) - math.pi


# --- constantes (seção 3.1) -------------------------------------------------


def test_reference_constants():
    assert R_S == 1.0
    assert M == 0.5
    assert R_PHOTON == 1.5
    assert B_CRIT == pytest.approx(3.0 * math.sqrt(3.0) / 2.0, rel=1e-15)
    assert R_ISCO == 3.0


# --- invariantes (seção 3.2) ------------------------------------------------


def test_invariants_of_known_state():
    x = np.array([3.0, 4.0, 0.0])
    v = np.array([0.0, 1.0, 2.0])
    # x × v = (8, −6, 3)
    h = math.sqrt(64.0 + 36.0 + 9.0)
    r = 5.0
    eps = 0.5 * 5.0 - h * h / (2.0 * r**3)
    assert angular_momentum(x, v) == pytest.approx(h, rel=1e-15)
    assert energy_eps(x, v) == pytest.approx(eps, rel=1e-15)
    assert energy_at_infinity(x, v) == pytest.approx(math.sqrt(2.0 * eps), rel=1e-15)
    assert impact_parameter(x, v) == pytest.approx(h / math.sqrt(2.0 * eps), rel=1e-15)
    assert axial_angular_momentum(x, v) == pytest.approx(-6.0, rel=1e-15)


# --- captura e escape -------------------------------------------------------


def test_b_255_is_captured():
    _, _, (status, x, _, _, _) = shoot(2.55)
    assert status == CAPTURED
    assert np.linalg.norm(x) < R_S


def test_b_265_escapes():
    _, _, (status, x, v, _, _) = shoot(2.65)
    assert status == ESCAPED
    assert np.linalg.norm(x) > R_MAX
    assert np.dot(x, v) > 0.0


# --- deflexão ---------------------------------------------------------------


def test_weak_field_deflection_b50():
    b = 50.0
    expected = 2.0 / b + 15.0 * math.pi / (16.0 * b * b)
    alpha = deflection(b)
    assert abs(alpha - expected) / expected < 0.01


@pytest.mark.parametrize("delta", [1e-4, 1e-6])
def test_strong_field_deflection(delta):
    alpha = deflection(B_CRIT * (1.0 + delta))
    assert abs(alpha - strong_field_deflection(delta)) < 0.05


@pytest.mark.parametrize("sign, expected_status", [(+1.0, ESCAPED), (-1.0, CAPTURED)])
def test_near_critical_rays_wind_more_than_4pi(sign, expected_status):
    delta = 1e-6
    _, _, (status, _, _, phi_swept, _) = shoot(B_CRIT * (1.0 + sign * delta))
    assert status == expected_status
    assert phi_swept > 4.0 * math.pi


# --- conservação ------------------------------------------------------------


def shoot_test_ray(b, trace):
    return shoot(b, trace=trace)[2]


def shoot_disk_ray(x_start, trace):
    """Raio vertical descendo em x = x_start rumo ao disco [3, 12] (matriz I/O do plano)."""
    x0 = np.array([x_start, 100.0, 0.0])
    v0 = np.array([0.0, -1.0, 0.0])
    return trace_ray(x0, v0, DISK_R_MAX, TEST_INTEGRATOR, TEST_DISK, trace)


@pytest.mark.parametrize(
    "shoot_fn, param",
    [(shoot_test_ray, b) for b in (2.55, 2.65, 50.0)]
    + [(shoot_test_ray, B_CRIT * (1.0 + d)) for d in (1e-4, 1e-6, -1e-6)]
    + [(shoot_disk_ray, x) for x in (8.0, 20.0)],
)
def test_h_and_eps_are_conserved(shoot_fn, param):
    buffer = np.zeros((TEST_INTEGRATOR.max_steps + 1, 6))
    _, _, _, _, n_steps = shoot_fn(param, buffer)
    states = buffer[: n_steps + 1]
    xs, vs = states[:, :3], states[:, 3:]
    h = np.linalg.norm(np.cross(xs, vs), axis=1)
    r = np.linalg.norm(xs, axis=1)
    eps = 0.5 * np.sum(vs * vs, axis=1) - h * h / (2.0 * r**3)
    assert np.max(np.abs(h / h[0] - 1.0)) < 1e-4
    assert np.max(np.abs(eps / eps[0] - 1.0)) < 1e-4


# --- disco e limite de passos (matriz I/O do plano) -------------------------


def test_disk_hit_stops_at_interpolated_crossing():
    buffer = np.zeros((TEST_INTEGRATOR.max_steps + 1, 6))
    status, x, v, _, n_steps = shoot_disk_ray(8.0, buffer)
    assert status == DISK

    # Refaz o último passo RK4 a partir do estado anterior ao cruzamento e confere que o
    # ponto devolvido é a interpolação linear em y = 0 entre as duas pontas do passo.
    prev = buffer[n_steps - 1]
    cfg = TEST_INTEGRATOR
    dt = step_size(
        np.linalg.norm(prev[:3]),
        cfg.step_factor,
        cfg.photon_step_factor,
        cfg.photon_r_min,
        cfg.photon_r_max,
        cfg.dt_min,
        cfg.dt_max,
    )
    h2 = float(np.sum(np.cross(prev[:3], prev[3:]) ** 2))
    end = np.array(rk4_step(*prev, h2, dt))
    assert prev[1] > 0.0 > end[1]
    t = prev[1] / (prev[1] - end[1])
    expected = prev + t * (end - prev)
    np.testing.assert_allclose(x, expected[:3], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(v, expected[3:], rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(buffer[n_steps], np.concatenate([x, v]))

    # A lente puxa o raio para dentro. Em primeira ordem, o deslocamento lateral até y = 0
    # é ∫ y · 3M b³/(y² + b²)^(5/2) dy = M (para y₀ ≫ b), logo r ≈ b − M = 7,5.
    assert np.linalg.norm(x) == pytest.approx(8.0 - M, abs=0.05)


def test_disk_miss_continues_and_escapes():
    status, x, _, _, _ = shoot_disk_ray(20.0, None)
    assert status == ESCAPED
    assert x[1] < 0.0


def test_step_limit_reports_max_steps():
    integrator = replace(IntegratorConfig(), max_steps=10)
    _, _, (status, _, _, _, n_steps) = shoot(2.65, integrator=integrator)
    assert status == MAX_STEPS
    assert n_steps == 10
