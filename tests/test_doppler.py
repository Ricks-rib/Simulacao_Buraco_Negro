"""Testes da Fase 5a: fator de desvio g (Doppler + redshift), atraso da luz e textura.

O g da seção 3.8 é conferido contra g = (p·u_obs)/(p·u_em) montado explicitamente na
métrica de Schwarzschild, sem usar a fórmula. Os testes de GPU usam um contexto
standalone e dão ``skip`` sem OpenGL 3.3.
"""

import math
from dataclasses import replace

import moderngl
import numpy as np
import pytest

from blackhole import compare, render_cpu
from blackhole.camera import Camera, pixel_offset
from blackhole.config import (
    CameraConfig,
    CheckerConfig,
    ColorConfig,
    CompareConfig,
    DiskConfig,
    DiskTextureConfig,
    GLCheckConfig,
    GpuConfig,
    IntegratorConfig,
    RenderConfig,
    StarfieldConfig,
)
from blackhole.physics import (
    DISK,
    ESCAPED,
    MAX_STEPS,
    M,
    R_S,
    axial_angular_momentum,
    disk_temperature,
    energy_at_infinity,
    keplerian_omega,
    peak_radius,
    redshift_factor,
    static_blueshift,
    trace_ray,
)
from blackhole.render_cpu import BACKGROUNDS, render, shade_sample, trace_fields
from blackhole.render_gpu import GpuRenderer
from blackhole.scene import (
    blackbody_rgb,
    blackbody_table,
    disk_radiance,
    disk_texture,
    star_radiance,
)

COLOR = ColorConfig()
DISK_CFG = DiskConfig()
TEX = DiskTextureConfig()
STARS = StarfieldConfig()
CHECKER = CheckerConfig()
CAM = CameraConfig()
GPU = GpuConfig()
CMP = CompareConfig()
# Resolução reduzida (mesma razão de aspecto do padrão).
SMALL_W, SMALL_H = 320, 180


@pytest.fixture(scope="module")
def table():
    return blackbody_table(COLOR)


@pytest.fixture(scope="module")
def camera():
    return Camera.from_config(CAM)


def texture(r, phi, t_emit, cfg=TEX):
    return disk_texture(
        r,
        phi,
        t_emit,
        cfg.radial_cells,
        cfg.azimuthal_cells,
        cfg.octaves,
        cfg.contrast,
        cfg.seed,
    )


def radiance(r, table, g=1.0, m=1.0, t_peak=DISK_CFG.t_peak):
    return np.array(
        disk_radiance(r, DISK_CFG.r_in, t_peak, table, COLOR.bb_t_min, COLOR.bb_t_max, g, m)
    )


def bb(temperature, table):
    return np.array(blackbody_rgb(float(temperature), table, COLOR.bb_t_min, COLOR.bb_t_max))


def shade(status, x, v, dt, table, doppler, r_cam, t_obs=0.0, tex_on=False, background="stars"):
    return np.array(
        shade_sample(
            status,
            np.asarray(x, dtype=np.float64),
            np.asarray(v, dtype=np.float64),
            dt,
            DISK_CFG.r_in,
            DISK_CFG.t_peak,
            table,
            COLOR.bb_t_min,
            COLOR.bb_t_max,
            BACKGROUNDS[background],
            math.radians(CHECKER.cell_deg),
            CHECKER.color_a,
            CHECKER.color_b,
            float(STARS.grid_size),
            STARS.density,
            STARS.jitter,
            STARS.seed,
            STARS.sigma,
            STARS.intensity_min,
            STARS.intensity_max,
            STARS.t_min,
            STARS.t_max,
            doppler,
            r_cam,
            t_obs,
            tex_on,
            TEX.radial_cells,
            TEX.azimuthal_cells,
            TEX.octaves,
            TEX.contrast,
            TEX.seed,
        )
    )


# --- g contra p·u explícito na métrica (independente da fórmula) -------------


def photon_pt(x, v):
    """p^t do fóton pela condição nula da métrica de Schwarzschild.

    Com v = (dr/dλ) r̂ + v_⊥ (|v_⊥| = r·|dΩ/dλ|):
    −(1 − 1/r)(p^t)² + (p^r)²/(1 − 1/r) + |v_⊥|² = 0.
    """
    r = np.linalg.norm(x)
    rhat = x / r
    pr = float(np.dot(v, rhat))
    v_perp2 = float(np.dot(v, v)) - pr * pr
    lapse2 = 1.0 - R_S / r
    return math.sqrt((pr * pr / lapse2 + v_perp2) / lapse2)


def explicit_g(x_cam, v_cam, x_em, v_em):
    """g = (p·u_obs)/(p·u_em) montado em coordenadas de Schwarzschild.

    O fóton físico percorre o raio ao contrário: momento espacial −v, p^t > 0. A câmera
    é estática, u_obs = (1/√(1 − 1/r_c), 0, 0, 0). O gás tem u_em = u^t (1, 0, 0, Ω) no
    ângulo φ = atan2(−z, x) em torno de +y, com u^t da normalização u·u = −1.
    """
    r_c = np.linalg.norm(x_cam)
    p_dot_u_obs = -(1.0 - R_S / r_c) * photon_pt(x_cam, v_cam) / math.sqrt(1.0 - R_S / r_c)

    x, z = x_em[0], x_em[2]
    r = math.hypot(x, z)
    dx, dz = -v_em[0], -v_em[2]  # momento físico = −v
    p_phi = (z * dx - x * dz) / (r * r)  # dφ/dλ com φ = atan2(−z, x)
    omega = math.sqrt(M / r**3)
    g_tt = -(1.0 - R_S / r)
    g_pp = r * r  # no plano do disco, sin θ = 1 em relação a +y
    u_t = 1.0 / math.sqrt(-(g_tt + g_pp * omega * omega))
    p_dot_u_em = u_t * (g_tt * photon_pt(x_em, v_em) + g_pp * p_phi * omega)
    return p_dot_u_obs / p_dot_u_em


def disk_rays(camera, n_side=24):
    """Raios de uma grade de pixels que acertam o disco (integrador de referência)."""
    cfg = RenderConfig()
    r_max = 2.0 * camera.r_cam
    rays = []
    for j in np.linspace(0, cfg.height - 1, n_side).astype(int):
        for i in np.linspace(0, cfg.width - 1, n_side).astype(int):
            p, q = pixel_offset(i, j, 0.0, 0.0, cfg.width, cfg.height)
            v0 = camera.initial_velocity(p, q, cfg.height)
            status, x, v, _, _, _ = trace_ray(
                camera.position, v0, r_max, IntegratorConfig(), DISK_CFG
            )
            if status == DISK:
                rays.append((v0, x, v))
    return rays


def test_redshift_factor_matches_explicit_p_dot_u(camera):
    rays = disk_rays(camera)
    assert len(rays) > 30
    for v0, x, v in rays:
        r = math.hypot(x[0], x[2])
        l_y, energy = axial_angular_momentum(x, v), energy_at_infinity(x, v)
        g = redshift_factor(r, camera.r_cam, l_y, energy)
        assert g == pytest.approx(explicit_g(camera.position, v0, x, v), rel=1e-4)


def test_zero_axial_momentum():
    for r in (3.0, 4.5, 12.0):
        for r_cam in (5.0, 30.0, 500.0):
            expected = math.sqrt(1.0 - 1.5 / r) / math.sqrt(1.0 - 1.0 / r_cam)
            assert redshift_factor(r, r_cam, 0.0, 0.7) == pytest.approx(expected, rel=1e-14)


def test_keplerian_omega_and_static_blueshift():
    assert keplerian_omega(3.0) == pytest.approx(math.sqrt(1.0 / (2.0 * 27.0)), rel=1e-15)
    assert static_blueshift(30.0) == pytest.approx(1.0 / math.sqrt(1.0 - 1.0 / 30.0), rel=1e-15)
    # Luz vinda do infinito com E: a câmera estática mede −p·u_obs = E·g_∞.
    x_cam = np.array([30.0, 0.0, 0.0])
    v = np.array([-0.3, 0.2, 0.1])
    energy = energy_at_infinity(x_cam, v)
    measured = math.sqrt(1.0 - 1.0 / 30.0) * photon_pt(x_cam, v)
    assert measured == pytest.approx(energy * static_blueshift(30.0), rel=1e-12)


# --- radiância do disco (matriz I/O) ----------------------------------------


@pytest.mark.parametrize("g", [0.3, 1.0, 1.7])
def test_inner_edge_is_black_for_any_g(table, g):
    np.testing.assert_array_equal(radiance(DISK_CFG.r_in, table, g=g, m=1.3), [0.0, 0.0, 0.0])


def test_doppler_off_is_phase_4_radiance(table):
    """g = m = 1: exatamente (T/T_pico)⁴ · rgb(T) da seção 3.6."""
    for r in (3.5, peak_radius(DISK_CFG.r_in), 7.0, 12.0):
        temperature = disk_temperature(r, DISK_CFG.r_in, DISK_CFG.t_peak)
        ratio = temperature / DISK_CFG.t_peak
        expected = ratio * ratio * ratio * ratio * bb(temperature, table)
        np.testing.assert_array_equal(radiance(r, table), expected)


def test_observed_temperature_and_g4_brightness(table):
    r = peak_radius(DISK_CFG.r_in)
    g = 1.2
    expected = g**4 * bb(g * DISK_CFG.t_peak, table)
    np.testing.assert_allclose(radiance(r, table, g=g), expected, rtol=1e-12)
    np.testing.assert_allclose(radiance(r, table, g=g, m=0.5), 0.5 * expected, rtol=1e-12)


def test_observed_temperature_below_table(table):
    r = 3.3
    temperature = disk_temperature(r, DISK_CFG.r_in, DISK_CFG.t_peak)
    g = 0.5 * COLOR.bb_t_min / temperature
    assert g * temperature < COLOR.bb_t_min
    expected = g**4 * (temperature / DISK_CFG.t_peak) ** 4 * table[0]
    np.testing.assert_allclose(radiance(r, table, g=g), expected, rtol=1e-12)


def test_observed_temperature_above_table(table):
    r = peak_radius(DISK_CFG.r_in)
    g = 5.0
    assert g * DISK_CFG.t_peak > COLOR.bb_t_max
    np.testing.assert_allclose(radiance(r, table, g=g), g**4 * table[-1], rtol=1e-12)


def test_shade_sample_switches(table):
    x = np.array([2.0, 0.0, 5.0])
    v = np.array([0.4, -0.8, -0.2])
    r = math.hypot(x[0], x[2])
    g = redshift_factor(r, 30.0, axial_angular_momentum(x, v), energy_at_infinity(x, v))
    assert g != pytest.approx(1.0)
    # Doppler desligado: independente de v; textura desligada: m = 1.
    np.testing.assert_array_equal(shade(DISK, x, v, 40.0, table, False, 30.0), radiance(r, table))
    np.testing.assert_array_equal(
        shade(DISK, x, v, 40.0, table, True, 30.0), radiance(r, table, g=g)
    )
    m = texture(r, math.atan2(-x[2], x[0]), 100.0 - 40.0)
    np.testing.assert_array_equal(
        shade(DISK, x, v, 40.0, table, True, 30.0, t_obs=100.0, tex_on=True),
        radiance(r, table, g=g, m=m),
    )
    # Sombra continua preta.
    for status in (0, MAX_STEPS):
        assert not shade(status, x, v, 0.0, table, True, 30.0).any()


def test_both_faces_emit_equally_with_doppler_and_texture(camera, table):
    """L_y, E e φ não mudam ao inverter v_y: com Doppler e textura as faces emitem igual."""
    status, x, v, _, dts = trace_fields(camera, replace(RenderConfig(), width=80, height=45))
    hits = status == DISK
    assert np.any(v[hits][:, 1] < 0.0) and np.any(v[hits][:, 1] > 0.0)
    for j, i in np.argwhere(hits)[:20]:
        flipped = v[j, i] * np.array([1.0, -1.0, 1.0])
        args = (table, True, camera.r_cam, 100.0, True)
        lit = shade(DISK, x[j, i], v[j, i], dts[j, i], *args)
        np.testing.assert_array_equal(lit, shade(DISK, x[j, i], flipped, dts[j, i], *args))
    assert lit.any()


# --- lado que se aproxima -----------------------------------------------------


def test_approaching_side_has_g_above_one(camera):
    """Câmera em +x: o gás em z > 0 anda para +x (rumo à câmera), o de z < 0 se afasta."""
    status, x, v, phi, _ = trace_fields(camera, replace(RenderConfig(), width=200, height=112))
    primary = (status == DISK) & (np.abs(phi) < math.pi)
    g = np.full(status.shape, np.nan)
    for j, i in zip(*np.nonzero(primary)):
        r = math.hypot(x[j, i, 0], x[j, i, 2])
        l_y = axial_angular_momentum(x[j, i], v[j, i])
        g[j, i] = redshift_factor(r, camera.r_cam, l_y, energy_at_infinity(x[j, i], v[j, i]))
    approaching = primary & (x[:, :, 2] > 1.0)
    receding = primary & (x[:, :, 2] < -1.0)
    assert approaching.sum() > 50 and receding.sum() > 50
    assert np.nanmax(g[approaching]) > 1.0
    assert np.all(g[receding] < 1.0)
    assert np.nanmean(g[approaching]) > np.nanmean(g[receding])


def test_render_approaching_side_brighter_and_bluer(camera):
    """Critério de aceite: com Doppler, a metade do disco que se aproxima (z > 0, à esquerda
    na imagem) fica mais clara e mais azul que a espelhada; sem Doppler, as duas são iguais."""
    cfg = replace(RenderConfig(), width=200, height=112, background="checker")
    color = replace(COLOR, exposure=0.25)  # sem saturar no ACES
    status, x, _, _, _ = trace_fields(camera, cfg)
    assert np.all(camera.right[[0, 1]] == pytest.approx(0.0, abs=1e-12))
    assert camera.right[2] < 0.0  # +z aparece à esquerda

    def halves(image):
        img = image.astype(float)
        mirror = img[:, ::-1]
        both = (status == DISK) & (status[:, ::-1] == DISK)
        left = both & (x[:, :, 2] > 1.0)
        lum = img.sum(axis=2)
        blue = img[:, :, 2] / np.maximum(img[:, :, 0], 1.0)
        blue_m = mirror[:, :, 2] / np.maximum(mirror[:, :, 0], 1.0)
        return lum[left], lum[:, ::-1][left], blue[left], blue_m[left]

    with_doppler = render(camera, cfg, disk=DISK_CFG, color=color)
    lum_a, lum_r, blue_a, blue_r = halves(with_doppler)
    assert lum_a.size > 100
    assert lum_a.mean() > 1.2 * lum_r.mean()
    assert blue_a.mean() > blue_r.mean()

    without = render(camera, cfg, disk=replace(DISK_CFG, doppler=False), color=color)
    lum_a, lum_r, _, _ = halves(without)
    assert abs(lum_a.mean() - lum_r.mean()) < 0.02 * lum_a.mean()


# --- estrelas com g_∞ -----------------------------------------------------------


def star(d, table, shift, t_star=None):
    t_min = STARS.t_min if t_star is None else t_star
    t_max = STARS.t_max if t_star is None else t_star
    return np.array(
        star_radiance(
            np.asarray(d, dtype=np.float64),
            float(STARS.grid_size),
            STARS.density,
            STARS.jitter,
            STARS.seed,
            STARS.sigma,
            STARS.intensity_min,
            STARS.intensity_max,
            t_min,
            t_max,
            table,
            COLOR.bb_t_min,
            COLOR.bb_t_max,
            shift,
        )
    )


def test_stars_get_static_blueshift(table):
    t_star = 5_000.0
    shift = static_blueshift(3.0)
    rng = np.random.default_rng(1)
    lit = 0
    for d in rng.normal(size=(4000, 3)):
        base = star(d, table, 1.0, t_star)
        if not base.any():
            continue
        lit += 1
        # Mesma estrela: T × g_∞ e brilho × g_∞⁴.
        profile = base / bb(t_star, table)
        expected = shift**4 * profile * bb(shift * t_star, table)
        np.testing.assert_allclose(star(d, table, shift, t_star), expected, rtol=1e-12)
    assert lit > 0


def test_shade_sample_stars_and_checker_with_doppler(table):
    rng = np.random.default_rng(2)
    r_cam = 4.0
    x = np.array([100.0, 0.0, 0.0])
    for d in rng.normal(size=(2000, 3)):
        np.testing.assert_array_equal(
            shade(ESCAPED, x, d, 0.0, table, True, r_cam),
            star(d, table, static_blueshift(r_cam)),
        )
        np.testing.assert_array_equal(
            shade(ESCAPED, x, d, 0.0, table, True, r_cam, background="checker"),
            shade(ESCAPED, x, d, 0.0, table, False, r_cam, background="checker"),
        )


# --- textura --------------------------------------------------------------------


def test_texture_is_bounded_and_varies():
    rng = np.random.default_rng(3)
    values = np.array(
        [
            texture(r, phi, t)
            for r, phi, t in zip(
                rng.uniform(3.0, 12.0, 5000),
                rng.uniform(-math.pi, math.pi, 5000),
                rng.uniform(0.0, 1.0e3, 5000),
            )
        ]
    )
    assert values.min() >= 1.0 - TEX.contrast and values.max() <= 1.0 + TEX.contrast
    assert values.std() > 0.05


@pytest.mark.parametrize("t", [0.0, 37.5, 1.0e4])
def test_texture_periodic_at_seam(t):
    for r in (3.2, 5.0, 11.9):
        assert texture(r, -math.pi, t) == pytest.approx(texture(r, math.pi, t), abs=1e-9)
        # Continuidade dos dois lados da costura.
        eps = 1.0e-7
        left, right = texture(r, math.pi - eps, t), texture(r, -math.pi + eps, t)
        assert left == pytest.approx(right, abs=1e-4)


def test_texture_is_continuous_in_phi_and_r():
    phis = np.linspace(-math.pi, math.pi, 20001)
    ring = np.array([texture(6.0, p, 0.0) for p in phis])
    assert np.abs(np.diff(ring)).max() < 0.01
    radii = np.linspace(3.0, 12.0, 20001)
    ray = np.array([texture(r, 0.3, 0.0) for r in radii])
    assert np.abs(np.diff(ray)).max() < 0.01


@pytest.mark.parametrize("t", [0.0, 1.0e6])
def test_texture_co_rotates_with_keplerian_omega(t):
    rng = np.random.default_rng(4)
    samples = zip(
        rng.uniform(3.0, 12.0, 200), rng.uniform(-3.0, 3.0, 200), rng.uniform(0.0, 50.0, 200)
    )
    for r, phi, dt in samples:
        moved = phi + keplerian_omega(r) * dt
        assert texture(r, moved, t + dt) == pytest.approx(texture(r, phi, t), abs=1e-6)


def ring_shift(r, delta_t, n=4096):
    """Rotação (rad) do padrão do anel r entre t = 0 e t = δt, por correlação circular."""
    phis = np.linspace(-math.pi, math.pi, n, endpoint=False)
    a = np.array([texture(r, p, 0.0) for p in phis])
    b = np.array([texture(r, p, delta_t) for p in phis])
    a -= a.mean()
    b -= b.mean()
    corr = np.fft.ifft(np.fft.fft(b) * np.conj(np.fft.fft(a))).real
    return np.argmax(corr) * 2.0 * math.pi / n


def test_inner_rings_rotate_faster():
    delta_t = 10.0
    inner, outer = ring_shift(4.0, delta_t), ring_shift(10.0, delta_t)
    step = 2.0 * math.pi / 4096
    assert inner == pytest.approx(keplerian_omega(4.0) * delta_t, abs=2 * step)
    assert outer == pytest.approx(keplerian_omega(10.0) * delta_t, abs=2 * step)
    assert inner > 3.0 * outer


@pytest.mark.parametrize(
    "field, value",
    [
        ("radial_cells", 0.0),
        ("azimuthal_cells", 0),
        ("azimuthal_cells", 2.5),
        ("octaves", 0),
        ("contrast", 1.5),
    ],
)
def test_invalid_texture_config_raises(field, value):
    with pytest.raises(ValueError):
        render_cpu.validate_texture(replace(TEX, **{field: value}))


# --- atraso da luz Δt -------------------------------------------------------------


def radial_delay(r1, r2):
    """Δt de um raio radial entre r1 e r2: ∫ dr/(1 − 1/r) = |Δr| + |ln((r2 − 1)/(r1 − 1))|."""
    return abs(r2 - r1) + abs(math.log((r2 - 1.0) / (r1 - 1.0)))


def test_radial_ray_delay_outward():
    direction = np.array([1.0, 2.0, -2.0]) / 3.0
    r1 = 2.0
    status, x, _, _, _, dt = trace_ray(
        r1 * direction, 0.8 * direction, 100.0, IntegratorConfig(), DiskConfig(enabled=False)
    )
    assert status == ESCAPED
    r2 = float(np.linalg.norm(x))
    assert dt == pytest.approx((r2 - r1) + math.log((r2 - 1.0) / (r1 - 1.0)), rel=1e-5)


def test_radial_ray_delay_inward():
    direction = np.array([0.0, 0.6, 0.8])
    integrator = replace(IntegratorConfig(), max_steps=300)
    r1 = 50.0
    status, x, _, _, _, dt = trace_ray(
        r1 * direction, -direction, 1.0e3, integrator, DiskConfig(enabled=False)
    )
    assert status == MAX_STEPS
    r2 = float(np.linalg.norm(x))
    assert 1.2 < r2 < 20.0
    assert dt == pytest.approx(radial_delay(r1, r2), rel=1e-5)


def test_disk_delay_is_positive_and_plausible(camera):
    status, x, _, _, dt = trace_fields(camera, replace(RenderConfig(), width=80, height=45))
    hits = status == DISK
    # Da câmera (r = 30) ao disco (3 ≤ r ≤ 12): pelo menos a distância em linha reta.
    straight = np.linalg.norm(x[hits] - camera.position, axis=1)
    assert np.all(dt[hits] >= straight)
    assert np.all(dt[hits] < 200.0)


# --- render CPU: CLI --------------------------------------------------------------


def test_render_cpu_cli_flags(tmp_path, capsys):
    out = tmp_path / "t.png"
    args = ["--width", "16", "--height", "9", "--texture", "--time", "12.5", "--output", str(out)]
    assert render_cpu.main(args) == 0
    assert out.exists()
    assert "Doppler ligado" in capsys.readouterr().out
    assert render_cpu.main([*args, "--no-doppler"]) == 0
    assert "Doppler desligado" in capsys.readouterr().out


def test_texture_changes_render_with_time(camera):
    cfg = replace(RenderConfig(), width=80, height=45, background="checker")
    disk = replace(DISK_CFG, texture=True)
    a = render(camera, cfg, disk=disk, t_obs=0.0)
    b = render(camera, cfg, disk=disk, t_obs=50.0)
    plain = render(camera, cfg, disk=DISK_CFG)
    assert not np.array_equal(a, b)
    assert not np.array_equal(a, plain)


# --- CPU × GPU ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def renderer():
    try:
        ctx = moderngl.create_standalone_context(require=GLCheckConfig().min_version_code)
    except Exception as exc:  # moderngl/glcontext não expõem um tipo de erro específico
        pytest.skip(f"sem contexto OpenGL 3.3: {exc}")
    gpu_renderer = GpuRenderer(ctx)
    yield gpu_renderer
    gpu_renderer.release()
    ctx.release()


def camera_at(r_cam):
    return Camera.from_config(replace(CAM, r_cam=r_cam))


@pytest.mark.parametrize(
    "r_cam, doppler, tex_on, t_obs",
    [
        (30.0, True, False, 0.0),
        (10.0, True, False, 0.0),
        (30.0, True, True, 100.0),
        (10.0, True, True, 100.0),
        (30.0, False, True, 100.0),
        # Tempo longo: t em float32 na GPU (matriz I/O).
        (30.0, True, True, 1.0e5),
    ],
)
def test_gpu_matches_cpu_with_doppler_and_texture(renderer, r_cam, doppler, tex_on, t_obs):
    local = replace(renderer.disk, doppler=doppler)
    original = renderer.disk
    renderer.disk = local
    try:
        res = compare.compare_scene(
            renderer,
            camera_at(r_cam),
            SMALL_W,
            SMALL_H,
            GPU.integrator.max_steps,
            True,
            CMP,
            texture=tex_on,
            t_obs=t_obs,
        )
    finally:
        renderer.disk = original
    assert res.bad_fraction <= CMP.max_bad_fraction
    assert compare.channel_diff(res.cpu, res.gpu).mean() < 1.0


def test_gpu_doppler_off_equals_phase_4_shader(renderer):
    """B desligado: a GPU volta à imagem sem Doppler (a mesma da CPU sem Doppler)."""
    camera = camera_at(30.0)
    renderer.set_uniforms(camera, SMALL_W, SMALL_H, background="checker", doppler=False)
    off = renderer.render()
    renderer.set_uniforms(camera, SMALL_W, SMALL_H, background="checker", doppler=True)
    on = renderer.render()
    assert not np.array_equal(off, on)
    cpu = render(
        camera,
        replace(RenderConfig(), width=SMALL_W, height=SMALL_H),
        integrator=GPU.integrator,
        disk=replace(DISK_CFG, doppler=False),
        background="checker",
    )
    assert compare.channel_diff(cpu, off).mean() < 1.0


def test_gpu_texture_long_time_has_no_jumps(renderer):
    """t grande em float32: entre dois quadros próximos a imagem muda pouco, como com t pequeno."""
    camera = camera_at(30.0)
    frame_dt = 40.0 * M / 60.0  # um quadro a 60 FPS com o time_scale padrão

    def frame_change(t):
        renderer.set_uniforms(camera, SMALL_W, SMALL_H, background="checker", texture=True, time=t)
        a = renderer.render()
        renderer.set_uniforms(
            camera, SMALL_W, SMALL_H, background="checker", texture=True, time=t + frame_dt
        )
        b = renderer.render()
        return compare.channel_diff(a, b).mean()

    small, large = frame_change(100.0), frame_change(1.0e5)
    assert large < 3.0 * small + 0.5


def test_gpu_rejects_non_finite_time(renderer):
    with pytest.raises(ValueError):
        renderer.set_uniforms(camera_at(30.0), 8, 8, time=math.inf)
