"""Testes da Fase 3: perfil de temperatura, corpo negro, cadeia de cor e estrelas."""

import math
from dataclasses import replace

import numpy as np
import pytest

from blackhole.camera import Camera
from blackhole.config import (
    CameraConfig,
    ColorConfig,
    DiskConfig,
    RenderConfig,
    StarfieldConfig,
)
from blackhole.physics import R_ISCO, disk_temperature, peak_radius
from blackhole.render_cpu import render
from blackhole.scene import (
    SRGB_THRESHOLD,
    XYZ_TO_SRGB,
    aces,
    blackbody_rgb,
    blackbody_table,
    blackbody_temperatures,
    cell_hash,
    disk_radiance,
    encode_pixel,
    hash_to_unit,
    hash_u32,
    srgb_encode,
    star_radiance,
)

COLOR = ColorConfig()
DISK_CFG = DiskConfig()
STARS = StarfieldConfig()
MASK = 0xFFFFFFFF


@pytest.fixture(scope="module")
def table():
    return blackbody_table(COLOR)


def bb(temperature, table):
    return np.array(blackbody_rgb(float(temperature), table, COLOR.bb_t_min, COLOR.bb_t_max))


def luminance(rgb):
    """Y de uma cor em sRGB linear, pela inversa da matriz XYZ → sRGB."""
    return (np.linalg.inv(XYZ_TO_SRGB) @ rgb)[1]


# --- perfil de temperatura do disco (seção 3.6) -----------------------------


def test_peak_radius():
    assert peak_radius(R_ISCO) == pytest.approx(49.0 / 36.0 * 3.0)
    assert peak_radius(R_ISCO) == pytest.approx(4.0833, abs=1e-4)


def test_temperature_is_zero_at_inner_edge():
    assert disk_temperature(R_ISCO, R_ISCO, DISK_CFG.t_peak) == 0.0
    assert disk_temperature(0.5 * R_ISCO, R_ISCO, DISK_CFG.t_peak) == 0.0


def test_temperature_peak():
    r_in, t_peak = DISK_CFG.r_in, DISK_CFG.t_peak
    r_pk = peak_radius(r_in)
    assert disk_temperature(r_pk, r_in, t_peak) == pytest.approx(t_peak, rel=1e-14)
    # Máximo: menor dos dois lados, em toda a faixa do disco.
    radii = np.linspace(r_in, DISK_CFG.r_out, 2001)
    temps = np.array([disk_temperature(r, r_in, t_peak) for r in radii])
    assert temps.max() <= t_peak * (1.0 + 1e-12)
    assert disk_temperature(r_pk * 0.99, r_in, t_peak) < t_peak
    assert disk_temperature(r_pk * 1.01, r_in, t_peak) < t_peak
    assert np.all(temps[1:] > 0.0)


# --- tabela de corpo negro --------------------------------------------------


def test_blackbody_table_shape_and_log_spacing(table):
    assert table.shape == (COLOR.bb_table_size, 3)
    temps = blackbody_temperatures(COLOR)
    assert temps[0] == pytest.approx(COLOR.bb_t_min)
    assert temps[-1] == pytest.approx(COLOR.bb_t_max)
    np.testing.assert_allclose(np.diff(np.log(temps)), np.diff(np.log(temps))[0], rtol=1e-9)
    assert np.all(table >= 0.0)


def test_blackbody_table_has_unit_luminance_where_not_clipped(table):
    unclipped = np.all(table > 0.0, axis=1)
    # A faixa quente (onde nada é cortado) é a maior parte da tabela.
    assert unclipped.sum() > COLOR.bb_table_size // 2
    for rgb in table[unclipped]:
        assert luminance(rgb) == pytest.approx(1.0, rel=1e-9)


def test_blackbody_colors(table):
    cold = bb(1_000.0, table)
    hot = bb(40_000.0, table)
    white = bb(6_500.0, table)
    assert cold[0] > cold[2]
    assert hot[2] > hot[0]
    # 6 500 K fica perto do branco D65: canais a menos de 10% entre si.
    assert white.max() / white.min() < 1.10


def test_blackbody_interpolates_in_log_t(table):
    temps = blackbody_temperatures(COLOR)
    np.testing.assert_allclose(bb(temps[10], table), table[10], rtol=1e-9)
    mid = math.sqrt(temps[10] * temps[11])
    np.testing.assert_allclose(bb(mid, table), 0.5 * (table[10] + table[11]), rtol=1e-9)


@pytest.mark.parametrize("temperature", [0.0, 500.0, 999.0])
def test_blackbody_clamps_below_table(table, temperature):
    np.testing.assert_array_equal(bb(temperature, table), table[0])


@pytest.mark.parametrize("temperature", [40_000.0, 41_000.0, 1.0e6])
def test_blackbody_clamps_above_table(table, temperature):
    np.testing.assert_array_equal(bb(temperature, table), table[-1])


# --- radiância do disco ------------------------------------------------------


def radiance(r, table, t_peak=DISK_CFG.t_peak):
    return np.array(
        disk_radiance(r, DISK_CFG.r_in, t_peak, table, COLOR.bb_t_min, COLOR.bb_t_max)
    )


def test_disk_radiance_black_at_inner_edge(table):
    np.testing.assert_array_equal(radiance(DISK_CFG.r_in, table), [0.0, 0.0, 0.0])


def test_disk_radiance_at_peak_is_blackbody_color(table):
    r_pk = peak_radius(DISK_CFG.r_in)
    np.testing.assert_allclose(radiance(r_pk, table), bb(DISK_CFG.t_peak, table), rtol=1e-12)


def test_disk_radiance_above_table_keeps_t4_brightness(table):
    # T_pico acima da tabela: cor da extremidade quente, brilho (T/T_pico)⁴ = 1 no pico.
    t_peak = 50_000.0
    r_pk = peak_radius(DISK_CFG.r_in)
    np.testing.assert_allclose(radiance(r_pk, table, t_peak), table[-1], rtol=1e-12)


def test_disk_radiance_below_table_keeps_t4_brightness(table):
    # Perto de r_in, T < 1 000 K: cor da extremidade fria, brilho ainda (T/T_pico)⁴.
    r = DISK_CFG.r_in * (1.0 + 1e-6)
    temperature = disk_temperature(r, DISK_CFG.r_in, DISK_CFG.t_peak)
    assert 0.0 < temperature < COLOR.bb_t_min
    expected = (temperature / DISK_CFG.t_peak) ** 4 * table[0]
    np.testing.assert_allclose(radiance(r, table), expected, rtol=1e-12)


# --- ACES e sRGB ------------------------------------------------------------


def test_tone_mapping_zero_is_black():
    assert aces(0.0) == 0.0
    assert srgb_encode(0.0) == 0.0
    assert encode_pixel(0.0, 0.0, 0.0, COLOR.exposure) == (0.0, 0.0, 0.0)


def test_aces_monotonic_and_bounded():
    xs = np.concatenate(([0.0], np.logspace(-6, 3, 2000)))
    ys = np.array([aces(x) for x in xs])
    assert np.all(np.diff(ys) >= 0.0)
    assert ys.min() >= 0.0 and ys.max() <= 1.0


def test_srgb_monotonic_and_continuous():
    xs = np.linspace(0.0, 1.0, 5001)
    ys = np.array([srgb_encode(x) for x in xs])
    assert np.all(np.diff(ys) > 0.0)
    assert srgb_encode(1.0) == pytest.approx(1.0, abs=1e-12)
    # Os dois ramos se encontram em 0,0031308 (diferença de ~3e-8 pelos coeficientes).
    below = srgb_encode(SRGB_THRESHOLD)
    above = srgb_encode(math.nextafter(SRGB_THRESHOLD, 1.0))
    assert above == pytest.approx(below, abs=1e-6)


# --- hash e estrelas (seção 3.7) --------------------------------------------


def lowbias32_ref(x):
    """Referência em Python puro (int e máscara de 32 bits)."""
    x &= MASK
    x ^= x >> 16
    x = (x * 0x7FEB352D) & MASK
    x ^= x >> 15
    x = (x * 0x846CA68B) & MASK
    x ^= x >> 16
    return x


def cell_hash_ref(ix, iy, iz, seed):
    h = lowbias32_ref(seed & MASK)
    for i in (ix, iy, iz):
        h = lowbias32_ref(h ^ (i & MASK))
    return h


@pytest.mark.parametrize("x", [0, 1, 2, 12345, 0x7FFFFFFF, 0x80000000, 0xDEADBEEF, MASK])
def test_hash_matches_python_reference(x):
    h = hash_u32(np.uint64(x))
    assert int(h) == lowbias32_ref(x)
    assert 0 <= int(h) <= MASK


@pytest.mark.parametrize(
    "cell", [(0, 0, 0), (1, -1, 2), (-100, 57, -3), (99, 99, -99), (-1, -1, -1)]
)
@pytest.mark.parametrize("seed", [0, STARS.seed, MASK])
def test_cell_hash_matches_python_reference(cell, seed):
    assert int(cell_hash(*cell, seed)) == cell_hash_ref(*cell, seed)


def test_hash_to_unit_range():
    assert hash_to_unit(np.uint64(0)) == 0.0
    top = hash_to_unit(np.uint64(MASK))
    assert top < 1.0
    assert top == (MASK >> 8) / 2.0**24


def test_lit_cell_fraction_matches_density():
    n = 24
    lit = 0
    for ix in range(-n // 2, n // 2):
        for iy in range(-n // 2, n // 2):
            for iz in range(-n // 2, n // 2):
                lit += hash_to_unit(cell_hash(ix, iy, iz, STARS.seed)) < STARS.density
    fraction = lit / n**3
    sigma = math.sqrt(STARS.density * (1.0 - STARS.density) / n**3)
    assert abs(fraction - STARS.density) < 5.0 * sigma


def star(d, table, seed=STARS.seed):
    return np.array(
        star_radiance(
            np.asarray(d, dtype=np.float64),
            float(STARS.grid_size),
            STARS.density,
            STARS.jitter,
            seed,
            STARS.sigma,
            STARS.intensity_min,
            STARS.intensity_max,
            STARS.t_min,
            STARS.t_max,
            table,
            COLOR.bb_t_min,
            COLOR.bb_t_max,
        )
    )


def test_stars_are_deterministic_and_scale_free(table):
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(2000, 3))
    first = np.array([star(d, table) for d in dirs])
    second = np.array([star(3.7 * d, table) for d in dirs])  # d não precisa ser unitária
    np.testing.assert_allclose(first, second, rtol=1e-12, atol=0.0)
    assert np.all(first >= 0.0)
    assert np.any(first > 0.0)
    other_seed = np.array([star(d, table, seed=STARS.seed + 1) for d in dirs])
    assert not np.array_equal(first, other_seed)


def test_stars_render_is_deterministic():
    camera = Camera.from_config(CameraConfig())
    cfg = replace(RenderConfig(), width=80, height=45, background="stars")
    no_disk = DiskConfig(enabled=False)
    a = render(camera, cfg, disk=no_disk)
    b = render(camera, cfg, disk=no_disk)
    np.testing.assert_array_equal(a, b)
    assert a.any()


def test_invalid_background_raises():
    camera = Camera.from_config(CameraConfig())
    cfg = replace(RenderConfig(), width=4, height=4)
    with pytest.raises(ValueError):
        render(camera, cfg, background="galaxy")
    with pytest.raises(ValueError):
        render(camera, replace(cfg, background="galaxy"))
