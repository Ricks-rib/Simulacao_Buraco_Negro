"""Cena: colorimetria do disco e do fundo (PLANO, seções 3.6 e 3.7).

- Fundo de depuração: xadrez em latitude/longitude em torno de +y.
- Corpo negro: tabela de cor em sRGB linear com luminância Y = 1, construída uma vez em
  NumPy (Planck × CMF CIE 1931) e interpolada em ln T dentro dos kernels.
- Radiância do disco: (T/T_pico)⁴ · rgb(T), com T(r) de ``physics.disk_temperature``.
- Campo estelar procedural: hash inteiro de 32 bits da célula de direção, reproduzível em
  GLSL 3.3 com ``uint``.
- Cadeia de saída: exposição → ACES aproximado (Narkowicz) → gama sRGB, a mesma para o
  disco e para o fundo.

Tudo aqui é replicado em ``shaders/blackhole.frag`` com ``// ver scene.py: <nome>``.
"""

import math

import numpy as np
from numba import njit

from blackhole.config import ColorConfig
from blackhole.physics import disk_temperature

# --- xadrez de depuração (seção 3.7) ----------------------------------------


@njit(cache=True)
def checker_color(d, cell_rad, color_a, color_b):
    """Cor do xadrez celeste na direção d (não precisa ser unitária).

    longitude = atan2(d_z, d_x) e latitude = asin(d_y/|d|), em torno do eixo +y. A
    paridade de ⌊lon/cell⌋ + ⌊lat/cell⌋ escolhe ``color_a`` (par) ou ``color_b`` (ímpar).
    Com casas de 15° são 24 × 12 casas. Cores em [0, 1], tratadas como radiância linear.
    """
    norm = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    lon = math.atan2(d[2], d[0])
    lat = math.asin(min(1.0, max(-1.0, d[1] / norm)))
    parity = (int(math.floor(lon / cell_rad)) + int(math.floor(lat / cell_rad))) & 1
    if parity == 0:
        return color_a
    return color_b


# --- corpo negro (seção 3.6) ------------------------------------------------

# Constantes de Planck (SI, valores exatos do SI de 2019).
PLANCK_H = 6.62607015e-34  # J·s
LIGHT_C = 2.99792458e8  # m/s
BOLTZMANN_K = 1.380649e-23  # J/K
NM = 1.0e-9  # m por nm

# Ajuste analítico multilobo das CMF CIE 1931 2° (Wyman, Sloan e Shirley, 2013, JCGT 2(2)).
# Cada lobo é (peso, μ, σ à esquerda de μ, σ à direita de μ), com λ em nm.
CMF_X_LOBES = ((1.056, 599.8, 37.9, 31.0), (0.362, 442.0, 16.0, 26.7), (-0.065, 501.1, 20.4, 26.2))
CMF_Y_LOBES = ((0.821, 568.8, 46.9, 40.5), (0.286, 530.9, 16.3, 31.1))
CMF_Z_LOBES = ((1.217, 437.0, 11.8, 36.0), (0.681, 459.0, 26.0, 13.8))

# XYZ → sRGB linear (IEC 61966-2-1, primárias Rec. 709, branco D65).
XYZ_TO_SRGB = np.array(
    [
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ]
)


def _cmf_lobes(lam, lobes):
    """Soma de gaussianas assimétricas: Σ w · exp(−½ ((λ − μ)/σ)²), σ pelo lado de μ."""
    total = np.zeros_like(lam)
    for weight, mu, sigma_left, sigma_right in lobes:
        sigma = np.where(lam < mu, sigma_left, sigma_right)
        total += weight * np.exp(-0.5 * ((lam - mu) / sigma) ** 2)
    return total


def cie_cmf(lam_nm):
    """CMF CIE 1931 2° (x̄, ȳ, z̄) pelo ajuste de Wyman, Sloan e Shirley; λ em nm."""
    lam = np.asarray(lam_nm, dtype=np.float64)
    return _cmf_lobes(lam, CMF_X_LOBES), _cmf_lobes(lam, CMF_Y_LOBES), _cmf_lobes(lam, CMF_Z_LOBES)


def planck(lam_nm, temperature):
    """Radiância espectral de Planck B_λ(T) = 2hc²/λ⁵ / (exp(hc/(λ k T)) − 1), em SI."""
    lam = np.asarray(lam_nm, dtype=np.float64) * NM
    x = PLANCK_H * LIGHT_C / (lam * BOLTZMANN_K * temperature)
    return 2.0 * PLANCK_H * LIGHT_C**2 / lam**5 / np.expm1(x)


def blackbody_temperatures(cfg: ColorConfig):
    """Temperaturas da tabela: ``bb_table_size`` valores espaçados uniformemente em ln T."""
    return np.exp(
        np.linspace(math.log(cfg.bb_t_min), math.log(cfg.bb_t_max), cfg.bb_table_size)
    )


def blackbody_table(cfg: ColorConfig):
    """Tabela (N, 3) de cores de corpo negro em sRGB linear, com luminância Y = 1.

    Para cada T: XYZ = Σ B_λ(T) · (x̄, ȳ, z̄) · Δλ de ``wavelength_min_nm`` a
    ``wavelength_max_nm``, normalizado por Y e levado a sRGB linear pela matriz da
    IEC 61966-2-1. Componentes negativas (T fora do gamut sRGB) são cortadas em 0; só
    nessas entradas Y deixa de ser 1. Na GPU a tabela vira textura 1D.
    """
    n_lam = int(round((cfg.wavelength_max_nm - cfg.wavelength_min_nm) / cfg.wavelength_step_nm))
    lam = cfg.wavelength_min_nm + cfg.wavelength_step_nm * np.arange(n_lam + 1)
    xbar, ybar, zbar = cie_cmf(lam)
    table = np.empty((cfg.bb_table_size, 3))
    for k, temperature in enumerate(blackbody_temperatures(cfg)):
        spectrum = planck(lam, temperature)
        xyz = np.array([np.sum(spectrum * xbar), np.sum(spectrum * ybar), np.sum(spectrum * zbar)])
        table[k] = XYZ_TO_SRGB @ (xyz / xyz[1])
    return np.maximum(table, 0.0)


@njit(cache=True)
def blackbody_rgb(temperature, table, t_min, t_max):
    """Cor de corpo negro (sRGB linear, Y = 1) por interpolação linear da tabela em ln T.

    Fora de [t_min, t_max] (inclusive T ≤ 0) vale a cor da extremidade mais próxima.
    """
    n = table.shape[0]
    if temperature <= t_min:
        return table[0, 0], table[0, 1], table[0, 2]
    if temperature >= t_max:
        return table[n - 1, 0], table[n - 1, 1], table[n - 1, 2]
    u = (math.log(temperature) - math.log(t_min)) / (math.log(t_max) - math.log(t_min)) * (n - 1)
    k = min(int(math.floor(u)), n - 2)
    w = u - k
    return (
        (1.0 - w) * table[k, 0] + w * table[k + 1, 0],
        (1.0 - w) * table[k, 1] + w * table[k + 1, 1],
        (1.0 - w) * table[k, 2] + w * table[k + 1, 2],
    )


@njit(cache=True)
def disk_radiance(r, r_in, t_peak, table, t_min, t_max):
    """Radiância linear do disco no raio r: (T/T_pico)⁴ · rgb(T) (antes da exposição).

    T(r) vem de ``physics.disk_temperature``. A emissão é a mesma nas duas faces e não
    depende da direção do raio (sem Doppler nem redshift, que são da Fase 5). Em r_in,
    T = 0 e a radiância é 0 (preto).
    """
    temperature = disk_temperature(r, r_in, t_peak)
    ratio = temperature / t_peak
    brightness = ratio * ratio * ratio * ratio
    cr, cg, cb = blackbody_rgb(temperature, table, t_min, t_max)
    return brightness * cr, brightness * cg, brightness * cb


# --- tone mapping e gama (seção 3.6) ----------------------------------------

# ACES aproximado de Narkowicz (2015): (x(a x + b)) / (x(c x + d) + e).
ACES_A = 2.51
ACES_B = 0.03
ACES_C = 2.43
ACES_D = 0.59
ACES_E = 0.14

# Curva sRGB por partes (IEC 61966-2-1).
SRGB_THRESHOLD = 0.0031308
SRGB_LINEAR_SLOPE = 12.92
SRGB_SCALE = 1.055
SRGB_OFFSET = 0.055
SRGB_GAMMA = 2.4


@njit(cache=True)
def aces(x):
    """Tone mapping ACES aproximado (Narkowicz) de um canal linear ≥ 0, cortado em [0, 1]."""
    y = (x * (ACES_A * x + ACES_B)) / (x * (ACES_C * x + ACES_D) + ACES_E)
    return min(max(y, 0.0), 1.0)


@njit(cache=True)
def srgb_encode(x):
    """sRGB exato por partes: 12,92·x até 0,0031308; acima, 1,055·x^(1/2,4) − 0,055."""
    if x <= SRGB_THRESHOLD:
        return SRGB_LINEAR_SLOPE * x
    return SRGB_SCALE * x ** (1.0 / SRGB_GAMMA) - SRGB_OFFSET


@njit(cache=True)
def encode_pixel(r, g, b, exposure):
    """Cadeia de saída de uma radiância linear: exposição → ACES → sRGB, canal a canal.

    Devolve valores em [0, 1]; radiância 0 dá (0, 0, 0), então a sombra continua preta.
    """
    return (
        srgb_encode(aces(exposure * r)),
        srgb_encode(aces(exposure * g)),
        srgb_encode(aces(exposure * b)),
    )


# --- campo estelar procedural (seção 3.7) -----------------------------------

# Tudo em uint64 com máscara de 32 bits depois de cada multiplicação: o resultado é o
# mesmo de ``uint`` no GLSL 3.3. (No Numba, uint64 com literal int vira float64, por
# isso todo literal é np.uint64.)
MASK32 = np.uint64(0xFFFFFFFF)
_SHIFT_8 = np.uint64(8)
_SHIFT_15 = np.uint64(15)
_SHIFT_16 = np.uint64(16)
# Multiplicadores do lowbias32 (Chris Wellons, "hash-prospector").
LOWBIAS32_M1 = np.uint64(0x7FEB352D)
LOWBIAS32_M2 = np.uint64(0x846CA68B)
# 2²⁴: (h ≫ 8)/2²⁴ é um número em [0, 1) exato em float32.
UNIT_SCALE = 16777216.0


@njit(cache=True)
def hash_u32(x):
    """lowbias32: hash inteiro de 32 bits sem sinal (x em uint64 com valor < 2³²).

    x ^= x ≫ 16; x *= 0x7feb352d; x ^= x ≫ 15; x *= 0x846ca68b; x ^= x ≫ 16 (mod 2³²).
    """
    x = x & MASK32
    x = x ^ (x >> _SHIFT_16)
    x = (x * LOWBIAS32_M1) & MASK32
    x = x ^ (x >> _SHIFT_15)
    x = (x * LOWBIAS32_M2) & MASK32
    x = x ^ (x >> _SHIFT_16)
    return x


@njit(cache=True)
def _int_to_u32(i):
    """Inteiro com sinal → uint32 em complemento de dois, como ``uint(int)`` no GLSL."""
    return np.uint64(i & 0xFFFFFFFF)


@njit(cache=True)
def cell_hash(ix, iy, iz, seed):
    """Hash da célula (ix, iy, iz): lowbias32 encadeado, h = H(H(H(H(seed) ^ ix) ^ iy) ^ iz)."""
    h = hash_u32(_int_to_u32(seed))
    h = hash_u32(h ^ _int_to_u32(ix))
    h = hash_u32(h ^ _int_to_u32(iy))
    h = hash_u32(h ^ _int_to_u32(iz))
    return h


@njit(cache=True)
def hash_to_unit(h):
    """Número em [0, 1) a partir dos 24 bits altos do hash: (h ≫ 8)/2²⁴."""
    return float(h >> _SHIFT_8) / UNIT_SCALE


@njit(cache=True)
def star_radiance(
    d,
    grid_size,
    density,
    jitter,
    seed,
    sigma,
    intensity_min,
    intensity_max,
    star_t_min,
    star_t_max,
    table,
    t_min,
    t_max,
):
    """Radiância linear do campo estelar na direção d (não precisa ser unitária).

    A célula é c = ⌊d̂·N⌋ numa rede cúbica 3D (sem polos nem costuras de cubo). Com o hash
    h da célula, a sequência r₀, r₁, … sai de h, H(h), H(H(h)), …:

    - a estrela existe se r₀ < densidade;
    - centro ŝ = normalize((c + ½ + jitter·(r₁₂₃ − ½))/N);
    - intensidade log-uniforme (r₄) e temperatura log-uniforme (r₅) → ``blackbody_rgb``;
    - perfil gaussiano na corda: I · exp(−|d̂ − ŝ|²/(2σ²)).

    Só a própria célula é avaliada.
    """
    norm = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    dx, dy, dz = d[0] / norm, d[1] / norm, d[2] / norm
    cx = math.floor(dx * grid_size)
    cy = math.floor(dy * grid_size)
    cz = math.floor(dz * grid_size)

    h = cell_hash(int(cx), int(cy), int(cz), seed)
    if hash_to_unit(h) >= density:
        return 0.0, 0.0, 0.0
    h = hash_u32(h)
    jx = hash_to_unit(h)
    h = hash_u32(h)
    jy = hash_to_unit(h)
    h = hash_u32(h)
    jz = hash_to_unit(h)
    h = hash_u32(h)
    r_intensity = hash_to_unit(h)
    h = hash_u32(h)
    r_temperature = hash_to_unit(h)

    sx = cx + 0.5 + jitter * (jx - 0.5)
    sy = cy + 0.5 + jitter * (jy - 0.5)
    sz = cz + 0.5 + jitter * (jz - 0.5)
    s_norm = math.sqrt(sx * sx + sy * sy + sz * sz)
    ex = dx - sx / s_norm
    ey = dy - sy / s_norm
    ez = dz - sz / s_norm
    chord2 = ex * ex + ey * ey + ez * ez

    intensity = intensity_min * (intensity_max / intensity_min) ** r_intensity
    temperature = star_t_min * (star_t_max / star_t_min) ** r_temperature
    value = intensity * math.exp(-chord2 / (2.0 * sigma * sigma))
    cr, cg, cb = blackbody_rgb(temperature, table, t_min, t_max)
    return value * cr, value * cg, value * cb
