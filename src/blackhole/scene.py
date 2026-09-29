"""Cena: colorimetria do disco e do fundo (PLANO, seções 3.6 e 3.7).

- Fundo de depuração: xadrez em latitude/longitude em torno de +y.
- Corpo negro: tabela de cor em sRGB linear com luminância Y = 1, construída uma vez em
  NumPy (Planck × CMF CIE 1931) e interpolada em ln T dentro dos kernels.
- Radiância do disco: m · (g·T/T_pico)⁴ · rgb(g·T), com T(r) de
  ``physics.disk_temperature``, o fator de desvio g da seção 3.8 e a emissividade m da
  textura co-rotante (seção 3.9); g = m = 1 sem Doppler e sem textura.
- Campo estelar procedural: hash inteiro de 32 bits da célula de direção, reproduzível em
  GLSL 3.3 com ``uint``; com Doppler, blueshift g_∞ do observador estático.
- Textura do disco: ruído de valor periódico em φ, com o mesmo hash, no referencial que
  gira com Ω(r).
- Cadeia de saída: exposição → ACES aproximado (Narkowicz) → gama sRGB, a mesma para o
  disco e para o fundo.

Tudo aqui é replicado em ``shaders/blackhole.frag`` com ``// ver scene.py: <nome>``; a
cadeia de saída também em ``shaders/bloom_composite.frag`` e os pesos de luminância
(``LUMINANCE_WEIGHTS``, limiar do bloom clássico) em ``shaders/bloom_down.frag``.
"""

import math

import numpy as np
from numba import njit

from blackhole.config import ColorConfig
from blackhole.physics import disk_temperature, keplerian_omega

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
def disk_radiance(r, r_in, t_peak, table, t_min, t_max, g, emissivity):
    """Radiância linear observada do disco no raio r (antes da exposição).

        T_obs = g·T(r),    radiância = m · (T_obs/T_pico)⁴ · rgb(T_obs)

    T(r) vem de ``physics.disk_temperature`` (seção 3.6). g é o fator de desvio da seção
    3.8 (``physics.redshift_factor``): a cor segue a temperatura observada e o brilho
    bolométrico escala com g⁴·T⁴. m = ``emissivity`` é o fator da textura
    (``disk_texture``), que muda só a emissividade, não a temperatura. Com g = m = 1 é a
    radiância da Fase 3. Em r_in, T = 0 e a radiância é 0 (preto) com qualquer g.
    """
    temperature = g * disk_temperature(r, r_in, t_peak)
    ratio = temperature / t_peak
    brightness = emissivity * ratio * ratio * ratio * ratio
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


# Pesos da luminância relativa Y = 0,2126·R + 0,7152·G + 0,0722·B em sRGB linear
# (primárias Rec. 709, branco D65): a linha Y da inversa de XYZ_TO_SRGB. Usados pelo
# limiar do bloom clássico (shaders/bloom_down.frag, seção 3.10).
LUMINANCE_WEIGHTS = (0.2126, 0.7152, 0.0722)


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
    shift,
):
    """Radiância linear do campo estelar na direção d (não precisa ser unitária).

    A célula é c = ⌊d̂·N⌋ numa rede cúbica 3D (sem polos nem costuras de cubo). Com o hash
    h da célula, a sequência r₀, r₁, … sai de h, H(h), H(H(h)), …:

    - a estrela existe se r₀ < densidade;
    - centro ŝ = normalize((c + ½ + jitter·(r₁₂₃ − ½))/N);
    - intensidade log-uniforme (r₄) e temperatura log-uniforme (r₅) → ``blackbody_rgb``;
    - perfil gaussiano na corda: I · exp(−|d̂ − ŝ|²/(2σ²)).

    ``shift`` é o fator de desvio da luz vinda do infinito até a câmera: com Doppler,
    g_∞ = ``physics.static_blueshift(r_cam)``; sem ele, 1. A temperatura observada é
    shift·T e o brilho é multiplicado por shift⁴ (como no disco, seção 3.8).

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
    temperature = shift * star_t_min * (star_t_max / star_t_min) ** r_temperature
    shift2 = shift * shift
    value = shift2 * shift2 * intensity * math.exp(-chord2 / (2.0 * sigma * sigma))
    cr, cg, cb = blackbody_rgb(temperature, table, t_min, t_max)
    return value * cr, value * cg, value * cb


# --- textura do disco (seção 3.9) -------------------------------------------

# fbm: cada oitava multiplica as células por FBM_LACUNARITY (inteiro, para manter a
# periodicidade em φ) e a amplitude por FBM_GAIN.
FBM_LACUNARITY = 2
FBM_GAIN = 0.5
# Uma volta completa, em radianos.
TWO_PI = 2.0 * math.pi


@njit(cache=True)
def periodic_value_noise(u, w, n_w, octave, seed):
    """Ruído de valor suave em [0, 1) na grade inteira (u, w), periódico em w com período n_w.

    Os valores dos vértices saem de ``cell_hash(i, j mod n_w, octave, seed)`` e são
    interpolados com a curva suave s(f) = f²(3 − 2f) (contínua, com derivada contínua).
    w é levado a [0, n_w) por w − n_w·⌊w/n_w⌋ e o vizinho de j = n_w − 1 é j = 0, então
    o ruído é contínuo e periódico: nada muda ao trocar w por w + n_w.
    """
    w = w - n_w * math.floor(w / n_w)
    fu = u - math.floor(u)
    fw = w - math.floor(w)
    i0 = int(math.floor(u))
    j0 = int(math.floor(w)) % n_w
    j1 = (j0 + 1) % n_w
    v00 = hash_to_unit(cell_hash(i0, j0, octave, seed))
    v10 = hash_to_unit(cell_hash(i0 + 1, j0, octave, seed))
    v01 = hash_to_unit(cell_hash(i0, j1, octave, seed))
    v11 = hash_to_unit(cell_hash(i0 + 1, j1, octave, seed))
    su = fu * fu * (3.0 - 2.0 * fu)
    sw = fw * fw * (3.0 - 2.0 * fw)
    a = v00 + su * (v10 - v00)
    b = v01 + su * (v11 - v01)
    return a + sw * (b - a)


@njit(cache=True)
def disk_texture(r, phi, t_emit, radial_cells, azimuthal_cells, octaves, contrast, seed):
    """Fator de emissividade m ∈ [1 − c, 1 + c] da textura co-rotante do disco (seção 3.9).

    O gás no raio r gira com Ω(r) (``physics.keplerian_omega``), então a textura é fixa
    no ângulo co-rotante φ' = φ − Ω(r)·t_emit, com φ = atan2(−z, x) (cresce no sentido
    da rotação) e t_emit = t_obs − Δt o tempo coordenado da emissão. Como Ω cai com r,
    os anéis internos giram mais rápido e a textura se cisalha em espirais.

    fbm = Σ_k a_k·ruído_k / Σ_k a_k, com a_k = FBM_GAIN^k, e a oitava k usa
    radial_cells·L^k células por unidade de ln r e azimuthal_cells·L^k células por volta
    (L = FBM_LACUNARITY), sempre um número inteiro: periódico em φ. m = 1 + c·(2·fbm − 1).
    Exige octaves ≥ 1 (validado por quem chama).
    """
    u = math.log(r) * radial_cells
    turns = (phi - keplerian_omega(r) * t_emit) / TWO_PI
    # Só a fração da volta importa (n_w é inteiro); reduzir antes de multiplicar por n_w
    # preserva a precisão com t grande (float32 na GPU).
    turns -= math.floor(turns)
    total = 0.0
    norm = 0.0
    amplitude = 1.0
    scale = 1
    for k in range(octaves):
        n_w = azimuthal_cells * scale
        total += amplitude * periodic_value_noise(u * scale, turns * n_w, n_w, k, seed)
        norm += amplitude
        amplitude *= FBM_GAIN
        scale *= FBM_LACUNARITY
    fbm = total / norm
    return 1.0 + contrast * (2.0 * fbm - 1.0)
