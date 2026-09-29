"""Física de referência do fóton em Schwarzschild (CPU, float64).

Unidades G = c = 1 com r_s = 1 (M = 1/2); eixo do disco em +y. A geodésica nula é
integrada na forma cartesiana equivalente (PLANO, seção 3.2):

    a = −(3/2) · h² · x / |x|⁵,    h = |x × v| (constante ao longo do raio)

A órbita r(φ) coincide com a da geodésica (equação de Binet u'' + u = (3/2) u²).

Esta é a fonte da verdade da física: constantes, geodésica e integrador, invariantes,
tétrade do observador estático, raio de escape, fórmula de Synge e perfil de temperatura
do disco de acreção (seção 3.6) ficam aqui. A colorimetria (corpo negro, tone mapping,
sRGB) e as estrelas ficam em ``scene.py``.
``src/blackhole/shaders/blackhole.frag`` (Fase 4) replica cada uma com o comentário
``// ver physics.py: <nome>``, e ``uv run python -m blackhole.compare`` confere a GPU
(float32) contra esta referência (float64).
A projeção da câmera (``camera.py``) e o fundo (``scene.py``) também são replicados no
shader, com a mesma convenção ``// ver <arquivo>.py: <nome>``.
"""

import math

import numpy as np
from numba import njit

from blackhole.config import DiskConfig, IntegratorConfig

# --- constantes (seção 3.1) -------------------------------------------------

# Raio de Schwarzschild, que é também o horizonte de eventos.
R_S = 1.0
# Massa: r_s = 2M.
M = 0.5
# Esfera de fótons: r = 3M.
R_PHOTON = 1.5
# Parâmetro de impacto crítico b_c = 3√3·M, que é o raio da sombra vista do infinito.
B_CRIT = 3.0 * math.sqrt(3.0) / 2.0
# Borda interna do disco na órbita circular estável mais interna (ISCO): r = 6M.
R_ISCO = 3.0

# Coeficiente da aceleração: 3M = 3/2.
ACCEL_COEFF = 3.0 * M

# --- códigos de status do raio (seção 3.4) ----------------------------------

CAPTURED = 0  # r < r_s: pixel da sombra
ESCAPED = 1  # r > r_max e se afastando: amostra o fundo pela direção final de v
DISK = 2  # primeiro cruzamento de y = 0 com r_in ≤ r ≤ r_out
MAX_STEPS = 3  # limite de passos; o render trata como capturado


# --- equação de movimento e integrador (seções 3.2 e 3.5) -------------------


@njit(cache=True)
def acceleration(x, y, z, h2):
    """Aceleração a = −(3/2)·h²·x/r⁵ no ponto (x, y, z), com h2 = h²."""
    r2 = x * x + y * y + z * z
    r5 = r2 * r2 * math.sqrt(r2)
    s = -ACCEL_COEFF * h2 / r5
    return s * x, s * y, s * z


@njit(cache=True)
def rk4_step(x, y, z, vx, vy, vz, h2, dt):
    """Um passo RK4 clássico do sistema x' = v, v' = a(x). Devolve o novo estado."""
    a1x, a1y, a1z = acceleration(x, y, z, h2)

    hdt = 0.5 * dt
    a2x, a2y, a2z = acceleration(x + hdt * vx, y + hdt * vy, z + hdt * vz, h2)
    v2x, v2y, v2z = vx + hdt * a1x, vy + hdt * a1y, vz + hdt * a1z

    a3x, a3y, a3z = acceleration(x + hdt * v2x, y + hdt * v2y, z + hdt * v2z, h2)
    v3x, v3y, v3z = vx + hdt * a2x, vy + hdt * a2y, vz + hdt * a2z

    a4x, a4y, a4z = acceleration(x + dt * v3x, y + dt * v3y, z + dt * v3z, h2)
    v4x, v4y, v4z = vx + dt * a3x, vy + dt * a3y, vz + dt * a3z

    w = dt / 6.0
    nx = x + w * (vx + 2.0 * v2x + 2.0 * v3x + v4x)
    ny = y + w * (vy + 2.0 * v2y + 2.0 * v3y + v4y)
    nz = z + w * (vz + 2.0 * v2z + 2.0 * v3z + v4z)
    nvx = vx + w * (a1x + 2.0 * a2x + 2.0 * a3x + a4x)
    nvy = vy + w * (a1y + 2.0 * a2y + 2.0 * a3y + a4y)
    nvz = vz + w * (a1z + 2.0 * a2z + 2.0 * a3z + a4z)
    return nx, ny, nz, nvx, nvy, nvz


@njit(cache=True)
def step_size(r, step_factor, photon_step_factor, photon_r_min, photon_r_max, dt_min, dt_max):
    """Passo afim dt = k·r, reduzido na faixa da esfera de fótons e limitado a [dt_min, dt_max].

    A esfera de fótons é o ponto crítico de precisão: ali o raio é instável e os erros
    crescem exponencialmente com o ângulo percorrido.
    """
    k = step_factor
    if photon_r_min < r < photon_r_max:
        k *= photon_step_factor
    return min(max(k * r, dt_min), dt_max)


@njit(cache=True)
def _wrap_angle(d):
    """Leva um incremento angular para (−π, π]."""
    if d > math.pi:
        d -= 2.0 * math.pi
    elif d <= -math.pi:
        d += 2.0 * math.pi
    return d


@njit(cache=True)
def _record(trace, i, x, y, z, vx, vy, vz):
    if i < trace.shape[0]:
        trace[i, 0] = x
        trace[i, 1] = y
        trace[i, 2] = z
        trace[i, 3] = vx
        trace[i, 4] = vy
        trace[i, 5] = vz


@njit(cache=True)
def integrate_ray(
    x0,
    v0,
    r_max,
    max_steps,
    step_factor,
    photon_step_factor,
    photon_r_min,
    photon_r_max,
    dt_min,
    dt_max,
    disk_enabled,
    r_in,
    r_out,
    trace,
):
    """Integra um raio a partir de (x0, v0) até uma das condições de parada da seção 3.4.

    Checagens em cada passo, nesta ordem: cruzamento do disco (aconteceu dentro do passo;
    interpolação linear em y; vale o primeiro cruzamento com r_in ≤ r ≤ r_out), depois
    r < r_s, depois r > r_max com x·v > 0.

    ``trace`` é um array (N, 6) que recebe o estado (x, v) após cada passo, com o estado
    inicial na linha 0, até N linhas; o render passa ``np.empty((0, 6))``.

    Devolve ``(status, x, v, phi_swept, n_steps)``. ``phi_swept`` é o ângulo polar
    percorrido no plano orbital (normal x0 × v0), desenrolado passo a passo.
    """
    x, y, z = x0[0], x0[1], x0[2]
    vx, vy, vz = v0[0], v0[1], v0[2]

    # h é constante: calculado uma única vez.
    lx = y * vz - z * vy
    ly = z * vx - x * vz
    lz = x * vy - y * vx
    h2 = lx * lx + ly * ly + lz * lz
    h = math.sqrt(h2)

    # Base do plano orbital: e1 ao longo de x0, e2 = n × e1. Um raio radial (h = 0)
    # não define plano nem percorre ângulo.
    r = math.sqrt(x * x + y * y + z * z)
    track_phi = h > 0.0
    e1x = e1y = e1z = e2x = e2y = e2z = 0.0
    if track_phi:
        e1x, e1y, e1z = x / r, y / r, z / r
        nx_, ny_, nz_ = lx / h, ly / h, lz / h
        e2x = ny_ * e1z - nz_ * e1y
        e2y = nz_ * e1x - nx_ * e1z
        e2z = nx_ * e1y - ny_ * e1x
    phi_raw = 0.0
    phi_swept = 0.0

    _record(trace, 0, x, y, z, vx, vy, vz)

    for i in range(max_steps):
        dt = step_size(
            r, step_factor, photon_step_factor, photon_r_min, photon_r_max, dt_min, dt_max
        )
        nx, ny, nz, nvx, nvy, nvz = rk4_step(x, y, z, vx, vy, vz, h2, dt)

        if disk_enabled and ((y > 0.0 and ny <= 0.0) or (y < 0.0 and ny >= 0.0)):
            t = y / (y - ny)
            px = x + t * (nx - x)
            pz = z + t * (nz - z)
            pr = math.sqrt(px * px + pz * pz)
            if r_in <= pr <= r_out:
                pvx = vx + t * (nvx - vx)
                pvy = vy + t * (nvy - vy)
                pvz = vz + t * (nvz - vz)
                if track_phi:
                    p_raw = math.atan2(px * e2x + pz * e2z, px * e1x + pz * e1z)
                    phi_swept += _wrap_angle(p_raw - phi_raw)
                _record(trace, i + 1, px, 0.0, pz, pvx, pvy, pvz)
                return (
                    DISK,
                    np.array([px, 0.0, pz]),
                    np.array([pvx, pvy, pvz]),
                    phi_swept,
                    i + 1,
                )

        x, y, z, vx, vy, vz = nx, ny, nz, nvx, nvy, nvz
        r = math.sqrt(x * x + y * y + z * z)
        if track_phi:
            new_raw = math.atan2(x * e2x + y * e2y + z * e2z, x * e1x + y * e1y + z * e1z)
            phi_swept += _wrap_angle(new_raw - phi_raw)
            phi_raw = new_raw
        _record(trace, i + 1, x, y, z, vx, vy, vz)

        if r < R_S:
            return CAPTURED, np.array([x, y, z]), np.array([vx, vy, vz]), phi_swept, i + 1
        if r > r_max and x * vx + y * vy + z * vz > 0.0:
            return ESCAPED, np.array([x, y, z]), np.array([vx, vy, vz]), phi_swept, i + 1

    return MAX_STEPS, np.array([x, y, z]), np.array([vx, vy, vz]), phi_swept, max_steps


_NO_TRACE = np.empty((0, 6))


def trace_ray(x0, v0, r_max, integrator: IntegratorConfig, disk: DiskConfig, trace=None):
    """Wrapper de ``integrate_ray`` que recebe os parâmetros das dataclasses do ``config``."""
    return integrate_ray(
        np.asarray(x0, dtype=np.float64),
        np.asarray(v0, dtype=np.float64),
        float(r_max),
        int(integrator.max_steps),
        float(integrator.step_factor),
        float(integrator.photon_step_factor),
        float(integrator.photon_r_min),
        float(integrator.photon_r_max),
        float(integrator.dt_min),
        float(integrator.dt_max),
        bool(disk.enabled),
        float(disk.r_in),
        float(disk.r_out),
        _NO_TRACE if trace is None else trace,
    )


# --- invariantes do raio (seção 3.2) ----------------------------------------


@njit(cache=True)
def angular_momentum(x, v):
    """Momento angular h = |x × v|."""
    lx = x[1] * v[2] - x[2] * v[1]
    ly = x[2] * v[0] - x[0] * v[2]
    lz = x[0] * v[1] - x[1] * v[0]
    return math.sqrt(lx * lx + ly * ly + lz * lz)


@njit(cache=True)
def energy_eps(x, v):
    """"Energia" ε = |v|²/2 − h²/(2r³), conservada pela equação de movimento."""
    h = angular_momentum(x, v)
    r = math.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
    v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    return 0.5 * v2 - h * h / (2.0 * r**3)


@njit(cache=True)
def energy_at_infinity(x, v):
    """Energia no infinito E = √(|v|² − h²/r³) = √(2ε)."""
    return math.sqrt(2.0 * energy_eps(x, v))


@njit(cache=True)
def impact_parameter(x, v):
    """Parâmetro de impacto b = h / E."""
    return angular_momentum(x, v) / energy_at_infinity(x, v)


@njit(cache=True)
def axial_angular_momentum(x, v):
    """Componente axial do momento angular L_y = (x × v) · ŷ."""
    return x[2] * v[0] - x[0] * v[2]


@njit(cache=True)
def deflection_angle(x0, v0, x, v, phi_swept):
    """Deflexão α do raio entre o estado inicial e o final, sem erro de distância finita.

    A direção de v no plano orbital é φ + β, onde β é o ângulo de x para v
    (β = atan2(h, x·v), sempre em (0, π) porque h > 0 no sentido do movimento).
    Logo α = Δφ + β_f − β_i. Usar Δφ − π erraria em cerca de b/R.
    """
    beta_i = math.atan2(
        angular_momentum(x0, v0), x0[0] * v0[0] + x0[1] * v0[1] + x0[2] * v0[2]
    )
    beta_f = math.atan2(angular_momentum(x, v), x[0] * v[0] + x[1] * v[1] + x[2] * v[2])
    return phi_swept + beta_f - beta_i


# --- câmera de observador estático e escape (seções 3.3 e 3.4) --------------


@njit(cache=True)
def static_observer_velocity(x, n):
    """Velocidade inicial v₀ de um raio que parte de x na direção local n (tétrade 3.3).

    A câmera é um observador estático em r = |x|; n é a direção unitária do pixel no
    seu referencial local. Com r̂ = x/r, cos α = n·r̂ e n_⊥ = n − (n·r̂)·r̂:

        v₀ = √(1 − r_s/r) · cos α · r̂ + n_⊥

    Então h = r·sin α e E = √(1 − r_s/r), logo b = r·sin α / √(1 − r_s/r), que é a
    fórmula de Synge. Usar n direto como v erraria o tamanho da sombra. Exige r > r_s
    (a câmera valida isso).
    """
    r = math.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
    rx, ry, rz = x[0] / r, x[1] / r, x[2] / r
    cos_a = n[0] * rx + n[1] * ry + n[2] * rz
    s = math.sqrt(1.0 - R_S / r) * cos_a
    # s·r̂ + n_⊥ = n + (s − cos α)·r̂
    k = s - cos_a
    return np.array([n[0] + k * rx, n[1] + k * ry, n[2] + k * rz])


@njit(cache=True)
def escape_radius(r_cam, r_min, factor):
    """Raio de escape r_max = max(r_min, factor · r_cam) (seção 3.4).

    Com factor > 1, a câmera nunca fica fora da região integrada.
    """
    return max(r_min, factor * r_cam)


@njit(cache=True)
def shadow_angular_radius(r):
    """Raio angular α da sombra para um observador estático em r (fórmula de Synge).

    sin α = (b_c / r) · √(1 − r_s/r), com α medido a partir da direção do centro.
    Para r < 1,5 (dentro da esfera de fótons) a sombra cobre mais de meio céu e
    α = π − asin(...). Definida para r > r_s.

    O máximo exato de sin α é 1 (em r = 1,5); o ``min`` só absorve o arredondamento
    de float64 ali, que daria sin α = 1 + ε e asin = NaN.
    """
    alpha = math.asin(min(1.0, B_CRIT / r * math.sqrt(1.0 - R_S / r)))
    if r < R_PHOTON:
        alpha = math.pi - alpha
    return alpha


# --- disco de acreção (seção 3.6) -------------------------------------------

# r_pico/r_in: o máximo de f(r) = r^(−3/4)·(1 − √(r_in/r))^(1/4) sai de
# d ln f/dr = 0 ⇔ √(r_in/r) = 6/7, isto é, r = (49/36)·r_in.
PEAK_RADIUS_RATIO = 49.0 / 36.0


@njit(cache=True)
def peak_radius(r_in):
    """Raio do pico de temperatura do disco, r_pico = (49/36)·r_in (≈ 4,08 com r_in = 3)."""
    return PEAK_RADIUS_RATIO * r_in


@njit(cache=True)
def _disk_profile(r, r_in):
    """f(r) = r^(−3/4) · (1 − √(r_in/r))^(1/4), com f = 0 para r ≤ r_in."""
    if r <= r_in:
        return 0.0
    return r**-0.75 * (1.0 - math.sqrt(r_in / r)) ** 0.25


@njit(cache=True)
def disk_temperature(r, r_in, t_peak):
    """Temperatura do disco no raio r: T(r) = T_pico · f(r) / f(r_pico).

    Perfil simplificado da seção 3.6, com f(r) = r^(−3/4) · (1 − √(r_in/r))^(1/4). Vale
    T = 0 em r_in (e dentro dele) e T = T_pico em r_pico = (49/36)·r_in, que é o máximo.
    """
    return t_peak * _disk_profile(r, r_in) / _disk_profile(peak_radius(r_in), r_in)
