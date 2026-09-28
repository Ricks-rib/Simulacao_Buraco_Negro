"""Câmera pinhole de observador estático (PLANO, seção 3.3).

A câmera monta uma base ortonormal (forward, right, up) e gera, para cada pixel, a
direção unitária n no referencial local do observador estático. A conversão de n na
velocidade inicial v₀ do raio é feita pela tétrade de ``physics.static_observer_velocity``:
a direção do pixel nunca é usada diretamente como v.

Coordenadas de pixel centradas: p cresce para a direita e q para cima, em pixels, a
partir do centro da imagem. Para o pixel (i, j) com deslocamento de sub-amostra s:

    p = (i + 0,5 + s_x) − W/2,    q = H/2 − (j + 0,5 + s_y)

A escala é a mesma nos dois eixos (pixels quadrados), tan(FOV_v/2)/(H/2), o que dá
tan(FOV_h/2) = tan(FOV_v/2)·W/H e a relação θ = atan(p · tan(FOV_h/2)/(W/2)) da seção 3.3.
"""

import math

import numpy as np
from numba import njit

from blackhole.config import CameraConfig
from blackhole.physics import R_S, static_observer_velocity


@njit(cache=True)
def pixel_offset(i, j, s_x, s_y, width, height):
    """Coordenadas centradas (p, q) do pixel (i, j) com deslocamento de sub-amostra (s_x, s_y)."""
    p = (i + 0.5 + s_x) - 0.5 * width
    q = 0.5 * height - (j + 0.5 + s_y)
    return p, q


@njit(cache=True)
def pixel_direction(p, q, pixel_scale, forward, right, up):
    """Direção local unitária n = normalize(f + p·k·right + q·k·up), com k = pixel_scale.

    ``pixel_scale`` = tan(FOV_v/2)/(H/2) = tan(FOV_h/2)/(W/2). Função compartilhada pela
    câmera e pelo kernel do renderizador.
    """
    a = p * pixel_scale
    b = q * pixel_scale
    nx = forward[0] + a * right[0] + b * up[0]
    ny = forward[1] + a * right[1] + b * up[1]
    nz = forward[2] + a * right[2] + b * up[2]
    inv = 1.0 / math.sqrt(nx * nx + ny * ny + nz * nz)
    return np.array([nx * inv, ny * inv, nz * inv])


def _as_vec3(value, name):
    vec = np.asarray(value, dtype=np.float64)
    if vec.shape != (3,) or not np.all(np.isfinite(vec)):
        raise ValueError(f"{name} deve ser um vetor 3D finito, recebido {value!r}")
    return vec


class Camera:
    """Observador estático com projeção pinhole.

    Atributos: ``position``, ``forward``, ``right``, ``up`` (base ortonormal destra, com
    right = forward × up_mundo normalizado e up = right × forward) e ``fov_v_deg``.
    """

    def __init__(self, position, target, up, fov_v_deg, degenerate_tolerance=None):
        tol = (
            CameraConfig().degenerate_tolerance
            if degenerate_tolerance is None
            else degenerate_tolerance
        )
        self.position = _as_vec3(position, "position")
        self.target = _as_vec3(target, "target")
        world_up = _as_vec3(up, "up")

        r = float(np.linalg.norm(self.position))
        if r <= R_S:
            raise ValueError(
                f"câmera em r = {r:g} está dentro do horizonte (r ≤ {R_S:g}); "
                "o observador estático só existe fora dele"
            )
        if not 0.0 < fov_v_deg < 180.0:
            raise ValueError(f"FOV vertical deve estar em (0°, 180°), recebido {fov_v_deg}")

        forward = self.target - self.position
        f_norm = float(np.linalg.norm(forward))
        if f_norm <= tol:
            raise ValueError("alvo coincide com a posição da câmera")
        forward = forward / f_norm

        up_norm = float(np.linalg.norm(world_up))
        if up_norm <= tol:
            raise ValueError("vetor up nulo")
        right = np.cross(forward, world_up / up_norm)
        r_norm = float(np.linalg.norm(right))
        if r_norm <= tol:
            raise ValueError("vetor up paralelo à direção de visada")
        right = right / r_norm

        self.forward = forward
        self.right = right
        self.up = np.cross(right, forward)
        self.fov_v_deg = float(fov_v_deg)

    @classmethod
    def from_config(cls, cfg: CameraConfig):
        """Câmera orbital: posição em (r_cam, inclinação a partir de +y, azimute) em torno do alvo."""
        inc = math.radians(cfg.inclination_deg)
        az = math.radians(cfg.azimuth_deg)
        offset = cfg.r_cam * np.array(
            [math.sin(inc) * math.cos(az), math.cos(inc), math.sin(inc) * math.sin(az)]
        )
        target = np.asarray(cfg.target, dtype=np.float64)
        return cls(target + offset, target, cfg.up, cfg.fov_v_deg, cfg.degenerate_tolerance)

    @property
    def r_cam(self):
        return float(np.linalg.norm(self.position))

    def pixel_scale(self, height):
        """tan(FOV_v/2)/(H/2): tangente do ângulo por pixel a partir do centro."""
        return math.tan(math.radians(self.fov_v_deg) / 2.0) / (0.5 * height)

    def direction(self, p, q, height):
        """Direção local unitária n do ponto (p, q) em pixels centrados."""
        return pixel_direction(
            float(p), float(q), self.pixel_scale(height), self.forward, self.right, self.up
        )

    def initial_velocity(self, p, q, height):
        """v₀ do raio do ponto (p, q), convertida pela tétrade estática (seção 3.3)."""
        return static_observer_velocity(self.position, self.direction(p, q, height))
