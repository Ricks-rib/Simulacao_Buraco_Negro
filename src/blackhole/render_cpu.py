"""Renderizador CPU de referência (PLANO, Fase 2): float64, Numba paralelo.

Cada pixel dispara n×n sub-amostras em grade regular. Cada raio parte da câmera com a
velocidade da tétrade estática e é integrado por ``physics.integrate_ray``:

- CAPTURED ou MAX_STEPS: preto (sombra; o limite de passos é tratado como captura);
- ESCAPED: xadrez celeste pela direção final de v.

As cores são médias das sub-amostras em [0, 1] e vão direto para uint8 (a cadeia
linear + tone mapping é da Fase 3). O disco fica desligado nesta fase.

Uso: ``uv run python -m blackhole.render_cpu [--width ... --supersample ...]``
"""

import argparse
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from numba import njit, prange
from PIL import Image

from blackhole.camera import Camera, pixel_direction, pixel_offset
from blackhole.config import (
    CameraConfig,
    CheckerConfig,
    DiskConfig,
    IntegratorConfig,
    RenderConfig,
)
from blackhole.physics import (
    ESCAPED,
    escape_radius,
    integrate_ray,
    static_observer_velocity,
)
from blackhole.scene import checker_color

# Fase 2: sem disco (o render passa disk_enabled=False).
_DISK_OFF = DiskConfig(enabled=False)


@njit(cache=True, parallel=True)
def _render_kernel(
    position,
    forward,
    right,
    up,
    width,
    height,
    pixel_scale,
    supersample,
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
    cell_rad,
    color_a,
    color_b,
    no_trace,
):
    """Imagem float64 (H, W, 3) em [0, 1], com um laço ``prange`` sobre as linhas."""
    image = np.zeros((height, width, 3))
    inv_n = 1.0 / supersample
    inv_samples = inv_n * inv_n
    for j in prange(height):
        for i in range(width):
            acc_r = 0.0
            acc_g = 0.0
            acc_b = 0.0
            for sy in range(supersample):
                for sx in range(supersample):
                    # Centro da sub-célula: s = (k + 0,5)/n − 0,5 (s = 0 com 1×1).
                    s_x = (sx + 0.5) * inv_n - 0.5
                    s_y = (sy + 0.5) * inv_n - 0.5
                    p, q = pixel_offset(i, j, s_x, s_y, width, height)
                    n = pixel_direction(p, q, pixel_scale, forward, right, up)
                    v0 = static_observer_velocity(position, n)
                    status, _, v, _, _ = integrate_ray(
                        position,
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
                        no_trace,
                    )
                    # Qualquer outro status (CAPTURED, MAX_STEPS) é sombra: soma zero.
                    if status == ESCAPED:
                        c = checker_color(v, cell_rad, color_a, color_b)
                        acc_r += c[0]
                        acc_g += c[1]
                        acc_b += c[2]
            image[j, i, 0] = acc_r * inv_samples
            image[j, i, 1] = acc_g * inv_samples
            image[j, i, 2] = acc_b * inv_samples
    return image


def to_uint8(image):
    """Converte cores em [0, 1] para uint8 (arredondamento, com corte fora da faixa)."""
    return np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def render(
    camera: Camera,
    render_cfg: RenderConfig,
    integrator: IntegratorConfig | None = None,
    checker: CheckerConfig | None = None,
):
    """Renderiza a cena vista por ``camera`` e devolve um array uint8 (H, W, 3)."""
    integrator = IntegratorConfig() if integrator is None else integrator
    checker = CheckerConfig() if checker is None else checker
    if render_cfg.width < 1 or render_cfg.height < 1:
        raise ValueError("a imagem precisa de largura e altura ≥ 1")
    if render_cfg.supersample < 1:
        raise ValueError("supersample precisa ser ≥ 1")

    r_max = escape_radius(camera.r_cam, render_cfg.r_max_min, render_cfg.r_max_factor)
    image = _render_kernel(
        camera.position,
        camera.forward,
        camera.right,
        camera.up,
        int(render_cfg.width),
        int(render_cfg.height),
        camera.pixel_scale(render_cfg.height),
        int(render_cfg.supersample),
        float(r_max),
        int(integrator.max_steps),
        float(integrator.step_factor),
        float(integrator.photon_step_factor),
        float(integrator.photon_r_min),
        float(integrator.photon_r_max),
        float(integrator.dt_min),
        float(integrator.dt_max),
        bool(_DISK_OFF.enabled),
        float(_DISK_OFF.r_in),
        float(_DISK_OFF.r_out),
        math.radians(checker.cell_deg),
        tuple(float(c) for c in checker.color_a),
        tuple(float(c) for c in checker.color_b),
        np.empty((0, 6)),
    )
    return to_uint8(image)


def save_png(image, path):
    """Grava o array uint8 (H, W, 3) em PNG, criando a pasta se preciso."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)
    return path


def main(argv=None):
    cam_cfg = CameraConfig()
    render_cfg = RenderConfig()
    parser = argparse.ArgumentParser(description="Renderizador CPU de referência (Fase 2).")
    parser.add_argument("--width", type=int, default=render_cfg.width)
    parser.add_argument("--height", type=int, default=render_cfg.height)
    parser.add_argument(
        "--supersample", type=int, default=render_cfg.supersample, help="n para n×n"
    )
    parser.add_argument("--r-cam", type=float, default=cam_cfg.r_cam)
    parser.add_argument("--fov", type=float, default=cam_cfg.fov_v_deg, help="FOV vertical (°)")
    parser.add_argument(
        "--inclination", type=float, default=cam_cfg.inclination_deg, help="a partir de +y (°)"
    )
    parser.add_argument("--azimuth", type=float, default=cam_cfg.azimuth_deg, help="(°)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(render_cfg.output_dir) / render_cfg.output_name,
        help="arquivo PNG de saída",
    )
    args = parser.parse_args(argv)

    cam_cfg = replace(
        cam_cfg,
        r_cam=args.r_cam,
        fov_v_deg=args.fov,
        inclination_deg=args.inclination,
        azimuth_deg=args.azimuth,
    )
    render_cfg = replace(
        render_cfg, width=args.width, height=args.height, supersample=args.supersample
    )
    camera = Camera.from_config(cam_cfg)

    start = time.perf_counter()
    image = render(camera, render_cfg)
    elapsed = time.perf_counter() - start

    path = save_png(image, args.output)
    ss = render_cfg.supersample
    print(
        f"{render_cfg.width}x{render_cfg.height} ({ss}x{ss}), r_cam = {camera.r_cam:g}: "
        f"render em {elapsed:.2f} s, salvo em {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
