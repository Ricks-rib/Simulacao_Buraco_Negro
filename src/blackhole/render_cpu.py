"""Renderizador CPU de referência (PLANO, Fases 2 e 3): float64, Numba paralelo.

O render tem duas etapas:

1. ``_trace_kernel`` (``prange``): cada pixel dispara n×n sub-amostras em grade regular.
   Cada raio parte da câmera com a velocidade da tétrade estática, é integrado por
   ``physics.integrate_ray`` (com o disco) e guarda status, x, v, φ percorrido e o
   atraso da luz Δt.
2. ``_shade_kernel`` (``prange``): cada sub-amostra vira radiância linear:

   - CAPTURED ou MAX_STEPS: preto (sombra; o limite de passos é tratado como captura);
   - DISK: m · (g·T/T_pico)⁴ · rgb(g·T) no ponto do impacto (seções 3.6, 3.8 e 3.9), com
     g = 1 sem Doppler e m = 1 sem textura;
   - ESCAPED: fundo pela direção final de v, em xadrez ou estrelas (com Doppler, as
     estrelas recebem o blueshift g_∞ do observador estático; o xadrez não muda).

   A média das sub-amostras é feita em espaço linear e passa pela cadeia exposição →
   ACES → sRGB, a mesma para o disco e para o fundo (seção 3.6).

Uso: ``uv run python -m blackhole.render_cpu [--width ... --supersample ... --background ...
--no-doppler --texture --time ...]``
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
    ColorConfig,
    DiskConfig,
    DiskTextureConfig,
    IntegratorConfig,
    RenderConfig,
    StarfieldConfig,
)
from blackhole.physics import (
    DISK,
    ESCAPED,
    axial_angular_momentum,
    energy_at_infinity,
    escape_radius,
    integrate_ray,
    redshift_factor,
    static_blueshift,
    static_observer_velocity,
)
from blackhole.scene import (
    blackbody_table,
    checker_color,
    disk_radiance,
    disk_texture,
    encode_pixel,
    star_radiance,
)

# Modos de fundo aceitos em ``RenderConfig.background`` e o código passado aos kernels.
BACKGROUNDS = {"checker": 0, "stars": 1}
_BG_CHECKER = BACKGROUNDS["checker"]


@njit(cache=True, parallel=True)
def _trace_kernel(
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
    no_trace,
):
    """Integra as n×n sub-amostras de cada pixel, com um laço ``prange`` sobre as linhas.

    Devolve ``status`` (H, W, S), ``x`` e ``v`` (H, W, S, 3) finais, ``phi`` (H, W, S) e o
    atraso da luz ``dt`` (H, W, S), com S = n² e a sub-amostra k = sy·n + sx.
    """
    n_sub = supersample * supersample
    status = np.empty((height, width, n_sub), dtype=np.int64)
    xs = np.empty((height, width, n_sub, 3))
    vs = np.empty((height, width, n_sub, 3))
    phis = np.empty((height, width, n_sub))
    dts = np.empty((height, width, n_sub))
    inv_n = 1.0 / supersample
    for j in prange(height):
        for i in range(width):
            for sy in range(supersample):
                for sx in range(supersample):
                    # Centro da sub-célula: s = (k + 0,5)/n − 0,5 (s = 0 com 1×1).
                    s_x = (sx + 0.5) * inv_n - 0.5
                    s_y = (sy + 0.5) * inv_n - 0.5
                    p, q = pixel_offset(i, j, s_x, s_y, width, height)
                    n = pixel_direction(p, q, pixel_scale, forward, right, up)
                    v0 = static_observer_velocity(position, n)
                    st, x, v, phi, _, dt_coord = integrate_ray(
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
                    k = sy * supersample + sx
                    status[j, i, k] = st
                    phis[j, i, k] = phi
                    dts[j, i, k] = dt_coord
                    for c in range(3):
                        xs[j, i, k, c] = x[c]
                        vs[j, i, k, c] = v[c]
    return status, xs, vs, phis, dts


@njit(cache=True)
def shade_sample(
    status,
    x,
    v,
    dt_coord,
    r_in,
    t_peak,
    table,
    t_min,
    t_max,
    background,
    cell_rad,
    color_a,
    color_b,
    star_grid,
    star_density,
    star_jitter,
    star_seed,
    star_sigma,
    star_i_min,
    star_i_max,
    star_t_min,
    star_t_max,
    doppler,
    r_cam,
    t_obs,
    texture,
    tex_radial_cells,
    tex_azimuthal_cells,
    tex_octaves,
    tex_contrast,
    tex_seed,
):
    """Radiância linear (antes da exposição) de uma sub-amostra já integrada.

    DISK usa o ponto do impacto x = (px, 0, pz) e o v interpolado nele:

    - com ``doppler``, g = ``redshift_factor`` com L_y e E desse (x, v) (seção 3.8); sem
      ele, g = 1. L_y e E não mudam ao inverter v_y, então as duas faces emitem igual;
    - com ``texture``, m = ``disk_texture`` em φ = atan2(−z, x) e no tempo de emissão
      t_emit = t_obs − Δt (``dt_coord``, o atraso da luz do raio); sem ela, m = 1.

    ESCAPED amostra o fundo pela direção final de v; com ``doppler`` as estrelas recebem
    o blueshift g_∞ = ``static_blueshift(r_cam)``. Os demais status são sombra.
    """
    if status == DISK:
        r = math.sqrt(x[0] * x[0] + x[2] * x[2])
        g = 1.0
        if doppler:
            g = redshift_factor(
                r, r_cam, axial_angular_momentum(x, v), energy_at_infinity(x, v)
            )
        emissivity = 1.0
        if texture:
            emissivity = disk_texture(
                r,
                math.atan2(-x[2], x[0]),
                t_obs - dt_coord,
                tex_radial_cells,
                tex_azimuthal_cells,
                tex_octaves,
                tex_contrast,
                tex_seed,
            )
        return disk_radiance(r, r_in, t_peak, table, t_min, t_max, g, emissivity)
    if status == ESCAPED:
        if background == _BG_CHECKER:
            c = checker_color(v, cell_rad, color_a, color_b)
            return c[0], c[1], c[2]
        shift = static_blueshift(r_cam) if doppler else 1.0
        return star_radiance(
            v,
            star_grid,
            star_density,
            star_jitter,
            star_seed,
            star_sigma,
            star_i_min,
            star_i_max,
            star_t_min,
            star_t_max,
            table,
            t_min,
            t_max,
            shift,
        )
    return 0.0, 0.0, 0.0


@njit(cache=True, parallel=True)
def _shade_kernel(
    status,
    xs,
    vs,
    dts,
    exposure,
    r_in,
    t_peak,
    table,
    t_min,
    t_max,
    background,
    cell_rad,
    color_a,
    color_b,
    star_grid,
    star_density,
    star_jitter,
    star_seed,
    star_sigma,
    star_i_min,
    star_i_max,
    star_t_min,
    star_t_max,
    doppler,
    r_cam,
    t_obs,
    texture,
    tex_radial_cells,
    tex_azimuthal_cells,
    tex_octaves,
    tex_contrast,
    tex_seed,
):
    """Imagem (H, W, 3) em [0, 1]: média linear das sub-amostras → exposição → ACES → sRGB."""
    height, width, n_sub = status.shape
    image = np.zeros((height, width, 3))
    inv_samples = 1.0 / n_sub
    for j in prange(height):
        for i in range(width):
            acc_r = 0.0
            acc_g = 0.0
            acc_b = 0.0
            for k in range(n_sub):
                lr, lg, lb = shade_sample(
                    status[j, i, k],
                    xs[j, i, k],
                    vs[j, i, k],
                    dts[j, i, k],
                    r_in,
                    t_peak,
                    table,
                    t_min,
                    t_max,
                    background,
                    cell_rad,
                    color_a,
                    color_b,
                    star_grid,
                    star_density,
                    star_jitter,
                    star_seed,
                    star_sigma,
                    star_i_min,
                    star_i_max,
                    star_t_min,
                    star_t_max,
                    doppler,
                    r_cam,
                    t_obs,
                    texture,
                    tex_radial_cells,
                    tex_azimuthal_cells,
                    tex_octaves,
                    tex_contrast,
                    tex_seed,
                )
                acc_r += lr
                acc_g += lg
                acc_b += lb
            er, eg, eb = encode_pixel(
                acc_r * inv_samples, acc_g * inv_samples, acc_b * inv_samples, exposure
            )
            image[j, i, 0] = er
            image[j, i, 1] = eg
            image[j, i, 2] = eb
    return image


def to_uint8(image):
    """Converte cores em [0, 1] para uint8 (arredondamento, com corte fora da faixa)."""
    return np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def _validate(render_cfg: RenderConfig):
    if render_cfg.width < 1 or render_cfg.height < 1:
        raise ValueError("a imagem precisa de largura e altura ≥ 1")
    if render_cfg.supersample < 1:
        raise ValueError("supersample precisa ser ≥ 1")


def validate_texture(texture_cfg: DiskTextureConfig):
    """Parâmetros da textura do disco: células > 0 (azimutais inteiras), oitavas ≥ 1 e
    contraste em [0, 1] (a emissividade m = 1 + c·(2·fbm − 1) nunca fica negativa)."""
    if not texture_cfg.radial_cells > 0.0:
        raise ValueError("radial_cells precisa ser > 0")
    azimuthal = texture_cfg.azimuthal_cells
    if isinstance(azimuthal, bool) or int(azimuthal) != azimuthal or azimuthal < 1:
        raise ValueError("azimuthal_cells precisa ser um inteiro ≥ 1")
    octaves = texture_cfg.octaves
    if isinstance(octaves, bool) or int(octaves) != octaves or octaves < 1:
        raise ValueError("octaves precisa ser um inteiro ≥ 1")
    if not 0.0 <= texture_cfg.contrast <= 1.0:
        raise ValueError("contrast precisa estar em [0, 1]")
    return texture_cfg


def _trace(camera, render_cfg, integrator, disk, supersample):
    r_max = escape_radius(camera.r_cam, render_cfg.r_max_min, render_cfg.r_max_factor)
    return _trace_kernel(
        camera.position,
        camera.forward,
        camera.right,
        camera.up,
        int(render_cfg.width),
        int(render_cfg.height),
        camera.pixel_scale(render_cfg.height),
        int(supersample),
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
        np.empty((0, 6)),
    )


def trace_fields(
    camera: Camera,
    render_cfg: RenderConfig,
    integrator: IntegratorConfig | None = None,
    disk: DiskConfig | None = None,
):
    """Integra um raio por pixel (1×1, ignora ``supersample``) e devolve os campos brutos.

    Devolve ``(status, x, v, phi, dt)`` com formas (H, W), (H, W, 3), (H, W, 3), (H, W) e
    (H, W): o status da seção 3.4, a posição e a velocidade finais, o ângulo φ percorrido
    no plano orbital e o atraso da luz Δt. Serve aos testes estruturais do disco.
    """
    integrator = IntegratorConfig() if integrator is None else integrator
    disk = DiskConfig() if disk is None else disk
    _validate(render_cfg)
    status, xs, vs, phis, dts = _trace(camera, render_cfg, integrator, disk, 1)
    return status[:, :, 0], xs[:, :, 0], vs[:, :, 0], phis[:, :, 0], dts[:, :, 0]


def render(
    camera: Camera,
    render_cfg: RenderConfig,
    integrator: IntegratorConfig | None = None,
    checker: CheckerConfig | None = None,
    disk: DiskConfig | None = None,
    color: ColorConfig | None = None,
    starfield: StarfieldConfig | None = None,
    background: str | None = None,
    texture_cfg: DiskTextureConfig | None = None,
    t_obs: float = 0.0,
):
    """Renderiza a cena vista por ``camera`` e devolve um array uint8 (H, W, 3).

    ``background`` ("checker" ou "stars") substitui ``render_cfg.background`` quando dado.
    Doppler e textura vêm de ``disk.doppler`` e ``disk.texture``; ``texture_cfg`` dá os
    parâmetros da textura e ``t_obs`` o tempo coordenado da câmera (o gás gira com Ω(r)).
    """
    integrator = IntegratorConfig() if integrator is None else integrator
    checker = CheckerConfig() if checker is None else checker
    disk = DiskConfig() if disk is None else disk
    color = ColorConfig() if color is None else color
    starfield = StarfieldConfig() if starfield is None else starfield
    texture_cfg = validate_texture(DiskTextureConfig() if texture_cfg is None else texture_cfg)
    background = render_cfg.background if background is None else background
    if background not in BACKGROUNDS:
        raise ValueError(
            f"fundo {background!r} inválido; use um de {', '.join(sorted(BACKGROUNDS))}"
        )
    _validate(render_cfg)

    status, xs, vs, _, dts = _trace(
        camera, render_cfg, integrator, disk, render_cfg.supersample
    )
    image = _shade_kernel(
        status,
        xs,
        vs,
        dts,
        float(color.exposure),
        float(disk.r_in),
        float(disk.t_peak),
        blackbody_table(color),
        float(color.bb_t_min),
        float(color.bb_t_max),
        int(BACKGROUNDS[background]),
        math.radians(checker.cell_deg),
        tuple(float(c) for c in checker.color_a),
        tuple(float(c) for c in checker.color_b),
        float(starfield.grid_size),
        float(starfield.density),
        float(starfield.jitter),
        int(starfield.seed),
        float(starfield.sigma),
        float(starfield.intensity_min),
        float(starfield.intensity_max),
        float(starfield.t_min),
        float(starfield.t_max),
        bool(disk.doppler),
        float(camera.r_cam),
        float(t_obs),
        bool(disk.texture),
        float(texture_cfg.radial_cells),
        int(texture_cfg.azimuthal_cells),
        int(texture_cfg.octaves),
        float(texture_cfg.contrast),
        int(texture_cfg.seed),
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
    disk_cfg = DiskConfig()
    color_cfg = ColorConfig()
    parser = argparse.ArgumentParser(description="Renderizador CPU de referência.")
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
        "--background", choices=sorted(BACKGROUNDS), default=render_cfg.background
    )
    parser.add_argument("--no-disk", action="store_true", help="desliga o disco de acreção")
    parser.add_argument(
        "--no-doppler",
        action="store_true",
        help="desliga o fator de desvio g (Doppler + redshift) do disco e o blueshift das estrelas",
    )
    parser.add_argument(
        "--texture", action="store_true", help="liga a textura co-rotante do disco"
    )
    parser.add_argument(
        "--time",
        type=float,
        default=0.0,
        help="tempo coordenado da câmera t_obs (unidades r_s/c), para a textura",
    )
    parser.add_argument("--exposure", type=float, default=color_cfg.exposure)
    parser.add_argument(
        "--t-peak", type=float, default=disk_cfg.t_peak, help="temperatura de pico do disco (K)"
    )
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
        render_cfg,
        width=args.width,
        height=args.height,
        supersample=args.supersample,
        background=args.background,
    )
    disk_cfg = replace(
        disk_cfg,
        enabled=not args.no_disk,
        t_peak=args.t_peak,
        doppler=disk_cfg.doppler and not args.no_doppler,
        texture=disk_cfg.texture or args.texture,
    )
    color_cfg = replace(color_cfg, exposure=args.exposure)
    camera = Camera.from_config(cam_cfg)

    start = time.perf_counter()
    image = render(camera, render_cfg, disk=disk_cfg, color=color_cfg, t_obs=args.time)
    elapsed = time.perf_counter() - start

    path = save_png(image, args.output)
    ss = render_cfg.supersample
    disk_label = "com disco" if disk_cfg.enabled else "sem disco"
    doppler_label = "Doppler ligado" if disk_cfg.doppler else "Doppler desligado"
    texture_label = f"textura em t = {args.time:g}" if disk_cfg.texture else "sem textura"
    print(
        f"{render_cfg.width}x{render_cfg.height} ({ss}x{ss}), r_cam = {camera.r_cam:g}, "
        f"{disk_label}, {doppler_label}, {texture_label}, fundo {render_cfg.background}: "
        f"render em {elapsed:.2f} s, salvo em {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
