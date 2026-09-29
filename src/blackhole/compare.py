"""Comparação CPU (float64) × GPU (float32) no fundo em xadrez (PLANO, Fase 4).

Renderiza duas cenas, sem e com disco, com a mesma câmera, resolução e integrador (o
preset de tempo real de ``GpuConfig``) nos dois renderizadores, num contexto ModernGL
standalone (sem janela), e mede os critérios de aceite:

- em cada cena, no máximo ``max_bad_fraction`` dos pixels com diferença maior que
  ``bad_threshold``/255 em algum canal;
- na cena sem disco, raio da sombra (horizontal e vertical, pela faixa preta que passa
  pelo pixel central) igual em ±``shadow_tolerance_px``.

O disco usa o Doppler de ``DiskConfig.doppler`` (ligado por padrão, Fase 5a) e a textura
desligada (``compare_scene`` aceita ``texture`` e ``t_obs`` para os testes).

Grava em ``outputs/`` três PNGs (CPU, GPU e diferença), cada um com as duas cenas
empilhadas (sem disco em cima). Na diferença, |Δ| máximo por pixel aparece em cinza
amplificado e os pixels ruins em vermelho.

Uso: ``uv run python -m blackhole.compare [--width --height --steps --r-cam --inclination]``

Código de saída: 0 dentro do aceite; 1 fora dele ou sem contexto OpenGL 3.3; 2 com
argumentos inválidos (``argparse``).
"""

import argparse
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import moderngl
import numpy as np

from blackhole.camera import Camera
from blackhole.check_gl import format_version
from blackhole.config import (
    CameraConfig,
    CompareConfig,
    DiskConfig,
    GLCheckConfig,
    GpuConfig,
    RenderConfig,
)
from blackhole.render_cpu import render, save_png
from blackhole.render_gpu import GpuRenderer, validate_background, validate_max_steps

# Cenas comparadas: (nome, disco ligado). A sombra é medida na cena sem disco.
SCENES = (("sem disco", False), ("com disco", True))


@dataclass
class SceneResult:
    """Resultado de uma cena: imagens, pixels ruins e tempos de render (s)."""

    name: str
    disk_enabled: bool
    cpu: np.ndarray
    gpu: np.ndarray
    bad_mask: np.ndarray
    bad_fraction: float
    max_diff: int
    cpu_time: float
    gpu_time: float


def channel_diff(cpu, gpu):
    """Maior |Δ| entre os canais de cada pixel, em 1/255 (H, W)."""
    return np.abs(cpu.astype(np.int16) - gpu.astype(np.int16)).max(axis=2)


def shadow_radius_px(image):
    """Raio da sombra (horizontal, vertical) em pixels, pela faixa preta no centro.

    A faixa é a sequência contígua de pixels pretos na linha H//2 (horizontal) e na
    coluna W//2 (vertical) que contém o pixel (W//2, H//2); o raio é metade do seu
    comprimento. Devolve ``None`` se o pixel central não for preto.
    """
    height, width, _ = image.shape
    black = np.all(image == 0, axis=2)
    ci, cj = width // 2, height // 2
    if not black[cj, ci]:
        return None

    def run_length(line, center):
        lo = hi = center
        while lo > 0 and line[lo - 1]:
            lo -= 1
        while hi < line.size - 1 and line[hi + 1]:
            hi += 1
        return hi - lo + 1

    return 0.5 * run_length(black[cj, :], ci), 0.5 * run_length(black[:, ci], cj)


def diff_image(cpu, gpu, cfg: CompareConfig):
    """PNG de diferença: |Δ| máximo × ganho em cinza, pixels ruins em ``bad_color``."""
    diff = channel_diff(cpu, gpu)
    gray = np.clip(diff * cfg.diff_gain, 0, 255).astype(np.uint8)
    image = np.repeat(gray[:, :, None], 3, axis=2)
    image[diff > cfg.bad_threshold] = cfg.bad_color
    return image


def compare_scene(
    renderer: GpuRenderer,
    camera: Camera,
    width,
    height,
    max_steps,
    disk_enabled,
    cfg: CompareConfig,
    name="",
    texture=False,
    t_obs=0.0,
):
    """Renderiza uma cena na CPU e na GPU com o mesmo integrador e compara as imagens.

    O Doppler vem de ``renderer.disk.doppler``; a textura fica desligada, a menos que
    ``texture`` seja dado (com o tempo ``t_obs`` da câmera).
    """
    gpu_cfg = renderer.gpu
    integrator = replace(gpu_cfg.integrator, max_steps=max_steps)
    render_cfg = replace(renderer.render_cfg, width=width, height=height, supersample=1)
    disk = replace(renderer.disk, enabled=disk_enabled, texture=bool(texture))

    start = time.perf_counter()
    cpu = render(
        camera,
        render_cfg,
        integrator=integrator,
        checker=renderer.checker,
        disk=disk,
        color=renderer.color,
        starfield=renderer.starfield,
        background=cfg.background,
        texture_cfg=renderer.texture_cfg,
        t_obs=t_obs,
    )
    cpu_time = time.perf_counter() - start

    renderer.set_uniforms(
        camera,
        width,
        height,
        max_steps=max_steps,
        background=cfg.background,
        disk_enabled=disk_enabled,
        doppler=disk.doppler,
        texture=disk.texture,
        time=t_obs,
    )
    # O primeiro draw pode incluir a compilação preguiçosa do driver: fica fora do tempo.
    renderer.render()
    start = time.perf_counter()
    gpu = renderer.render()
    gpu_time = time.perf_counter() - start

    diff = channel_diff(cpu, gpu)
    bad = diff > cfg.bad_threshold
    return SceneResult(
        name=name,
        disk_enabled=disk_enabled,
        cpu=cpu,
        gpu=gpu,
        bad_mask=bad,
        bad_fraction=float(bad.mean()),
        max_diff=int(diff.max()),
        cpu_time=cpu_time,
        gpu_time=gpu_time,
    )


def run_comparison(renderer: GpuRenderer, camera: Camera, width, height, max_steps, cfg):
    """Compara as cenas de ``SCENES`` e devolve ``(resultados, raios, aprovado)``.

    ``raios`` é ``{"cpu": (h, v), "gpu": (h, v)}`` (valores ``None`` se o centro não for
    preto), medidos na cena sem disco.
    """
    validate_max_steps(max_steps, renderer.gpu)
    validate_background(cfg.background)
    results = [
        compare_scene(renderer, camera, width, height, max_steps, disk, cfg, name)
        for name, disk in SCENES
    ]
    shadow_scene = next(res for res in results if not res.disk_enabled)
    radii = {"cpu": shadow_radius_px(shadow_scene.cpu), "gpu": shadow_radius_px(shadow_scene.gpu)}

    ok = all(res.bad_fraction <= cfg.max_bad_fraction for res in results)
    if radii["cpu"] is None or radii["gpu"] is None:
        ok = False
    else:
        for axis in range(2):
            if abs(radii["cpu"][axis] - radii["gpu"][axis]) > cfg.shadow_tolerance_px:
                ok = False
    return results, radii, ok


def save_outputs(results, cfg: CompareConfig, output_dir=None):
    """Grava os PNGs de CPU, GPU e diferença, com as cenas empilhadas na vertical."""
    out = Path(cfg.output_dir if output_dir is None else output_dir)
    cpu = np.concatenate([res.cpu for res in results], axis=0)
    gpu = np.concatenate([res.gpu for res in results], axis=0)
    diff = np.concatenate([diff_image(res.cpu, res.gpu, cfg) for res in results], axis=0)
    return [
        save_png(cpu, out / cfg.cpu_name),
        save_png(gpu, out / cfg.gpu_name),
        save_png(diff, out / cfg.diff_name),
    ]


def _format_radii(radii):
    return "sem sombra no centro" if radii is None else f"{radii[0]:.1f} px x {radii[1]:.1f} px"


def main(argv=None):
    cmp_cfg = CompareConfig()
    gpu_cfg = GpuConfig()
    cam_cfg = CameraConfig()
    gl_cfg = GLCheckConfig()
    parser = argparse.ArgumentParser(description="Comparação CPU x GPU no fundo em xadrez.")
    parser.add_argument("--width", type=int, default=cmp_cfg.width)
    parser.add_argument("--height", type=int, default=cmp_cfg.height)
    parser.add_argument(
        "--steps",
        type=int,
        default=gpu_cfg.integrator.max_steps,
        help=f"limite de passos, de {gpu_cfg.min_steps} a {gpu_cfg.max_steps_cap}",
    )
    parser.add_argument("--r-cam", type=float, default=cam_cfg.r_cam)
    parser.add_argument(
        "--inclination", type=float, default=cam_cfg.inclination_deg, help="a partir de +y (°)"
    )
    args = parser.parse_args(argv)

    try:
        validate_max_steps(args.steps, gpu_cfg)
        if args.width < 1 or args.height < 1:
            raise ValueError("a imagem precisa de largura e altura ≥ 1")
        camera = Camera.from_config(
            replace(cam_cfg, r_cam=args.r_cam, inclination_deg=args.inclination)
        )
    except ValueError as exc:
        parser.error(str(exc))

    required = format_version(gl_cfg.min_version_code)
    try:
        ctx = moderngl.create_standalone_context(require=gl_cfg.min_version_code)
    except Exception as exc:  # moderngl/glcontext não expõem um tipo de erro específico
        print(f"ERRO: não foi possível criar um contexto OpenGL {required}: {exc}")
        return 1

    try:
        renderer_name = ctx.info["GL_RENDERER"]
        print(f"Renderer: {renderer_name}")
        if not renderer_name.upper().startswith(gl_cfg.expected_vendor.upper()):
            print(
                f"AVISO: o renderer não é {gl_cfg.expected_vendor}; "
                "veja `uv run python -m blackhole.check_gl`."
            )
        renderer = GpuRenderer(
            ctx, gpu=gpu_cfg, render_cfg=RenderConfig(), disk=DiskConfig()
        )
        try:
            results, radii, ok = run_comparison(
                renderer, camera, args.width, args.height, args.steps, cmp_cfg
            )
        finally:
            renderer.release()
    finally:
        ctx.release()

    integ = gpu_cfg.integrator
    print(
        f"{args.width}x{args.height}, r_cam = {camera.r_cam:g}, inclinação = "
        f"{args.inclination:g}°, {args.steps} passos, k = {integ.step_factor:g} "
        f"(x{integ.photon_step_factor:g} na esfera de fótons), fundo {cmp_cfg.background}, "
        f"Doppler {'ligado' if DiskConfig().doppler else 'desligado'}, sem textura"
    )
    limit = 100.0 * cmp_cfg.max_bad_fraction
    for res in results:
        print(
            f"[{res.name}] pixels ruins (> {cmp_cfg.bad_threshold}/255): "
            f"{100.0 * res.bad_fraction:.4f}% (limite {limit:g}%), dif. máx. = {res.max_diff}/255; "
            f"CPU {res.cpu_time:.3f} s, GPU {1000.0 * res.gpu_time:.2f} ms"
        )
    print(
        f"[sombra] CPU {_format_radii(radii['cpu'])}, GPU {_format_radii(radii['gpu'])} "
        f"(tolerância {cmp_cfg.shadow_tolerance_px:g} px)"
    )
    for path in save_outputs(results, cmp_cfg):
        print(f"Salvo: {path}")
    print("OK: dentro do aceite." if ok else "FALHOU: fora do aceite.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
