"""Gravação de vídeo com a câmera orbitando o buraco negro (PLANO, Fase 5).

Cria um contexto OpenGL standalone (sem janela) e percorre uma trajetória de câmera
determinística, um quadro por instante. Cada quadro passa pelo
``FramePipeline.render_image`` (o mesmo caminho da captura ``F12``: escala 1,0, sem HUD,
com bloom, Doppler e textura conforme a vista) e o ``imageio`` (extra opcional ``video``)
grava um MP4 em ``outputs/``.

Trajetória (``VideoConfig``): no instante s = i/N ∈ [0, 1), azimute = az₀ + 360°·s e
inclinação = incl₀ + A·sen(2π·s), com r_cam fixo. A volta fecha em posição, então o vídeo
roda em loop sem salto de câmera (a textura do disco gira de forma independente e não
repete). O relógio da textura (t_obs da câmera) avança 1/fps s real por quadro, na escala
``AppConfig.time_scale``: t_i = (i/fps)·time_scale·M.

Cada quadro é o de um **observador estático** na posição da trajetória (tétrade da seção
3.3), não o de um observador em movimento: não há aberração nem Doppler da câmera. Numa
órbita real em r = 30 a velocidade seria v ≈ √(M/r) ≈ 0,13c, com aberração visível; o
observador em movimento está fora do escopo (PLANO, seção 7).

Uso::

    uv sync --extra video
    uv run python -m blackhole.record
    uv run python -m blackhole.record --duration 2 --width 1280 --height 720
    uv run python -m blackhole.record --no-texture --no-doppler --no-bloom
    uv run python -m blackhole.record --output outputs/orbita.mp4

Código de saída: 0 com o vídeo gravado; 1 sem o extra ``video``, sem contexto OpenGL 3.3
ou com erro na gravação; 2 com argumentos inválidos (``argparse``).
"""

import argparse
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import moderngl

from blackhole.app import FramePipeline, OrbitState, ViewState, capture_path, renderer_warning
from blackhole.check_gl import format_version
from blackhole.config import (
    AppConfig,
    BloomConfig,
    CameraConfig,
    ColorConfig,
    DiskConfig,
    GLCheckConfig,
    GpuConfig,
    RenderConfig,
    VideoConfig,
)
from blackhole.physics import M

# Uma volta completa, em graus (azimute) e em radianos (fase da oscilação da inclinação).
_FULL_TURN_DEG = 360.0
_FULL_TURN_RAD = 2.0 * math.pi
# Menor lado aceito (o yuv420p subamostra a cor em blocos de 2×2).
_MIN_SIDE = 2

MISSING_EXTRA_MESSAGE = (
    "ERRO: a gravação de vídeo precisa do extra `video` (imageio e imageio-ffmpeg). "
    "Instale com `uv sync --extra video`."
)


class VideoExtraMissing(RuntimeError):
    """O extra opcional ``video`` (``imageio``/``imageio-ffmpeg``) não está instalado."""


class GLContextUnavailable(RuntimeError):
    """Não foi possível criar o contexto OpenGL standalone."""


def load_imageio():
    """Importa ``imageio.v2`` só no caminho da gravação; ``VideoExtraMissing`` se faltar."""
    try:
        import imageio.v2 as iio
        import imageio_ffmpeg  # noqa: F401  (o plugin FFMPEG do imageio depende dele)
    except ImportError as exc:
        raise VideoExtraMissing(f"{MISSING_EXTRA_MESSAGE} ({exc})") from exc
    return iio


# --- trajetória e tempo (sem GL) --------------------------------------------------


def frame_count(video: VideoConfig):
    """N = round(duração × fps) quadros."""
    return int(round(video.duration_s * video.fps))


def orbit_center(video: VideoConfig, cam: CameraConfig):
    """(r_cam, incl₀, az₀) da trajetória: ``VideoConfig``, ou a ``CameraConfig`` se ``None``."""
    r_cam = cam.r_cam if video.r_cam is None else video.r_cam
    incl0 = video.inclination_center_deg
    if incl0 is None:
        incl0 = cam.inclination_deg
    az0 = cam.azimuth_deg if video.azimuth_start_deg is None else video.azimuth_start_deg
    return float(r_cam), float(incl0), float(az0)


def validate_video(video: VideoConfig, cam: CameraConfig, app: AppConfig):
    """``ValueError`` com a causa se a ``VideoConfig`` não produz um vídeo válido."""
    for name in ("width", "height"):
        side = getattr(video, name)
        if not isinstance(side, int) or isinstance(side, bool) or side < _MIN_SIDE:
            raise ValueError(f"{name} precisa ser um inteiro ≥ {_MIN_SIDE} (recebido {side!r})")
        if side % 2:
            raise ValueError(f"{name} = {side} é ímpar; o {video.pixel_format} exige tamanho par")
        if video.macro_block_size > 1 and side % video.macro_block_size:
            raise ValueError(
                f"{name} = {side} não é múltiplo de macro_block_size = {video.macro_block_size}"
            )
    if video.macro_block_size < 1:
        raise ValueError("macro_block_size precisa ser ≥ 1")
    if not (math.isfinite(video.fps) and video.fps > 0.0):
        raise ValueError(f"fps precisa ser > 0 (recebido {video.fps!r})")
    if not (math.isfinite(video.duration_s) and video.duration_s > 0.0):
        raise ValueError(f"a duração precisa ser > 0 s (recebida {video.duration_s!r})")
    if frame_count(video) < 1:
        raise ValueError(
            f"duração {video.duration_s:g} s × {video.fps:g} fps não chega a um quadro"
        )
    if not (video.crf_min <= video.crf <= video.crf_max):
        raise ValueError(f"crf precisa estar em [{video.crf_min}, {video.crf_max}]")
    if video.progress_every_frames < 1:
        raise ValueError("progress_every_frames precisa ser ≥ 1")

    r_cam, incl0, az0 = orbit_center(video, cam)
    amplitude = video.inclination_amplitude_deg
    if not all(math.isfinite(v) for v in (r_cam, incl0, az0, amplitude)):
        raise ValueError("a trajetória precisa de r_cam, inclinação e azimute finitos")
    if not (app.r_cam_min <= r_cam <= app.r_cam_max):
        raise ValueError(
            f"r_cam = {r_cam:g} fora da faixa [{app.r_cam_min:g}, {app.r_cam_max:g}]"
        )
    low, high = incl0 - abs(amplitude), incl0 + abs(amplitude)
    if low < app.inclination_min_deg or high > app.inclination_max_deg:
        raise ValueError(
            f"inclinação {incl0:g}° ± {abs(amplitude):g}° sai da faixa "
            f"[{app.inclination_min_deg:g}°, {app.inclination_max_deg:g}°]"
        )


def orbit_at(i, video: VideoConfig, cam: CameraConfig, app: AppConfig):
    """Posição da câmera no quadro i (s = i/N): um ``OrbitState`` de observador estático.

    azimute = az₀ + 360°·s e inclinação = incl₀ + A·sen(2π·s). Com i = N a posição volta
    à do quadro 0 (a volta fecha).
    """
    s = i / frame_count(video)
    r_cam, incl0, az0 = orbit_center(video, cam)
    state = OrbitState(
        r_cam,
        incl0 + video.inclination_amplitude_deg * math.sin(_FULL_TURN_RAD * s),
        az0 + _FULL_TURN_DEG * s,
    )
    # Só leva o azimute a [0°, 360°): r_cam e a inclinação já foram validados.
    state.clamp(app)
    return state


def time_at(i, video: VideoConfig, app: AppConfig):
    """Relógio da textura no quadro i: t = (i/fps)·time_scale·M (como ``advance_time``)."""
    return (i / video.fps) * app.time_scale * M


def video_view(
    video: VideoConfig,
    app: AppConfig,
    gpu: GpuConfig,
    color: ColorConfig,
    disk: DiskConfig,
    render_cfg: RenderConfig,
    bloom: BloomConfig,
):
    """Vista inicial do app com a textura da ``VideoConfig`` e Doppler/bloom sobrepostos
    quando a ``VideoConfig`` os define (``None`` = padrão do app). Sem HUD."""
    view = ViewState.initial(app, gpu, color, disk, render_cfg, bloom)
    view.texture = bool(video.texture)
    if video.doppler is not None:
        view.doppler = bool(video.doppler)
    if video.bloom is not None:
        view.bloom = bool(video.bloom)
    view.hud_visible = False
    return view


def frame_view(i, base_view: ViewState, video: VideoConfig, app: AppConfig):
    """Cópia da vista com o relógio no instante do quadro i."""
    return replace(base_view, time=time_at(i, video, app))


def render_frame(pipeline: FramePipeline, i, base_view, video, cam, app):
    """Quadro i: ``render_image`` com a câmera e o tempo do instante i (= captura F12)."""
    camera = orbit_at(i, video, cam, app).camera(cam)
    view = frame_view(i, base_view, video, app)
    return pipeline.render_image(camera, view, (video.width, video.height))


# --- gravação (GL + imageio) --------------------------------------------------------


def _open_writer(iio, path: Path, video: VideoConfig):
    return iio.get_writer(
        str(path),
        format="FFMPEG",
        mode="I",
        fps=float(video.fps),
        codec=video.codec,
        pixelformat=video.pixel_format,
        # Sem o "quality" do imageio: a qualidade vai pelo CRF do H.264.
        quality=None,
        output_params=["-crf", str(video.crf)],
        macro_block_size=video.macro_block_size,
        ffmpeg_log_level=video.ffmpeg_log_level,
    )


def record_video(
    output,
    video: VideoConfig | None = None,
    *,
    cam: CameraConfig | None = None,
    app: AppConfig | None = None,
    gpu: GpuConfig | None = None,
    color: ColorConfig | None = None,
    disk: DiskConfig | None = None,
    render_cfg: RenderConfig | None = None,
    bloom: BloomConfig | None = None,
    ctx: moderngl.Context | None = None,
    log=print,
):
    """Grava o vídeo da órbita em ``output`` e devolve o ``Path``.

    Ordem: valida a config (``ValueError``) → importa o ``imageio``
    (``VideoExtraMissing``) → cria o contexto standalone (``GLContextUnavailable``) →
    renderiza e codifica os N quadros. Com ``ctx`` dado, usa esse contexto e não o
    libera. Se a gravação falhar ou for interrompida, o arquivo parcial é apagado.
    """
    video = VideoConfig() if video is None else video
    cam = CameraConfig() if cam is None else cam
    app = AppConfig() if app is None else app
    gpu = GpuConfig() if gpu is None else gpu
    color = ColorConfig() if color is None else color
    disk = DiskConfig() if disk is None else disk
    render_cfg = RenderConfig() if render_cfg is None else render_cfg
    bloom = BloomConfig() if bloom is None else bloom
    validate_video(video, cam, app)
    iio = load_imageio()

    owns_ctx = ctx is None
    if owns_ctx:
        gl_cfg = GLCheckConfig()
        try:
            ctx = moderngl.create_standalone_context(require=gl_cfg.min_version_code)
        except Exception as exc:  # moderngl/glcontext não expõem um tipo de erro específico
            required = format_version(gl_cfg.min_version_code)
            raise GLContextUnavailable(
                f"ERRO: não foi possível criar um contexto OpenGL {required}: {exc}"
            ) from exc

    path = Path(output)
    n = frame_count(video)
    try:
        name = ctx.info["GL_RENDERER"]
        log(f"Renderer: {name}")
        warning = renderer_warning(name)
        if warning is not None:
            log(f"{warning}; veja `uv run python -m blackhole.check_gl`.")
        base_view = video_view(video, app, gpu, color, disk, render_cfg, bloom)
        r_cam, incl0, _ = orbit_center(video, cam)
        log(
            f"{video.width}x{video.height}, {video.fps:g} fps, {video.duration_s:g} s "
            f"({n} quadros), {video.codec} {video.pixel_format} crf {video.crf}; órbita em "
            f"r_cam = {r_cam:g}, inclinação {incl0:g}° ± {video.inclination_amplitude_deg:g}°; "
            f"Doppler {'ligado' if base_view.doppler else 'desligado'}, textura "
            f"{'ligada' if base_view.texture else 'desligada'}, bloom "
            f"{base_view.bloom_mode if base_view.bloom else 'desligado'}"
        )
        pipeline = FramePipeline(
            ctx, app, gpu=gpu, render_cfg=render_cfg, disk=disk, color=color, bloom=bloom
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = _open_writer(iio, path, video)
            done = False
            start = time.perf_counter()
            try:
                for i in range(n):
                    writer.append_data(render_frame(pipeline, i, base_view, video, cam, app))
                    if (i + 1) % video.progress_every_frames == 0 or i + 1 == n:
                        elapsed = time.perf_counter() - start
                        eta = elapsed / (i + 1) * (n - i - 1)
                        log(
                            f"quadro {i + 1}/{n} ({100.0 * (i + 1) / n:.0f}%), "
                            f"{elapsed:.1f} s, faltam ~{eta:.1f} s"
                        )
                done = True
            finally:
                try:
                    writer.close()
                except Exception:
                    path.unlink(missing_ok=True)
                    if done:
                        raise
                if not done:
                    path.unlink(missing_ok=True)
            elapsed = time.perf_counter() - start
        finally:
            pipeline.release()
    finally:
        if owns_ctx:
            ctx.release()
    log(f"Vídeo salvo: {path} ({n} quadros em {elapsed:.1f} s)")
    return path


# --- linha de comando ---------------------------------------------------------------


def build_parser(video: VideoConfig | None = None):
    video = VideoConfig() if video is None else video
    parser = argparse.ArgumentParser(
        description="Grava um vídeo com a câmera (observador estático) orbitando o buraco negro."
    )
    parser.add_argument("--width", type=int, default=video.width, help="largura (par)")
    parser.add_argument("--height", type=int, default=video.height, help="altura (par)")
    parser.add_argument("--fps", type=float, default=video.fps, help="quadros por segundo")
    parser.add_argument(
        "--duration", type=float, default=video.duration_s, help="duração da volta (s)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="arquivo de saída (padrão: outputs/<prefixo>_<data>_<hora>.mp4)",
    )
    parser.add_argument(
        "--no-texture", action="store_true", help="desliga a textura co-rotante do disco"
    )
    parser.add_argument("--no-doppler", action="store_true", help="desliga Doppler e redshift")
    parser.add_argument("--no-bloom", action="store_true", help="desliga o bloom")
    return parser


def main(argv=None):
    """Grava o vídeo e devolve o código de saída (0, 1 ou 2)."""
    video_cfg = VideoConfig()
    cam, app, render_cfg = CameraConfig(), AppConfig(), RenderConfig()
    parser = build_parser(video_cfg)
    args = parser.parse_args(argv)
    video = replace(
        video_cfg,
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration_s=args.duration,
        texture=video_cfg.texture and not args.no_texture,
        doppler=False if args.no_doppler else video_cfg.doppler,
        bloom=False if args.no_bloom else video_cfg.bloom,
    )
    try:
        validate_video(video, cam, app)
    except ValueError as exc:
        parser.error(str(exc))

    output = args.output
    if output is None:
        output = capture_path(
            render_cfg.output_dir, video.file_prefix, suffix=video.file_suffix
        )
    try:
        record_video(output, video, cam=cam, app=app, render_cfg=render_cfg)
    except (VideoExtraMissing, GLContextUnavailable) as exc:
        print(exc)
        return 1
    except (OSError, RuntimeError, ValueError, moderngl.Error) as exc:
        print(f"ERRO na gravação: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
