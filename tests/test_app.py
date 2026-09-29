"""Testes das Fases 4b e 5b: app em tempo real (lógica pura, pipeline com e sem bloom,
HUD, captura, reload e benchmark).

A lógica da órbita, dos limites e do teclado roda sem GL. Os testes do pipeline, do
``reload`` e do benchmark usam um contexto standalone e dão ``skip`` sem OpenGL 3.3.
"""

import math
import os
import shutil
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime

import moderngl
import numpy as np
import pytest

from blackhole import app as app_mod
from blackhole.app import (
    CHAR_ACTIONS,
    Controls,
    FramePipeline,
    OrbitState,
    ShaderWatcher,
    capture_path,
    char_action,
    hud_lines,
    internal_size,
    render_hud_image,
)
from blackhole.config import AppConfig, BloomConfig, CameraConfig, GLCheckConfig, GpuConfig
from blackhole.physics import M
from blackhole.render_gpu import GpuRenderer, shader_path

APP = AppConfig()
GPU = GpuConfig()
CAM = CameraConfig()
BLOOM = BloomConfig()


@pytest.fixture(scope="module")
def ctx():
    try:
        context = moderngl.create_standalone_context(require=GLCheckConfig().min_version_code)
    except Exception as exc:  # moderngl/glcontext não expõem um tipo de erro específico
        pytest.skip(f"sem contexto OpenGL 3.3: {exc}")
    yield context
    context.release()


@pytest.fixture(scope="module")
def pipeline(ctx):
    frame = FramePipeline(ctx, APP)
    yield frame
    frame.release()


@contextmanager
def offscreen(ctx, size):
    """FBO alvo RGBA8; na saída devolve ``ctx.fbo`` ao anterior antes de liberar o alvo."""
    before = ctx.fbo
    fbo = ctx.framebuffer(color_attachments=[ctx.renderbuffer(size, components=4)])
    try:
        yield fbo
    finally:
        ctx.fbo = before
        for attachment in fbo.color_attachments:
            attachment.release()
        fbo.release()


def read_rgb(fbo):
    """RGB uint8 (H, W, 3) com a linha 0 no topo, como ``GpuRenderer.render()``."""
    width, height = fbo.size
    data = fbo.read(components=3, alignment=1)
    return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)[::-1]


# --- configuração ------------------------------------------------------------


def test_app_config_defaults_match_plan():
    assert (APP.r_cam_min, APP.r_cam_max) == (2.0, 500.0)
    assert 0.0 < APP.inclination_min_deg < APP.inclination_max_deg < 180.0
    assert APP.steps_increment == 128
    assert APP.benchmark_size == (1920, 1080) and APP.benchmark_target_fps == 60.0
    assert APP.scale_min <= APP.scale == 1.0 <= APP.scale_max
    assert app_mod.BlackHoleApp.aspect_ratio is None
    assert app_mod.BlackHoleApp.gl_version == (3, 3)


# --- matriz I/O: limites -------------------------------------------------------


@pytest.mark.parametrize("r_start, clicks", [(APP.r_cam_min, +1), (APP.r_cam_max, -1)])
def test_zoom_wheel_stays_at_limit(r_start, clicks):
    orbit = OrbitState(r_start, 80.0, 0.0)
    for _ in range(5):
        orbit.zoom_wheel(clicks, APP)
    assert orbit.r_cam == r_start


@pytest.mark.parametrize("r_start, zoom_out", [(APP.r_cam_min, -1), (APP.r_cam_max, +1)])
def test_zoom_keys_stay_at_limit(r_start, zoom_out):
    controls = Controls(APP)
    controls.orbit.r_cam = r_start
    controls.keys(0, 0, zoom_out, 1.0)
    assert controls.orbit.r_cam == r_start


def test_zoom_is_multiplicative():
    orbit = OrbitState(30.0, 80.0, 0.0)
    orbit.zoom_wheel(+1, APP)
    assert orbit.r_cam == pytest.approx(30.0 / APP.zoom_wheel_factor)
    orbit.zoom_wheel(-2, APP)
    assert orbit.r_cam == pytest.approx(30.0 * APP.zoom_wheel_factor)
    orbit.zoom(math.log(2.0), APP)
    assert orbit.r_cam == pytest.approx(60.0 * APP.zoom_wheel_factor)


@pytest.mark.parametrize("dy", [+1.0e5, -1.0e5])
def test_orbit_past_pole_is_clamped_and_camera_valid(dy):
    controls = Controls(APP)
    controls.drag(0.0, dy)
    inc = controls.orbit.inclination_deg
    assert inc in (APP.inclination_min_deg, APP.inclination_max_deg)
    camera = controls.camera()  # não pode levantar "up paralelo à visada"
    assert camera.r_cam == pytest.approx(CAM.r_cam)


def test_drag_and_arrow_directions():
    controls = Controls(APP)
    controls.drag(10.0, 0.0)
    assert controls.orbit.azimuth_deg == pytest.approx(10.0 * APP.orbit_mouse_deg_per_px)
    controls.drag(0.0, 10.0)  # para baixo: a câmera sobe
    assert controls.orbit.inclination_deg < CAM.inclination_deg
    controls.reset_camera()
    controls.keys(0, 1, 0, 0.1)  # seta para cima: a câmera sobe
    assert controls.orbit.inclination_deg == pytest.approx(
        CAM.inclination_deg - APP.orbit_key_deg_per_s * 0.1
    )
    controls.keys(-1, 0, 0, 0.1)  # seta para a esquerda: azimute cresce
    assert controls.orbit.azimuth_deg == pytest.approx(APP.orbit_key_deg_per_s * 0.1)


def test_azimuth_wraps():
    orbit = OrbitState(30.0, 80.0, 350.0)
    orbit.orbit(20.0, 0.0, APP)
    assert orbit.azimuth_deg == pytest.approx(10.0)


@pytest.mark.parametrize(
    "start, action", [(GPU.max_steps_cap, "steps_up"), (GPU.min_steps, "steps_down")]
)
def test_steps_stay_at_limit(start, action):
    controls = Controls(APP)
    controls.view.max_steps = start
    assert controls.apply(action)
    assert controls.view.max_steps == start


def test_steps_increment_from_default():
    controls = Controls(APP)
    controls.apply("steps_up")
    assert controls.view.max_steps == GPU.integrator.max_steps + APP.steps_increment
    for _ in range(20):
        controls.apply("steps_down")
    assert controls.view.max_steps == GPU.min_steps


def test_exposure_factor_and_limits():
    controls = Controls(APP)
    start = controls.view.exposure
    controls.apply("exposure_up")
    assert controls.view.exposure == pytest.approx(start * APP.exposure_factor)
    for _ in range(200):
        controls.apply("exposure_up")
    assert controls.view.exposure == APP.exposure_max
    for _ in range(400):
        controls.apply("exposure_down")
    assert controls.view.exposure == APP.exposure_min


def test_toggles_and_reset():
    controls = Controls(APP)
    view = controls.view
    disk, background, hud = view.disk_enabled, view.background, view.hud_visible
    doppler, texture = view.doppler, view.texture
    bloom, bloom_mode = view.bloom, view.bloom_mode
    for action in (
        "toggle_disk",
        "toggle_background",
        "toggle_hud",
        "toggle_doppler",
        "toggle_texture",
        "toggle_bloom",
        "toggle_bloom_mode",
    ):
        assert controls.apply(action)
    assert (view.disk_enabled, view.hud_visible) == (not disk, not hud)
    assert (view.doppler, view.texture) == (not doppler, not texture)
    assert view.bloom == (not bloom) and view.bloom_mode != bloom_mode
    assert view.background != background
    controls.apply("toggle_background")
    assert view.background == background

    controls.drag(100.0, 50.0)
    controls.orbit.zoom(1.0, APP)
    controls.apply("steps_up")
    controls.apply("reset")
    orbit = controls.orbit
    assert (orbit.r_cam, orbit.inclination_deg, orbit.azimuth_deg) == (
        CAM.r_cam,
        CAM.inclination_deg,
        CAM.azimuth_deg,
    )
    # R só volta a câmera; a vista fica como estava.
    assert view.max_steps == GPU.integrator.max_steps + APP.steps_increment
    assert not controls.apply("fly")


def test_doppler_texture_defaults_and_clock():
    """Doppler começa ligado e a textura desligada (DiskConfig); o relógio avança
    time_scale M por segundo e o R não o zera."""
    controls = Controls(APP)
    view = controls.view
    assert (view.doppler, view.texture, view.time) == (True, False, 0.0)
    view.advance_time(0.5, APP)
    assert view.time == pytest.approx(APP.time_scale * M * 0.5)
    controls.apply("reset")
    assert view.time == pytest.approx(APP.time_scale * M * 0.5)


def test_bloom_defaults_and_mode_toggle():
    """O app abre com o bloom ligado em glare; K alterna o modo também com o bloom
    desligado, e o modo escolhido vale quando ligar."""
    controls = Controls(APP)
    view = controls.view
    assert (view.bloom, view.bloom_mode) == (BLOOM.enabled, BLOOM.mode) == (True, "glare")
    controls.apply("toggle_bloom")
    assert not view.bloom
    controls.apply("toggle_bloom_mode")
    assert view.bloom_mode == "classic" and not view.bloom
    controls.apply("toggle_bloom")
    assert (view.bloom, view.bloom_mode) == (True, "classic")
    controls.apply("toggle_bloom_mode")
    assert view.bloom_mode == "glare"
    # R não mexe no bloom.
    controls.apply("reset")
    assert (view.bloom, view.bloom_mode) == (True, "glare")


def test_bloom_initial_state_from_config():
    controls = Controls(APP, bloom=replace(BLOOM, enabled=False, mode="classic"))
    assert (controls.view.bloom, controls.view.bloom_mode) == (False, "classic")


# --- teclado -----------------------------------------------------------------


@pytest.mark.parametrize(
    "char, action",
    [
        ("+", "steps_up"),
        ("-", "steps_down"),
        ("−", "steps_down"),
        ("[", "exposure_down"),
        ("]", "exposure_up"),
        ("d", "toggle_disk"),
        ("D", "toggle_disk"),
        ("g", "toggle_background"),
        ("G", "toggle_background"),
        ("h", "toggle_hud"),
        ("H", "toggle_hud"),
        ("r", "reset"),
        ("R", "reset"),
        ("b", "toggle_doppler"),
        ("B", "toggle_doppler"),
        ("t", "toggle_texture"),
        ("T", "toggle_texture"),
        ("l", "toggle_bloom"),
        ("L", "toggle_bloom"),
        ("k", "toggle_bloom_mode"),
        ("K", "toggle_bloom_mode"),
    ],
)
def test_char_action_mapping(char, action):
    assert char_action(char) == action


@pytest.mark.parametrize("char", ["w", "s", "W", "x", "", "++", None])
def test_char_action_ignores_others(char):
    assert char_action(char) is None


def test_every_char_action_is_handled():
    for action in set(CHAR_ACTIONS.values()):
        assert Controls(APP).apply(action), action


# --- tamanhos, captura e vigia do shader ----------------------------------------


@pytest.mark.parametrize("buffer_size", [(0, 0), (0, 720), (1280, 0)])
def test_internal_size_minimized_is_none(buffer_size):
    assert internal_size(buffer_size, 1.0) is None


def test_internal_size_scales_and_is_at_least_one():
    assert internal_size((1920, 1080), 1.0) == (1920, 1080)
    assert internal_size((1920, 1080), 0.5) == (960, 540)
    assert internal_size((2560, 1440), 0.75) == (1920, 1080)
    assert internal_size((1, 1), APP.scale_min) == (1, 1)


def test_capture_path_has_prefix_date_and_time():
    path = capture_path("outputs", "captura", datetime(2026, 9, 28, 21, 5, 7, 123456))
    assert path.as_posix() == "outputs/captura_20260928_210507_123.png"


def test_shader_watcher_detects_change_once(tmp_path):
    frag = tmp_path / "a.frag"
    frag.write_text("x", encoding="utf-8")
    watcher = ShaderWatcher([frag])
    assert not watcher.changed()
    stat = frag.stat()
    os.utime(frag, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert watcher.changed()
    assert not watcher.changed()
    frag.unlink()
    assert watcher.changed()


# --- argumentos ------------------------------------------------------------------


@pytest.mark.parametrize(
    "value", [str(APP.scale_min / 2), str(APP.scale_max * 2), "nan", "abc", "0"]
)
def test_invalid_scale_exits_2(value):
    with pytest.raises(SystemExit) as exc:
        app_mod.main(["--scale", value, "--benchmark"])
    assert exc.value.code == 2


def test_parser_accepts_scale_dev_benchmark():
    args = app_mod.build_parser(APP).parse_args(["--scale", "0.5", "--dev", "--benchmark"])
    assert (args.scale, args.dev, args.benchmark) == (0.5, True, True)


# --- HUD (sem GL) ----------------------------------------------------------------


def test_hud_lines_show_required_fields():
    controls = Controls(APP)
    lines = hud_lines(controls, 61.5, (2560, 1440), 0.5, "AVISO: GPU errada")
    text = "\n".join(t for t, _ in lines)
    for needle in (
        "FPS 61.5",
        f"passos {GPU.integrator.max_steps}",
        "r_cam 30.00",
        "incl 80.0",
        "escala 0.50",
        "interno 1280x720",
        "buffer 2560x1440",
        "exposicao",
        "disco",
        "fundo",
        "doppler ligado",
        "textura desligada",
        "t 0.0",
        "bloom ligado",
        "modo glare",
    ):
        assert needle in text, needle
    assert lines[-1] == ("AVISO: GPU errada", True)
    controls.apply("toggle_bloom")
    controls.apply("toggle_bloom_mode")
    text = "\n".join(t for t, _ in hud_lines(controls, 60.0, (2560, 1440), 1.0))
    assert "bloom desligado" in text and "modo classico" in text
    # A fonte padrão do Pillow só tem ASCII e "°".
    assert all(ch.isascii() or ch == "°" for ch in text)


def test_hud_image_scales_with_dpi():
    lines = [("FPS 60.0", False), ("AVISO", True)]
    small = render_hud_image(lines, APP, 1.0)
    large = render_hud_image(lines, APP, 2.0)
    assert small.mode == "RGBA"
    assert large.size[0] > small.size[0] and large.size[1] > small.size[1]
    alpha = np.asarray(small)[..., 3]
    assert alpha.min() >= APP.hud_background[3]


def test_renderer_warning():
    assert app_mod.renderer_warning("NVIDIA GeForce RTX 5070/PCIe/SSE2") is None
    warning = app_mod.renderer_warning("AMD Radeon(TM) Graphics")
    assert warning is not None and "AMD Radeon" in warning and warning.isascii()


# --- pipeline (GL) ----------------------------------------------------------------

SMALL = (96, 54)


def test_pipeline_scale_one_matches_gpu_renderer(ctx):
    """Bloom desligado: o quadro é o da Fase 5a (o do GpuRenderer), bit a bit, e o
    pipeline nem cria a textura HDR."""
    controls = Controls(APP)
    controls.view.background = "checker"
    controls.view.bloom = False
    camera, view = controls.camera(), controls.view
    pipeline = FramePipeline(ctx, APP)
    try:
        with offscreen(ctx, SMALL) as target:
            assert pipeline.render(target, camera, view, SMALL, 1.0, show_hud=False)
            frame = read_rgb(target)
        assert pipeline.hdr_texture is None and pipeline.bloom.level_sizes == []
        reference = _gpu_frame(pipeline.renderer, camera, view)
    finally:
        pipeline.release()
    np.testing.assert_array_equal(frame, reference)


def _gpu_frame(renderer, camera, view):
    """Quadro do ``GpuRenderer.render()`` (sem bloom) com os parâmetros da vista."""
    renderer.set_uniforms(
        camera,
        *SMALL,
        max_steps=view.max_steps,
        background=view.background,
        disk_enabled=view.disk_enabled,
        exposure=view.exposure,
        doppler=view.doppler,
        texture=view.texture,
        time=view.time,
    )
    return renderer.render()


@pytest.mark.parametrize("mode", ["glare", "classic"])
def test_pipeline_bloom_changes_frame_and_capture_matches(ctx, pipeline, mode):
    """Com bloom, o quadro muda (halo) e a captura (render_image) é o mesmo quadro."""
    controls = Controls(APP)
    view = controls.view
    view.bloom, view.bloom_mode = True, mode
    camera = controls.camera()
    size = (192, 108)
    with offscreen(ctx, size) as target:
        assert pipeline.render(target, camera, view, size, 1.0, show_hud=False)
        frame = read_rgb(target)
    assert pipeline.hdr_texture.size == size
    assert pipeline.bloom.level_sizes[0] == (size[0] // 2, size[1] // 2)
    before = ctx.fbo
    captured = pipeline.render_image(camera, view, size)
    assert ctx.fbo is before
    np.testing.assert_array_equal(captured, frame)
    no_bloom = _gpu_frame(pipeline.renderer, camera, view)
    assert not np.array_equal(frame, no_bloom)
    # O bloom só acrescenta luz perto das fontes; a média muda pouco.
    assert abs(frame.astype(float).mean() - no_bloom.astype(float).mean()) < 10.0


def test_pipeline_bloom_at_internal_scale(ctx, pipeline):
    """Escala 0,5: o bloom roda no tamanho interno e o resultado é esticado no alvo."""
    controls = Controls(APP)
    controls.view.background = "checker"
    controls.view.disk_enabled = False
    inner = internal_size(SMALL, 0.5)
    with offscreen(ctx, SMALL) as target:
        pipeline.render(target, controls.camera(), controls.view, SMALL, 0.5, show_hud=False)
        frame = read_rgb(target)
    assert pipeline.hdr_texture.size == inner == pipeline.scene_texture.size
    assert pipeline.bloom.level_sizes[0] == (inner[0] // 2, inner[1] // 2)
    # A imagem cobre o alvo todo; a sombra no centro fica bem mais escura que o céu (o
    # glare do xadrez, que cobre o céu inteiro, a clareia um pouco neste quadro minúsculo).
    center = frame[SMALL[1] // 2, SMALL[0] // 2].astype(int).sum()
    for j, i in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        assert frame[j, i].astype(int).sum() > 3 * center


def test_pipeline_invalid_bloom_config_raises(ctx):
    with pytest.raises(ValueError):
        FramePipeline(ctx, APP, bloom=replace(BLOOM, levels=0))


def test_capture_goes_through_pipeline_with_bloom(ctx, pipeline, tmp_path):
    """F12: o PNG é o quadro do pipeline (com bloom, sem HUD) no tamanho do buffer."""
    from PIL import Image

    from blackhole.config import RenderConfig

    class FakeApp:
        pass

    fake = FakeApp()
    fake.pipeline = pipeline
    fake.controls = Controls(APP)
    fake.controls.view.hud_visible = True
    fake.render_cfg = replace(RenderConfig(), output_dir=str(tmp_path))
    fake.app_cfg = APP
    size = (160, 90)
    pipeline.hud.update(0.0, 1.0, lambda: [("FPS 60.0", False)], force=True)
    app_mod.BlackHoleApp._capture(fake, size)
    files = list(tmp_path.glob(f"{APP.capture_prefix}_*.png"))
    assert len(files) == 1
    saved = np.asarray(Image.open(files[0]).convert("RGB"))
    expected = pipeline.render_image(fake.controls.camera(), fake.controls.view, size)
    np.testing.assert_array_equal(saved, expected)
    view = fake.controls.view
    view.bloom = False
    assert not np.array_equal(saved, pipeline.render_image(fake.controls.camera(), view, size))


def test_pipeline_scaled_fills_target(ctx, pipeline):
    controls = Controls(APP)
    controls.view.background = "checker"
    controls.view.disk_enabled = False  # o disco passa na frente do centro da sombra
    controls.view.bloom = False  # o glare do céu em xadrez clareia a sombra num quadro minúsculo
    with offscreen(ctx, SMALL) as target:
        pipeline.render(target, controls.camera(), controls.view, SMALL, 0.5, show_hud=False)
        frame = read_rgb(target)
    assert pipeline.scene_texture.size == internal_size(SMALL, 0.5)
    # Sombra no centro, fundo (não preto) nos quatro cantos: a imagem cobre o alvo todo.
    assert not frame[SMALL[1] // 2, SMALL[0] // 2].any()
    for j, i in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        assert frame[j, i].any()


def test_pipeline_minimized_skips_frame(ctx, pipeline):
    controls = Controls(APP)
    with offscreen(ctx, (4, 4)) as target:
        assert not pipeline.render(target, controls.camera(), controls.view, (0, 0), 1.0)


def test_hud_opaque_only_in_corner(ctx, pipeline):
    controls = Controls(APP)
    controls.view.background = "checker"
    camera, view = controls.camera(), controls.view
    size = (320, 180)
    pipeline.hud.update(0.0, 1.0, lambda: [("FPS 60.0", False), ("passos 512", False)], force=True)
    with offscreen(ctx, size) as target:
        pipeline.render(target, camera, view, size, 1.0, show_hud=False)
        plain = read_rgb(target)
        pipeline.render(target, camera, view, size, 1.0, show_hud=True)
        with_hud = read_rgb(target)

    x, y, w, h = pipeline.hud.viewport(size)
    # Retângulo do HUD em linhas a partir do topo (a leitura já foi invertida).
    top = size[1] - (y + h)
    changed = np.any(plain != with_hud, axis=2)
    inside = np.zeros_like(changed)
    inside[top : top + h, x : x + w] = True
    assert x == APP.hud_margin and top == APP.hud_margin
    assert not changed[~inside].any()
    assert changed[inside].mean() > 0.5
    # O fundo do HUD escurece a cena (alpha do fundo > 0) sem apagá-la por completo.
    region_plain = plain[inside].astype(int).sum()
    region_hud = with_hud[inside].astype(int).sum()
    assert region_hud < region_plain


def test_hud_update_throttled_and_only_on_change(ctx):
    hud = app_mod.HudOverlay(ctx, APP)
    try:
        calls = []

        def lines():
            calls.append(1)
            return [("FPS 1", False)]

        assert hud.update(0.0, 1.0, lines)
        assert not hud.update(APP.hud_refresh_s / 2, 1.0, lines)  # antes do intervalo
        assert len(calls) == 1
        assert not hud.update(APP.hud_refresh_s, 1.0, lines)  # mesmo texto: sem upload
        assert len(calls) == 2
        assert hud.update(2 * APP.hud_refresh_s, 1.0, lambda: [("FPS 2", False)])
    finally:
        hud.release()


# --- reload (GL) -------------------------------------------------------------------


@pytest.fixture
def shader_copy(tmp_path):
    for name in (GPU.vertex_shader, GPU.fragment_shader):
        shutil.copy(shader_path(name, GPU), tmp_path / name)
    # shader_dir absoluto: Path(módulo) / absoluto = o caminho absoluto.
    return replace(GPU, shader_dir=str(tmp_path)), tmp_path


def test_reload_error_keeps_old_program_then_recovers(ctx, shader_copy):
    gpu, folder = shader_copy
    renderer = GpuRenderer(ctx, gpu=gpu)
    try:
        camera = Controls(APP).camera()
        renderer.set_uniforms(camera, *SMALL, background="checker")
        reference = renderer.render()

        frag = folder / gpu.fragment_shader
        good = frag.read_text(encoding="utf-8")
        broken = good.replace("void main() {", "void main() { erro_de_sintaxe", 1)
        frag.write_text(broken, encoding="utf-8")
        program = renderer.program
        assert not renderer.reload()
        assert renderer.last_error
        assert renderer.program is program
        np.testing.assert_array_equal(renderer.render(), reference)

        frag.write_text(good, encoding="utf-8")
        assert renderer.reload()
        assert renderer.last_error is None
        assert renderer.program is not program
        renderer.set_uniforms(camera, *SMALL, background="checker")
        np.testing.assert_array_equal(renderer.render(), reference)
    finally:
        renderer.release()


def test_reload_missing_file_keeps_old_program(ctx, shader_copy):
    gpu, folder = shader_copy
    renderer = GpuRenderer(ctx, gpu=gpu)
    try:
        program = renderer.program
        (folder / gpu.fragment_shader).unlink()
        assert not renderer.reload()
        assert renderer.program is program and renderer.last_error
    finally:
        renderer.release()


def test_reload_rejects_program_without_frame_uniform(ctx, shader_copy):
    """u_exposure otimizado pelo compilador: reload() recusa e o programa antigo segue."""
    gpu, folder = shader_copy
    renderer = GpuRenderer(ctx, gpu=gpu)
    try:
        camera = Controls(APP).camera()
        renderer.set_uniforms(camera, *SMALL, background="checker")
        reference = renderer.render()

        frag = folder / gpu.fragment_shader
        good = frag.read_text(encoding="utf-8")
        uses = {
            "vec4(u_exposure * radiance, 1.0)": "vec4(radiance, 1.0)",
            "encode_pixel(radiance, u_exposure)": "encode_pixel(radiance, 1.0)",
        }
        hardcoded = good
        for use, fixed in uses.items():
            assert use in good
            hardcoded = hardcoded.replace(use, fixed)
        frag.write_text(hardcoded, encoding="utf-8")
        program = renderer.program
        assert not renderer.reload()
        assert "u_exposure" in renderer.last_error
        assert renderer.program is program
        renderer.set_uniforms(camera, *SMALL, background="checker")
        np.testing.assert_array_equal(renderer.render(), reference)
    finally:
        renderer.release()


# --- benchmark (GL) ------------------------------------------------------------------


class _SharedContext:
    """Contexto da fixture entregue ao ``run_benchmark``, que não pode liberá-lo."""

    def __init__(self, context, renderer_name=None):
        self._ctx = context
        self.info = dict(context.info)
        if renderer_name is not None:
            self.info["GL_RENDERER"] = renderer_name

    def release(self):
        pass

    def __getattr__(self, name):
        return getattr(self._ctx, name)

    def __setattr__(self, name, value):
        # Escritas (ex.: ctx.fbo = anterior) vão para o contexto real.
        if name in ("_ctx", "info"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._ctx, name, value)


SHORT = replace(APP, benchmark_size=(64, 36), benchmark_frames=3, benchmark_warmup=1)


def test_benchmark_short_passes(ctx, monkeypatch, capsys):
    shared = _SharedContext(ctx)
    monkeypatch.setattr(app_mod.moderngl, "create_standalone_context", lambda **kw: shared)
    before = ctx.fbo
    assert app_mod.run_benchmark(replace(SHORT, benchmark_target_fps=1.0), 1.0) == 0
    assert ctx.fbo is before
    out = capsys.readouterr().out
    assert "FPS" in out and "Renderer" in out and "OK" in out


def test_benchmark_forces_bloom_on(ctx, monkeypatch, capsys):
    """O benchmark mede com o bloom ligado mesmo se a BloomConfig começar desligada."""
    shared = _SharedContext(ctx)
    monkeypatch.setattr(app_mod.moderngl, "create_standalone_context", lambda **kw: shared)
    monkeypatch.setattr(app_mod, "BloomConfig", lambda: replace(BLOOM, enabled=False))
    rendered = []
    original = FramePipeline.render

    def spy(self, target, camera, view, *args, **kwargs):
        rendered.append(view.bloom)
        return original(self, target, camera, view, *args, **kwargs)

    monkeypatch.setattr(FramePipeline, "render", spy)
    assert app_mod.run_benchmark(replace(SHORT, benchmark_target_fps=1.0), 1.0) == 0
    assert rendered and all(rendered)
    assert "bloom (glare) ligados" in capsys.readouterr().out


def test_benchmark_slow_exits_1(ctx, monkeypatch, capsys):
    shared = _SharedContext(ctx, "AMD Radeon(TM) Graphics")
    monkeypatch.setattr(app_mod.moderngl, "create_standalone_context", lambda **kw: shared)
    before = ctx.fbo
    assert app_mod.run_benchmark(replace(SHORT, benchmark_target_fps=1.0e12), 1.0) == 1
    assert ctx.fbo is before
    out = capsys.readouterr().out
    assert "FALHOU" in out and "AVISO" in out


def test_benchmark_without_gl_context_exits_1(monkeypatch, capsys):
    def no_context(*args, **kwargs):
        raise RuntimeError("driver ausente")

    monkeypatch.setattr(app_mod.moderngl, "create_standalone_context", no_context)
    assert app_mod.run_benchmark(SHORT) == 1
    assert "ERRO" in capsys.readouterr().out
