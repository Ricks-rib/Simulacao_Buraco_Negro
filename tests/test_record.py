"""Testes da Fase 5c: gravação de vídeo com a câmera orbitando o buraco negro.

A trajetória, o relógio e a validação rodam sem GL. Os testes de quadro usam um contexto
standalone e dão ``skip`` sem OpenGL 3.3; a gravação real dá ``skip`` sem o extra
``video`` (``imageio``).
"""

import math
import sys
from dataclasses import replace

import moderngl
import numpy as np
import pytest

from blackhole import record as record_mod
from blackhole.app import FramePipeline, OrbitState, ViewState
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
from blackhole.record import (
    frame_count,
    orbit_at,
    render_frame,
    time_at,
    validate_video,
    video_view,
)

APP = AppConfig()
CAM = CameraConfig()
VIDEO = VideoConfig()
# Vídeo mínimo dos testes com GL: 64×36, 10 fps, 0,5 s = 5 quadros.
TINY = replace(VIDEO, width=64, height=36, fps=10.0, duration_s=0.5)


@pytest.fixture(scope="module")
def ctx():
    try:
        context = moderngl.create_standalone_context(require=GLCheckConfig().min_version_code)
    except Exception as exc:  # moderngl/glcontext não expõem um tipo de erro específico
        pytest.skip(f"sem contexto OpenGL 3.3: {exc}")
    yield context
    context.release()


def _base_view(video=VIDEO):
    return video_view(
        video, APP, GpuConfig(), ColorConfig(), DiskConfig(), RenderConfig(), BloomConfig()
    )


# --- configuração ----------------------------------------------------------------


def test_video_config_defaults_match_plan():
    assert (VIDEO.width, VIDEO.height, VIDEO.fps, VIDEO.duration_s) == (1920, 1080, 60.0, 20.0)
    assert frame_count(VIDEO) == 1200
    assert (VIDEO.codec, VIDEO.pixel_format) == ("libx264", "yuv420p")
    assert VIDEO.inclination_amplitude_deg == 15.0
    assert VIDEO.file_suffix == ".mp4"
    validate_video(VIDEO, CAM, APP)


def test_video_view_defaults():
    """Textura ligada no vídeo; Doppler e bloom seguem os padrões do app; sem HUD."""
    view = _base_view()
    assert view.texture and not view.hud_visible
    assert view.doppler == DiskConfig().doppler
    assert (view.bloom, view.bloom_mode) == (BloomConfig().enabled, BloomConfig().mode)
    off = _base_view(replace(VIDEO, texture=False, doppler=False, bloom=False))
    assert (off.texture, off.doppler, off.bloom) == (False, False, False)


# --- trajetória e relógio --------------------------------------------------------


def test_first_frame_is_initial_camera():
    orbit = orbit_at(0, VIDEO, CAM, APP)
    initial = OrbitState.initial(CAM, APP)
    assert (orbit.r_cam, orbit.inclination_deg, orbit.azimuth_deg) == (
        initial.r_cam,
        initial.inclination_deg,
        initial.azimuth_deg,
    )


def test_orbit_closes_for_loop():
    """O quadro N (fora do vídeo) coincide em posição com o quadro 0."""
    n = frame_count(VIDEO)
    first, after_last = orbit_at(0, VIDEO, CAM, APP), orbit_at(n, VIDEO, CAM, APP)
    assert after_last.r_cam == first.r_cam
    assert after_last.inclination_deg == pytest.approx(first.inclination_deg, abs=1e-9)
    assert after_last.azimuth_deg == pytest.approx(first.azimuth_deg, abs=1e-9)
    cam_first, cam_after = first.camera(CAM), after_last.camera(CAM)
    np.testing.assert_allclose(cam_after.position, cam_first.position, atol=1e-9)


def test_orbit_full_turn_and_inclination_oscillation():
    n = frame_count(VIDEO)
    states = [orbit_at(i, VIDEO, CAM, APP) for i in range(n)]
    incl = np.array([s.inclination_deg for s in states])
    amp = VIDEO.inclination_amplitude_deg
    # Uma oscilação: máximo em s = 1/4 (abaixo do disco), mínimo em s = 3/4 (acima).
    assert incl.max() == pytest.approx(CAM.inclination_deg + amp)
    assert incl.min() == pytest.approx(CAM.inclination_deg - amp)
    assert np.argmax(incl) == n // 4 and np.argmin(incl) == 3 * n // 4
    assert APP.inclination_min_deg <= incl.min() and incl.max() <= APP.inclination_max_deg
    # O azimute cresce 360°/N por quadro e cobre a volta inteira sem repetir.
    az = np.array([s.azimuth_deg for s in states])
    np.testing.assert_allclose(np.diff(az), 360.0 / n, rtol=1e-9)
    assert az[0] == 0.0 and az[-1] == pytest.approx(360.0 * (n - 1) / n)
    assert all(s.r_cam == CAM.r_cam for s in states)


def test_orbit_center_overrides_and_wrap():
    video = replace(VIDEO, r_cam=50.0, inclination_center_deg=90.0, azimuth_start_deg=300.0)
    n = frame_count(video)
    orbit = orbit_at(n // 2, video, CAM, APP)
    assert orbit.r_cam == 50.0
    assert orbit.inclination_deg == pytest.approx(90.0, abs=1e-9)
    assert orbit.azimuth_deg == pytest.approx(120.0)  # 300° + 180°, levado a [0°, 360°)


def test_time_per_frame_matches_advance_time():
    view = _base_view()
    for i in range(1, 11):
        view.advance_time(1.0 / VIDEO.fps, APP)
        assert time_at(i, VIDEO, APP) == pytest.approx(view.time, rel=1e-12)
    assert time_at(0, VIDEO, APP) == 0.0
    # Um segundo de vídeo = time_scale M no relógio.
    assert time_at(int(VIDEO.fps), VIDEO, APP) == pytest.approx(APP.time_scale * M)


# --- validação ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "changes",
    [
        {"width": 1921},
        {"height": 1079},
        {"width": 1},
        {"height": 0},
        {"fps": 0.0},
        {"fps": -30.0},
        {"fps": math.nan},
        {"duration_s": 0.0},
        {"duration_s": -1.0},
        {"duration_s": math.inf},
        {"duration_s": 1.0e-4},  # não chega a um quadro
        {"inclination_amplitude_deg": 100.0},  # 80° ± 100° sai de [1°, 179°]
        {"r_cam": 1.0},
        {"crf": 60},
    ],
)
def test_invalid_video_config_raises(changes):
    with pytest.raises(ValueError):
        validate_video(replace(VIDEO, **changes), CAM, APP)


def test_odd_size_refused_before_opening_file(tmp_path, monkeypatch):
    """Tamanho ímpar: ValueError antes de importar o imageio ou criar o contexto."""

    def fail(*args, **kwargs):
        raise AssertionError("não deveria chegar aqui")

    monkeypatch.setattr(record_mod, "load_imageio", fail)
    monkeypatch.setattr(record_mod.moderngl, "create_standalone_context", fail)
    out = tmp_path / "v.mp4"
    with pytest.raises(ValueError, match="ímpar"):
        record_mod.record_video(out, replace(TINY, width=65))
    assert not out.exists()


@pytest.mark.parametrize(
    "args",
    [
        ["--width", "1921"],
        ["--height", "7"],
        ["--fps", "0"],
        ["--duration", "-2"],
        ["--width", "abc"],
    ],
)
def test_cli_invalid_arguments_exit_2(args, tmp_path, monkeypatch):
    monkeypatch.setattr(record_mod, "load_imageio", lambda: pytest.fail("importou o imageio"))
    with pytest.raises(SystemExit) as exc:
        record_mod.main([*args, "--output", str(tmp_path / "v.mp4")])
    assert exc.value.code == 2
    assert not (tmp_path / "v.mp4").exists()


# --- falta do extra e do contexto ------------------------------------------------------


def test_missing_extra_exits_1(tmp_path, monkeypatch, capsys):
    """Sem o imageio: nada é gravado, mensagem com `uv sync --extra video`, saída 1."""
    monkeypatch.setitem(sys.modules, "imageio", None)
    monkeypatch.setitem(sys.modules, "imageio.v2", None)

    def no_context(*args, **kwargs):
        raise AssertionError("o contexto não deveria ser criado")

    monkeypatch.setattr(record_mod.moderngl, "create_standalone_context", no_context)
    out = tmp_path / "v.mp4"
    code = record_mod.main(["--width", "64", "--height", "36", "--output", str(out)])
    assert code == 1
    assert "uv sync --extra video" in capsys.readouterr().out
    assert not out.exists()


def test_missing_imageio_ffmpeg_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    with pytest.raises(record_mod.VideoExtraMissing, match="uv sync --extra video"):
        record_mod.load_imageio()


def test_without_gl_context_exits_1(tmp_path, monkeypatch, capsys):
    def no_context(*args, **kwargs):
        raise RuntimeError("driver ausente")

    monkeypatch.setattr(record_mod, "load_imageio", lambda: object())
    monkeypatch.setattr(record_mod.moderngl, "create_standalone_context", no_context)
    out = tmp_path / "v.mp4"
    code = record_mod.main(["--width", "64", "--height", "36", "--output", str(out)])
    assert code == 1
    out_text = capsys.readouterr().out
    assert "ERRO" in out_text and "OpenGL 3.3" in out_text
    assert not out.exists()


# --- quadro = captura (GL) ---------------------------------------------------------------


def test_frame_equals_render_image(ctx):
    """O quadro i é, bit a bit, o render_image (captura F12) com a câmera e o tempo do
    instante i."""
    pipeline = FramePipeline(ctx, APP)
    try:
        base = _base_view(TINY)
        i = 3
        frame = render_frame(pipeline, i, base, TINY, CAM, APP)
        n = frame_count(TINY)
        s = i / n
        camera = OrbitState(
            CAM.r_cam,
            CAM.inclination_deg + TINY.inclination_amplitude_deg * math.sin(2.0 * math.pi * s),
            CAM.azimuth_deg + 360.0 * s,
        ).camera(CAM)
        view = ViewState.initial(
            APP, GpuConfig(), ColorConfig(), DiskConfig(), RenderConfig(), BloomConfig()
        )
        view.texture = True
        view.time = i / TINY.fps * APP.time_scale * M
        expected = pipeline.render_image(camera, view, (TINY.width, TINY.height))
        np.testing.assert_array_equal(frame, expected)
        assert frame.shape == (TINY.height, TINY.width, 3)
        # Outro instante: a câmera e o relógio mudam o quadro.
        assert not np.array_equal(frame, render_frame(pipeline, 0, base, TINY, CAM, APP))
    finally:
        pipeline.release()


# --- gravação real (GL + imageio) -----------------------------------------------------------


def test_record_short_video_reads_back(ctx, tmp_path, capsys):
    iio = pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")
    out = tmp_path / "sub" / "orbita.mp4"
    before = ctx.fbo
    path = record_mod.record_video(out, TINY, ctx=ctx)
    assert path == out and out.exists() and out.stat().st_size > 0
    assert ctx.fbo is before
    printed = capsys.readouterr().out
    assert "Renderer" in printed and f"{frame_count(TINY)}/{frame_count(TINY)}" in printed

    reader = iio.get_reader(str(out))
    try:
        frames = [np.asarray(frame) for frame in reader]
    finally:
        reader.close()
    assert len(frames) == frame_count(TINY) == 5
    assert all(f.shape == (TINY.height, TINY.width, 3) for f in frames)

    # O primeiro quadro do vídeo é o render_image do instante 0, a menos da compressão.
    pipeline = FramePipeline(ctx, APP)
    try:
        expected = render_frame(pipeline, 0, _base_view(TINY), TINY, CAM, APP)
    finally:
        pipeline.release()
    diff = np.abs(frames[0].astype(int) - expected.astype(int)).mean()
    assert diff < 12.0


def test_record_failure_removes_partial_file(ctx, tmp_path, monkeypatch):
    pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")
    calls = []
    original = FramePipeline.render_image

    def flaky(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise moderngl.Error("falha simulada")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(FramePipeline, "render_image", flaky)
    out = tmp_path / "v.mp4"
    with pytest.raises(moderngl.Error):
        record_mod.record_video(out, TINY, ctx=ctx)
    assert not out.exists()
