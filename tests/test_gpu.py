"""Testes da Fase 4a: shader GPU (float32) validado contra a CPU (float64).

Os testes que precisam de OpenGL 3.3 usam um contexto standalone e dão ``skip`` se ele não
puder ser criado (driver ausente ou versão < 3.3). Os testes de texto do ``.frag``
(referências cruzadas e constantes) e da matriz I/O que não tocam o GL rodam sempre.
"""

import math
import re
from dataclasses import replace

import moderngl
import numpy as np
import pytest

from blackhole import bloom as bloom_mod
from blackhole import camera as camera_mod
from blackhole import compare, physics, render_cpu, scene
from blackhole.camera import Camera
from blackhole.config import (
    BloomConfig,
    CameraConfig,
    CompareConfig,
    DiskConfig,
    GLCheckConfig,
    GpuConfig,
    IntegratorConfig,
    RenderConfig,
)
from blackhole.render_gpu import GpuRenderer, inject_defines, shader_path

GPU = GpuConfig()
CMP = CompareConfig()
# Resolução reduzida (mesma razão de aspecto do padrão) para os testes de comparação.
SMALL_W, SMALL_H = 320, 180


def camera_at(r_cam, **kwargs):
    return Camera.from_config(replace(CameraConfig(), r_cam=r_cam, **kwargs))


def frag_source(name=None):
    return shader_path(GPU.fragment_shader if name is None else name).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ctx():
    try:
        context = moderngl.create_standalone_context(require=GLCheckConfig().min_version_code)
    except Exception as exc:  # moderngl/glcontext não expõem um tipo de erro específico
        pytest.skip(f"sem contexto OpenGL 3.3: {exc}")
    yield context
    context.release()


@pytest.fixture(scope="module")
def renderer(ctx):
    gpu_renderer = GpuRenderer(ctx)
    yield gpu_renderer
    gpu_renderer.release()


# --- configuração (Decisão do integrador) -----------------------------------


def test_realtime_preset_and_cpu_defaults():
    integ = GPU.integrator
    assert (integ.step_factor, integ.photon_step_factor, integ.max_steps) == (0.05, 0.2, 512)
    # Na esfera de fótons o passo fica em 0,01·r.
    assert integ.step_factor * integ.photon_step_factor == pytest.approx(0.01)
    assert GPU.min_steps <= integ.max_steps <= GPU.max_steps_cap == 1024
    # Os padrões da referência da CPU não mudam.
    ref = IntegratorConfig()
    assert (ref.step_factor, ref.photon_step_factor, ref.max_steps) == (0.01, 0.1, 20_000)


# --- texto do shader: referências cruzadas e constantes ----------------------

# Funções portadas (Code Map do plano da 4a), por arquivo de origem.
PORTED = {
    physics: (
        "acceleration",
        "rk4_step",
        "step_size",
        "integrate_ray",
        "static_observer_velocity",
        "escape_radius",
        "disk_temperature",
        "peak_radius",
        "_time_rate",
        "energy_at_infinity",
        "axial_angular_momentum",
        "keplerian_omega",
        "static_blueshift",
        "redshift_factor",
    ),
    scene: (
        "checker_color",
        "blackbody_table",
        "blackbody_rgb",
        "disk_radiance",
        "aces",
        "srgb_encode",
        "encode_pixel",
        "hash_u32",
        "cell_hash",
        "hash_to_unit",
        "star_radiance",
        "periodic_value_noise",
        "disk_texture",
    ),
    camera_mod: ("pixel_offset", "pixel_direction"),
    render_cpu: ("shade_sample",),
}


def test_frag_cross_references_ported_functions():
    source = frag_source()
    for module, names in PORTED.items():
        filename = module.__name__.rsplit(".", 1)[1] + ".py"
        for name in names:
            assert hasattr(module, name), f"{filename} não tem {name}"
            assert f"ver {filename}: {name}" in source, f"falta // ver {filename}: {name}"


def _frag_constants(source):
    """``const <tipo> NOME = expr;`` do .frag, avaliados em ordem (expr pode citar anteriores).

    ``vec3(a, b, c)`` vira a tupla ``(a, b, c)``.
    """
    values = {}
    pattern = r"const\s+(?:float|int|uint|vec3)\s+(\w+)\s*=\s*([^;]+);"
    for name, expr in re.findall(pattern, source):
        expr = re.sub(r"\b(0[xX][0-9a-fA-F]+)[uU]\b", r"\1", expr.strip())
        expr = re.sub(r"^vec3\((.*)\)$", r"(\1)", expr)
        values[name] = eval(expr, {"__builtins__": {}}, dict(values))  # noqa: S307
    return values


def test_frag_constants_match_python():
    consts = _frag_constants(frag_source())
    expected = {
        "R_S": physics.R_S,
        "M": physics.M,
        "ACCEL_COEFF": physics.ACCEL_COEFF,
        "PEAK_RADIUS_RATIO": physics.PEAK_RADIUS_RATIO,
        "CIRCULAR_ORBIT_COEFF": physics.CIRCULAR_ORBIT_COEFF,
        "CAPTURED": physics.CAPTURED,
        "ESCAPED": physics.ESCAPED,
        "DISK": physics.DISK,
        "MAX_STEPS": physics.MAX_STEPS,
        "BG_CHECKER": render_cpu.BACKGROUNDS["checker"],
        "BG_STARS": render_cpu.BACKGROUNDS["stars"],
        "ACES_A": scene.ACES_A,
        "ACES_B": scene.ACES_B,
        "ACES_C": scene.ACES_C,
        "ACES_D": scene.ACES_D,
        "ACES_E": scene.ACES_E,
        "SRGB_THRESHOLD": scene.SRGB_THRESHOLD,
        "SRGB_LINEAR_SLOPE": scene.SRGB_LINEAR_SLOPE,
        "SRGB_SCALE": scene.SRGB_SCALE,
        "SRGB_OFFSET": scene.SRGB_OFFSET,
        "SRGB_GAMMA": scene.SRGB_GAMMA,
        "LOWBIAS32_M1": int(scene.LOWBIAS32_M1),
        "LOWBIAS32_M2": int(scene.LOWBIAS32_M2),
        "UNIT_SCALE": scene.UNIT_SCALE,
        "FBM_LACUNARITY": scene.FBM_LACUNARITY,
        "FBM_GAIN": scene.FBM_GAIN,
        "TWO_PI": scene.TWO_PI,
    }
    assert set(consts) == set(expected)
    for name, value in expected.items():
        assert consts[name] == pytest.approx(value, rel=1e-12), name


# --- shaders do bloom (Fase 5b): referências cruzadas e constantes ------------

BLOOM_CFG = BloomConfig()
ACES_SRGB = {
    name: getattr(scene, name)
    for name in (
        "ACES_A",
        "ACES_B",
        "ACES_C",
        "ACES_D",
        "ACES_E",
        "SRGB_THRESHOLD",
        "SRGB_LINEAR_SLOPE",
        "SRGB_SCALE",
        "SRGB_OFFSET",
        "SRGB_GAMMA",
    )
}
TENT = {
    name: getattr(bloom_mod, name)
    for name in ("TENT_CENTER_WEIGHT", "TENT_EDGE_WEIGHT", "TENT_CORNER_WEIGHT")
}
# Shader → (referências "ver <arquivo>.py: <nome>", constantes esperadas). Um comentário
# pode citar um grupo ("ver bloom.py: A, B, C"): a referência exigida é a primeira do grupo
# e as constantes são todas conferidas pelo valor.
BLOOM_SHADERS = {
    BLOOM_CFG.down_shader: (
        {
            scene: ("LUMINANCE_WEIGHTS",),
            bloom_mod: ("DOWN_CENTER_WEIGHT", "PREFILTER_NONE"),
        },
        {
            "LUMINANCE_WEIGHTS": scene.LUMINANCE_WEIGHTS,
            "DOWN_CENTER_WEIGHT": bloom_mod.DOWN_CENTER_WEIGHT,
            "DOWN_INNER_WEIGHT": bloom_mod.DOWN_INNER_WEIGHT,
            "DOWN_EDGE_WEIGHT": bloom_mod.DOWN_EDGE_WEIGHT,
            "DOWN_CORNER_WEIGHT": bloom_mod.DOWN_CORNER_WEIGHT,
            "PREFILTER_NONE": bloom_mod.PREFILTER_NONE,
            "PREFILTER_THRESHOLD": bloom_mod.PREFILTER_THRESHOLD,
        },
    ),
    BLOOM_CFG.up_shader: (
        {bloom_mod: ("TENT_CENTER_WEIGHT",)},
        TENT,
    ),
    BLOOM_CFG.composite_shader: (
        {
            scene: ("aces", "srgb_encode", "encode_pixel"),
            bloom_mod: ("MODES", "TENT_CENTER_WEIGHT"),
        },
        {
            "MODE_GLARE": bloom_mod.MODES["glare"],
            "MODE_CLASSIC": bloom_mod.MODES["classic"],
            **TENT,
            **ACES_SRGB,
        },
    ),
}


@pytest.mark.parametrize("shader", sorted(BLOOM_SHADERS))
def test_bloom_shader_cross_references(shader):
    source = frag_source(shader)
    for module, names in BLOOM_SHADERS[shader][0].items():
        filename = module.__name__.rsplit(".", 1)[1] + ".py"
        for name in names:
            assert hasattr(module, name), f"{filename} não tem {name}"
            assert f"ver {filename}: {name}" in source, f"{shader}: falta // ver {filename}: {name}"


@pytest.mark.parametrize("shader", sorted(BLOOM_SHADERS))
def test_bloom_shader_constants_match_python(shader):
    consts = _frag_constants(frag_source(shader))
    expected = BLOOM_SHADERS[shader][1]
    assert set(consts) == set(expected)
    for name, value in expected.items():
        assert consts[name] == pytest.approx(value, rel=1e-12), name


def test_hdr_output_is_exposed_linear_radiance(renderer):
    """u_hdr_output: o alvo RGBA16F recebe exposição × radiância, e encode_pixel dela é a
    saída RGBA8 de sempre (±1/255)."""
    ctx = renderer.ctx
    camera = camera_at(30.0)
    width, height = 64, 36
    exposure = 1.7
    texture = ctx.texture((width, height), 4, dtype="f4")
    fbo = ctx.framebuffer(color_attachments=[texture])
    before = ctx.fbo
    try:
        renderer.set_uniforms(camera, width, height, exposure=exposure, hdr=True)
        fbo.use()
        fbo.viewport = (0, 0, width, height)
        renderer.draw()
        data = fbo.read(components=3, alignment=1, dtype="f4")
    finally:
        ctx.fbo = before
        fbo.release()
        texture.release()
    hdr = np.frombuffer(data, dtype=np.float32).reshape(height, width, 3)[::-1]
    assert hdr.max() > 1.0  # radiância linear, sem o corte do ACES
    renderer.set_uniforms(camera, width, height, exposure=exposure)
    ldr = renderer.render().astype(int)
    encoded = np.array(
        [[scene.encode_pixel(*map(float, px), 1.0) for px in row] for row in hdr]
    )
    assert np.abs(np.round(encoded * 255.0) - ldr).max() <= 1


def test_inject_defines_after_version_line():
    out = inject_defines("#version 330 core\nvoid main() {}\n", {"MAX_STEPS_CAP": 1024})
    assert out.splitlines()[:2] == ["#version 330 core", "#define MAX_STEPS_CAP 1024"]
    with pytest.raises(ValueError):
        inject_defines("void main() {}\n", {"MAX_STEPS_CAP": 1024})


# --- matriz I/O --------------------------------------------------------------


@pytest.mark.parametrize("steps", [GPU.min_steps - 1, GPU.max_steps_cap + 1])
def test_steps_out_of_range_raise(renderer, steps):
    with pytest.raises(ValueError):
        renderer.set_uniforms(camera_at(30.0), 8, 8, max_steps=steps)


@pytest.mark.parametrize("steps", [GPU.min_steps, GPU.max_steps_cap])
def test_steps_at_range_limits_accepted(renderer, steps):
    renderer.set_uniforms(camera_at(30.0), 8, 8, max_steps=steps, background="checker")


def test_invalid_background_raises(renderer):
    with pytest.raises(ValueError):
        renderer.set_uniforms(camera_at(30.0), 8, 8, background="galaxy")


def test_preset_out_of_range_raises_on_construction(ctx):
    bad = replace(GPU, integrator=replace(GPU.integrator, max_steps=GPU.max_steps_cap + 1))
    with pytest.raises(ValueError):
        GpuRenderer(ctx, gpu=bad)


def test_compare_cli_rejects_steps_out_of_range():
    with pytest.raises(SystemExit) as exc:
        compare.main(["--steps", str(GPU.min_steps - 1)])
    assert exc.value.code == 2


def test_compare_without_gl_context_exits_1(monkeypatch, capsys):
    def no_context(*args, **kwargs):
        raise RuntimeError("driver ausente")

    monkeypatch.setattr(compare.moderngl, "create_standalone_context", no_context)
    assert compare.main([]) == 1
    assert "ERRO" in capsys.readouterr().out


class _SharedContext:
    """Contexto da fixture entregue ao ``compare.main``, que não pode liberá-lo.

    Criar e liberar um segundo contexto standalone deixaria o da fixture sem estar
    current. ``renderer_name`` troca o ``GL_RENDERER`` (simula outra GPU).
    """

    def __init__(self, context, renderer_name=None):
        self._ctx = context
        self.info = dict(context.info)
        if renderer_name is not None:
            self.info["GL_RENDERER"] = renderer_name

    def release(self):
        pass

    def __getattr__(self, name):
        return getattr(self._ctx, name)


# Resolução mínima para os testes do main() do compare (só o fluxo importa aqui).
TINY_ARGS = ["--width", "64", "--height", "36"]


def test_compare_non_nvidia_renderer_warns_and_runs(ctx, monkeypatch, tmp_path, capsys):
    shared = _SharedContext(ctx, "AMD Radeon(TM) Graphics")
    monkeypatch.setattr(compare.moderngl, "create_standalone_context", lambda **kw: shared)
    monkeypatch.chdir(tmp_path)
    assert compare.main(TINY_ARGS) == 0
    out = capsys.readouterr().out
    assert "AVISO" in out and "AMD Radeon(TM) Graphics" in out


def test_compare_acceptance_failure_prints_saves_and_exits_1(ctx, monkeypatch, tmp_path, capsys):
    shared = _SharedContext(ctx)
    monkeypatch.setattr(compare.moderngl, "create_standalone_context", lambda **kw: shared)
    # Fração máxima negativa: nenhuma cena passa, sem mexer nas tolerâncias do aceite real.
    failing = replace(CMP, max_bad_fraction=-1.0)
    monkeypatch.setattr(compare, "CompareConfig", lambda: failing)
    monkeypatch.chdir(tmp_path)
    assert compare.main(TINY_ARGS) == 1
    out = capsys.readouterr().out
    assert "pixels ruins" in out and "[sombra]" in out and "FALHOU" in out
    for name in (CMP.cpu_name, CMP.gpu_name, CMP.diff_name):
        assert (tmp_path / CMP.output_dir / name).exists()


# --- comparação CPU × GPU ----------------------------------------------------


def test_render_shape_and_dtype(renderer):
    renderer.set_uniforms(camera_at(30.0), 33, 17, background="checker")
    image = renderer.render()
    assert image.shape == (17, 33, 3)
    assert image.dtype == np.uint8


def test_render_restores_previous_framebuffer(renderer):
    """render() usa um FBO próprio e devolve o framebuffer anterior (a janela na 4b)."""
    ctx = renderer.ctx
    before = ctx.fbo
    other = ctx.simple_framebuffer((4, 4))
    try:
        other.use()
        renderer.set_uniforms(camera_at(30.0), 8, 8, background="checker")
        renderer.render()
        assert ctx.fbo == other
    finally:
        # Não deixa um FBO liberado como "anterior" para os próximos testes.
        ctx.fbo = before
        other.release()


@pytest.mark.parametrize("r_cam", [10.0, 30.0])
def test_gpu_matches_cpu_within_acceptance(renderer, r_cam):
    """Critérios da Fase 4 em resolução reduzida, nas cenas sem e com disco."""
    results, radii, ok = compare.run_comparison(
        renderer, camera_at(r_cam), SMALL_W, SMALL_H, GPU.integrator.max_steps, CMP
    )
    for res in results:
        assert res.bad_fraction <= CMP.max_bad_fraction, res.name
    assert radii["cpu"] is not None and radii["gpu"] is not None
    for axis in range(2):
        assert abs(radii["cpu"][axis] - radii["gpu"][axis]) <= CMP.shadow_tolerance_px
    assert ok

    # A sombra medida é a da fórmula de Synge (±1 px), não só igual nos dois lados.
    camera = camera_at(r_cam)
    expected_px = math.tan(physics.shadow_angular_radius(r_cam)) / camera.pixel_scale(SMALL_H)
    assert radii["gpu"][0] == pytest.approx(expected_px, abs=1.0)


def test_gpu_disk_is_visible_and_orientation_matches(renderer):
    """A cena com disco não é simétrica em cima/embaixo: pega uma leitura invertida."""
    results, _, _ = compare.run_comparison(
        renderer, camera_at(30.0), SMALL_W, SMALL_H, GPU.integrator.max_steps, CMP
    )
    plain, disk = results
    assert not plain.disk_enabled and disk.disk_enabled
    assert not np.array_equal(plain.gpu, disk.gpu)
    assert not np.array_equal(disk.gpu, disk.gpu[::-1])
    assert compare.channel_diff(disk.cpu, disk.gpu).mean() < 1.0


def test_gpu_step_limit_pixels_are_black(renderer):
    """Com 256 passos e k minúsculo, nenhum raio chega a r_max: tudo é MAX_STEPS (preto)."""
    tiny = replace(GPU, integrator=replace(GPU.integrator, step_factor=1.0e-4))
    local = GpuRenderer(renderer.ctx, gpu=tiny)
    try:
        local.set_uniforms(camera_at(30.0), 32, 18, max_steps=GPU.min_steps, background="checker")
        assert not local.render().any()
    finally:
        local.release()


# --- estrelas na GPU ---------------------------------------------------------


def test_gpu_stars_not_black_and_deterministic(renderer):
    camera = camera_at(30.0)
    renderer.set_uniforms(camera, 400, 225, background="stars", disk_enabled=False)
    first = renderer.render()
    second = renderer.render()
    np.testing.assert_array_equal(first, second)
    assert first.any()

    # Estrutural (não é o aceite): as estrelas acesas coincidem quase todas com as da CPU,
    # o que confere o hash em uint. O float32 pode mudar a célula de algumas direções.
    cpu = render_cpu.render(
        camera,
        replace(RenderConfig(), width=400, height=225),
        integrator=GPU.integrator,
        disk=DiskConfig(enabled=False),
        background="stars",
    )
    lit_cpu = cpu.any(axis=2)
    lit_gpu = first.any(axis=2)
    assert (lit_cpu & lit_gpu).sum() >= 0.9 * lit_cpu.sum()


# --- PNGs do compare ---------------------------------------------------------


def test_save_outputs_writes_three_pngs(renderer, tmp_path):
    results, _, _ = compare.run_comparison(
        renderer, camera_at(30.0), 64, 36, GPU.integrator.max_steps, CMP
    )
    paths = compare.save_outputs(results, CMP, output_dir=tmp_path)
    assert [p.name for p in paths] == [CMP.cpu_name, CMP.gpu_name, CMP.diff_name]
    assert all(p.exists() for p in paths)


def test_diff_image_marks_bad_pixels():
    cpu = np.zeros((2, 2, 3), dtype=np.uint8)
    gpu = cpu.copy()
    gpu[0, 0] = (CMP.bad_threshold + 1, 0, 0)
    gpu[1, 1] = (0, CMP.bad_threshold, 0)
    diff = compare.diff_image(cpu, gpu, CMP)
    np.testing.assert_array_equal(diff[0, 0], CMP.bad_color)
    assert diff[1, 1, 0] == diff[1, 1, 1] == diff[1, 1, 2] > 0
    np.testing.assert_array_equal(diff[0, 1], (0, 0, 0))


def test_shadow_radius_px_on_synthetic_image():
    image = np.full((9, 11, 3), 200, dtype=np.uint8)
    image[4, 3:8] = 0  # 5 px na horizontal
    image[2:7, 5] = 0  # 5 px na vertical
    assert compare.shadow_radius_px(image) == (2.5, 2.5)
    image[4, 5] = 1
    assert compare.shadow_radius_px(image) is None
