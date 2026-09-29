"""Renderizador GPU (PLANO, Fase 4): fragment shader GLSL 330 em float32, via ModernGL.

``GpuRenderer`` encapsula o programa ``shaders/fullscreen.vert`` + ``shaders/blackhole.frag``
(em ``src/blackhole/shaders/``, carregados por caminho relativo a este módulo), a tabela de
corpo negro como textura N×1 RGB32F sem filtragem e o render num framebuffer RGBA8.
Não cria janela nem contexto: recebe um ``moderngl.Context`` pronto (standalone no
``compare``, da janela no app da Fase 4b).

Uso típico::

    renderer = GpuRenderer(ctx)
    renderer.set_uniforms(camera, width, height, background="checker")
    image = renderer.render()  # uint8 (H, W, 3), linha 0 no topo, como na CPU

O shader replica a física de ``physics.py``, a câmera de ``camera.py`` e a cena de
``scene.py`` (ver os comentários ``// ver <arquivo>.py: <nome>`` no ``.frag``).
"""

import math
from pathlib import Path

import moderngl
import numpy as np

from blackhole.camera import Camera
from blackhole.config import (
    CheckerConfig,
    ColorConfig,
    DiskConfig,
    DiskTextureConfig,
    GpuConfig,
    RenderConfig,
    StarfieldConfig,
)
from blackhole.render_cpu import BACKGROUNDS, validate_texture
from blackhole.scene import blackbody_table

# Componentes do framebuffer (RGBA8, não sRGB) e da leitura (RGB).
_FBO_COMPONENTS = 4
_READ_COMPONENTS = 3
# Vértices do triângulo de tela cheia gerado por gl_VertexID em fullscreen.vert.
_FULLSCREEN_VERTICES = 3
# Unidade de textura da tabela de corpo negro.
_BB_TEXTURE_UNIT = 0
# Uniforms enviados a cada quadro por set_uniforms; reload() exige todos no programa novo.
FRAME_UNIFORMS = (
    "u_resolution",
    "u_cam_pos",
    "u_cam_forward",
    "u_cam_right",
    "u_cam_up",
    "u_pixel_scale",
    "u_max_steps",
    "u_background",
    "u_disk_enabled",
    "u_exposure",
    "u_doppler",
    "u_disk_texture",
    "u_time",
)


def shader_path(name, gpu: GpuConfig | None = None):
    """Caminho de um shader em ``src/blackhole/<GpuConfig.shader_dir>/``."""
    gpu = GpuConfig() if gpu is None else gpu
    return Path(__file__).resolve().parent / gpu.shader_dir / name


def inject_defines(source, defines):
    """Insere ``#define NOME valor`` logo após a linha ``#version`` do shader."""
    first, sep, rest = source.partition("\n")
    if not first.startswith("#version"):
        raise ValueError("o shader precisa começar com a linha #version")
    lines = "".join(f"#define {name} {value}\n" for name, value in defines.items())
    return first + sep + lines + rest


def validate_max_steps(max_steps, gpu: GpuConfig):
    """``max_steps`` precisa estar em [min_steps, max_steps_cap] (matriz I/O)."""
    if isinstance(max_steps, bool) or int(max_steps) != max_steps:
        raise ValueError(f"max_steps precisa ser inteiro, recebido {max_steps!r}")
    if not gpu.min_steps <= max_steps <= gpu.max_steps_cap:
        raise ValueError(
            f"max_steps = {max_steps} fora da faixa [{gpu.min_steps}, {gpu.max_steps_cap}]"
        )
    return int(max_steps)


def validate_background(background):
    """``background`` precisa ser um dos modos de ``render_cpu.BACKGROUNDS``."""
    if background not in BACKGROUNDS:
        raise ValueError(
            f"fundo {background!r} inválido; use um de {', '.join(sorted(BACKGROUNDS))}"
        )
    return background


class GpuRenderer:
    """Programa ModernGL do buraco negro, com uniforms vindos das dataclasses do ``config``.

    Os parâmetros fixos (integrador, tabela de corpo negro, xadrez, estrelas, textura do
    disco, raio de escape) são enviados na construção; ``set_uniforms`` envia os que mudam
    por quadro (câmera, tamanho, passos, fundo, disco, exposição, Doppler, textura, tempo).
    ``reload`` recompila os shaders do disco sem perder o programa em uso se houver erro
    (modo dev do app).
    """

    def __init__(
        self,
        ctx: moderngl.Context,
        gpu: GpuConfig | None = None,
        render_cfg: RenderConfig | None = None,
        checker: CheckerConfig | None = None,
        disk: DiskConfig | None = None,
        color: ColorConfig | None = None,
        starfield: StarfieldConfig | None = None,
        texture_cfg: DiskTextureConfig | None = None,
    ):
        self.ctx = ctx
        self.gpu = GpuConfig() if gpu is None else gpu
        self.render_cfg = RenderConfig() if render_cfg is None else render_cfg
        self.checker = CheckerConfig() if checker is None else checker
        self.disk = DiskConfig() if disk is None else disk
        self.color = ColorConfig() if color is None else color
        self.starfield = StarfieldConfig() if starfield is None else starfield
        self.texture_cfg = validate_texture(
            DiskTextureConfig() if texture_cfg is None else texture_cfg
        )
        # O preset precisa caber na faixa aceita; falha já na construção.
        validate_max_steps(self.gpu.integrator.max_steps, self.gpu)

        self.program = self._compile()
        table = np.ascontiguousarray(blackbody_table(self.color), dtype="f4")
        self.bb_texture = ctx.texture((table.shape[0], 1), 3, data=table.tobytes(), dtype="f4")
        self.bb_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.bb_texture.repeat_x = False
        self.bb_texture.repeat_y = False
        self.vao = ctx.vertex_array(self.program, [])
        self._fbo = None
        self._size = None
        # Mensagem do último reload() que falhou (None depois de um reload bem-sucedido).
        self.last_error = None
        self._set_static_uniforms()

    # --- construção ---------------------------------------------------------

    def shader_paths(self):
        """Caminhos do vertex e do fragment shader (vigiados pelo modo dev do app)."""
        return (
            shader_path(self.gpu.vertex_shader, self.gpu),
            shader_path(self.gpu.fragment_shader, self.gpu),
        )

    def _compile(self):
        vert_path, frag_path = self.shader_paths()
        vert = vert_path.read_text(encoding="utf-8")
        frag = frag_path.read_text(encoding="utf-8")
        frag = inject_defines(frag, {"MAX_STEPS_CAP": int(self.gpu.max_steps_cap)})
        return self.ctx.program(vertex_shader=vert, fragment_shader=frag)

    def reload(self):
        """Recompila os shaders do disco (modo dev do app).

        Em sucesso troca programa e VAO, reenvia os uniforms fixos e devolve ``True``. Em
        erro (compilação, leitura do arquivo, uniform fixo ou de ``FRAME_UNIFORMS`` ausente) guarda a mensagem em
        ``last_error``, mantém o programa antigo e devolve ``False``. Os uniforms por
        quadro precisam de um novo ``set_uniforms`` depois de um reload bem-sucedido.
        """
        try:
            program = self._compile()
        except (moderngl.Error, OSError, UnicodeDecodeError, ValueError) as exc:
            self.last_error = str(exc)
            return False
        try:
            self._set_static_uniforms(program)
        except (KeyError, moderngl.Error) as exc:
            program.release()
            self.last_error = f"uniform fixo ausente ou inválido: {exc}"
            return False
        # Um uniform por quadro otimizado pelo compilador faria o próximo set_uniforms falhar.
        missing = [name for name in FRAME_UNIFORMS if name not in set(program)]
        if missing:
            program.release()
            self.last_error = f"uniform por quadro ausente: {', '.join(missing)}"
            return False
        old_program, old_vao = self.program, self.vao
        self.program = program
        self.vao = self.ctx.vertex_array(program, [])
        old_vao.release()
        old_program.release()
        self.last_error = None
        return True

    def _set(self, name, value, program=None):
        (self.program if program is None else program)[name].value = value

    def _set_static_uniforms(self, program=None):
        """Envia os uniforms fixos a ``program`` (padrão: o programa atual)."""

        def put(name, value):
            self._set(name, value, program)

        integ = self.gpu.integrator
        put("u_step_factor", float(integ.step_factor))
        put("u_photon_step_factor", float(integ.photon_step_factor))
        put("u_photon_r_min", float(integ.photon_r_min))
        put("u_photon_r_max", float(integ.photon_r_max))
        put("u_dt_min", float(integ.dt_min))
        put("u_dt_max", float(integ.dt_max))

        put("u_r_max_min", float(self.render_cfg.r_max_min))
        put("u_r_max_factor", float(self.render_cfg.r_max_factor))

        put("u_disk_r_in", float(self.disk.r_in))
        put("u_disk_r_out", float(self.disk.r_out))
        put("u_disk_t_peak", float(self.disk.t_peak))

        put("u_bb_table", _BB_TEXTURE_UNIT)
        put("u_bb_t_min", float(self.color.bb_t_min))
        put("u_bb_t_max", float(self.color.bb_t_max))

        put("u_cell_rad", math.radians(self.checker.cell_deg))
        put("u_checker_a", tuple(float(c) for c in self.checker.color_a))
        put("u_checker_b", tuple(float(c) for c in self.checker.color_b))

        stars = self.starfield
        put("u_star_grid", float(stars.grid_size))
        put("u_star_density", float(stars.density))
        put("u_star_jitter", float(stars.jitter))
        put("u_star_seed", int(stars.seed))
        put("u_star_sigma", float(stars.sigma))
        put("u_star_i_min", float(stars.intensity_min))
        put("u_star_i_max", float(stars.intensity_max))
        put("u_star_t_min", float(stars.t_min))
        put("u_star_t_max", float(stars.t_max))

        tex = self.texture_cfg
        put("u_tex_radial_cells", float(tex.radial_cells))
        put("u_tex_azimuthal_cells", int(tex.azimuthal_cells))
        put("u_tex_octaves", int(tex.octaves))
        put("u_tex_contrast", float(tex.contrast))
        put("u_tex_seed", int(tex.seed))

    # --- por quadro ---------------------------------------------------------

    def set_uniforms(
        self,
        camera: Camera,
        width: int,
        height: int,
        max_steps: int | None = None,
        background: str | None = None,
        disk_enabled: bool | None = None,
        exposure: float | None = None,
        doppler: bool | None = None,
        texture: bool | None = None,
        time: float = 0.0,
    ):
        """Envia câmera, tamanho, passos, fundo, disco, exposição, Doppler, textura e tempo.

        Omitidos, ``max_steps`` vem de ``GpuConfig.integrator``, ``background`` de
        ``RenderConfig``, ``disk_enabled``, ``doppler`` e ``texture`` de ``DiskConfig`` e
        ``exposure`` de ``ColorConfig``. ``time`` é o tempo coordenado t_obs da câmera
        (unidades r_s/c), usado pela textura. Levanta ``ValueError`` com passos fora de
        [256, 1024], fundo desconhecido, tempo não finito ou tamanho < 1.
        """
        max_steps = self.gpu.integrator.max_steps if max_steps is None else max_steps
        max_steps = validate_max_steps(max_steps, self.gpu)
        background = validate_background(
            self.render_cfg.background if background is None else background
        )
        disk_enabled = self.disk.enabled if disk_enabled is None else bool(disk_enabled)
        exposure = self.color.exposure if exposure is None else float(exposure)
        doppler = self.disk.doppler if doppler is None else bool(doppler)
        texture = self.disk.texture if texture is None else bool(texture)
        time = float(time)
        if not math.isfinite(time):
            raise ValueError(f"tempo precisa ser finito, recebido {time!r}")
        width, height = int(width), int(height)
        if width < 1 or height < 1:
            raise ValueError("a imagem precisa de largura e altura ≥ 1")

        values = {
            "u_resolution": (float(width), float(height)),
            "u_cam_pos": tuple(float(c) for c in camera.position),
            "u_cam_forward": tuple(float(c) for c in camera.forward),
            "u_cam_right": tuple(float(c) for c in camera.right),
            "u_cam_up": tuple(float(c) for c in camera.up),
            "u_pixel_scale": float(camera.pixel_scale(height)),
            "u_max_steps": max_steps,
            "u_background": int(BACKGROUNDS[background]),
            "u_disk_enabled": disk_enabled,
            "u_exposure": exposure,
            "u_doppler": doppler,
            "u_disk_texture": texture,
            "u_time": time,
        }
        for name in FRAME_UNIFORMS:
            self._set(name, values[name])
        self._size = (width, height)

    def draw(self):
        """Desenha o triângulo de tela cheia no framebuffer e viewport atuais."""
        self.bb_texture.use(_BB_TEXTURE_UNIT)
        self.vao.render(moderngl.TRIANGLES, vertices=_FULLSCREEN_VERTICES)

    def render(self):
        """Renderiza num FBO RGBA8 do tamanho do último ``set_uniforms`` e lê a imagem.

        Devolve uint8 (H, W, 3) com a linha 0 no topo (a leitura do OpenGL começa
        embaixo e é invertida aqui), no mesmo formato de ``render_cpu.render``.
        """
        if self._size is None:
            raise RuntimeError("chame set_uniforms antes de render")
        width, height = self._size
        if self._fbo is None or self._fbo.size != self._size:
            self._release_fbo()
            self._fbo = self.ctx.framebuffer(
                color_attachments=[
                    self.ctx.renderbuffer(self._size, components=_FBO_COMPONENTS, dtype="f1")
                ]
            )
        # Restaura o framebuffer que estava ligado (a janela no app da 4b) depois da leitura.
        # Num contexto standalone recém-criado ctx.fbo é None: detect_framebuffer() embrulha o
        # framebuffer ligado no GL (o padrão), para o FBO próprio nunca ficar como ctx.fbo.
        previous = self.ctx.fbo if self.ctx.fbo is not None else self.ctx.detect_framebuffer()
        self._fbo.use()
        try:
            self._fbo.viewport = (0, 0, width, height)
            self.draw()
            data = self._fbo.read(components=_READ_COMPONENTS, alignment=1, dtype="f1")
        finally:
            previous.use()
        image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, _READ_COMPONENTS)
        return np.ascontiguousarray(image[::-1])

    def _release_fbo(self):
        if self._fbo is not None:
            for attachment in self._fbo.color_attachments:
                attachment.release()
            self._fbo.release()
            self._fbo = None

    def release(self):
        """Libera FBO, VAO, textura e programa (o contexto é de quem o criou)."""
        self._release_fbo()
        self.vao.release()
        self.bb_texture.release()
        self.program.release()
