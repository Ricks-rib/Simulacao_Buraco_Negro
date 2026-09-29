"""App em tempo real (PLANO, Fase 4): janela, câmera orbital, controles e HUD.

Monta um ``WindowConfig`` do ``moderngl-window`` sobre o ``GpuRenderer`` da Fase 4a. A
cena é desenhada num FBO com textura no tamanho interno (buffer físico × escala) e
apresentada na janela pelo passe ``shaders/present.frag`` com filtro linear. Com o bloom
ligado (Fase 5b, PLANO seção 3.10), a cena sai em HDR linear (RGBA16F), passa pelo
``BloomPass`` (cadeia de mips, glare ou clássico, ACES + sRGB) e só então chega à textura
RGBA8 da cena. O HUD é texto do Pillow numa textura RGBA, desenhada no canto superior
esquerdo com blend alpha.

A lógica pura (órbita, limites, mapeamento de teclas, tamanhos, caminho da captura) fica
separada da janela e é testada sem GL; ``FramePipeline`` desenha em qualquer framebuffer
alvo (a janela no app, um FBO offscreen no ``--benchmark`` e nos testes).

Uso::

    uv run python -m blackhole.app                 # janela
    uv run python -m blackhole.app --scale 0.75    # escala de resolução interna
    uv run python -m blackhole.app --dev           # recarrega o shader quando o arquivo muda
    uv run python -m blackhole.app --benchmark     # FPS em 1920x1080 offscreen; sai com 0 ou 1

Controles: arrastar com o mouse ou setas (órbita), roda ou W/S (distância), D (disco),
B (Doppler e redshift), T (textura co-rotante do disco), L (bloom), K (bloom glare ↔
clássico), G (xadrez/estrelas), + / - (passos), [ / ] (exposição), H (HUD), R (câmera
inicial), F12 (captura em ``outputs/``, com bloom e sem HUD), Esc (sair).

O relógio da textura é o tempo coordenado t_obs da câmera, que avança
``AppConfig.time_scale`` M por segundo real (frame_time limitado a ``max_input_dt_s``).
"""

import argparse
import math
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import moderngl
import moderngl_window as mglw
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from blackhole.bloom import MODES as BLOOM_MODES
from blackhole.bloom import BloomPass, hdr_texture
from blackhole.camera import Camera
from blackhole.check_gl import format_version, is_expected_renderer
from blackhole.config import (
    AppConfig,
    BloomConfig,
    CameraConfig,
    ColorConfig,
    DiskConfig,
    GLCheckConfig,
    GpuConfig,
    RenderConfig,
)
from blackhole.physics import M
from blackhole.render_cpu import BACKGROUNDS, save_png
from blackhole.render_gpu import GpuRenderer, shader_path

# Vértices do triângulo de tela cheia (fullscreen.vert), componentes RGBA das texturas e
# RGB da leitura da captura.
_FULLSCREEN_VERTICES = 3
_RGBA = 4
_RGB = 3
# Unidade de textura do passe de apresentação (a 0 é a tabela de corpo negro do renderer).
_PRESENT_TEXTURE_UNIT = 1

# Caractere digitado → ação. O texto independe do layout (ABNT2) e inclui o teclado
# numérico; "−" é o sinal de menos Unicode. Letras em minúscula (char_action normaliza).
CHAR_ACTIONS = {
    "+": "steps_up",
    "-": "steps_down",
    "−": "steps_down",
    "[": "exposure_down",
    "]": "exposure_up",
    "b": "toggle_doppler",
    "d": "toggle_disk",
    "g": "toggle_background",
    "h": "toggle_hud",
    "k": "toggle_bloom_mode",
    "l": "toggle_bloom",
    "r": "reset",
    "t": "toggle_texture",
}

# Nome do fundo no HUD (a fonte padrão do Pillow só tem ASCII e "°": sem acentos).
BACKGROUND_LABELS = {"checker": "xadrez", "stars": "estrelas"}
# Nome do modo do bloom no HUD (ASCII).
BLOOM_MODE_LABELS = {"glare": "glare", "classic": "classico"}


def on_off(flag, feminine=False):
    """Estado de um modo no HUD (ASCII)."""
    if feminine:
        return "ligada" if flag else "desligada"
    return "ligado" if flag else "desligado"


# --- lógica pura (sem GL) -----------------------------------------------------


def clamp(value, low, high):
    """``value`` limitado a [low, high]."""
    return min(max(value, low), high)


def char_action(char):
    """Ação de ``CHAR_ACTIONS`` para o caractere digitado, ou ``None``."""
    if not isinstance(char, str) or len(char) != 1:
        return None
    return CHAR_ACTIONS.get(char.lower())


def internal_size(buffer_size, scale):
    """Tamanho interno da cena: buffer físico × escala, cada lado ≥ 1.

    Devolve ``None`` com a janela minimizada (buffer com algum lado < 1): o quadro é pulado.
    """
    width, height = int(buffer_size[0]), int(buffer_size[1])
    if width < 1 or height < 1:
        return None
    return max(1, round(width * scale)), max(1, round(height * scale))


def capture_path(output_dir, prefix, now=None):
    """``<output_dir>/<prefix>_AAAAMMDD_HHMMSS_mmm.png`` (data e hora da captura)."""
    now = datetime.now() if now is None else now
    stamp = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
    return Path(output_dir) / f"{prefix}_{stamp}.png"


@dataclass
class OrbitState:
    """Câmera orbital: distância, inclinação a partir de +y e azimute (graus)."""

    r_cam: float
    inclination_deg: float
    azimuth_deg: float

    @classmethod
    def initial(cls, cam: CameraConfig, app: AppConfig):
        state = cls(float(cam.r_cam), float(cam.inclination_deg), float(cam.azimuth_deg))
        state.clamp(app)
        return state

    def clamp(self, app: AppConfig):
        """Aplica 2 ≤ r_cam ≤ 500, a inclinação longe dos polos e o azimute em [0°, 360°)."""
        self.r_cam = clamp(self.r_cam, app.r_cam_min, app.r_cam_max)
        self.inclination_deg = clamp(
            self.inclination_deg, app.inclination_min_deg, app.inclination_max_deg
        )
        self.azimuth_deg %= 360.0

    def orbit(self, d_azimuth_deg, d_inclination_deg, app: AppConfig):
        self.azimuth_deg += d_azimuth_deg
        self.inclination_deg += d_inclination_deg
        self.clamp(app)

    def zoom(self, log_factor, app: AppConfig):
        """Zoom multiplicativo r ← r·exp(log_factor): passo proporcional à distância."""
        self.r_cam *= math.exp(log_factor)
        self.clamp(app)

    def zoom_wheel(self, clicks, app: AppConfig):
        """Roda para cima (clicks > 0) aproxima: r ← r·f^(−clicks)."""
        self.zoom(-clicks * math.log(app.zoom_wheel_factor), app)

    def camera(self, cam: CameraConfig):
        return Camera.from_config(
            replace(
                cam,
                r_cam=self.r_cam,
                inclination_deg=self.inclination_deg,
                azimuth_deg=self.azimuth_deg,
            )
        )


@dataclass
class ViewState:
    """Parâmetros de render ajustados pelo teclado (passos, exposição, fundo, disco, HUD,
    Doppler, textura, bloom e modo do bloom) e o relógio da textura (tempo coordenado
    t_obs, unidades r_s/c)."""

    max_steps: int
    exposure: float
    background: str
    disk_enabled: bool
    hud_visible: bool
    doppler: bool
    texture: bool
    bloom: bool
    bloom_mode: str
    time: float = 0.0

    @classmethod
    def initial(
        cls,
        app: AppConfig,
        gpu: GpuConfig,
        color: ColorConfig,
        disk: DiskConfig,
        render_cfg: RenderConfig,
        bloom: BloomConfig | None = None,
    ):
        bloom = BloomConfig() if bloom is None else bloom
        return cls(
            max_steps=int(gpu.integrator.max_steps),
            exposure=float(color.exposure),
            background=render_cfg.background,
            disk_enabled=bool(disk.enabled),
            hud_visible=bool(app.hud_visible),
            doppler=bool(disk.doppler),
            texture=bool(disk.texture),
            bloom=bool(bloom.enabled),
            bloom_mode=bloom.mode,
        )

    def advance_time(self, dt_s, app: AppConfig):
        """Avança o relógio da textura: t ← t + time_scale·M·dt (time_scale em M/s)."""
        self.time += app.time_scale * M * dt_s

    def change_steps(self, direction, app: AppConfig, gpu: GpuConfig):
        """± ``steps_increment``, limitado a [min_steps, max_steps_cap] (256 a 1024)."""
        steps = self.max_steps + direction * app.steps_increment
        self.max_steps = int(clamp(steps, gpu.min_steps, gpu.max_steps_cap))

    def change_exposure(self, direction, app: AppConfig):
        """Exposição × ``exposure_factor``^direction, limitada à faixa da ``AppConfig``."""
        exposure = self.exposure * app.exposure_factor**direction
        self.exposure = clamp(exposure, app.exposure_min, app.exposure_max)

    def toggle_background(self):
        modes = tuple(BACKGROUNDS)
        self.background = modes[(modes.index(self.background) + 1) % len(modes)]

    def toggle_bloom_mode(self):
        """``K``: glare ↔ clássico (vale também com o bloom desligado)."""
        modes = tuple(BLOOM_MODES)
        self.bloom_mode = modes[(modes.index(self.bloom_mode) + 1) % len(modes)]


class Controls:
    """Estado da câmera e da vista, com as ações do teclado e do mouse (PLANO, Fase 4)."""

    def __init__(
        self,
        app: AppConfig | None = None,
        cam: CameraConfig | None = None,
        gpu: GpuConfig | None = None,
        color: ColorConfig | None = None,
        disk: DiskConfig | None = None,
        render_cfg: RenderConfig | None = None,
        bloom: BloomConfig | None = None,
    ):
        self.app = AppConfig() if app is None else app
        self.cam = CameraConfig() if cam is None else cam
        self.gpu = GpuConfig() if gpu is None else gpu
        self.orbit = OrbitState.initial(self.cam, self.app)
        self.view = ViewState.initial(
            self.app,
            self.gpu,
            ColorConfig() if color is None else color,
            DiskConfig() if disk is None else disk,
            RenderConfig() if render_cfg is None else render_cfg,
            BloomConfig() if bloom is None else bloom,
        )

    def reset_camera(self):
        """``R``: câmera de volta à posição inicial (a vista não muda)."""
        self.orbit = OrbitState.initial(self.cam, self.app)

    def apply(self, action):
        """Executa uma ação de ``CHAR_ACTIONS``; devolve ``False`` se ela não existir."""
        view, app = self.view, self.app
        if action == "steps_up":
            view.change_steps(+1, app, self.gpu)
        elif action == "steps_down":
            view.change_steps(-1, app, self.gpu)
        elif action == "exposure_up":
            view.change_exposure(+1, app)
        elif action == "exposure_down":
            view.change_exposure(-1, app)
        elif action == "toggle_disk":
            view.disk_enabled = not view.disk_enabled
        elif action == "toggle_doppler":
            view.doppler = not view.doppler
        elif action == "toggle_texture":
            view.texture = not view.texture
        elif action == "toggle_bloom":
            view.bloom = not view.bloom
        elif action == "toggle_bloom_mode":
            view.toggle_bloom_mode()
        elif action == "toggle_background":
            view.toggle_background()
        elif action == "toggle_hud":
            view.hud_visible = not view.hud_visible
        elif action == "reset":
            self.reset_camera()
        else:
            return False
        return True

    def drag(self, dx, dy):
        """Arrastar "pega" a cena: para a direita a câmera vai para a esquerda (azimute
        cresce); para baixo (dy > 0) a câmera sobe (inclinação diminui)."""
        sens = self.app.orbit_mouse_deg_per_px
        self.orbit.orbit(dx * sens, -dy * sens, self.app)

    def keys(self, right, up, zoom_out, dt):
        """Controles contínuos: setas movem a câmera (direita: azimute diminui; cima:
        inclinação diminui); ``zoom_out`` = S − W afasta. Cada entrada vale −1, 0 ou 1."""
        app = self.app
        angle = app.orbit_key_deg_per_s * dt
        if right or up:
            self.orbit.orbit(-right * angle, -up * angle, app)
        if zoom_out:
            self.orbit.zoom(zoom_out * app.zoom_key_rate * dt, app)

    def camera(self):
        return self.orbit.camera(self.cam)


class FpsCounter:
    """Quadros por segundo médios entre duas amostras."""

    def __init__(self):
        self._frames = 0
        self._start = None
        self.fps = 0.0

    def tick(self, now):
        if self._start is None:
            self._start = now
        else:
            self._frames += 1

    def sample(self, now):
        if self._start is not None and now > self._start and self._frames > 0:
            self.fps = self._frames / (now - self._start)
            self._frames = 0
            self._start = now
        return self.fps


class ShaderWatcher:
    """Detecta mudança (mtime) em arquivos de shader, para a recarga do modo dev."""

    def __init__(self, paths):
        self.paths = tuple(Path(p) for p in paths)
        self._stamps = self._read()

    def _read(self):
        stamps = []
        for path in self.paths:
            try:
                stamps.append(path.stat().st_mtime_ns)
            except OSError:
                stamps.append(None)
        return stamps

    def changed(self):
        """``True`` uma vez por mudança detectada desde a última chamada."""
        stamps = self._read()
        if stamps == self._stamps:
            return False
        self._stamps = stamps
        return True


def hud_lines(controls, fps, buffer_size, scale, renderer_warning=None, dev_status=None):
    """Linhas do HUD como ``(texto, aviso)``. Só ASCII e "°" (fonte padrão do Pillow)."""
    orbit, view = controls.orbit, controls.view
    inner = internal_size(buffer_size, scale) or (0, 0)
    lines = [
        (f"FPS {fps:.1f}", False),
        (f"passos {view.max_steps}", False),
        (
            f"r_cam {orbit.r_cam:.2f}   incl {orbit.inclination_deg:.1f}°   "
            f"azim {orbit.azimuth_deg:.1f}°",
            False,
        ),
        (f"escala {scale:.2f}   interno {inner[0]}x{inner[1]}", False),
        (f"buffer {buffer_size[0]}x{buffer_size[1]}", False),
        (f"exposicao {view.exposure:.3g}", False),
        (
            f"disco {on_off(view.disk_enabled)}   "
            f"fundo {BACKGROUND_LABELS.get(view.background, view.background)}",
            False,
        ),
        (
            f"doppler {on_off(view.doppler)}   textura {on_off(view.texture, True)}   "
            f"t {view.time:.1f}",
            False,
        ),
        (
            f"bloom {on_off(view.bloom)}   "
            f"modo {BLOOM_MODE_LABELS.get(view.bloom_mode, view.bloom_mode)}",
            False,
        ),
    ]
    if dev_status is not None:
        lines.append((dev_status, dev_status.startswith("shader: ERRO")))
    if renderer_warning is not None:
        lines.append((renderer_warning, True))
    return lines


def render_hud_image(lines, app: AppConfig, scale=1.0):
    """Imagem RGBA do HUD: fundo semitransparente e uma linha de texto por entrada.

    ``scale`` multiplica fonte, margens e espaçamento (pixels lógicos → físicos).
    """
    font = ImageFont.load_default(max(1, round(app.hud_font_size * scale)))
    pad = max(0, round(app.hud_padding * scale))
    spacing = max(0, round(app.hud_line_spacing * scale))
    ascent, descent = font.getmetrics()
    line_height = ascent + descent
    widths = [int(math.ceil(font.getlength(text))) for text, _ in lines] or [0]
    width = max(widths) + 2 * pad
    height = len(lines) * line_height + max(0, len(lines) - 1) * spacing + 2 * pad
    image = Image.new("RGBA", (max(1, width), max(1, height)), app.hud_background)
    draw = ImageDraw.Draw(image)
    for i, (text, warning) in enumerate(lines):
        color = app.hud_warning_color if warning else app.hud_text_color
        draw.text((pad, pad + i * (line_height + spacing)), text, font=font, fill=color)
    return image


def hud_scale(wnd):
    """Escala do HUD: ``pixel_ratio`` (buffer/janela) ou a escala DPI do sistema.

    No Windows o pyglet cria a janela DPI-aware com buffer = janela (pixel_ratio 1); a
    escala DPI do sistema (``pyglet.window.Window.scale``) mantém o texto legível. Com a
    janela minimizada (tamanho 0) vale 1.
    """
    width = wnd.size[0]
    ratio = wnd.buffer_size[0] / width if width > 0 else 1.0
    dpi = getattr(getattr(wnd, "_window", None), "scale", None)
    if isinstance(dpi, (int, float)) and dpi > 0:
        ratio = max(ratio, float(dpi))
    return ratio if ratio > 0 else 1.0


# --- GL ------------------------------------------------------------------------


class HudOverlay:
    """Textura RGBA do HUD, redesenhada só quando o texto muda (no máximo a cada
    ``hud_refresh_s``)."""

    def __init__(self, ctx: moderngl.Context, app: AppConfig):
        self.ctx = ctx
        self.app = app
        self.texture = None
        # Escala da textura atual (fonte e margem em pixels físicos).
        self.scale = 1.0
        self._key = None
        self._last_update = None

    def invalidate(self):
        """Força a próxima ``update`` a reconstruir as linhas (ex.: depois de um resize)."""
        self._last_update = None

    def update(self, now, scale, lines_fn, force=False):
        """Chama ``lines_fn()`` se já passou ``hud_refresh_s`` e sobe a textura se mudou.

        Devolve ``True`` quando a textura foi redesenhada.
        """
        due = self._last_update is None or now - self._last_update >= self.app.hud_refresh_s
        if not (force or due):
            return False
        self._last_update = now
        lines = lines_fn()
        key = (tuple(lines), scale)
        if key == self._key and self.texture is not None:
            return False
        self._key = key
        self.scale = scale
        image = render_hud_image(lines, self.app, scale)
        # O Pillow começa no topo; a textura do OpenGL começa embaixo.
        data = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).tobytes()
        if self.texture is None or self.texture.size != image.size:
            self._release_texture()
            self.texture = self.ctx.texture(image.size, _RGBA, data=data)
            self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.texture.repeat_x = False
            self.texture.repeat_y = False
        else:
            self.texture.write(data)
        return True

    def viewport(self, buffer_size):
        """Retângulo (x, y, w, h) do HUD no canto superior esquerdo, origem embaixo."""
        if self.texture is None:
            return None
        margin = round(self.app.hud_margin * self.scale)
        width, height = self.texture.size
        return (margin, int(buffer_size[1]) - margin - height, width, height)

    def _release_texture(self):
        if self.texture is not None:
            self.texture.release()
            self.texture = None

    def release(self):
        self._release_texture()


class FramePipeline:
    """Quadro completo: cena no FBO interno → present no alvo → HUD no canto.

    ``render`` desenha em qualquer framebuffer alvo (janela ou offscreen) de tamanho
    ``buffer_size`` em pixels físicos. A textura da cena tem filtro linear, esticada na
    apresentação quando a escala ≠ 1.

    Com ``view.bloom``, a cena vai para uma textura HDR RGBA16F no tamanho interno e o
    ``BloomPass`` compõe glare ou clássico + ACES + sRGB na textura RGBA8 da cena. Sem
    bloom, a textura HDR nem é criada e o quadro é o da Fase 5a, bit a bit.
    """

    def __init__(
        self,
        ctx: moderngl.Context,
        app: AppConfig | None = None,
        gpu: GpuConfig | None = None,
        render_cfg: RenderConfig | None = None,
        disk: DiskConfig | None = None,
        color: ColorConfig | None = None,
        bloom: BloomConfig | None = None,
    ):
        self.ctx = ctx
        self.app = AppConfig() if app is None else app
        self.renderer = GpuRenderer(ctx, gpu=gpu, render_cfg=render_cfg, disk=disk, color=color)
        gpu_cfg = self.renderer.gpu
        # Valida a BloomConfig (ValueError) e compila os shaders do bloom; as texturas HDR
        # só são criadas no primeiro quadro com bloom.
        try:
            self.bloom = BloomPass(ctx, bloom, gpu_cfg)
        except Exception:
            self.renderer.release()
            raise
        vert = shader_path(gpu_cfg.vertex_shader, gpu_cfg).read_text(encoding="utf-8")
        frag = shader_path(self.app.present_shader, gpu_cfg).read_text(encoding="utf-8")
        self.present = ctx.program(vertex_shader=vert, fragment_shader=frag)
        self.present["u_texture"].value = _PRESENT_TEXTURE_UNIT
        self.present_vao = ctx.vertex_array(self.present, [])
        self.hud = HudOverlay(ctx, self.app)
        self.scene_texture = None
        self.scene_fbo = None
        self.hdr_texture = None
        self.hdr_fbo = None

    def _hdr_target(self, size):
        if self.hdr_texture is None or self.hdr_texture.size != size:
            self._release_hdr()
            self.hdr_texture = hdr_texture(self.ctx, size)
            self.hdr_fbo = self.ctx.framebuffer(color_attachments=[self.hdr_texture])
        return self.hdr_fbo

    def _scene_target(self, size):
        if self.scene_texture is None or self.scene_texture.size != size:
            self._release_scene()
            self.scene_texture = self.ctx.texture(size, _RGBA, dtype="f1")
            self.scene_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.scene_texture.repeat_x = False
            self.scene_texture.repeat_y = False
            self.scene_fbo = self.ctx.framebuffer(color_attachments=[self.scene_texture])
        return self.scene_fbo

    def _present(self, target, texture, viewport):
        target.viewport = viewport
        self.present["u_viewport"].value = tuple(float(v) for v in viewport)
        texture.use(_PRESENT_TEXTURE_UNIT)
        self.present_vao.render(moderngl.TRIANGLES, vertices=_FULLSCREEN_VERTICES)

    def render(self, target, camera: Camera, view: ViewState, buffer_size, scale, show_hud=True):
        """Desenha o quadro em ``target``; devolve ``False`` (sem desenhar) se minimizado."""
        size = internal_size(buffer_size, scale)
        if size is None:
            return False
        buffer_size = (int(buffer_size[0]), int(buffer_size[1]))
        ctx = self.ctx
        ctx.disable(moderngl.BLEND)

        scene = self._scene_target(size)
        # Com bloom, o raio vai para a textura HDR; sem ele, direto para a da cena (RGBA8).
        ray_target = self._hdr_target(size) if view.bloom else scene
        ray_target.use()
        ray_target.viewport = (0, 0, *size)
        self.renderer.set_uniforms(
            camera,
            size[0],
            size[1],
            max_steps=view.max_steps,
            background=view.background,
            disk_enabled=view.disk_enabled,
            exposure=view.exposure,
            doppler=view.doppler,
            texture=view.texture,
            time=view.time,
            hdr=view.bloom,
        )
        self.renderer.draw()
        if view.bloom:
            self.bloom.render(self.hdr_texture, scene, mode=view.bloom_mode)

        target.use()
        full = (0, 0, *buffer_size)
        self._present(target, self.scene_texture, full)

        hud_rect = self.hud.viewport(buffer_size) if show_hud else None
        if hud_rect is not None:
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            try:
                self._present(target, self.hud.texture, hud_rect)
            finally:
                ctx.disable(moderngl.BLEND)
        target.viewport = full
        return True

    def render_image(self, camera: Camera, view: ViewState, size):
        """Quadro no tamanho ``size``, escala 1,0 e sem HUD, num alvo offscreen próprio.

        Passa pelo mesmo pipeline do quadro (com bloom, se ``view.bloom``). Devolve uint8
        (H, W, 3) com a linha 0 no topo, como ``GpuRenderer.render()``, e religa o
        framebuffer que estava em uso (a janela no app).
        """
        size = (int(size[0]), int(size[1]))
        if size[0] < 1 or size[1] < 1:
            raise ValueError("a imagem precisa de largura e altura ≥ 1")
        ctx = self.ctx
        previous = ctx.fbo if ctx.fbo is not None else ctx.detect_framebuffer()
        target = ctx.framebuffer(color_attachments=[ctx.renderbuffer(size, components=_RGBA)])
        try:
            self.render(target, camera, view, size, 1.0, show_hud=False)
            data = target.read(components=_RGB, alignment=1)
        finally:
            previous.use()
            for attachment in target.color_attachments:
                attachment.release()
            target.release()
        image = np.frombuffer(data, dtype=np.uint8).reshape(size[1], size[0], _RGB)
        return np.ascontiguousarray(image[::-1])

    def _release_hdr(self):
        if self.hdr_fbo is not None:
            self.hdr_fbo.release()
            self.hdr_fbo = None
        if self.hdr_texture is not None:
            self.hdr_texture.release()
            self.hdr_texture = None

    def _release_scene(self):
        if self.scene_fbo is not None:
            self.scene_fbo.release()
            self.scene_fbo = None
        if self.scene_texture is not None:
            self.scene_texture.release()
            self.scene_texture = None

    def release(self):
        self._release_scene()
        self._release_hdr()
        self.bloom.release()
        self.hud.release()
        self.present_vao.release()
        self.present.release()
        self.renderer.release()


# --- argumentos -----------------------------------------------------------------


def _scale_type(app: AppConfig):
    def parse(text):
        try:
            value = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"escala inválida: {text!r}") from None
        if not (math.isfinite(value) and app.scale_min <= value <= app.scale_max):
            raise argparse.ArgumentTypeError(
                f"escala {text} fora da faixa [{app.scale_min:g}, {app.scale_max:g}]"
            )
        return value

    return parse


def add_app_arguments(parser, app: AppConfig):
    parser.add_argument(
        "--scale",
        type=_scale_type(app),
        default=app.scale,
        help=f"escala de resolução interna, de {app.scale_min:g} a {app.scale_max:g}",
    )
    parser.add_argument(
        "--dev", action="store_true", help="recarrega o shader quando o arquivo muda"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help=(
            f"mede o FPS em {app.benchmark_size[0]}x{app.benchmark_size[1]} offscreen e sai "
            f"com 0 se >= {app.benchmark_target_fps:g} FPS"
        ),
    )


def build_parser(app: AppConfig | None = None):
    """Parser do ``moderngl-window`` mais ``--scale``, ``--dev`` e ``--benchmark``."""
    parser = mglw.create_parser()
    add_app_arguments(parser, AppConfig() if app is None else app)
    return parser


def renderer_warning(renderer_name, gl_cfg: GLCheckConfig | None = None):
    """Aviso (console e HUD) quando o renderer não é o esperado; ``None`` se for."""
    gl_cfg = GLCheckConfig() if gl_cfg is None else gl_cfg
    if is_expected_renderer(renderer_name, gl_cfg):
        return None
    name = renderer_name.encode("ascii", "ignore").decode("ascii")
    return f"AVISO: GPU errada, renderer {name} (esperado {gl_cfg.expected_vendor})"


# --- janela ---------------------------------------------------------------------

_APP = AppConfig()
_GL = GLCheckConfig()


class BlackHoleApp(mglw.WindowConfig):
    """Janela do ``moderngl-window`` com a câmera orbital e o HUD (PLANO, Fase 4)."""

    title = _APP.title
    window_size = _APP.window_size
    vsync = _APP.vsync
    resizable = _APP.resizable
    # Sem tarjas: a imagem preenche o buffer inteiro, qualquer que seja a razão de aspecto.
    aspect_ratio = None
    gl_version = divmod(_GL.min_version_code // 10, 10)
    # O present cobre o buffer inteiro: limpar antes seria trabalho à toa.
    clear_color = None

    @classmethod
    def add_arguments(cls, parser):
        add_app_arguments(parser, _APP)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.app_cfg = _APP
        self.scale = float(self.argv.scale)
        self.render_cfg = RenderConfig()
        self.controls = Controls(self.app_cfg, render_cfg=self.render_cfg)
        self.pipeline = FramePipeline(self.ctx, self.app_cfg, render_cfg=self.render_cfg)
        self.fps = FpsCounter()

        name = self.ctx.info["GL_RENDERER"]
        print(f"Renderer: {name}")
        self.renderer_warning = renderer_warning(name)
        if self.renderer_warning is not None:
            print(f"{self.renderer_warning}; veja `uv run python -m blackhole.check_gl`.")

        self.watcher = None
        self._next_poll = 0.0
        if self.argv.dev:
            self.watcher = ShaderWatcher(self.pipeline.renderer.shader_paths())
            print("Modo dev: o shader é recarregado quando o arquivo muda.")
        self._capture_requested = False
        self._closed = False

    # --- quadro ---

    def on_render(self, time_s, frame_time):
        if self._closed:
            return
        self.fps.tick(time_s)
        self._poll_shader(time_s)
        dt = clamp(frame_time, 0.0, self.app_cfg.max_input_dt_s)
        self._continuous_input(dt)
        self.controls.view.advance_time(dt, self.app_cfg)

        buffer_size = tuple(self.wnd.buffer_size)
        if internal_size(buffer_size, self.scale) is None:
            return  # janela minimizada
        if self._capture_requested:
            self._capture_requested = False
            self._capture(buffer_size)

        view = self.controls.view
        if view.hud_visible:
            self.pipeline.hud.update(
                time_s, hud_scale(self.wnd), lambda: self._hud_lines(time_s, buffer_size)
            )
        self.pipeline.render(
            self.wnd.fbo,
            self.controls.camera(),
            view,
            buffer_size,
            self.scale,
            show_hud=view.hud_visible,
        )

    def _hud_lines(self, now, buffer_size):
        dev_status = None
        if self.watcher is not None:
            error = self.pipeline.renderer.last_error
            dev_status = "shader: ERRO (ver console)" if error else "modo dev: recarga ligada"
        return hud_lines(
            self.controls,
            self.fps.sample(now),
            buffer_size,
            self.scale,
            self.renderer_warning,
            dev_status,
        )

    def _continuous_input(self, dt):
        keys, pressed = self.wnd.keys, self.wnd.is_key_pressed
        right = int(pressed(keys.RIGHT)) - int(pressed(keys.LEFT))
        up = int(pressed(keys.UP)) - int(pressed(keys.DOWN))
        zoom_out = int(pressed(keys.S)) - int(pressed(keys.W))
        self.controls.keys(right, up, zoom_out, dt)

    def _poll_shader(self, now):
        if self.watcher is None or now < self._next_poll:
            return
        self._next_poll = now + self.app_cfg.shader_poll_s
        if not self.watcher.changed():
            return
        renderer = self.pipeline.renderer
        if renderer.reload():
            print("Shader recarregado.")
        else:
            print("ERRO ao recompilar o shader; o programa anterior continua em uso:")
            print(renderer.last_error)
        self.pipeline.hud.invalidate()

    def _capture(self, buffer_size):
        """F12: render extra pelo pipeline do quadro (com o bloom, se ligado) no tamanho do
        buffer, escala 1,0 e sem HUD, gravado em ``outputs/``."""
        path = capture_path(self.render_cfg.output_dir, self.app_cfg.capture_prefix)
        try:
            image = self.pipeline.render_image(
                self.controls.camera(), self.controls.view, buffer_size
            )
            save_png(image, path)
        except (OSError, ValueError, moderngl.Error) as exc:
            print(f"ERRO na captura: {exc}")
            return
        print(f"Captura salva: {path}")

    # --- eventos ---

    def on_resize(self, width, height):
        # O tamanho é lido de wnd.buffer_size a cada quadro; só o HUD precisa de refresh.
        self.pipeline.hud.invalidate()

    def on_key_event(self, key, action, modifiers):
        keys = self.wnd.keys
        if key == keys.F12 and action == keys.ACTION_PRESS:
            self._capture_requested = True

    def on_unicode_char_entered(self, char):
        action = char_action(char)
        if action is not None and self.controls.apply(action):
            self.pipeline.hud.invalidate()

    def on_mouse_drag_event(self, x, y, dx, dy):
        self.controls.drag(dx, dy)

    def on_mouse_scroll_event(self, x_offset, y_offset):
        self.controls.orbit.zoom_wheel(y_offset, self.app_cfg)

    def on_close(self):
        if not self._closed:
            self._closed = True
            self.pipeline.release()


# --- benchmark ------------------------------------------------------------------


def run_benchmark(app: AppConfig | None = None, scale=None):
    """Pipeline completo (cena interna + bloom + present + HUD) num alvo offscreen.

    Tamanho ``benchmark_size``, ``ctx.finish()`` por quadro, aquecimento descartado.
    Doppler, textura e bloom ficam ligados (o caso mais caro; o bloom no modo inicial da
    ``BloomConfig``) e o relógio da textura avança
    ``1/benchmark_target_fps`` s por quadro. Devolve 0 com FPS médio ≥
    ``benchmark_target_fps``, 1 abaixo dele ou sem OpenGL 3.3.
    """
    app = AppConfig() if app is None else app
    scale = app.scale if scale is None else float(scale)
    gl_cfg = GLCheckConfig()
    required = format_version(gl_cfg.min_version_code)
    try:
        ctx = moderngl.create_standalone_context(require=gl_cfg.min_version_code)
    except Exception as exc:  # moderngl/glcontext não expõem um tipo de erro específico
        print(f"ERRO: não foi possível criar um contexto OpenGL {required}: {exc}")
        return 1

    size = (int(app.benchmark_size[0]), int(app.benchmark_size[1]))
    try:
        name = ctx.info["GL_RENDERER"]
        print(f"Renderer: {name}")
        warning = renderer_warning(name, gl_cfg)
        if warning is not None:
            print(f"{warning}; veja `uv run python -m blackhole.check_gl`.")
        controls = Controls(app)
        camera, view = controls.camera(), controls.view
        view.doppler = True
        view.texture = True
        view.bloom = True
        frame_dt = 1.0 / app.benchmark_target_fps
        pipeline = FramePipeline(ctx, app)
        target = ctx.framebuffer(color_attachments=[ctx.renderbuffer(size, components=_RGBA)])
        previous = ctx.fbo
        try:
            pipeline.hud.update(0.0, 1.0, lambda: hud_lines(controls, 0.0, size, scale, warning))
            for _ in range(app.benchmark_warmup):
                view.advance_time(frame_dt, app)
                pipeline.render(target, camera, view, size, scale, show_hud=view.hud_visible)
                ctx.finish()
            start = time.perf_counter()
            for _ in range(app.benchmark_frames):
                view.advance_time(frame_dt, app)
                pipeline.render(target, camera, view, size, scale, show_hud=view.hud_visible)
                ctx.finish()
            elapsed = time.perf_counter() - start
        finally:
            # O alvo fica ligado depois de render(); não deixa um FBO liberado em ctx.fbo.
            ctx.fbo = previous
            for attachment in target.color_attachments:
                attachment.release()
            target.release()
            pipeline.release()
    finally:
        ctx.release()

    fps = app.benchmark_frames / elapsed if elapsed > 0 else math.inf
    inner = internal_size(size, scale)
    ok = fps >= app.benchmark_target_fps
    print(
        f"{size[0]}x{size[1]}, escala {scale:g} (interno {inner[0]}x{inner[1]}), "
        f"{view.max_steps} passos, Doppler, textura e bloom ({view.bloom_mode}) ligados, "
        f"{app.benchmark_frames} quadros (+{app.benchmark_warmup} de aquecimento)"
    )
    print(
        f"FPS médio: {fps:.1f} ({1000.0 / fps:.2f} ms/quadro), alvo {app.benchmark_target_fps:g}"
    )
    print("OK: dentro do aceite." if ok else "FALHOU: abaixo do FPS alvo.")
    return 0 if ok else 1


def main(argv=None):
    """Abre a janela, ou roda o ``--benchmark`` e devolve o código de saída."""
    args_list = sys.argv[1:] if argv is None else list(argv)
    args = build_parser(_APP).parse_args(args_list)
    if args.benchmark:
        return run_benchmark(_APP, args.scale)
    mglw.run_window_config(BlackHoleApp, args=args_list)
    return 0


if __name__ == "__main__":
    sys.exit(main())
