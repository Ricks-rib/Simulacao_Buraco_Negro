"""Bloom do app em tempo real (PLANO, seção 3.10), só na GPU.

A cena chega em HDR linear (radiância × exposição, RGBA16F; ``blackhole.frag`` com
``u_hdr_output``). ``BloomPass`` calcula B = PSF ∗ fonte por uma cadeia de mips no estilo
de "Next Generation Post Processing in Call of Duty" (Jimenez, 2014):

1. downsample de 13 taps (``shaders/bloom_down.frag``) da cena até o nível N, cada nível
   com metade do tamanho do anterior; no primeiro nível, por pixel da cena, valores não
   finitos viram 0 e o modo clássico aplica o filtro de brilho c·max(0, L − limiar)/L;
2. upsample em tenda 3×3 (``shaders/bloom_up.frag``) do nível N até o 1, somando cada
   nível ao de cima por blend aditivo: o nível 1 fica com a soma dos N níveis borrados;
3. composição (``shaders/bloom_composite.frag``): B = tenda(nível 1)/N, depois
   glare, (1 − k_g)·I + k_g·B, ou clássico, I + k_c·B, e ACES + sRGB (``scene.py``).

A soma de N níveis dividida por N é uma PSF de média 1 com cauda longa (soma de
gaussianas de larguras ~2^i px), próxima da cauda ~1/θ² do glare do olho. Os pesos dos
filtros somam 1 e as texturas usam clamp-to-edge, então a média se conserva de forma
aproximada. Não há réplica na CPU: a validação é por testes de propriedade
(``tests/test_bloom.py``).
"""

import math

import moderngl

from blackhole.config import BloomConfig, GpuConfig
from blackhole.render_gpu import shader_path

# Modo do bloom → código de ``u_mode`` em bloom_composite.frag.
MODES = {"glare": 0, "classic": 1}
# Código de ``u_prefilter`` em bloom_down.frag: sem filtro ou limiar do clássico.
PREFILTER_NONE = 0
PREFILTER_THRESHOLD = 1

# Filtro de 13 taps do downsample (Jimenez, 2014), em texels da fonte: centro (0, 0);
# quatro taps internos em (±1, ±1); quatro de borda em (±2, 0) e (0, ±2); quatro de canto
# em (±2, ±2). Soma: 1/8 + 4·1/8 + 4·1/16 + 4·1/32 = 1.
DOWN_CENTER_WEIGHT = 0.125
DOWN_INNER_WEIGHT = 0.125
DOWN_EDGE_WEIGHT = 0.0625
DOWN_CORNER_WEIGHT = 0.03125

# Tenda 3×3 do upsample e da composição: (1 2 1 / 2 4 2 / 1 2 1)/16. Soma: 1.
TENT_CENTER_WEIGHT = 0.25
TENT_EDGE_WEIGHT = 0.125
TENT_CORNER_WEIGHT = 0.0625

# Formato das texturas HDR (cena e níveis): RGBA float16.
HDR_COMPONENTS = 4
HDR_DTYPE = "f2"
# Unidades de textura do bloom (0 é a tabela de corpo negro e 1 o present do app).
_SOURCE_UNIT = 2
_BLOOM_UNIT = 3
# Vértices do triângulo de tela cheia (fullscreen.vert).
_FULLSCREEN_VERTICES = 3


def _is_int(value):
    return not isinstance(value, bool) and isinstance(value, int)


def _is_real(value):
    """Número real (int ou float), sem aceitar bool."""
    return not isinstance(value, bool) and isinstance(value, (int, float))


def validate_bloom(cfg: BloomConfig):
    """Parâmetros do bloom: modo conhecido, forças em [0, 1], limiar ≥ 0 e níveis ≥ 1.

    Levanta ``ValueError`` com mensagem clara (matriz I/O da Fase 5b).
    """
    if cfg.mode not in MODES:
        raise ValueError(f"modo de bloom {cfg.mode!r} inválido; use um de {', '.join(MODES)}")
    for name in ("glare_strength", "classic_strength"):
        value = getattr(cfg, name)
        if not (_is_real(value) and 0.0 <= value <= 1.0):
            raise ValueError(f"{name} precisa estar em [0, 1], recebido {value!r}")
    threshold = cfg.threshold
    if not (_is_real(threshold) and math.isfinite(threshold) and threshold >= 0):
        raise ValueError(f"threshold precisa ser finito e ≥ 0, recebido {threshold!r}")
    if not (_is_int(cfg.levels) and cfg.levels >= 1):
        raise ValueError(f"levels precisa ser um inteiro ≥ 1, recebido {cfg.levels!r}")
    return cfg


def mip_sizes(size, levels):
    """Tamanhos dos níveis 1..N da cadeia a partir da cena de tamanho ``size``.

    Cada nível tem metade do anterior (divisão inteira, cada lado ≥ 1). A cadeia tem no
    máximo ``levels`` níveis e para no primeiro nível de 1×1 px; o nível 1 sempre existe
    (com uma cena de 1×1 ele também é 1×1 e B = I).
    """
    width, height = int(size[0]), int(size[1])
    if width < 1 or height < 1:
        raise ValueError("a cena precisa de largura e altura ≥ 1")
    sizes = []
    while len(sizes) < levels:
        width, height = max(1, width // 2), max(1, height // 2)
        sizes.append((width, height))
        if (width, height) == (1, 1):
            break
    return sizes


def hdr_texture(ctx: moderngl.Context, size):
    """Textura RGBA16F com filtro linear e clamp-to-edge (cena HDR e níveis da cadeia)."""
    texture = ctx.texture(size, HDR_COMPONENTS, dtype=HDR_DTYPE)
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    texture.repeat_x = False
    texture.repeat_y = False
    return texture


class BloomPass:
    """Cadeia de mips do bloom e composição com ACES + sRGB (PLANO, seção 3.10).

    Valida a ``BloomConfig`` na construção (``ValueError``). As texturas dos níveis são
    criadas no primeiro ``render`` e reutilizadas até o tamanho da cena mudar.
    """

    def __init__(
        self,
        ctx: moderngl.Context,
        cfg: BloomConfig | None = None,
        gpu: GpuConfig | None = None,
    ):
        self.ctx = ctx
        self.cfg = validate_bloom(BloomConfig() if cfg is None else cfg)
        self.gpu = GpuConfig() if gpu is None else gpu
        vert = shader_path(self.gpu.vertex_shader, self.gpu).read_text(encoding="utf-8")
        self._programs = []
        self._vaos = []
        try:
            self.down, self.down_vao = self._program(vert, self.cfg.down_shader)
            self.up, self.up_vao = self._program(vert, self.cfg.up_shader)
            self.composite, self.composite_vao = self._program(vert, self.cfg.composite_shader)
        except Exception:
            self._release_programs()
            raise
        self.down["u_source"].value = _SOURCE_UNIT
        self.down["u_threshold"].value = float(self.cfg.threshold)
        self.up["u_source"].value = _SOURCE_UNIT
        self.composite["u_scene"].value = _SOURCE_UNIT
        self.composite["u_bloom"].value = _BLOOM_UNIT
        self._size = None
        self._levels = []  # (textura, framebuffer) dos níveis 1..N

    def _program(self, vert, frag_name):
        frag = shader_path(frag_name, self.gpu).read_text(encoding="utf-8")
        program = self.ctx.program(vertex_shader=vert, fragment_shader=frag)
        self._programs.append(program)
        vao = self.ctx.vertex_array(program, [])
        self._vaos.append(vao)
        return program, vao

    @property
    def level_sizes(self):
        """Tamanhos dos níveis em uso (vazio antes do primeiro ``render``)."""
        return [texture.size for texture, _ in self._levels]

    def strength(self, mode):
        """k_g (glare) ou k_c (clássico) da ``BloomConfig``."""
        return self.cfg.glare_strength if mode == "glare" else self.cfg.classic_strength

    def _ensure_levels(self, size):
        if self._size == size:
            return
        self._release_levels()
        for level_size in mip_sizes(size, self.cfg.levels):
            texture = hdr_texture(self.ctx, level_size)
            self._levels.append((texture, self.ctx.framebuffer(color_attachments=[texture])))
        self._size = size

    def render(self, source, target, viewport=None, mode=None, encode=True):
        """Bloom de ``source`` (textura HDR) composto em ``target``.

        ``viewport`` é o retângulo (x, y, w, h) do alvo, do tamanho de ``source`` (padrão:
        (0, 0, tamanho da fonte)). ``mode`` é ``"glare"`` ou ``"classic"`` (padrão: o da
        ``BloomConfig``). Com ``encode`` a saída passa por ACES + sRGB (alvo RGBA8); sem
        ele é a radiância linear (alvo float, para os testes). Deixa o blend desligado e
        ``target`` ligado.
        """
        mode = self.cfg.mode if mode is None else mode
        if mode not in MODES:
            raise ValueError(f"modo de bloom {mode!r} inválido; use um de {', '.join(MODES)}")
        size = tuple(source.size)
        viewport = (0, 0, *size) if viewport is None else tuple(int(v) for v in viewport)
        if tuple(viewport[2:]) != size:
            raise ValueError("o viewport da composição precisa ter o tamanho da fonte")
        self._ensure_levels(size)
        ctx = self.ctx
        ctx.disable(moderngl.BLEND)

        # Downsample: cena → nível 1 → … → nível N.
        previous = source
        for index, (texture, fbo) in enumerate(self._levels):
            fbo.use()
            fbo.viewport = (0, 0, *texture.size)
            prefilter = PREFILTER_THRESHOLD if index == 0 and mode == "classic" else PREFILTER_NONE
            self.down["u_first_level"].value = index == 0
            self.down["u_prefilter"].value = prefilter
            self.down["u_source_texel"].value = (1.0 / previous.size[0], 1.0 / previous.size[1])
            self.down["u_target_size"].value = (float(texture.size[0]), float(texture.size[1]))
            previous.use(_SOURCE_UNIT)
            self.down_vao.render(moderngl.TRIANGLES, vertices=_FULLSCREEN_VERTICES)
            previous = texture

        # Upsample: nível N → … → nível 1, somando por blend aditivo.
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        try:
            for index in range(len(self._levels) - 1, 0, -1):
                small = self._levels[index][0]
                texture, fbo = self._levels[index - 1]
                fbo.use()
                fbo.viewport = (0, 0, *texture.size)
                self.up["u_source_texel"].value = (1.0 / small.size[0], 1.0 / small.size[1])
                self.up["u_target_size"].value = (float(texture.size[0]), float(texture.size[1]))
                small.use(_SOURCE_UNIT)
                self.up_vao.render(moderngl.TRIANGLES, vertices=_FULLSCREEN_VERTICES)
        finally:
            ctx.disable(moderngl.BLEND)

        # Composição no alvo.
        bloom = self._levels[0][0]
        target.use()
        target.viewport = viewport
        program = self.composite
        program["u_bloom_texel"].value = (1.0 / bloom.size[0], 1.0 / bloom.size[1])
        program["u_viewport"].value = tuple(float(v) for v in viewport)
        program["u_mode"].value = MODES[mode]
        program["u_strength"].value = float(self.strength(mode))
        program["u_level_count"].value = float(len(self._levels))
        program["u_encode"].value = bool(encode)
        source.use(_SOURCE_UNIT)
        bloom.use(_BLOOM_UNIT)
        self.composite_vao.render(moderngl.TRIANGLES, vertices=_FULLSCREEN_VERTICES)

    def _release_levels(self):
        for texture, fbo in self._levels:
            fbo.release()
            texture.release()
        self._levels = []
        self._size = None

    def _release_programs(self):
        for vao in self._vaos:
            vao.release()
        for program in self._programs:
            program.release()
        self._vaos = []
        self._programs = []

    def release(self):
        """Libera níveis, VAOs e programas (o contexto é de quem o criou)."""
        self._release_levels()
        self._release_programs()
