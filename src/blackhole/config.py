"""Parâmetros do projeto, concentrados em dataclasses (sem números mágicos no código).

Cada fase acrescenta aqui as suas dataclasses (física, disco, câmera, qualidade, cor).
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GLCheckConfig:
    """Requisitos verificados por ``check_gl``."""

    # Versão mínima do OpenGL no formato do moderngl (330 = OpenGL 3.3 core).
    min_version_code: int = 330
    # Prefixo esperado no GL_RENDERER da GPU dedicada (RTX 5070).
    expected_vendor: str = "NVIDIA"


@dataclass(frozen=True)
class IntegratorConfig:
    """Integrador RK4 de passo adaptativo dt = k·r (PLANO, seção 3.5)."""

    # Fator k do passo dt = k·r fora da faixa da esfera de fótons.
    step_factor: float = 0.01
    # Multiplicador de k dentro da faixa da esfera de fótons, onde o raio é instável.
    photon_step_factor: float = 0.1
    # Faixa radial em que o passo é reduzido (envolve r = 1,5).
    photon_r_min: float = 1.2
    photon_r_max: float = 2.0
    # Limites absolutos do passo afim.
    dt_min: float = 1.0e-5
    dt_max: float = 10.0
    # Passos antes de desistir; o raio é então tratado como capturado (seção 3.4).
    max_steps: int = 20_000


@dataclass(frozen=True)
class DiskConfig:
    """Disco de acreção fino no plano y = 0 (PLANO, seção 3.6)."""

    # Borda interna na ISCO (r = 3) e borda externa.
    r_in: float = 3.0
    r_out: float = 12.0
    enabled: bool = True
    # Temperatura no pico do perfil T(r), em kelvin, atingida em r_pico = (49/36)·r_in.
    t_peak: float = 10_000.0
    # Fator de desvio g da seção 3.8 (Doppler + redshift) no disco e blueshift g_∞ do
    # observador estático nas estrelas. Desligado = visual "Interstellar" (tecla B no app).
    doppler: bool = True
    # Textura procedural de emissividade co-rotante com Ω(r) (DiskTextureConfig). A
    # referência da CPU fica sem textura; tecla T no app, --texture no render CPU.
    texture: bool = False


@dataclass(frozen=True)
class DiskTextureConfig:
    """Textura procedural do disco (PLANO, seção 3.9): ruído de valor em (ln r, φ').

    φ' = φ − Ω(r)·t_emit é o ângulo no referencial que gira com o gás. A emissividade é
    multiplicada por m = 1 + contraste·(2·fbm − 1) ∈ [1 − contraste, 1 + contraste].
    """

    # Células por unidade de ln r na primeira oitava (≈ 17 células radiais de r = 3 a 12).
    radial_cells: float = 12.0
    # Células azimutais por volta na primeira oitava (inteiro: a textura é periódica em φ).
    # Com 24 células, a célula tem arco 2πr/24 ≈ 0,26·r contra Δr ≈ r/12: alongada em φ.
    azimuthal_cells: int = 24
    # Oitavas do fbm (cada uma dobra as células nos dois eixos e divide a amplitude por 2).
    octaves: int = 4
    # Contraste c da emissividade, em [0, 1].
    contrast: float = 0.7
    # Semente do hash (inteiro de 32 bits sem sinal).
    seed: int = 4242


@dataclass(frozen=True)
class CameraConfig:
    """Câmera orbital de observador estático (PLANO, seção 3.3).

    A posição é dada em coordenadas esféricas em torno do alvo: distância r_cam,
    inclinação medida a partir do eixo +y (eixo do disco) e azimute medido no plano
    xz a partir de +x, no sentido de +z.
    """

    r_cam: float = 30.0
    # FOV vertical em graus; o horizontal sai da razão de aspecto da imagem.
    fov_v_deg: float = 60.0
    # 80° em relação a +y: levemente acima do plano do disco.
    inclination_deg: float = 80.0
    azimuth_deg: float = 0.0
    # Ponto para onde a câmera olha e vetor "para cima" do mundo.
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    # Abaixo desta norma, forward × up é tratado como degenerado (up paralelo à visada).
    degenerate_tolerance: float = 1.0e-9


@dataclass(frozen=True)
class RenderConfig:
    """Renderizador CPU de referência (PLANO, Fase 2)."""

    width: int = 800
    height: int = 450
    # Sub-amostras por eixo (n×n por pixel): 1 para medidas, 2 para imagens finais.
    supersample: int = 1
    # Raio de escape r_max = max(r_max_min, r_max_factor · r_cam) (seção 3.4).
    r_max_min: float = 50.0
    r_max_factor: float = 2.0
    output_dir: str = "outputs"
    output_name: str = "render_cpu.png"
    # Fundo dos raios que escapam: "checker" (depuração) ou "stars" (seção 3.7).
    background: str = "stars"


@dataclass(frozen=True)
class CheckerConfig:
    """Fundo de depuração: xadrez em latitude/longitude em torno de +y (seção 3.7)."""

    # Casa de 15°: 24 casas em longitude × 12 em latitude.
    cell_deg: float = 15.0
    # Cores das casas em [0, 1] (RGB). Nenhuma é preta, para não confundir com a sombra.
    color_a: tuple[float, float, float] = (0.92, 0.92, 0.92)
    color_b: tuple[float, float, float] = (0.20, 0.35, 0.70)


@dataclass(frozen=True)
class ColorConfig:
    """Cadeia de cor (PLANO, seção 3.6): corpo negro, exposição, ACES e sRGB."""

    # Multiplicador da radiância linear antes do tone mapping (disco e fundo).
    exposure: float = 1.0
    # Faixa da tabela de corpo negro, em kelvin; fora dela vale a cor da extremidade.
    bb_t_min: float = 1_000.0
    bb_t_max: float = 40_000.0
    # Número de entradas da tabela, espaçadas uniformemente em ln T.
    bb_table_size: int = 256
    # Integração de Planck × CMF CIE 1931: faixa visível e passo, em nm.
    wavelength_min_nm: float = 380.0
    wavelength_max_nm: float = 780.0
    wavelength_step_nm: float = 1.0


@dataclass(frozen=True)
class StarfieldConfig:
    """Campo estelar procedural determinístico (PLANO, seção 3.7).

    A direção unitária d̂ cai na célula ⌊d̂·N⌋ de uma rede cúbica 3D. Cada célula tem no
    máximo uma estrela, decidida por um hash inteiro de 32 bits dos índices da célula.
    """

    # N: células por unidade de comprimento; a célula subtende ≈ 1/N rad no céu.
    grid_size: int = 100
    # Probabilidade de uma célula conter estrela.
    density: float = 0.05
    # Deslocamento máximo do centro da estrela, em frações da célula (0 = no centro).
    jitter: float = 0.4
    # Semente do hash (inteiro de 32 bits sem sinal).
    seed: int = 1337
    # Largura σ do perfil gaussiano, medida na corda |d̂ − ŝ| (≈ radianos).
    sigma: float = 0.0015
    # Intensidade de pico, log-uniforme entre os limites (radiância linear).
    intensity_min: float = 0.05
    intensity_max: float = 3.0
    # Temperatura de cor, log-uniforme entre os limites, em kelvin.
    t_min: float = 2_500.0
    t_max: float = 30_000.0


def _realtime_integrator():
    """Preset do integrador de tempo real (Fase 4a, Decisão do integrador).

    k = 0,05 fora da esfera de fótons e 0,05·0,2 = 0,01 dentro dela, com 512 passos. Os
    padrões de ``IntegratorConfig`` (referência da CPU) não mudam.
    """
    return IntegratorConfig(step_factor=0.05, photon_step_factor=0.2, max_steps=512)


@dataclass(frozen=True)
class GpuConfig:
    """Renderizador GPU (fragment shader GLSL 330, float32) (PLANO, Fase 4).

    A GPU e as duas renderizações do ``compare`` usam o mesmo ``integrator``; o
    ``max_steps`` dele é o limite real (uniform) e ``max_steps_cap`` é o limite de
    compilação do laço (``#define MAX_STEPS_CAP``).
    """

    integrator: IntegratorConfig = field(default_factory=_realtime_integrator)
    # Faixa aceita para o número de passos (uniform ``u_max_steps``).
    min_steps: int = 256
    max_steps_cap: int = 1024
    # Shaders em ``src/blackhole/<shader_dir>/``, carregados por caminho relativo ao módulo.
    shader_dir: str = "shaders"
    vertex_shader: str = "fullscreen.vert"
    fragment_shader: str = "blackhole.frag"


@dataclass(frozen=True)
class BloomConfig:
    """Bloom do app em tempo real (PLANO, seção 3.10), só na GPU.

    A cena é renderizada em HDR linear (radiância × exposição). Uma cadeia de mips calcula
    B = PSF ∗ fonte, com PSF de média 1 (média dos ``levels`` níveis borrados), e o
    resultado entra antes do ACES + sRGB:

    - ``"glare"``: saída = (1 − k_g)·I + k_g·B, sem limiar (a energia média se conserva);
    - ``"classic"``: saída = I + k_c·B, com fonte I·max(0, L − limiar)/L (soma energia).

    Validado por ``bloom.validate_bloom`` na construção do ``BloomPass``.
    """

    # Estado inicial no app (teclas L e K): bloom ligado e modo "glare" ou "classic".
    enabled: bool = True
    mode: str = "glare"
    # k_g: fração da luz espalhada pela PSF no modo glare, em [0, 1].
    glare_strength: float = 0.05
    # k_c: ganho do halo no modo clássico, em [0, 1].
    classic_strength: float = 0.5
    # Limiar de luminância linear do clássico, na radiância já multiplicada pela exposição
    # (1,0 ≈ branco depois do ACES), ≥ 0.
    threshold: float = 1.0
    # Níveis da cadeia de mips (cada um com metade do tamanho do anterior), ≥ 1. A cadeia
    # para antes no nível de 1×1 px.
    levels: int = 6
    # Shaders do bloom, em GpuConfig.shader_dir.
    down_shader: str = "bloom_down.frag"
    up_shader: str = "bloom_up.frag"
    composite_shader: str = "bloom_composite.frag"


@dataclass(frozen=True)
class AppConfig:
    """App em tempo real (PLANO, Fase 4): janela, câmera orbital, controles, HUD e benchmark.

    O estado inicial (e o reset com ``R``) vem das outras dataclasses: câmera de
    ``CameraConfig``, passos de ``GpuConfig.integrator``, exposição de ``ColorConfig``,
    disco de ``DiskConfig`` e fundo de ``RenderConfig``.
    """

    # Janela: título, tamanho inicial (pixels lógicos) e vsync.
    title: str = "Buraco Negro (Schwarzschild)"
    window_size: tuple[int, int] = (1280, 720)
    vsync: bool = True
    resizable: bool = True

    # Escala de resolução interna (cena = buffer × escala), fixa durante a execução.
    scale: float = 1.0
    scale_min: float = 0.25
    scale_max: float = 2.0

    # Distância da câmera: 2 ≤ r_cam ≤ 500 (PLANO, seção 3.3).
    r_cam_min: float = 2.0
    r_cam_max: float = 500.0
    # Inclinação (a partir de +y) longe dos polos: a Camera recusa up paralelo à visada.
    inclination_min_deg: float = 1.0
    inclination_max_deg: float = 179.0

    # Órbita: graus por pixel lógico arrastado com o mouse e graus por segundo nas setas.
    orbit_mouse_deg_per_px: float = 0.25
    orbit_key_deg_per_s: float = 90.0
    # Zoom multiplicativo: r ← r·f^(∓1) por clique da roda e r ← r·exp(∓taxa·dt) com W/S.
    zoom_wheel_factor: float = 1.1
    zoom_key_rate: float = 1.0
    # Maior frame_time usado nos controles contínuos e no relógio (evita saltos depois de
    # uma pausa), s.
    max_input_dt_s: float = 0.1

    # Relógio da textura do disco: tempo coordenado t_obs, em unidades de M, que passa por
    # segundo real. Com 40 M/s (20 r_s/c por segundo), a borda interna (r = 3, período
    # 2π/Ω ≈ 46 r_s/c) dá uma volta em ≈ 2,3 s e a externa (r = 12) em ≈ 18 s.
    time_scale: float = 40.0

    # Passos: incremento de + / − (a faixa é [GpuConfig.min_steps, max_steps_cap]).
    steps_increment: int = 128
    # Exposição: multiplicada/dividida pelo fator em ] / [, limitada à faixa.
    exposure_factor: float = 1.25
    exposure_min: float = 1.0 / 64.0
    exposure_max: float = 64.0

    # HUD: fonte padrão do Pillow (tamanho em pixels lógicos × pixel_ratio), margem até o
    # canto da janela e espaçamento interno (pixels lógicos), cores RGBA e atualização.
    hud_font_size: int = 16
    hud_margin: int = 10
    hud_padding: int = 6
    hud_line_spacing: int = 4
    hud_text_color: tuple[int, int, int, int] = (255, 255, 255, 255)
    hud_warning_color: tuple[int, int, int, int] = (255, 190, 60, 255)
    hud_background: tuple[int, int, int, int] = (0, 0, 0, 160)
    hud_refresh_s: float = 0.25
    hud_visible: bool = True

    # Modo desenvolvimento (--dev): intervalo de sondagem dos arquivos de shader, s.
    shader_poll_s: float = 0.5

    # Captura (F12): PNG em RenderConfig.output_dir, com prefixo + data e hora.
    capture_prefix: str = "captura"

    # Benchmark (--benchmark): alvo offscreen, quadros medidos, aquecimento e FPS alvo.
    benchmark_size: tuple[int, int] = (1920, 1080)
    benchmark_frames: int = 300
    benchmark_warmup: int = 30
    benchmark_target_fps: float = 60.0

    # Passe de apresentação (textura da cena ou do HUD → janela), em GpuConfig.shader_dir.
    present_shader: str = "present.frag"


@dataclass(frozen=True)
class CompareConfig:
    """Comparação CPU (float64) × GPU (float32) no fundo em xadrez (PLANO, Fase 4)."""

    width: int = 800
    height: int = 450
    # Pixel ruim: diferença > bad_threshold (em 1/255) em algum canal.
    bad_threshold: int = 10
    # Fração máxima de pixels ruins por cena.
    max_bad_fraction: float = 0.01
    # Diferença máxima do raio da sombra, em pixels (horizontal e vertical).
    shadow_tolerance_px: float = 1.0
    # Fundo das duas cenas comparadas (o campo estelar não entra na comparação numérica).
    background: str = "checker"
    # PNG de diferença: |Δ| máximo por pixel × ganho em cinza; pixels ruins nesta cor.
    diff_gain: float = 8.0
    bad_color: tuple[int, int, int] = (255, 0, 0)
    output_dir: str = "outputs"
    # Cada PNG empilha as duas cenas: sem disco em cima, com disco embaixo.
    cpu_name: str = "compare_cpu.png"
    gpu_name: str = "compare_gpu.png"
    diff_name: str = "compare_diff.png"
