"""Parâmetros do projeto, concentrados em dataclasses (sem números mágicos no código).

Cada fase acrescenta aqui as suas dataclasses (física, disco, câmera, qualidade, cor).
"""

from dataclasses import dataclass


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
