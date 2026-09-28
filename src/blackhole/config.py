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
