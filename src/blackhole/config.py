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
