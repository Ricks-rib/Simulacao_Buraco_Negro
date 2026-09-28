"""Verificação do contexto OpenGL: versão, vendor, renderer e executáveis do Python.

Uso: ``uv run python -m blackhole.check_gl``

Em máquinas com GPU híbrida (RTX 5070 + Radeon integrada), o Windows pode
entregar o contexto OpenGL à GPU integrada. Este módulo avisa quando o
renderer não é NVIDIA e imprime os dois executáveis que precisam ser marcados
como "Alto desempenho" em Configurações → Sistema → Tela → Gráficos: o
``.venv\\Scripts\\python.exe`` apenas inicia o interpretador base
(``sys._base_executable``).

Código de saída: 0 com OpenGL suficiente (mesmo com aviso de GPU), 1 se não
for possível criar um contexto com a versão mínima de ``GLCheckConfig``.
"""

import sys

import moderngl

from blackhole.config import GLCheckConfig


def format_version(version_code: int) -> str:
    """Converte o código do moderngl (ex.: 330) em texto "3.3"."""
    return f"{version_code // 100}.{version_code % 100 // 10}"


def main() -> int:
    cfg = GLCheckConfig()
    required = format_version(cfg.min_version_code)

    print(f"Python (venv):  {sys.executable}")
    print(f"Python (base):  {sys._base_executable}")

    # Com require=, o moderngl já recusa (com exceção) um contexto abaixo da versão mínima.
    try:
        ctx = moderngl.create_standalone_context(require=cfg.min_version_code)
    except Exception as exc:  # moderngl/glcontext não expõem um tipo de erro específico
        print(f"ERRO: não foi possível criar um contexto OpenGL {required}: {exc}")
        return 1

    try:
        info = ctx.info
        renderer = info["GL_RENDERER"]
        print(f"OpenGL:         {format_version(ctx.version_code)} ({info['GL_VERSION']})")
        print(f"Vendor:         {info['GL_VENDOR']}")
        print(f"Renderer:       {renderer}")
    finally:
        ctx.release()

    # O driver nativo da NVIDIA começa o renderer com "NVIDIA". Camadas de tradução
    # (ex.: "D3D12 (NVIDIA GeForce ...)" do GLOn12) também contam como GPU errada.
    if not renderer.upper().startswith(cfg.expected_vendor.upper()):
        print()
        print(f"AVISO: o renderer não é {cfg.expected_vendor}; o contexto está na GPU errada.")
        print("Marque os dois executáveis acima como 'Alto desempenho' em")
        print("Configurações > Sistema > Tela > Gráficos (ou no painel de controle NVIDIA).")
    else:
        print(f"OK: OpenGL >= {required} na GPU {cfg.expected_vendor}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
