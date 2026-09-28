# Buraco Negro

Simulação 3D em tempo real de um buraco negro de Schwarzschild (sombra, lente gravitacional e disco de acreção), feita por ray tracing de geodésicas nulas. O plano completo, com a física, as fases e os critérios de aceite, está em [PLANO_BURACO_NEGRO.md](PLANO_BURACO_NEGRO.md).

## Requisitos

- [uv](https://docs.astral.sh/uv/). Ele gerencia o Python, o ambiente virtual e as dependências.
- Python 3.12, fixado em `.python-version`. Se não estiver instalado, o `uv` baixa sozinho.
- GPU com OpenGL 3.3 ou superior. A referência é uma NVIDIA GeForce RTX 5070.

Não use `pip`, `python -m venv` nem ative o `.venv` à mão: tudo passa pelo `uv`.

## Instalação

```powershell
uv sync
```

Esse comando cria o `.venv` e instala as dependências nas versões travadas no `uv.lock`. O pacote `blackhole` é instalado em modo editável. Para incluir o extra de gravação de vídeo, que só é usado na Fase 5:

```powershell
uv sync --extra video
```

O `uv sync` deixa o ambiente idêntico ao que foi pedido. Por isso, um `uv sync` sem `--extra video` desinstala o `imageio` e o `imageio-ffmpeg`. Enquanto precisar do vídeo, passe o `--extra video` em todo `uv sync`.

## Execução

| Tarefa | Comando |
|---|---|
| Verificar a GPU | `uv run python -m blackhole.check_gl` |
| Rodar os testes | `uv run pytest` |
| Render CPU | `uv run python -m blackhole.render_cpu` |
| App em tempo real | `uv run python -m blackhole.app` |
| Comparar CPU e GPU | `uv run python -m blackhole.compare` |

Os módulos de render, do app e da comparação chegam nas próximas fases. As imagens e os vídeos gerados vão para `outputs/`, que fica fora do git.

## GPU dedicada (notebooks e máquinas com GPU híbrida)

Em máquinas com duas GPUs, como uma RTX 5070 e uma Radeon integrada, o Windows pode entregar o contexto OpenGL à GPU integrada. Para verificar qual está em uso, rode:

```powershell
uv run python -m blackhole.check_gl
```

O comando mostra a versão do OpenGL, o vendor, o renderer e os dois executáveis do Python. Se o renderer não for NVIDIA, ele emite um aviso. Nesse caso:

1. Abra **Configurações → Sistema → Tela → Gráficos**.
2. Adicione os **dois** executáveis que o `check_gl` mostra: `Python (venv)` e `Python (base)`. O `.venv\Scripts\python.exe` só inicia o interpretador base, por isso os dois precisam ser marcados.
3. Em cada um, escolha **Alto desempenho**. Outra opção é configurar os dois no painel de controle NVIDIA.
4. Rode o `check_gl` de novo e confirme que o renderer aparece como `NVIDIA GeForce RTX 5070`.
