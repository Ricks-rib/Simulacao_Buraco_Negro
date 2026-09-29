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

O app em tempo real chega na Fase 4b. As imagens e os vídeos gerados vão para `outputs/`, que fica fora do git.

### Shader e comparação CPU × GPU

Os shaders ficam em `src/blackhole/shaders/` (`fullscreen.vert` e `blackhole.frag`). O `blackhole.frag` replica a física de `physics.py`, a câmera de `camera.py` e a cena de `scene.py` em float32; cada função traz o comentário `// ver <arquivo>.py: <nome>`. Mudou de um lado, muda do outro.

O `compare` renderiza o fundo em xadrez sem e com disco na CPU (float64) e na GPU (float32), com a mesma câmera, resolução e integrador (o preset de tempo real de `GpuConfig`), e confere o aceite da Fase 4: no máximo 1% dos pixels com diferença acima de 10/255 em algum canal e o raio da sombra igual em ±1 px. Confirme antes com o `check_gl` que o renderer é a NVIDIA.

```powershell
uv run python -m blackhole.compare
uv run python -m blackhole.compare --r-cam 10 --steps 1024
```

Opções: `--width`, `--height`, `--steps` (256 a 1024), `--r-cam` e `--inclination`. O comando imprime o renderer, as métricas e os tempos de CPU e GPU, grava `compare_cpu.png`, `compare_gpu.png` e `compare_diff.png` em `outputs/` (as duas cenas empilhadas, e os pixels ruins em vermelho na diferença) e sai com 0 dentro do aceite, 1 fora dele ou sem contexto OpenGL 3.3, ou 2 com argumentos inválidos.

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
