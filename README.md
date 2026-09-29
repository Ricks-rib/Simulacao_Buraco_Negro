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
| Vídeo da órbita (extra `video`) | `uv run python -m blackhole.record` |

As imagens e os vídeos gerados vão para `outputs/`, que fica fora do git.

### Render CPU

```powershell
uv run python -m blackhole.render_cpu
uv run python -m blackhole.render_cpu --texture --time 100
uv run python -m blackhole.render_cpu --no-doppler --supersample 2
```

Por padrão, o disco tem o fator de desvio g da seção 3.8 do plano (Doppler e redshift): o lado que se aproxima fica mais claro e mais azul. Com o Doppler ligado, as estrelas de fundo recebem o blueshift do observador estático. Opções da Fase 5: `--no-doppler` desliga o Doppler (visual "Interstellar"), `--texture` liga a textura co-rotante do disco e `--time` dá o tempo coordenado da câmera t_obs, em unidades r_s/c, para a posição da textura. As demais opções são `--width`, `--height`, `--supersample`, `--r-cam`, `--fov`, `--inclination`, `--azimuth`, `--background`, `--no-disk`, `--exposure`, `--t-peak` e `--output`.

### App em tempo real

```powershell
uv run python -m blackhole.app
uv run python -m blackhole.app --scale 0.75
uv run python -m blackhole.app --dev
uv run python -m blackhole.app --benchmark
```

A janela abre em 1280×720, redimensionável, e a imagem sempre preenche o buffer físico (sem tarjas, inclusive com escala DPI do Windows acima de 100%). A cena é renderizada no tamanho interno (buffer × escala) e esticada na janela com filtro linear. O HUD, no canto superior esquerdo, mostra FPS, passos, r_cam, inclinação, azimute, escala, tamanho interno e do buffer físico, exposição, disco, fundo, Doppler, textura, o tempo t_obs e o estado e o modo do bloom. Se o renderer não for NVIDIA, o app avisa no console e no HUD.

| Entrada | Ação |
|---|---|
| Arrastar com o mouse / setas | órbita ao redor do buraco negro |
| Roda do mouse / `W` `S` | distância (2 ≤ r_cam ≤ 500, zoom proporcional à distância) |
| `D` | disco ligado/desligado |
| `B` | Doppler e redshift ligados/desligados (desligado = visual "Interstellar") |
| `T` | textura co-rotante do disco ligada/desligada (os anéis internos giram mais rápido) |
| `L` | bloom ligado/desligado |
| `K` | bloom glare ↔ clássico (vale também com o bloom desligado) |
| `G` | alternância entre xadrez e estrelas |
| `+` / `-` (inclusive no teclado numérico) | passos de integração (256 a 1024, de 128 em 128) |
| `[` / `]` | exposição |
| `H` | HUD ligado/desligado |
| `R` | câmera na posição inicial |
| `F12` | captura em `outputs/captura_<data>_<hora>.png` (tamanho do buffer, escala 1,0, com bloom, sem HUD) |
| `Esc` | saída |

A inclinação fica entre 1° e 179°, longe dos polos. O app começa com o Doppler ligado e a textura desligada (`DiskConfig`). O relógio da textura avança `AppConfig.time_scale` M por segundo (40 M/s por padrão: a borda interna dá uma volta em cerca de 2,3 s). Todos os parâmetros (sensibilidades, limites, HUD, benchmark, relógio) estão em `AppConfig`, no `config.py`, e os da textura em `DiskTextureConfig`.

O bloom (seção 3.10 do plano) só existe no app: a cena é renderizada em HDR linear e uma cadeia de mips espalha a luz antes do tone mapping. O modo **glare** (padrão) espalha uma fração fixa de toda a luz, sem limiar, e conserva a energia média, como a PSF de uma lente ou do olho. O modo **clássico** só espalha o excesso de luminância acima de um limiar e soma energia. O app abre com o bloom ligado em glare; os parâmetros (forças, limiar, níveis) estão em `BloomConfig`. O render CPU e o `compare` não têm bloom.

Opções:

- `--scale`: escala de resolução interna, fixa durante a execução (0,25 a 2,0; padrão 1,0). Fora da faixa, o comando sai com erro de argumento (código 2).
- `--dev`: modo desenvolvimento. O app recompila o `blackhole.frag` (e o `fullscreen.vert`) quando o arquivo muda. Com erro de compilação, o programa anterior continua em uso e o erro é impresso no console uma vez por mudança.
- `--benchmark`: não abre janela. Renderiza o pipeline completo (cena no FBO interno, apresentação e HUD) num alvo offscreen de 1920×1080, com 512 passos, Doppler, textura e bloom ligados e `ctx.finish()` por quadro, descarta o aquecimento, imprime o renderer e o FPS médio e sai com 0 se o FPS for ≥ 60 (1 caso contrário). Não depende do monitor, da escala DPI nem do vsync. Confirme antes com o `check_gl` que o renderer é a NVIDIA.

### Shader e comparação CPU × GPU

Os shaders ficam em `src/blackhole/shaders/` (`fullscreen.vert`, `blackhole.frag`, o `present.frag` do app e os `bloom_*.frag` do bloom). O `blackhole.frag` replica a física de `physics.py`, a câmera de `camera.py` e a cena de `scene.py` em float32; cada função traz o comentário `// ver <arquivo>.py: <nome>`. Mudou de um lado, muda do outro.

O `compare` renderiza o fundo em xadrez sem e com disco (Doppler de `DiskConfig`, textura desligada) na CPU (float64) e na GPU (float32), com a mesma câmera, resolução e integrador (o preset de tempo real de `GpuConfig`), e confere o aceite da Fase 4: no máximo 1% dos pixels com diferença acima de 10/255 em algum canal e o raio da sombra igual em ±1 px. Confirme antes com o `check_gl` que o renderer é a NVIDIA.

```powershell
uv run python -m blackhole.compare
uv run python -m blackhole.compare --r-cam 10 --steps 1024
```

Opções: `--width`, `--height`, `--steps` (256 a 1024), `--r-cam` e `--inclination`. O comando imprime o renderer, as métricas e os tempos de CPU e GPU, grava `compare_cpu.png`, `compare_gpu.png` e `compare_diff.png` em `outputs/` (as duas cenas empilhadas, e os pixels ruins em vermelho na diferença) e sai com 0 dentro do aceite, 1 fora dele ou sem contexto OpenGL 3.3, ou 2 com argumentos inválidos.

### Vídeo da câmera orbitando

Precisa do extra `video` (`imageio` e `imageio-ffmpeg`). Sem ele, o comando não grava nada, pede o `uv sync --extra video` e sai com 1; o resto do projeto e os testes não dependem dele (os testes de gravação dão skip).

```powershell
uv sync --extra video
uv run python -m blackhole.record
uv run python -m blackhole.record --duration 2 --width 1280 --height 720
uv run python -m blackhole.record --no-texture --no-doppler --no-bloom --output outputs/orbita.mp4
```

Não abre janela: cria um contexto OpenGL sem janela, renderiza cada quadro pelo mesmo caminho da captura `F12` do app (escala 1,0, sem HUD) e grava um MP4 em `outputs/orbita_<data>_<hora>.mp4` (H.264, yuv420p, CRF 18). Padrão: 1920×1080, 60 fps, 20 s (1 200 quadros). A câmera dá uma volta completa em azimute enquanto a inclinação oscila uma vez entre 65° e 95° (80° ± 15°), passando acima e abaixo do plano do disco; a volta fecha, então o vídeo roda em loop sem salto de posição. A textura co-rotante do disco fica ligada, e Doppler e bloom seguem os padrões do app. O relógio da textura avança `AppConfig.time_scale` M por segundo de vídeo.

Opções: `--width` e `--height` (pares, por causa do yuv420p), `--fps`, `--duration` (s), `--output`, `--no-texture`, `--no-doppler` e `--no-bloom`. Argumentos inválidos saem com 2; sem o extra, sem contexto OpenGL 3.3 ou com erro na gravação, com 1 (o arquivo parcial é apagado). Os parâmetros (codec, CRF, trajetória, prefixo do arquivo) estão em `VideoConfig`, no `config.py`. Confirme antes com o `check_gl` que o renderer é a NVIDIA.

Limitação: cada quadro é o de um observador **estático** na posição da trajetória (a mesma câmera do app), não o de um observador em órbita. Não há aberração nem Doppler da câmera; numa órbita real em r = 30 (v ≈ 0,13c) a aberração seria visível. O observador em movimento está fora do escopo (seção 7 do plano).

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
