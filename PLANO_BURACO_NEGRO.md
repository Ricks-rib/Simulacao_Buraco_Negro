# Plano de Implementação: Simulação 3D de Buraco Negro com Sombra

## 1. Objetivo

Renderização interativa em 3D de um buraco negro de Schwarzschild (sem rotação), com:

- sombra e anel de fótons fisicamente corretos;
- lente gravitacional sobre fundo estrelado;
- disco de acreção visível por cima e por baixo da sombra (efeito "Interstellar");
- câmera orbital controlada por mouse e teclado, em tempo real.

## 2. Decisão de arquitetura

A técnica é ray tracing de geodésicas nulas, com um raio de luz integrado por pixel, de trás para a frente, da câmera até o destino. O custo é de cerca de 10⁶ pixels × centenas de passos de integração por quadro, o que é inviável em Python puro em tempo real.

A estratégia usa duas camadas:

- **Renderizador de referência em CPU** (NumPy + Numba, float64): valida a física e gera imagens estáticas de alta qualidade. É lento, mas é a fonte da verdade.
- **Renderizador em tempo real em GPU** (fragment shader GLSL via ModernGL, float32): recebe a mesma física, portada para o shader. O Python cuida da janela, da câmera, dos controles e dos parâmetros.

Critério de consistência: a saída do shader é comparada com a da CPU para a mesma câmera, resolução e número máximo de passos (seção 6, Fase 4).

## 3. Modelo físico

### 3.1 Unidades e valores de referência

Unidades: G = c = 1, com raio de Schwarzschild r_s = 1 (portanto massa M = 1/2).

| Grandeza | Valor |
|---|---|
| Horizonte de eventos | r = 1 |
| Esfera de fótons | r = 1,5 |
| Parâmetro de impacto crítico (raio da sombra vista de longe) | b_c = 3√3/2 ≈ 2,598 |
| Borda interna do disco (ISCO) | r = 3 |
| Raio angular da sombra para observador estático em r (fórmula de Synge) | sin α = (b_c / r) · √(1 − 1/r) |
| Velocidade angular kepleriana no disco | Ω(r) = √(M / r³) = √(1 / (2r³)) |

Eixo de simetria do disco: +y. O disco gira com momento angular no sentido +y.

### 3.2 Equação de movimento do fóton

A forma cartesiana é equivalente à geodésica nula de Schwarzschild:

```
a = −(3/2) · h² · x / |x|⁵
h = |x × v|   (constante ao longo do raio)
```

Interpretação exata: com x = r · r̂ e parâmetro afim λ, a velocidade cartesiana é v = (dr/dλ) · r̂ + r · (dφ/dλ) · φ̂. A órbita r(φ) coincide com a geodésica (equação de Binet u'' + u = (3/2) u², com u = 1/r).

Invariantes do raio:

| Invariante | Expressão | Uso |
|---|---|---|
| Momento angular | h = \|x × v\| | teste de conservação |
| "Energia" | ε = \|v\|²/2 − h²/(2r³) | teste de conservação |
| Energia no infinito | E = √(\|v\|² − h²/r³) = √(2ε) | cálculo de b e do Doppler |
| Parâmetro de impacto | b = h / E | classificação do raio |
| Componente axial do momento angular | L_y = (x × v) · ŷ | Doppler (Fase 5) |

### 3.3 Câmera (observador estático)

A câmera é um observador estático em r_cam, com projeção pinhole no seu referencial local. As direções do pixel **não** podem ser usadas diretamente como v cartesiano, porque a fórmula de Synge vale no referencial local e a câmera "plana" erra o tamanho da sombra (≈ 1,7% em r = 30, ≈ 3,4% em r = 15, ≈ 5% em r = 10).

Conversão da direção local para v inicial (tétrade estática):

```
n     = direção unitária do pixel no referencial local
cos α = n · r̂                      (componente radial)
n_⊥   = n − (n · r̂) · r̂           (componente tangencial)
v₀    = √(1 − 1/r_cam) · cos α · r̂ + n_⊥
```

Com isso, b = r_cam · sin α / √(1 − 1/r_cam), que reproduz a fórmula de Synge.

Parâmetros padrão (em `config.py`):

| Parâmetro | Padrão |
|---|---|
| Distância inicial r_cam | 30 |
| Limites de distância | 2 ≤ r_cam ≤ 500 |
| FOV vertical | 60° |
| Inclinação inicial | 80° em relação ao eixo +y (levemente acima do disco) |
| Resolução do render CPU | 800×450 |

Projeção: um pixel com coordenada horizontal p (em pixels, a partir do centro) corresponde ao ângulo θ = atan(p · tan(FOV_h/2) / (W/2)). Essa relação é usada para medir a sombra nas imagens.

### 3.4 Condições de parada

- r < 1: raio capturado, pixel preto (sombra).
- r > r_max e raio se afastando (x · v > 0): raio escapou, com amostragem do fundo pela direção final de v. O valor é r_max = max(50, 2 · r_cam), para que a câmera nunca fique fora da região integrada.
- Cruzamento do plano do disco (y = 0) com r_in ≤ r ≤ r_out: cor do disco. O primeiro cruzamento é o válido (disco fino, opaco e visível dos dois lados).
- Limite máximo de passos atingido: o pixel é tratado como capturado. Viés conhecido: raios muito próximos de b_c ficam pretos e aumentam levemente a sombra. Por isso, CPU e GPU usam o mesmo limite na comparação.

### 3.5 Integrador

- RK4 com passo proporcional à distância (dt = k · r), com limites mínimo e máximo em `config.py`.
- Passo reduzido nas proximidades da esfera de fótons (1,2 < r < 2).
- Cruzamento do disco detectado por mudança de sinal de y entre passos, com interpolação linear do ponto de impacto.
- CPU em float64 e GPU em float32 (o GLSL 3.3 não tem float64). Nessa formulação não há singularidade em r = 1 (só em r = 0). O ponto crítico de precisão é a esfera de fótons, não o horizonte.

### 3.6 Disco de acreção

- r_in = 3 (ISCO) e r_out = 12, ambos configuráveis.
- Perfil de temperatura simplificado: T(r) = T_pico · f(r) / f(r_pico), com f(r) = r^(−3/4) · (1 − √(r_in/r))^(1/4). O pico fica em r_pico = (49/36) · r_in ≈ 4,08 e T = 0 em r_in.
- T_pico padrão: 10 000 K (configurável).
- Cor: cromaticidade de corpo negro por tabela precomputada (1 000 K a 40 000 K, normalizada em luminância).
- Brilho: I ∝ (T / T_pico)⁴, multiplicado pela exposição.
- Saída: tone mapping (ACES aproximado) seguido de correção de gama sRGB. A mesma cadeia se aplica ao fundo.

### 3.7 Fundo

- Modo depuração: esfera celeste em xadrez, em latitude/longitude, com 24 × 12 casas de 15°. Evidencia a distorção da lente e é o modo usado nas comparações CPU × GPU.
- Modo final: campo estelar procedural determinístico (hash inteiro da célula de direção, com a mesma função na CPU e na GPU) ou textura equiretangular (panorama da Via Láctea, com verificação de licença). Ele não entra na comparação numérica, porque float32 e float64 podem divergir no hash.

### 3.8 Doppler e redshift (Fase 5)

Fator de desvio para emissor em órbita kepleriana e observador estático:

```
g = √(1 − 1,5/r) / [ √(1 − 1/r_cam) · (1 − Ω(r) · λ) ]
λ = −L_y / E        (sinal negativo porque o raio é traçado de trás para a frente)
```

- √(1 − 1,5/r) = 1/u^t do emissor, que inclui redshift gravitacional e Doppler transverso.
- (1 − Ω·λ) é o Doppler longitudinal: o lado que se aproxima tem g > 1.
- 1/√(1 − 1/r_cam) é o blueshift da luz recebida pelo observador estático.
- Cor pela temperatura observada T_obs = g · T.
- Brilho ∝ g⁴ · T⁴ (intensidade bolométrica).
- Efeito esperado: lado do disco que se aproxima mais brilhante e azulado.

## 4. Stack

- `uv`: gerenciamento do Python, do ambiente virtual e das dependências (sem `pip` nem `venv` manual).
- Python 3.12, fixado com `uv python pin 3.12` (arquivo `.python-version`). O padrão do `py` nesta máquina é o 3.14, ainda arriscado para o Numba.
- `numpy`, `numba`: física e renderizador CPU.
- `pillow`, `matplotlib`: gravação e comparação de imagens.
- `moderngl`, `moderngl-window`: janela OpenGL 3.3+ e shader (a API do `moderngl-window` mudou na 3.x; o código segue a versão travada).
- `pytest`: testes da física, no grupo de desenvolvimento (`uv add --dev pytest`).
- `imageio`, `imageio-ffmpeg`: gravação de vídeo, como extra opcional `video` (Fase 5).

Dependências declaradas no `pyproject.toml` e versões exatas travadas no `uv.lock`, que é versionado no git. Comandos padrão:

| Tarefa | Comando |
|---|---|
| Criar ou atualizar o ambiente | `uv sync` |
| Adicionar dependência | `uv add <pacote>` |
| Rodar testes | `uv run pytest` |
| Rodar um módulo | `uv run python -m blackhole.<módulo>` |

### Sistema-alvo

| Componente | Máquina de desenvolvimento |
|---|---|
| Sistema | Windows 11 Pro (código multiplataforma) |
| CPU | AMD Ryzen 7 9800X3D (8 núcleos, 16 threads) |
| GPU | NVIDIA GeForce RTX 5070 12 GB (dedicada) + AMD Radeon integrada |
| RAM | 32 GB |

**Atenção, GPU híbrida:** o Windows pode entregar o contexto OpenGL à Radeon integrada. O `check_gl.py` informa o renderer em uso e emite um aviso se não for a NVIDIA. Correção: em Configurações → Sistema → Tela → Gráficos, o executável do Python é marcado como "Alto desempenho" (ou o painel de controle NVIDIA é usado). No Windows, o `.venv\Scripts\python.exe` apenas inicia o interpretador base; por isso, o `check_gl.py` também imprime `sys._base_executable`, e os dois executáveis são marcados.

## 5. Estrutura de pastas

O projeto fica na raiz do diretório atual (`buraco_negro/`), sem subpasta extra.

```
buraco_negro/
├── CLAUDE.md
├── PLANO_BURACO_NEGRO.md
├── README.md
├── pyproject.toml        # projeto uv: dependências, grupo dev, extra video, layout src/
├── uv.lock               # versões travadas (versionado)
├── .python-version       # 3.12 (uv python pin)
├── .gitignore            # pastas de agentes/skills, .venv/, outputs/, __pycache__/, cache do Numba
├── src/blackhole/
│   ├── __init__.py
│   ├── config.py         # parâmetros (dataclass): física, disco, câmera, qualidade, cor
│   ├── physics.py        # aceleração, passo RK4, integração de um raio, invariantes
│   ├── camera.py         # posição, orientação, FOV, tétrade estática → v inicial por pixel
│   ├── scene.py          # disco, corpo negro, tone mapping, fundo (xadrez/estrelas)
│   ├── render_cpu.py     # renderizador de referência (Numba, paralelo)
│   ├── app.py            # aplicação em tempo real (ModernGL)
│   ├── check_gl.py       # versão OpenGL, vendor e renderer da GPU
│   ├── compare.py        # comparação CPU × GPU (contexto sem janela)
│   └── shaders/
│       ├── fullscreen.vert
│       └── blackhole.frag
├── tests/
│   ├── test_physics.py
│   └── test_camera.py
├── assets/               # texturas de fundo
└── outputs/              # imagens e vídeos gerados (fora do git)
```

## 6. Fases

### Fase 0: Setup

- `git init` (o `.gitignore` já existe).
- `uv init --package` (layout `src/`, com build backend), `uv python pin 3.12`, `uv add` das dependências da seção 4 e `uv sync`. O `uv` instala o próprio pacote em modo editável.
- README com instruções de instalação (`uv sync`), execução (`uv run ...`) e configuração da GPU dedicada.
- `check_gl.py`: impressão da versão OpenGL, do vendor, do renderer e do executável base do Python, com aviso se o renderer não for NVIDIA.

**Aceite:**
- `uv run python -m blackhole.check_gl` informa OpenGL ≥ 3.3 e renderer NVIDIA GeForce RTX 5070.
- `uv run pytest` executa sem erro (ainda sem testes).
- `uv.lock` gerado e versionado.

### Fase 1: Física e testes (CPU)

- `physics.py`: constantes, aceleração, passo RK4, integração de um raio com as condições de parada da seção 3.4 e funções dos invariantes da seção 3.2. Numba com `cache=True`.
- Raios de teste: início em x = (−R, b, 0) com v = (1, 0, 0), R = 10⁴ e r_max = 2 · 10⁴ (parâmetros próprios dos testes, independentes do padrão de renderização).
- Testes em `test_physics.py`:
  - raio com b = 2,55 capturado; raio com b = 2,65 escapa;
  - campo fraco, b = 50: deflexão comparada com α = 2/b + 15π/(16 b²), erro < 1%;
  - campo forte, b = b_c · (1 + δ) com δ = 10⁻⁴ e 10⁻⁶: deflexão comparada com a aproximação logarítmica α = −ln(δ) + ln[216(7 − 4√3)] − π, erro absoluto < 0,05 rad;
  - com δ = 10⁻⁶: ângulo total percorrido > 4π (várias voltas perto de r = 1,5), tanto para o raio que escapa (+δ) quanto para o capturado (−δ);
  - conservação de h e de ε ao longo do raio, com erro relativo < 1e-4.

**Aceite:** todos os testes passando.

### Fase 2: Renderizador CPU de referência

- `camera.py`: posição, alvo, vetor up, FOV e conversão pela tétrade estática da seção 3.3.
- `render_cpu.py`: loop paralelo por pixel (`numba.prange`), com sombra preta, fundo em xadrez e supersampling opcional (1×1 para medidas, 2×2 para imagens finais).
- Saída em PNG na pasta `outputs/`.
- Testes em `test_camera.py`:
  - o raio disparado no ângulo de Synge tem b = b_c (erro relativo < 1e-6);
  - raio angular da sombra renderizada em r_cam = 10, 30 e 100 compatível com a fórmula de Synge (erro < 2%), medido na linha central com a relação pixel → ângulo da seção 3.3.

**Aceite:**
- Testes das Fases 1 e 2 passando.
- Distorção visível do xadrez e anel de Einstein presente.
- Imagem de 800×450 (1×1) gerada em ≤ 30 s, com o cache do Numba já compilado.

### Fase 3: Disco de acreção (CPU)

- `scene.py`: parâmetros do disco, perfil de temperatura, tabela de corpo negro, exposição, tone mapping, gama sRGB e fundo estrelado procedural.
- Câmera na inclinação padrão de 80° em relação ao eixo.

**Aceite:** imagem com disco deformado passando por cima da sombra, imagem secundária do disco visível por baixo e anel de fótons fino na borda da sombra.

### Fase 4: Tempo real (GPU)

- `fullscreen.vert`: quad de tela cheia.
- `blackhole.frag`: a mesma física da CPU, com loop de passos fixo (256 a 1024, configurável), passo adaptativo e saída antecipada.
- `app.py`: janela com `moderngl-window`. Uniforms: posição e orientação da câmera, FOV, parâmetros do disco, exposição, modo de fundo e número de passos.
- Controles (câmera orbital):

| Entrada | Ação |
|---|---|
| Arrastar com mouse / setas | órbita ao redor do buraco negro |
| Roda do mouse / `W` `S` | distância (limitada a 2 ≤ r_cam ≤ 500) |
| `D` | disco ligado/desligado |
| `G` | alternância entre xadrez e estrelas |
| `+` / `−` (inclusive no teclado numérico) | qualidade (número de passos) |
| `[` / `]` | exposição |
| `H` | HUD ligado/desligado |
| `R` | câmera na posição inicial |
| `F12` | captura de tela para `outputs/` |
| `Esc` | saída |

- HUD com FPS, número de passos, r_cam, inclinação e modos ativos.
- Escala de resolução interna, para manter o FPS.
- Tratamento de redimensionamento da janela e de DPI alto do Windows.
- Recarga automática do shader quando o `.frag` muda (modo desenvolvimento).
- `compare.py`: contexto ModernGL sem janela (standalone), render CPU × GPU no modo xadrez, com mesma câmera, resolução (escala 1,0) e número máximo de passos.

**Aceite:**
- ≥ 60 FPS em 1920×1080, escala 1,0 e 512 passos, na RTX 5070 (confirmada pelo `check_gl`).
- `compare.py`: raio da sombra idêntico em ±1 px, e no máximo 1% dos pixels com diferença > 10/255 em algum canal (esperada só na vizinhança do anel de fótons).

### Fase 5: Realismo (opcional)

- Fator de desvio g da seção 3.8, com tecla `B` para ligar e desligar (o visual de Interstellar omite o Doppler de propósito).
- Pós-processamento com bloom.
- Opcional: textura procedural no disco girando com velocidade kepleriana (disco ainda fino).
- Gravação de vídeo com a câmera orbitando o buraco negro (`imageio-ffmpeg`).

## 7. Fora do escopo

- Buraco negro com rotação (métrica de Kerr), como possível evolução futura.
- Câmera em voo livre (WASD) e observador em movimento (aberração relativística), como possível evolução futura.
- Disco volumétrico ou turbulento.
- Relatividade numérica (fusões e ondas gravitacionais).

## 8. Riscos

| Risco | Mitigação |
|---|---|
| Contexto OpenGL na GPU integrada | aviso no `check_gl.py` e configuração de "Alto desempenho" no Windows |
| Incompatibilidade do Numba com Python recente | Python fixado em 3.12 com `uv python pin` |
| Precisão float32 perto da esfera de fótons | passo reduzido em 1,2 < r < 2; comparação com a CPU em float64 |
| Limite de passos escurece raios quase críticos | mesmo limite na CPU e na GPU ao comparar; qualidade ajustável |
| Anel de fótons serrilhado | supersampling na CPU; escala de resolução > 1 opcional na GPU |
| Mudanças de API do `moderngl-window` | versão travada no `uv.lock` |

## 9. Instruções de execução para o Claude Code

- Regras permanentes do projeto em `CLAUDE.md`.
- Execução fase a fase, com parada ao fim de cada fase para revisão e um resumo do que foi feito e de como testar.
- Commit ao fim de cada fase, após a revisão do usuário.
- Nenhuma fase iniciada com testes da fase anterior falhando.
- Física centralizada: constantes e fórmulas documentadas em `physics.py` e replicadas no shader, com comentário de referência cruzada.
- Parâmetros concentrados em `config.py`, sem números mágicos espalhados.
- Sem otimização prematura antes da Fase 4.
- Em caso de dúvida sobre a física: pergunta antes de qualquer suposição.
