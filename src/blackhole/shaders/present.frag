#version 330 core
// Passe de apresentação do app (PLANO, Fase 4): copia uma textura para o retângulo do
// viewport atual, esticando-a (cena na escala interna → buffer da janela) ou 1:1 (HUD).
// Usa o triângulo de tela cheia de fullscreen.vert, que não tem varyings: a coordenada
// da textura sai de gl_FragCoord e do retângulo do viewport (ver app.py: FramePipeline).
// Não há física aqui; a imagem da cena já sai de blackhole.frag em sRGB.

uniform sampler2D u_texture;
// Viewport em pixels do framebuffer: (x, y, largura, altura), origem embaixo à esquerda.
uniform vec4 u_viewport;

out vec4 f_color;

void main() {
    vec2 uv = (gl_FragCoord.xy - u_viewport.xy) / u_viewport.zw;
    f_color = texture(u_texture, uv);
}
