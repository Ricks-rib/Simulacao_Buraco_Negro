#version 330 core
// Bloom (PLANO, seção 3.10): upsample de um nível da cadeia de mips.
//
// Tenda 3×3 (pesos 1 2 1 / 2 4 2 / 1 2 1, divididos por 16) sobre o nível menor, somada
// ao nível de destino por blend aditivo (ONE, ONE; ver bloom.py: BloomPass). Depois do
// último upsample, o nível 1 guarda a soma dos N níveis borrados (Jimenez, 2014).
//
// Triângulo de fullscreen.vert, viewport (0, 0, tamanho do nível): uv por gl_FragCoord.

// ver bloom.py: TENT_CENTER_WEIGHT, TENT_EDGE_WEIGHT, TENT_CORNER_WEIGHT (tenda 3×3)
const float TENT_CENTER_WEIGHT = 0.25;
const float TENT_EDGE_WEIGHT = 0.125;
const float TENT_CORNER_WEIGHT = 0.0625;

// Nível menor (já acumulado), com filtro linear e clamp-to-edge.
uniform sampler2D u_source;
// 1/tamanho da fonte, em uv por texel: os taps da tenda ficam a 1 texel da fonte.
uniform vec2 u_source_texel;
// Tamanho do nível de destino, em pixels.
uniform vec2 u_target_size;

out vec4 f_color;

// Tenda 3×3 em uv, com passo de 1 texel da fonte.
vec3 tent(vec2 uv) {
    vec2 s = u_source_texel;
    vec3 center = texture(u_source, uv).rgb;
    vec3 edges = texture(u_source, uv + vec2(-s.x, 0.0)).rgb
        + texture(u_source, uv + vec2(s.x, 0.0)).rgb
        + texture(u_source, uv + vec2(0.0, -s.y)).rgb
        + texture(u_source, uv + vec2(0.0, s.y)).rgb;
    vec3 corners = texture(u_source, uv + vec2(-s.x, -s.y)).rgb
        + texture(u_source, uv + vec2(s.x, -s.y)).rgb
        + texture(u_source, uv + vec2(-s.x, s.y)).rgb
        + texture(u_source, uv + vec2(s.x, s.y)).rgb;
    return TENT_CENTER_WEIGHT * center + TENT_EDGE_WEIGHT * edges
        + TENT_CORNER_WEIGHT * corners;
}

void main() {
    f_color = vec4(tent(gl_FragCoord.xy / u_target_size), 1.0);
}
