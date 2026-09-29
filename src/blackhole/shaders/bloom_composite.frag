#version 330 core
// Bloom (PLANO, seção 3.10): composição da cena HDR com o bloom, seguida de ACES + sRGB.
//
// B = tenda(nível 1)/N é a PSF de média 1 aplicada à fonte (a soma dos N níveis borrados
// dividida por N). Modos (ver bloom.py: MODES):
//   glare:    saída = (1 − k)·I + k·B   (sem limiar; a energia média se conserva)
//   clássico: saída = I + k·B           (B vem da fonte filtrada pelo limiar)
// I já é radiância × exposição (blackhole.frag com u_hdr_output), então a cadeia de saída
// usa exposição 1. u_encode = false grava a radiância linear (testes de energia).
//
// Triângulo de fullscreen.vert; uv por gl_FragCoord e pelo retângulo do viewport.

// ver bloom.py: MODES (código de u_mode)
const int MODE_GLARE = 0;
const int MODE_CLASSIC = 1;

// ver bloom.py: TENT_CENTER_WEIGHT, TENT_EDGE_WEIGHT, TENT_CORNER_WEIGHT (tenda 3×3)
const float TENT_CENTER_WEIGHT = 0.25;
const float TENT_EDGE_WEIGHT = 0.125;
const float TENT_CORNER_WEIGHT = 0.0625;

// ver scene.py: ACES_A ... ACES_E (ACES aproximado de Narkowicz)
const float ACES_A = 2.51;
const float ACES_B = 0.03;
const float ACES_C = 2.43;
const float ACES_D = 0.59;
const float ACES_E = 0.14;

// ver scene.py: SRGB_THRESHOLD ... SRGB_GAMMA (curva sRGB por partes, IEC 61966-2-1)
const float SRGB_THRESHOLD = 0.0031308;
const float SRGB_LINEAR_SLOPE = 12.92;
const float SRGB_SCALE = 1.055;
const float SRGB_OFFSET = 0.055;
const float SRGB_GAMMA = 2.4;

// Cena HDR (RGBA16F), do mesmo tamanho do viewport: lida texel a texel.
uniform sampler2D u_scene;
// Nível 1 da cadeia, com a soma dos N níveis (filtro linear, clamp-to-edge).
uniform sampler2D u_bloom;
// 1/tamanho do nível 1, em uv por texel.
uniform vec2 u_bloom_texel;
// Viewport em pixels do alvo: (x, y, largura, altura), origem embaixo à esquerda.
uniform vec4 u_viewport;
uniform int u_mode;
// k_g ou k_c (ver config.py: BloomConfig.glare_strength, classic_strength).
uniform float u_strength;
// N: número de níveis da cadeia (normaliza a PSF para média 1).
uniform float u_level_count;
uniform bool u_encode;

out vec4 f_color;

// ver scene.py: aces
float aces(float x) {
    float y = (x * (ACES_A * x + ACES_B)) / (x * (ACES_C * x + ACES_D) + ACES_E);
    return clamp(y, 0.0, 1.0);
}

// ver scene.py: srgb_encode
float srgb_encode(float x) {
    if (x <= SRGB_THRESHOLD) {
        return SRGB_LINEAR_SLOPE * x;
    }
    return SRGB_SCALE * pow(x, 1.0 / SRGB_GAMMA) - SRGB_OFFSET;
}

// ver scene.py: encode_pixel
vec3 encode_pixel(vec3 rgb, float exposure) {
    return vec3(
        srgb_encode(aces(exposure * rgb.r)),
        srgb_encode(aces(exposure * rgb.g)),
        srgb_encode(aces(exposure * rgb.b))
    );
}

// Tenda 3×3 no nível 1, com passo de 1 texel dele (a mesma de bloom_up.frag).
vec3 tent(vec2 uv) {
    vec2 s = u_bloom_texel;
    vec3 center = texture(u_bloom, uv).rgb;
    vec3 edges = texture(u_bloom, uv + vec2(-s.x, 0.0)).rgb
        + texture(u_bloom, uv + vec2(s.x, 0.0)).rgb
        + texture(u_bloom, uv + vec2(0.0, -s.y)).rgb
        + texture(u_bloom, uv + vec2(0.0, s.y)).rgb;
    vec3 corners = texture(u_bloom, uv + vec2(-s.x, -s.y)).rgb
        + texture(u_bloom, uv + vec2(s.x, -s.y)).rgb
        + texture(u_bloom, uv + vec2(-s.x, s.y)).rgb
        + texture(u_bloom, uv + vec2(s.x, s.y)).rgb;
    return TENT_CENTER_WEIGHT * center + TENT_EDGE_WEIGHT * edges
        + TENT_CORNER_WEIGHT * corners;
}

void main() {
    vec2 local = gl_FragCoord.xy - u_viewport.xy;
    vec3 scene = texelFetch(u_scene, ivec2(local), 0).rgb;
    vec3 bloom = tent(local / u_viewport.zw) / u_level_count;
    vec3 color;
    if (u_mode == MODE_CLASSIC) {
        color = scene + u_strength * bloom;
    } else {
        color = (1.0 - u_strength) * scene + u_strength * bloom;
    }
    if (u_encode) {
        // A exposição já está na cena HDR: a cadeia de saída usa exposição 1.
        f_color = vec4(encode_pixel(color, 1.0), 1.0);
    } else {
        f_color = vec4(color, 1.0);
    }
}
