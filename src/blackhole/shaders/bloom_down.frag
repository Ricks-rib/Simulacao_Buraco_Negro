#version 330 core
// Bloom (PLANO, seção 3.10): downsample de um nível da cadeia de mips.
//
// Filtro de 13 taps de "Next Generation Post Processing in Call of Duty" (Jimenez, 2014):
// com filtragem linear, cada tap cobre 2×2 texels da fonte e o conjunto cobre 6×6, o que
// evita a cintilação de estrelas subpixel de um box 2×2. Os pesos somam 1, então a média
// se conserva (energia do modo glare). Sem a média de Karis, que quebraria a conservação.
//
// No primeiro nível (u_first_level) cada tap lê os seus 4 texels da cena um a um e os
// interpola como o texture() faria: antes da média, valores não finitos (NaN, inf) viram
// 0 (um pixel ruim não se espalha nem apaga os vizinhos) e, no clássico
// (u_prefilter = PREFILTER_THRESHOLD), cada texel passa pelo filtro de brilho
// c·max(0, L − limiar)/L, na radiância já multiplicada pela exposição.
//
// Triângulo de fullscreen.vert, viewport (0, 0, tamanho do nível): uv por gl_FragCoord.

// ver scene.py: LUMINANCE_WEIGHTS (luminância relativa em sRGB linear, Rec. 709)
const vec3 LUMINANCE_WEIGHTS = vec3(0.2126, 0.7152, 0.0722);

// ver bloom.py: DOWN_CENTER_WEIGHT ... DOWN_CORNER_WEIGHT (pesos do filtro de 13 taps)
const float DOWN_CENTER_WEIGHT = 0.125;
const float DOWN_INNER_WEIGHT = 0.125;
const float DOWN_EDGE_WEIGHT = 0.0625;
const float DOWN_CORNER_WEIGHT = 0.03125;

// ver bloom.py: PREFILTER_NONE, PREFILTER_THRESHOLD (código de u_prefilter)
const int PREFILTER_NONE = 0;
const int PREFILTER_THRESHOLD = 1;

// Nível anterior (ou a cena HDR, no primeiro nível), com filtro linear e clamp-to-edge.
uniform sampler2D u_source;
// 1/tamanho da fonte, em uv por texel.
uniform vec2 u_source_texel;
// Tamanho do nível de destino, em pixels.
uniform vec2 u_target_size;
// Primeiro nível (fonte = cena HDR): guarda e filtro de brilho por texel.
uniform bool u_first_level;
uniform int u_prefilter;
// Limiar de luminância linear do clássico (ver config.py: BloomConfig.threshold).
uniform float u_threshold;

out vec4 f_color;

// Filtro de brilho do clássico: c·max(0, L − limiar)/L; L <= 0 (sombra) dá 0.
vec3 bright_pass(vec3 c) {
    float luminance = dot(c, LUMINANCE_WEIGHTS);
    if (luminance <= 0.0) {
        return vec3(0.0);
    }
    return c * (max(luminance - u_threshold, 0.0) / luminance);
}

// Um texel da cena no primeiro nível: não finito vira 0 e, no clássico, o filtro de
// brilho vale por pixel (antes de qualquer média).
vec3 first_level_texel(ivec2 p, ivec2 size) {
    vec3 c = texelFetch(u_source, clamp(p, ivec2(0), size - 1), 0).rgb;
    if (any(isnan(c)) || any(isinf(c))) {
        return vec3(0.0);
    }
    if (u_prefilter == PREFILTER_THRESHOLD) {
        return bright_pass(c);
    }
    return c;
}

vec3 tap(vec2 uv, vec2 offset_texels) {
    vec2 at = uv + offset_texels * u_source_texel;
    if (!u_first_level) {
        return texture(u_source, at).rgb;
    }
    // Primeiro nível: a mesma interpolação bilinear do texture() (clamp-to-edge), mas
    // com os 4 texels lidos um a um, para o filtro e a guarda valerem por pixel.
    ivec2 size = textureSize(u_source, 0);
    vec2 p = at * vec2(size) - 0.5;
    vec2 base = floor(p);
    vec2 f = p - base;
    ivec2 i = ivec2(base);
    vec3 c00 = first_level_texel(i, size);
    vec3 c10 = first_level_texel(i + ivec2(1, 0), size);
    vec3 c01 = first_level_texel(i + ivec2(0, 1), size);
    vec3 c11 = first_level_texel(i + ivec2(1, 1), size);
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_target_size;

    vec3 a = tap(uv, vec2(-2.0, 2.0));
    vec3 b = tap(uv, vec2(0.0, 2.0));
    vec3 c = tap(uv, vec2(2.0, 2.0));
    vec3 d = tap(uv, vec2(-2.0, 0.0));
    vec3 e = tap(uv, vec2(0.0, 0.0));
    vec3 f = tap(uv, vec2(2.0, 0.0));
    vec3 g = tap(uv, vec2(-2.0, -2.0));
    vec3 h = tap(uv, vec2(0.0, -2.0));
    vec3 i = tap(uv, vec2(2.0, -2.0));
    vec3 j = tap(uv, vec2(-1.0, 1.0));
    vec3 k = tap(uv, vec2(1.0, 1.0));
    vec3 l = tap(uv, vec2(-1.0, -1.0));
    vec3 m = tap(uv, vec2(1.0, -1.0));

    vec3 color = DOWN_CENTER_WEIGHT * e
        + DOWN_INNER_WEIGHT * (j + k + l + m)
        + DOWN_EDGE_WEIGHT * (b + d + f + h)
        + DOWN_CORNER_WEIGHT * (a + c + g + i);
    f_color = vec4(color, 1.0);
}
