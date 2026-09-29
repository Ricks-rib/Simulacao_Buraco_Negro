#version 330 core
// Buraco negro de Schwarzschild por ray tracing de geodésicas nulas (PLANO, Fase 4).
//
// Port 1:1, em float32, da referência da CPU (float64): cada função traz o comentário
// "ver <arquivo>.py: <nome>" da função Python que replica. Mudou de um lado, muda do outro.
// Unidades G = c = 1, r_s = 1 (M = 1/2); eixo do disco em +y.
//
// MAX_STEPS_CAP (limite de compilação do laço) é injetado por render_gpu.py com #define
// logo após a linha #version; o limite real é o uniform u_max_steps (<= MAX_STEPS_CAP).
#ifndef MAX_STEPS_CAP
#error "MAX_STEPS_CAP precisa ser definido por render_gpu.py"
#endif

// --- constantes (ver physics.py e scene.py; test_gpu.py confere os valores) ---

// ver physics.py: R_S (horizonte de eventos)
const float R_S = 1.0;
// ver physics.py: M (r_s = 2M)
const float M = 0.5;
// ver physics.py: ACCEL_COEFF (3M)
const float ACCEL_COEFF = 3.0 * M;
// ver physics.py: PEAK_RADIUS_RATIO (r_pico / r_in = 49/36)
const float PEAK_RADIUS_RATIO = 49.0 / 36.0;
// ver physics.py: CIRCULAR_ORBIT_COEFF (3M: u^t = 1/√(1 − 3M/r) da órbita circular)
const float CIRCULAR_ORBIT_COEFF = 3.0 * M;

// ver physics.py: CAPTURED, ESCAPED, DISK, MAX_STEPS (status do raio, seção 3.4)
const int CAPTURED = 0;
const int ESCAPED = 1;
const int DISK = 2;
const int MAX_STEPS = 3;

// ver render_cpu.py: BACKGROUNDS (código do fundo em u_background)
const int BG_CHECKER = 0;
const int BG_STARS = 1;

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

// ver scene.py: LOWBIAS32_M1, LOWBIAS32_M2, UNIT_SCALE (hash lowbias32)
const uint LOWBIAS32_M1 = 0x7FEB352Du;
const uint LOWBIAS32_M2 = 0x846CA68Bu;
const float UNIT_SCALE = 16777216.0;

// ver scene.py: FBM_LACUNARITY, FBM_GAIN, TWO_PI (textura do disco)
const int FBM_LACUNARITY = 2;
const float FBM_GAIN = 0.5;
const float TWO_PI = 6.283185307179586;

// --- uniforms (preenchidos por render_gpu.py: GpuRenderer.set_uniforms) ---

// Câmera: tamanho da imagem, posição e base ortonormal (ver camera.py: Camera).
uniform vec2 u_resolution;
uniform vec3 u_cam_pos;
uniform vec3 u_cam_forward;
uniform vec3 u_cam_right;
uniform vec3 u_cam_up;
// tan(FOV_v/2)/(H/2) (ver camera.py: Camera.pixel_scale).
uniform float u_pixel_scale;

// Raio de escape (ver config.py: RenderConfig.r_max_min, r_max_factor).
uniform float u_r_max_min;
uniform float u_r_max_factor;

// Integrador (ver config.py: IntegratorConfig, preset GpuConfig.integrator).
uniform int u_max_steps;
uniform float u_step_factor;
uniform float u_photon_step_factor;
uniform float u_photon_r_min;
uniform float u_photon_r_max;
uniform float u_dt_min;
uniform float u_dt_max;

// Disco (ver config.py: DiskConfig).
uniform bool u_disk_enabled;
uniform float u_disk_r_in;
uniform float u_disk_r_out;
uniform float u_disk_t_peak;
// Fator de desvio g (disco) e blueshift g_∞ (estrelas), por quadro (ver config.py:
// DiskConfig.doppler).
uniform bool u_doppler;
// Textura co-rotante do disco, por quadro (ver config.py: DiskConfig.texture), e tempo
// coordenado da câmera t_obs (unidades r_s/c).
uniform bool u_disk_texture;
uniform float u_time;
// Parâmetros fixos da textura (ver config.py: DiskTextureConfig).
uniform float u_tex_radial_cells;
uniform int u_tex_azimuthal_cells;
uniform int u_tex_octaves;
uniform float u_tex_contrast;
uniform uint u_tex_seed;

// Cor (ver config.py: ColorConfig). A tabela de corpo negro é uma textura N×1 RGB32F
// sem filtragem, lida com texelFetch (ver scene.py: blackbody_table).
uniform sampler2D u_bb_table;
uniform float u_bb_t_min;
uniform float u_bb_t_max;
uniform float u_exposure;
// Saída HDR linear para o bloom do app, por quadro (ver render_gpu.py: set_uniforms, hdr):
// ligada, grava u_exposure·radiância sem ACES/sRGB num alvo RGBA16F; desligada, a cadeia
// de saída completa num alvo RGBA8.
uniform bool u_hdr_output;

// Fundo (ver config.py: CheckerConfig, StarfieldConfig).
uniform int u_background;
uniform float u_cell_rad;
uniform vec3 u_checker_a;
uniform vec3 u_checker_b;
uniform float u_star_grid;
uniform float u_star_density;
uniform float u_star_jitter;
uniform uint u_star_seed;
uniform float u_star_sigma;
uniform float u_star_i_min;
uniform float u_star_i_max;
uniform float u_star_t_min;
uniform float u_star_t_max;

out vec4 f_color;

// --- câmera (seção 3.3) ---

// ver camera.py: pixel_offset
// gl_FragCoord.xy já é o centro do pixel (i + 0,5); a origem do framebuffer é embaixo,
// então q = gl_FragCoord.y − H/2 cresce para cima, como na CPU (render_gpu.py inverte a
// imagem lida na vertical).
vec2 pixel_offset(vec2 frag_coord, vec2 resolution) {
    return frag_coord - 0.5 * resolution;
}

// ver camera.py: pixel_direction
vec3 pixel_direction(vec2 pq, float pixel_scale, vec3 forward, vec3 right, vec3 up) {
    float a = pq.x * pixel_scale;
    float b = pq.y * pixel_scale;
    return normalize(forward + a * right + b * up);
}

// ver physics.py: static_observer_velocity
// v₀ = √(1 − r_s/r) · cos α · r̂ + n_⊥ = n + (s − cos α)·r̂, com s = √(1 − r_s/r) · cos α.
vec3 static_observer_velocity(vec3 x, vec3 n) {
    float r = length(x);
    vec3 rhat = x / r;
    float cos_a = dot(n, rhat);
    float s = sqrt(1.0 - R_S / r) * cos_a;
    float k = s - cos_a;
    return n + k * rhat;
}

// ver physics.py: escape_radius
float escape_radius(float r_cam, float r_min, float factor) {
    return max(r_min, factor * r_cam);
}

// --- equação de movimento e integrador (seções 3.2 e 3.5) ---

// ver physics.py: acceleration
// a = −(3/2)·h²·x/r⁵, com h2 = h².
vec3 acceleration(vec3 x, float h2) {
    float r2 = dot(x, x);
    float r5 = r2 * r2 * sqrt(r2);
    return (-ACCEL_COEFF * h2 / r5) * x;
}

// ver physics.py: rk4_step
void rk4_step(vec3 x, vec3 v, float h2, float dt, out vec3 nx, out vec3 nv) {
    vec3 a1 = acceleration(x, h2);

    float hdt = 0.5 * dt;
    vec3 a2 = acceleration(x + hdt * v, h2);
    vec3 v2 = v + hdt * a1;

    vec3 a3 = acceleration(x + hdt * v2, h2);
    vec3 v3 = v + hdt * a2;

    vec3 a4 = acceleration(x + dt * v3, h2);
    vec3 v4 = v + dt * a3;

    float w = dt / 6.0;
    nx = x + w * (v + 2.0 * v2 + 2.0 * v3 + v4);
    nv = v + w * (a1 + 2.0 * a2 + 2.0 * a3 + a4);
}

// ver physics.py: step_size
// dt = k·r, com k reduzido na faixa da esfera de fótons e limitado a [dt_min, dt_max].
float step_size(float r) {
    float k = u_step_factor;
    if (u_photon_r_min < r && r < u_photon_r_max) {
        k *= u_photon_step_factor;
    }
    return min(max(k * r, u_dt_min), u_dt_max);
}

// ver physics.py: _time_rate
// dt/dλ = E/(1 − r_s/r) (atraso da luz); 0 com r <= r_s.
float time_rate(float r, float energy) {
    float lapse2 = 1.0 - R_S / r;
    if (lapse2 <= 0.0) {
        return 0.0;
    }
    return energy / lapse2;
}

// ver physics.py: integrate_ray
// Checagens em cada passo, nesta ordem: cruzamento do disco (interpolação linear em y;
// vale o primeiro com r_in <= r <= r_out), depois r < r_s, depois r > r_max com x·v > 0.
// Parada antecipada por return. Em DISK, x e v recebem o ponto de impacto interpolado.
// dt_coord recebe o atraso da luz Δt (trapézio de E/(1 − r_s/r) em cada passo; no passo
// do disco, só a fração t até o impacto).
// O ângulo φ percorrido e o traço do raio não são portados (só servem aos testes da CPU).
int integrate_ray(inout vec3 x, inout vec3 v, float r_max, out float dt_coord) {
    vec3 l = cross(x, v);
    float h2 = dot(l, l);
    float r = length(x);
    float energy = sqrt(max(dot(v, v) - h2 / (r * r * r), 0.0));
    float rate = time_rate(r, energy);
    dt_coord = 0.0;

    for (int i = 0; i < MAX_STEPS_CAP; ++i) {
        if (i >= u_max_steps) {
            break;
        }
        float dt = step_size(r);
        vec3 nx;
        vec3 nv;
        rk4_step(x, v, h2, dt, nx, nv);

        if (u_disk_enabled && ((x.y > 0.0 && nx.y <= 0.0) || (x.y < 0.0 && nx.y >= 0.0))) {
            float t = x.y / (x.y - nx.y);
            float px = x.x + t * (nx.x - x.x);
            float pz = x.z + t * (nx.z - x.z);
            float pr = sqrt(px * px + pz * pz);
            if (u_disk_r_in <= pr && pr <= u_disk_r_out) {
                dt_coord += 0.5 * t * dt * (rate + time_rate(pr, energy));
                x = vec3(px, 0.0, pz);
                v = v + t * (nv - v);
                return DISK;
            }
        }

        x = nx;
        v = nv;
        r = length(x);
        float new_rate = time_rate(r, energy);
        dt_coord += 0.5 * dt * (rate + new_rate);
        rate = new_rate;
        if (r < R_S) {
            return CAPTURED;
        }
        if (r > r_max && dot(x, v) > 0.0) {
            return ESCAPED;
        }
    }
    return MAX_STEPS;
}

// --- disco de acreção (seção 3.6) ---

// ver physics.py: peak_radius
float peak_radius(float r_in) {
    return PEAK_RADIUS_RATIO * r_in;
}

// ver physics.py: _disk_profile
// f(r) = r^(−3/4) · (1 − √(r_in/r))^(1/4), com f = 0 para r <= r_in.
float disk_profile(float r, float r_in) {
    if (r <= r_in) {
        return 0.0;
    }
    return pow(r, -0.75) * pow(1.0 - sqrt(r_in / r), 0.25);
}

// ver physics.py: disk_temperature
float disk_temperature(float r, float r_in, float t_peak) {
    return t_peak * disk_profile(r, r_in) / disk_profile(peak_radius(r_in), r_in);
}

// ver scene.py: blackbody_rgb
// Interpolação linear manual em ln T (a textura não tem filtragem); fora de
// [t_min, t_max] vale a cor da extremidade mais próxima.
vec3 blackbody_rgb(float temperature) {
    int n = textureSize(u_bb_table, 0).x;
    if (temperature <= u_bb_t_min) {
        return texelFetch(u_bb_table, ivec2(0, 0), 0).rgb;
    }
    if (temperature >= u_bb_t_max) {
        return texelFetch(u_bb_table, ivec2(n - 1, 0), 0).rgb;
    }
    float u = (log(temperature) - log(u_bb_t_min)) / (log(u_bb_t_max) - log(u_bb_t_min))
        * float(n - 1);
    int k = min(int(floor(u)), n - 2);
    float w = u - float(k);
    return (1.0 - w) * texelFetch(u_bb_table, ivec2(k, 0), 0).rgb
        + w * texelFetch(u_bb_table, ivec2(k + 1, 0), 0).rgb;
}

// ver scene.py: disk_radiance
// T_obs = g·T(r); m · (T_obs/T_pico)⁴ · rgb(T_obs), antes da exposição (g = m = 1 sem
// Doppler e sem textura).
vec3 disk_radiance(float r, float g, float emissivity) {
    float temperature = g * disk_temperature(r, u_disk_r_in, u_disk_t_peak);
    float ratio = temperature / u_disk_t_peak;
    float brightness = emissivity * ratio * ratio * ratio * ratio;
    return brightness * blackbody_rgb(temperature);
}

// --- Doppler e redshift (seção 3.8) ---

// ver physics.py: energy_at_infinity
// E = √(|v|² − h²/r³) = √(2ε).
float energy_at_infinity(vec3 x, vec3 v) {
    vec3 l = cross(x, v);
    float r = length(x);
    return sqrt(dot(v, v) - dot(l, l) / (r * r * r));
}

// ver physics.py: axial_angular_momentum
// L_y = (x × v) · ŷ.
float axial_angular_momentum(vec3 x, vec3 v) {
    return x.z * v.x - x.x * v.z;
}

// ver physics.py: keplerian_omega
// Ω(r) = √(M/r³), no sentido de φ = atan2(−z, x).
float keplerian_omega(float r) {
    return sqrt(M / (r * r * r));
}

// ver physics.py: static_blueshift
// g_∞ = 1/√(1 − r_s/r_cam).
float static_blueshift(float r_cam) {
    return 1.0 / sqrt(1.0 - R_S / r_cam);
}

// ver physics.py: redshift_factor
// g = √(1 − 1,5/r) / [√(1 − 1/r_cam) · (1 − Ω(r)·λ)], λ = −L_y/E (raio traçado de trás
// para a frente: o fóton físico tem momento −v).
float redshift_factor(float r, float r_cam, float l_y, float energy) {
    float lam = -l_y / energy;
    return sqrt(1.0 - CIRCULAR_ORBIT_COEFF / r)
        / (sqrt(1.0 - R_S / r_cam) * (1.0 - keplerian_omega(r) * lam));
}

// --- fundo (seção 3.7) ---

// ver scene.py: checker_color
vec3 checker_color(vec3 d) {
    float norm = length(d);
    float lon = atan(d.z, d.x);
    float lat = asin(clamp(d.y / norm, -1.0, 1.0));
    int parity = (int(floor(lon / u_cell_rad)) + int(floor(lat / u_cell_rad))) & 1;
    if (parity == 0) {
        return u_checker_a;
    }
    return u_checker_b;
}

// ver scene.py: hash_u32
// lowbias32; a aritmética de uint é mod 2³², igual à máscara de 32 bits da CPU.
uint hash_u32(uint x) {
    x ^= x >> 16u;
    x *= LOWBIAS32_M1;
    x ^= x >> 15u;
    x *= LOWBIAS32_M2;
    x ^= x >> 16u;
    return x;
}

// ver scene.py: cell_hash
// uint(int) é o complemento de dois (ver scene.py: _int_to_u32).
uint cell_hash(int ix, int iy, int iz, uint seed) {
    uint h = hash_u32(seed);
    h = hash_u32(h ^ uint(ix));
    h = hash_u32(h ^ uint(iy));
    h = hash_u32(h ^ uint(iz));
    return h;
}

// ver scene.py: hash_to_unit
float hash_to_unit(uint h) {
    return float(h >> 8u) / UNIT_SCALE;
}

// ver scene.py: star_radiance
// shift = g_∞ com Doppler (T × shift, brilho × shift⁴), 1 sem ele.
vec3 star_radiance(vec3 d, float shift) {
    vec3 dn = d / length(d);
    vec3 c = floor(dn * u_star_grid);

    uint h = cell_hash(int(c.x), int(c.y), int(c.z), u_star_seed);
    if (hash_to_unit(h) >= u_star_density) {
        return vec3(0.0);
    }
    h = hash_u32(h);
    float jx = hash_to_unit(h);
    h = hash_u32(h);
    float jy = hash_to_unit(h);
    h = hash_u32(h);
    float jz = hash_to_unit(h);
    h = hash_u32(h);
    float r_intensity = hash_to_unit(h);
    h = hash_u32(h);
    float r_temperature = hash_to_unit(h);

    vec3 s = c + 0.5 + u_star_jitter * (vec3(jx, jy, jz) - 0.5);
    vec3 e = dn - s / length(s);
    float chord2 = dot(e, e);

    float intensity = u_star_i_min * pow(u_star_i_max / u_star_i_min, r_intensity);
    float temperature = shift * u_star_t_min * pow(u_star_t_max / u_star_t_min, r_temperature);
    float shift2 = shift * shift;
    float value = shift2 * shift2 * intensity
        * exp(-chord2 / (2.0 * u_star_sigma * u_star_sigma));
    return value * blackbody_rgb(temperature);
}

// --- textura do disco (seção 3.9) ---

// ver scene.py: periodic_value_noise
// Ruído de valor suave em [0, 1), periódico em w com período n_w (inteiro).
float periodic_value_noise(float u, float w, int n_w, int octave, uint seed) {
    float period = float(n_w);
    w = w - period * floor(w / period);
    float fu = u - floor(u);
    float fw = w - floor(w);
    int i0 = int(floor(u));
    int j0 = int(floor(w)) % n_w;
    int j1 = (j0 + 1) % n_w;
    float v00 = hash_to_unit(cell_hash(i0, j0, octave, seed));
    float v10 = hash_to_unit(cell_hash(i0 + 1, j0, octave, seed));
    float v01 = hash_to_unit(cell_hash(i0, j1, octave, seed));
    float v11 = hash_to_unit(cell_hash(i0 + 1, j1, octave, seed));
    float su = fu * fu * (3.0 - 2.0 * fu);
    float sw = fw * fw * (3.0 - 2.0 * fw);
    float a = v00 + su * (v10 - v00);
    float b = v01 + su * (v11 - v01);
    return a + sw * (b - a);
}

// ver scene.py: disk_texture
// m = 1 + c·(2·fbm − 1) no ângulo co-rotante φ' = φ − Ω(r)·t_emit, em (ln r, φ').
float disk_texture(float r, float phi, float t_emit) {
    float u = log(r) * u_tex_radial_cells;
    float turns = (phi - keplerian_omega(r) * t_emit) / TWO_PI;
    turns -= floor(turns);
    float total = 0.0;
    float norm = 0.0;
    float amplitude = 1.0;
    int scale = 1;
    for (int k = 0; k < u_tex_octaves; ++k) {
        int n_w = u_tex_azimuthal_cells * scale;
        total += amplitude
            * periodic_value_noise(u * float(scale), turns * float(n_w), n_w, k, u_tex_seed);
        norm += amplitude;
        amplitude *= FBM_GAIN;
        scale *= FBM_LACUNARITY;
    }
    float fbm = total / norm;
    return 1.0 + u_tex_contrast * (2.0 * fbm - 1.0);
}

// --- cadeia de saída (seção 3.6) ---

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

// ver render_cpu.py: shade_sample
// DISK usa o ponto de impacto e o v interpolado nele: g com Doppler (L_y e E desse x, v),
// m com textura (φ = atan2(−z, x), t_emit = t_obs − Δt). ESCAPED amostra o fundo pela
// direção final de v (estrelas com g_∞ se houver Doppler; o xadrez não muda). CAPTURED e
// MAX_STEPS são sombra (preto).
vec3 shade_sample(int status, vec3 x, vec3 v, float dt_coord) {
    float r_cam = length(u_cam_pos);
    if (status == DISK) {
        float r = sqrt(x.x * x.x + x.z * x.z);
        float g = 1.0;
        if (u_doppler) {
            g = redshift_factor(
                r, r_cam, axial_angular_momentum(x, v), energy_at_infinity(x, v)
            );
        }
        float emissivity = 1.0;
        if (u_disk_texture) {
            emissivity = disk_texture(r, atan(-x.z, x.x), u_time - dt_coord);
        }
        return disk_radiance(r, g, emissivity);
    }
    if (status == ESCAPED) {
        if (u_background == BG_CHECKER) {
            return checker_color(v);
        }
        float shift = u_doppler ? static_blueshift(r_cam) : 1.0;
        return star_radiance(v, shift);
    }
    return vec3(0.0);
}

void main() {
    vec2 pq = pixel_offset(gl_FragCoord.xy, u_resolution);
    vec3 n = pixel_direction(pq, u_pixel_scale, u_cam_forward, u_cam_right, u_cam_up);
    vec3 x = u_cam_pos;
    vec3 v = static_observer_velocity(x, n);
    float r_max = escape_radius(length(u_cam_pos), u_r_max_min, u_r_max_factor);

    float dt_coord;
    int status = integrate_ray(x, v, r_max, dt_coord);
    vec3 radiance = shade_sample(status, x, v, dt_coord);
    if (u_hdr_output) {
        // Radiância linear × exposição; o bloom e o ACES + sRGB vêm depois
        // (bloom_composite.frag).
        f_color = vec4(u_exposure * radiance, 1.0);
    } else {
        // Radiância linear → exposição → ACES → sRGB; o FBO RGBA8 (não sRGB) arredonda.
        f_color = vec4(encode_pixel(radiance, u_exposure), 1.0);
    }
}
