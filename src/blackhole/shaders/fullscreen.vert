#version 330 core
// Triângulo de tela cheia sem buffer de vértices (PLANO, Fase 4): os três vértices saem
// de gl_VertexID e cobrem o retângulo [-1, 1]² do clip space. O fragment shader usa só
// gl_FragCoord, então não há varyings.

void main() {
    vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
