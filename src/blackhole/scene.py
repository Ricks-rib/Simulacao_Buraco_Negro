"""Cena: fundo celeste (e, na Fase 3, disco, corpo negro e tone mapping).

Fase 2: só o fundo de depuração, um xadrez em latitude/longitude em torno de +y
(PLANO, seção 3.7), amostrado pela direção final de v dos raios que escapam.
"""

import math

from numba import njit


@njit(cache=True)
def checker_color(d, cell_rad, color_a, color_b):
    """Cor do xadrez celeste na direção d (não precisa ser unitária).

    longitude = atan2(d_z, d_x) e latitude = asin(d_y/|d|), em torno do eixo +y. A
    paridade de ⌊lon/cell⌋ + ⌊lat/cell⌋ escolhe ``color_a`` (par) ou ``color_b`` (ímpar).
    Com casas de 15° são 24 × 12 casas. Cores em [0, 1].
    """
    norm = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    lon = math.atan2(d[2], d[0])
    lat = math.asin(min(1.0, max(-1.0, d[1] / norm)))
    parity = (int(math.floor(lon / cell_rad)) + int(math.floor(lat / cell_rad))) & 1
    if parity == 0:
        return color_a
    return color_b
