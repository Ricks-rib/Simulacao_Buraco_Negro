"""Teste de fumaça da Fase 0: o pacote instala e os parâmetros do check_gl são coerentes."""

import blackhole
from blackhole.check_gl import format_version
from blackhole.config import GLCheckConfig


def test_package_imports():
    assert blackhole.__doc__


def test_gl_check_config_requires_opengl_33():
    cfg = GLCheckConfig()
    assert cfg.min_version_code >= 330
    assert format_version(cfg.min_version_code) == "3.3"
    assert cfg.expected_vendor == "NVIDIA"


def test_format_version():
    assert format_version(330) == "3.3"
    assert format_version(460) == "4.6"
