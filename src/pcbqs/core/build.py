"""Build the optional QS backend: python -m pcbqs.core.build."""
from emsdev.build import build_core
build_core(__file__, "pcbqs_core.cpp", arch_env="PCBQS_ARCH", includes_here=True)
