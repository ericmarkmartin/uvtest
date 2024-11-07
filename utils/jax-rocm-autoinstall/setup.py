import os
from setuptools import setup, find_packages
import sys

# Check if /opt/rocm exists
rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")

jax_version = (0, 4, 33)
jax_version_str = ".".join(str(v) for v in jax_version)

root_url = "https://github.com/ROCm/jax/releases/download/rocm-jax-v{jax_v}/"
jaxlib_template_url = (
    "jaxlib-{jax_v}-cp3{pyminor}-cp3{pyminor}-manylinux_2_28_x86_64.whl"
)
pjrt_template_url = "jax_rocm60_pjrt-{jax_v}-py3-none-manylinux_2_28_x86_64.whl"
plugin_template_url = (
    "jax_rocm60_plugin-{jax_v}-cp3{pyminor}-cp3{pyminor}-manylinux_2_28_x86_64.whl"
)

rocm_dependencies = []

if os.path.exists(rocm_path):
    if sys.version_info >= (3, 13):
        raise RuntimeError(f"Unsupported Python version: {sys.version}")

    # Add the specific wheel from the URL if /opt/rocm exists
    pyminor = sys.version_info.minor

    jaxlib_url = (root_url + jaxlib_template_url).format(jax_v=jax_version_str, pyminor=pyminor)
    pjrt_url = (root_url + pjrt_template_url).format(jax_v=jax_version_str)
    plugin_url = (root_url + plugin_template_url).format(jax_v=jax_version_str, pyminor=pyminor)

    rocm_dependencies.append(f"jaxlib @ {jaxlib_url}")
    rocm_dependencies.append(f"jax_rocm60_pjrt @ {pjrt_url}")
    rocm_dependencies.append(f"jax_rocm60_plugin @ {plugin_url}")
    rocm_dependencies.append(f"jax == {jax_version_str}")

setup(
    name="jax-rocm-autoinstall",
    version=jax_version_str,
    description="Automatically install jaxlib ROCM extensions if ROCM is present.",
    authors=[{"name": "Filippo Vicentini", "email": "filippovicentini@gmail.com"}],
    readme="README.md",
    requires_python=">=3.10",
    packages=find_packages(),
    install_requires=rocm_dependencies,
)
