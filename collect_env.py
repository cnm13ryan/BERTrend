# ruff: noqa
# Adapted from https://github.com/pytorch/pytorch/blob/main/torch/utils/collect_env.py
#
# Run `poetry install` (or any other Poetry-managed command that
# creates / activates the project virtual-environment) first,
# then simply
#
#     poetry run python collect_env_poetry.py
#
# The script inspects the active Poetry environment, the host
# system, CUDA/ROCm, CPU, etc.  It also parses pyproject.toml so
# that only *your* declared dependencies are reported.

import argparse
import datetime
import json
import locale
import os
import re
import subprocess
import sys
from collections import namedtuple

import tomli
from packaging.requirements import Requirement

try:
    from torch.utils.collect_env import (
        get_gcc_version,
        get_clang_version,
        get_cmake_version,
        get_conda_packages,
        get_os,
        get_libc_version,
        get_cpu_info,
        get_running_cuda_version,
        get_gpu_info,
        get_nvidia_driver_version,
        get_cudnn_version,
        summarize_build_flags,
        get_gpu_topo,
        get_cachingallocator_config,
        get_cuda_module_loading_config,
        is_xnnpack_available,
        get_rocm_version,
        get_neuron_sdk_version,
    )
except Exception:
    # PyTorch not available -→ fall back to “N/A” stubs so the script still runs
    def _na(*_a, **_kw):
        return "N/A"

    get_gcc_version = get_clang_version = get_cmake_version = _na
    get_conda_packages = lambda *_a, **_kw: ""
    get_os = get_libc_version = get_cpu_info = _na
    get_running_cuda_version = get_gpu_info = get_nvidia_driver_version = _na
    get_cudnn_version = get_rocm_version = get_neuron_sdk_version = _na
    summarize_build_flags = lambda: ""
    get_gpu_topo = _na
    get_cachingallocator_config = get_cuda_module_loading_config = _na
    is_xnnpack_available = lambda: "N/A"


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Collect environment information.")
    p.add_argument(
        "-p",
        "--pyproject",
        default="pyproject.toml",
        help="Path to the pyproject.toml file.",
    )
    return p.parse_args()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def run(cmd):
    """Return (rc, stdout, stderr). Works with list‐or‐string commands."""
    shell = isinstance(cmd, str)
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell
    )
    raw_out, raw_err = p.communicate()
    enc = "oem" if sys.platform == "win32" else locale.getpreferredencoding()
    return p.returncode, raw_out.decode(enc).strip(), raw_err.decode(enc).strip()


def run_and_read_all(cmd):
    rc, out, _ = run(cmd)
    return out if rc == 0 else None


def run_and_parse_first_match(cmd, pat):
    rc, out, _ = run(cmd)
    if rc != 0:
        return None
    m = re.search(pat, out)
    return m.group(1) if m else None


def run_and_first_line(cmd):
    rc, out, _ = run(cmd)
    return out.split("\n")[0] if rc == 0 and out else None


# -----------------------------------------------------------------------------
# Poetry integration
# -----------------------------------------------------------------------------
DEFAULT_POETRY_PATTERNS = {
    "torch",
    "numpy",
    "triton",
    "optree",
    "onnx",
    "nccl",
    "transformers",
    "pynvml",
    "nvidia",
    "zmq",
}


def get_project_dependencies(pyproject_path):
    """Return the *canonicalised* package names declared in pyproject.toml."""
    try:
        with open(pyproject_path, "rb") as fh:
            data = tomli.load(fh)
        project = data.get("project", {})  # or tool.poetry.* for pre-PEP-621
        deps = project.get("dependencies", [])
        pkgs = []
        for dep in deps:
            try:
                pkgs.append(Requirement(dep).name.lower())
            except Exception as exc:
                print(f"Could not parse dependency '{dep}': {exc}")
        return set(pkgs)
    except Exception as exc:
        print(f"Failed to read {pyproject_path}: {exc}")
        return set()


def get_poetry_packages(package_names):
    """
    Returns
    -------
    (poetry_version, formatted_output)
    """
    poetry_version = (
        run_and_first_line(["poetry", "--version"]) or "Poetry not found"
    )  # --version is a documented global flag :contentReference[oaicite:0]{index=0}

    raw = run_and_read_all(["poetry", "show", "--no-ansi"])
    # `poetry show` is the recommended way to list installed packages :contentReference[oaicite:1]{index=1}
    if not raw:
        return poetry_version, ""

    chosen = []
    for line in raw.splitlines():
        m = re.match(r"^(\S+)\s+([^\s]+)\s+", line)
        if not m:
            continue
        name, ver = m.groups()
        if name.lower() in package_names:
            chosen.append(f"{name.lower()}=={ver}")
    return poetry_version, "\n".join(chosen)


# -----------------------------------------------------------------------------
# System collectors (unchanged except for Poetry/uv removal)
# -----------------------------------------------------------------------------
try:
    import torch

    TORCH_AVAILABLE = True
except (ImportError, OSError, AttributeError, NameError):
    TORCH_AVAILABLE = False


# (Many helper functions from the original file are kept verbatim – trimmed here
# for brevity.  Nothing changes in their behaviour.)


# -----------------------------------------------------------------------------
# Aggregate + pretty-print
# -----------------------------------------------------------------------------
EnvInfo = namedtuple(
    "EnvInfo",
    [
        # PyTorch / system
        "torch_version",
        "is_debug_build",
        "cuda_compiled_version",
        "gcc_version",
        "clang_version",
        "cmake_version",
        "os",
        "libc_version",
        "python_version",
        "python_platform",
        "is_cuda_available",
        "cuda_runtime_version",
        "cuda_module_loading",
        "nvidia_driver_version",
        "nvidia_gpu_models",
        "cudnn_version",
        # Manager-specific
        "poetry_version",
        "poetry_packages",
        "conda_packages",
        # HIP / ROCm / neuron
        "hip_compiled_version",
        "hip_runtime_version",
        "miopen_runtime_version",
        "caching_allocator_config",
        "is_xnnpack_available",
        "cpu_info",
        "rocm_version",
        "neuron_sdk_version",
        # Build extras
        "build_flags",
        "gpu_topology",
    ],
)

ENV_FMT = """
PyTorch version: {torch_version}
Is debug build: {is_debug_build}
CUDA used to build PyTorch: {cuda_compiled_version}
ROCM used to build PyTorch: {hip_compiled_version}

OS: {os}
GCC version: {gcc_version}
Clang version: {clang_version}
CMake version: {cmake_version}
Libc version: {libc_version}

Python version: {python_version}
Python platform: {python_platform}
Is CUDA available: {is_cuda_available}
CUDA runtime version: {cuda_runtime_version}
CUDA_MODULE_LOADING set to: {cuda_module_loading}
GPU models and configuration: {nvidia_gpu_models}
Nvidia driver version: {nvidia_driver_version}
cuDNN version: {cudnn_version}
HIP runtime version: {hip_runtime_version}
MIOpen runtime version: {miopen_runtime_version}
Is XNNPACK available: {is_xnnpack_available}

CPU:
{cpu_info}

ROCM Version: {rocm_version}
Neuron SDK Version: {neuron_sdk_version}
Build Flags:
{build_flags}
GPU Topology:
{gpu_topology}

Versions of relevant libraries:
{poetry_packages}
{conda_packages}
""".strip()


def pretty(env):
    d = env._asdict()

    # helpers
    def _none_to_txt(txt="Could not collect"):
        for k, v in d.items():
            if v is None:
                d[k] = txt

    def _bool_to_yesno():
        for k, v in d.items():
            if v is True:
                d[k] = "Yes"
            elif v is False:
                d[k] = "No"

    def _indent_packages(tag, key):
        if d[key]:
            d[key] = "[{}] ".format(tag) + d[key].replace("\n", f"\n[{tag}] ")

    def _maybe_multiline(key):
        v = d[key]
        if v and "\n" in v:
            d[key] = "\n" + v + "\n"

    _bool_to_yesno()
    _none_to_txt()

    for k in ("poetry_packages", "conda_packages"):
        if not d[k]:
            d[k] = "No relevant packages"
    _indent_packages("poetry", "poetry_packages")
    _indent_packages("conda", "conda_packages")

    for k in ("nvidia_gpu_models", "gpu_topology"):
        _maybe_multiline(k)

    return ENV_FMT.format(**d)


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def collect_env(pyproject):
    # Functions from the original script (get_os, get_cpu_info, etc.)
    # are assumed to be present. They are unchanged.
    pkgs = get_project_dependencies(pyproject)
    poetry_version, proj_pkg_blob = get_poetry_packages(pkgs)

    # -- PyTorch section ------------------------------------------------------
    if TORCH_AVAILABLE:
        torch_version = torch.__version__
        is_debug = torch.version.debug
        cuda_compiled = torch.version.cuda or "N/A"
        cuda_avail = torch.cuda.is_available()
        hip_compiled = (
            (torch.version.hip or "N/A") if hasattr(torch.version, "hip") else "N/A"
        )
    else:
        torch_version = is_debug = cuda_compiled = cuda_avail = hip_compiled = "N/A"

    # -- Populate EnvInfo -----------------------------------------------------
    return EnvInfo(
        torch_version=torch_version,
        is_debug_build=is_debug,
        cuda_compiled_version=cuda_compiled,
        gcc_version=get_gcc_version(run),
        clang_version=get_clang_version(run),
        cmake_version=get_cmake_version(run),
        os=get_os(run),
        libc_version=get_libc_version(),
        python_version=f"{sys.version.replace(chr(10),' ')} ({sys.maxsize.bit_length()+1}-bit runtime)",
        python_platform=sys.platform,
        is_cuda_available=cuda_avail,
        cuda_runtime_version=get_running_cuda_version(run),
        cuda_module_loading=get_cuda_module_loading_config(),
        nvidia_driver_version=get_nvidia_driver_version(run),
        nvidia_gpu_models=get_gpu_info(run),
        cudnn_version=get_cudnn_version(run),
        poetry_version=poetry_version,
        poetry_packages=proj_pkg_blob,
        conda_packages=get_conda_packages(run),  # kept for users inside Conda
        hip_compiled_version=hip_compiled,
        hip_runtime_version="N/A",
        miopen_runtime_version="N/A",
        caching_allocator_config=get_cachingallocator_config(),
        is_xnnpack_available=is_xnnpack_available(),
        cpu_info=get_cpu_info(run),
        rocm_version=get_rocm_version(run),
        neuron_sdk_version=get_neuron_sdk_version(run),
        build_flags=summarize_build_flags(),
        gpu_topology=get_gpu_topo(run),
    )


def main():
    args = parse_args()
    info = collect_env(args.pyproject)
    print(pretty(info))

    # Optional: PyTorch crash-handler snippet unchanged
    if (
        TORCH_AVAILABLE
        and hasattr(torch, "utils")
        and hasattr(torch.utils, "_crash_handler")
    ):
        dump_dir = torch.utils._crash_handler.DEFAULT_MINIDUMP_DIR
        if sys.platform == "linux" and os.path.exists(dump_dir):
            dumps = [os.path.join(dump_dir, f) for f in os.listdir(dump_dir)]
            if dumps:
                latest = max(dumps, key=os.path.getctime)
                ts = datetime.datetime.fromtimestamp(os.path.getctime(latest)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                print(
                    f"\n*** Detected a minidump at {latest} created on {ts}, "
                    "if this is related to your bug please include it when filing a report ***",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()
