# Installation

This guide walks you through setting up the environment and installing the PRIME package.

---

## Prerequisites

- **Operating System**: Linux (recommended for GPU acceleration) or macOS.
- **Python**: Version 3.10+ (tested with Python 3.13 / 3.14).
- **Conda**: [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).
- **GPU (Optional but Recommended)**: NVIDIA GPU with CUDA support for accelerated ESM embedding inference.

---

## 1. Environment Setup

Create the Conda environment using the repository's environment file:

```bash
conda env create -f tl_env.yml
conda activate tl_env
```

---

## 2. Package Installation

Install the PRIME package in editable mode from the repository root:

```bash
cd prime
pip install -e .
```

This installs the package and registers the three command-line tools in your PATH:
- `prime-process`
- `prime-uncert`
- `prime-predict`

---

## 3. Verify Installation

Check that the CLI tools are available and display their help menus:

```bash
prime-process --help
prime-uncert --help
prime-predict --help
```
