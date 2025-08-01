# ⚙️ vLLM Debugging in Low-Memory Environment (Manual Build & Runtime Setup)

This guide documents a low-level, manual method to build and run `vLLM` from source without relying on standard Python packaging. It's especially helpful in machines with limited system memory (e.g., 64GB RAM), where the unified build process would typically fail due to OOM errors.

---

## 🖥️ System Setup

* **GPU**: RTX 4090
* **RAM**: 64GB
* **OS**: Linux
* **Python**: 3.11.13
* **CUDA**: 12.4
* **vLLM**: Latest source from `main` branch
* **Base path**: `/home/wukong`

---

## 🚫 What Was Avoided

* Avoided `python setup.py develop` or `pip install -e .`, which caused out-of-memory (OOM) failures during compilation.
* No binary wheel installation via PyPI — all components are built and linked manually.

---

## ✅ Manual Build & Execution Steps

### 0. Environment preparation

```
conda create -n vllm-dev python=3.11
conda activate vllm-dev
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements/build.txt
pip install -r requirements/common.txt
```

### 1. Inject source path using `.pth` file

This lets Python recognize `~/git.repo/vllm` without installing the package:

```bash
export SITE_DIR=$(python -m site --user-site)
mkdir -p "$SITE_DIR"
echo "/home/wukong/git.repo/vllm" > "$SITE_DIR/vllm-dev.pth"
```

---

### 2. Manually build dynamic libraries via CMake

Avoids large monolithic builds:

```bash
cd ~/git.repo/vllm/
mkdir -p build && cd build
cmake .. -DVLLM_PYTHON_EXECUTABLE=$(which python)
cmake --build . -j8
```

---

### 3. Link built libraries into `vllm/`

```bash
cd ~/git.repo/vllm/vllm
ln -sf ../build/_C.abi3.so .
ln -sf ../build/_moe_C.abi3.so .
ln -sf ../build/_vllm_fa2_C.abi3.so .
ln -sf ../build/_vllm_fa3_C.abi3.so .
ln -sf ../build/cumem_allocator.abi3.so .
```

Also link FlashAttention-specific libraries:

```bash
cd ~/git.repo/vllm/vllm/vllm_flash_attn
ln -s ../../build/vllm-flash-attn/_vllm_fa2_C.abi3.so _vllm_fa2_C.abi3.so
ln -s ../../build/vllm-flash-attn/_vllm_fa3_C.abi3.so _vllm_fa3_C.abi3.so
```

---

### 4. Copy FlashAttention Python files

```bash
cp -Rf ~/git.repo/vllm/.deps/vllm-flash-attn-src/vllm_flash_attn/* ~/git.repo/vllm/vllm/vllm_flash_attn/
```

---

### 5. Manually install runtime dependencies

```bash
pip install numba==0.61.2
git clone https://github.com/flashinfer-ai/flashinfer
cd flashinfer
pip install -e .
```

### 6. Injection of metadata

In `~/git.repo/vllm/`, create a file `inject_vllm_metadata.py` as below.

```bash
$cat inject_vllm_metadata.py 
import site
import sys
from pathlib import Path
import vllm

site_packages = next(p for p in site.getsitepackages() if "site-packages" in p and str(sys.prefix) in p)
dist_info_dir = Path(site_packages) / "vllm-0.0.0.dev0.dist-info"
dist_info_dir.mkdir(parents=True, exist_ok=True)

(dist_info_dir / "METADATA").write_text("Name: vllm\nVersion: 0.0.0.dev0\n")
(dist_info_dir / "top_level.txt").write_text("vllm\n")
(dist_info_dir / "RECORD").write_text("")

print(f"✅ Injected fake metadata into: {dist_info_dir}")
```

Then run it like this:
```bash
python inject_vllm_metadata.py
```

---

### 7. Launch vLLM server

```bash
python -m vllm.entrypoints.openai.api_server --model facebook/opt-125m
```

#### ✅ Expect to see logs like:

```
INFO Using FlashInfer for top-p & top-k sampling.
INFO Using Flash Attention backend on V1 engine.
```

