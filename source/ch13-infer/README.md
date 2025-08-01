# ✅ vLLM 手动调试运行 OPT-125M 成功记录（内存受限环境）

这种绕开标准构建链、模块路径注入与动态库补丁的方式，是调试和自定义开发 vLLM 的强力手段。这套方法尤其适合开发者机器内存不够的场景。

## 🖥️ 环境信息

* GPU：RTX 4090
* 内存：64GB（曾因编译 OOM）
* 系统：Linux
* Python：3.11.13
* CUDA：12.4
* vLLM：最新 main 分支源码
* Home: /home/wukong

## 🚫 未采用的方式

* **没有使用 `python setup.py develop` 或 `pip install -e`**，因为内存不足会 OOM。
* 没有使用 PyPI 安装包，完全从源码构建。

## ✅ 实际步骤

0. env preparation

```
conda create -n vllm-dev python=3.11
conda activate vllm-dev
pip install -r requirements/build.txt
pip install -r requirements/common.txt
```

1. **使用 `.pth` 路径方式** 注入 `~/git.repo/vllm` 到 `site-packages`，绕过打包流程。

```
export SITE_DIR=$(python -m site --user-site)
mkdir -p "$SITE_DIR"
echo "/home/wukong/git.repo/vllm" > "$SITE_DIR/vllm-dev.pth"
```

2. **手动使用 CMake 构建动态库**，避免 setup 时统一编译 OOM。

```
cd ~/git.repo/vllm/
mkdir -p build && cd build
cmake .. -DVLLM_PYTHON_EXECUTABLE=$(which python)
cmake --build . -j8
```

3. **手动创建 symlink for dynamic libs**：

   ```bash
    cd ~/git.repo/vllm/vllm
    ln -sf ../build/_C.abi3.so .
    ln -sf ../build/_moe_C.abi3.so .
    ln -sf ../build/_vllm_fa2_C.abi3.so .
    ln -sf ../build/_vllm_fa3_C.abi3.so .
    ln -sf ../build/cumem_allocator.abi3.so .
   ```

   ```bash
   cd ~/git.repo/vllm/vllm/vllm_flash_attn
   ln -s ../../build/vllm-flash-attn/_vllm_fa2_C.abi3.so _vllm_fa2_C.abi3.so
   ln -s ../../build/vllm-flash-attn/_vllm_fa3_C.abi3.so _vllm_fa3_C.abi3.so
   ```

4. **复制 flash attention 构建产物**：

   ```bash
   cp -Rf ~/git.repo/vllm/.deps/vllm-flash-attn-src/vllm_flash_attn/* ~/git.repo/vllm/vllm/vllm_flash_attn/
   ```

5. **手动安装依赖**：

   ```bash
   pip install numba==0.61.2
   git clone https://github.com/flashinfer-ai/flashinfer
   cd flashinfer
   pip install -e flashinfer
   ```

6. **成功运行**：

   ```bash
   python -m vllm.entrypoints.openai.api_server --model facebook/opt-125m
   ```

   日志验证：

   ```
   INFO Using FlashInfer for top-p & top-k sampling.
   INFO Using Flash Attention backend on V1 engine.
   ```


