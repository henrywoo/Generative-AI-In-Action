import torch
import subprocess
import re


def get_gpu_info():
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_name": None,
        "compute_capability": None,
        "supports_tensor_cores": False,
        "architecture": None,
        "mxfp4_possible": False
    }

    if not info["cuda_available"]:
        return info

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)

    info["device_name"] = props.name
    major, minor = props.major, props.minor
    info["compute_capability"] = f"{major}.{minor}"

    # Tensor Core available for compute capability >= 7.0
    info["supports_tensor_cores"] = (major >= 7)

    # Architecture name (Ampere, Ada, Hopper etc.)
    try:
        result = subprocess.check_output("nvidia-smi -q | grep 'Architecture'", shell=True)
        match = re.search(r'Architecture\s*:\s*(\w+)', result.decode())
        if match:
            arch = match.group(1)
            info["architecture"] = arch

            # MXFP4 only on Hopper (H100) and Ada (RTX 4090 etc.)
            if arch in ["Ada", "Hopper"]:
                info["mxfp4_possible"] = True
    except:
        info["architecture"] = "Unknown"

    return info


def print_gpu_info(info):
    if not info["cuda_available"]:
        print("CUDA not available. MXFP4 unsupported.")
        return

    print(f"GPU Name           : {info['device_name']}")
    print(f"Compute Capability: {info['compute_capability']}")
    print(f"Tensor Cores      : {'✅ Yes' if info['supports_tensor_cores'] else '❌ No'}")
    print(f"GPU Architecture  : {info['architecture']}")
    print(f"MXFP4 Supported   : {'✅ Yes' if info['mxfp4_possible'] else '❌ No'}")


if __name__ == "__main__":
    info = get_gpu_info()
    print_gpu_info(info)
