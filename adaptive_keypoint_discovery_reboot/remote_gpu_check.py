#!/usr/bin/env python
"""在 VS Code 的远程终端中检查训练所需的 GPU 与 PyTorch CUDA 环境。

本脚本只读取环境信息，不安装软件，也不会启动训练。
"""

from __future__ import annotations

import json
import platform
import shutil
import socket
import subprocess
import sys
from typing import Any


def run(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        return {
            "available": True,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}


def torch_report() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # torch may not be installed yet
        return {"imported": False, "error": repr(exc)}

    report: dict[str, Any] = {
        "imported": True,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
    }
    try:
        report["cudnn_version"] = torch.backends.cudnn.version()
        report["cudnn_error"] = None
    except Exception as exc:
        report["cudnn_version"] = None
        report["cudnn_error"] = repr(exc)
    devices: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": prop.name,
                "memory_gb": round(prop.total_memory / 1024**3, 2),
                "compute_capability": f"{prop.major}.{prop.minor}",
            }
        )
    report["devices"] = devices

    if report["cuda_available"] and report["cudnn_error"] is None:
        try:
            sample = torch.zeros((1, 3, 32, 32), device="cuda")
            kernel = torch.zeros((4, 3, 3, 3), device="cuda")
            torch.nn.functional.conv2d(sample, kernel)
            torch.cuda.synchronize()
            report["cuda_conv_smoke"] = True
            report["cuda_conv_error"] = None
        except Exception as exc:
            report["cuda_conv_smoke"] = False
            report["cuda_conv_error"] = repr(exc)
    else:
        report["cuda_conv_smoke"] = False
        report["cuda_conv_error"] = "skipped because CUDA or cuDNN is unavailable"
    return report


def main() -> None:
    nvidia_smi = shutil.which("nvidia-smi")
    report = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "nvidia_smi_path": nvidia_smi,
        "nvidia_smi_query": run(
            [
                nvidia_smi or "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader",
            ]
        ),
        "torch": torch_report(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    torch_info = report["torch"]
    if not nvidia_smi:
        print("\n[结论] 当前终端找不到 nvidia-smi：先确认 VS Code 确实连到 GPU 服务器。")
    elif not torch_info.get("imported"):
        print("\n[结论] GPU 驱动可检查，但当前 Python 环境尚未成功导入 PyTorch。")
    elif not torch_info.get("cuda_available"):
        print("\n[结论] 当前 PyTorch 不能使用 CUDA；常见原因是装了 CPU 版 PyTorch或环境不对。")
    elif torch_info.get("cudnn_error"):
        print("\n[结论] CUDA 可见，但 cuDNN 初始化失败；先处理动态库版本冲突，不能开始训练。")
    elif not torch_info.get("cuda_conv_smoke"):
        print("\n[结论] CUDA 可见，但最小卷积测试失败；不能开始训练。")
    else:
        print("\n[结论] 当前 Python 环境能够调用 CUDA/cuDNN，最小卷积测试通过，可进入训练前依赖检查。")


if __name__ == "__main__":
    main()
