"""Install a forwarding API recorder into this diagnostic's disposable wheel environment."""

import hashlib
import json
import os
import subprocess
from pathlib import Path


def install_trace(runtime_folder, output, env):
    output.mkdir(parents=True, exist_ok=True)
    library = Path(runtime_folder) / "libMoltenVK.dylib"
    original = library.with_name("libMoltenVK.unmodified.dylib")
    assert library.is_file() and not original.exists()
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    assert digest == "45f81ce6bd517b6bc904f252b84ebe30177e791143144cc81eb0a1e99e840c0a"
    source = Path(__file__).with_suffix(".cpp").resolve()
    forwarding = output / "libMoltenVK.trace.dylib"
    command = [
        "clang++",
        "-std=c++17",
        "-O2",
        "-dynamiclib",
        "-Wall",
        "-Wextra",
        "-Werror",
        str(source),
        "-o",
        str(forwarding),
    ]
    print("Building Vulkan API recorder:", command, flush=True)
    subprocess.run(command, check=True)
    library.rename(original)
    library.write_bytes(forwarding.read_bytes())
    env["QD_TRACE_MOLTENVK_REAL"] = str(original.resolve())
    env["QD_TRACE_VULKAN_DIR"] = str(output.resolve())
    (output / "trace-install.json").write_text(
        json.dumps(
            dict(
                original_sha256=digest,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                metal_header_sha256=hashlib.sha256(
                    source.with_name("vulkan_metal_stats.h").read_bytes()
                ).hexdigest(),
                wrapper_sha256=hashlib.sha256(forwarding.read_bytes()).hexdigest(),
                command=command,
                controller_pid=os.getpid(),
            ),
            indent=2,
        )
        + "\n"
    )
