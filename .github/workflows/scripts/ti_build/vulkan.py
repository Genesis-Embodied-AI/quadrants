# -*- coding: utf-8 -*-

# -- stdlib --
import os
import platform
import subprocess

# -- third party --
# -- own --
from .dep import download_dep
from .misc import banner, get_cache_home, path_prepend

VULKAN_VERSION = "1.4.321.1"


# -- code --
@banner(f"Setup Vulkan {VULKAN_VERSION}")
def setup_vulkan():
    u = platform.uname()
    match (u.system, u.machine):
        case ("Linux", "x86_64"):
            url = f"https://sdk.lunarg.com/sdk/download/{VULKAN_VERSION}/linux/vulkansdk-linux-x86_64-{VULKAN_VERSION}.tar.xz"
            prefix = get_cache_home() / f"vulkan-{VULKAN_VERSION}"

            download_dep(url, prefix, strip=1)
            sdk = prefix / "x86_64"
            os.environ["VULKAN_SDK"] = str(sdk)
            path_prepend("PATH", sdk / "bin")
            path_prepend("LD_LIBRARY_PATH", sdk / "lib")
            os.environ["VK_LAYER_PATH"] = str(sdk / "share" / "vulkan" / "explicit_layer.d")
        case ("Linux", "arm64") | ("Linux", "aarch64"):
            url = "https://github.com/Genesis-Embodied-AI/quadrants-sdk-builds/releases/download/vulkan-sdk-1.4.321.1-202509161414/vulkansdk-ubuntu-22.04-arm-1.4.321.1.tar.xz"
            prefix = get_cache_home() / f"vulkan-arm-{VULKAN_VERSION}"

            download_dep(url, prefix, strip=1)
            sdk = prefix / "x86_64"
            os.environ["VULKAN_SDK"] = str(sdk)
            path_prepend("PATH", sdk / "bin")
            path_prepend("LD_LIBRARY_PATH", sdk / "lib")
            os.environ["VK_LAYER_PATH"] = str(sdk / "share" / "vulkan" / "explicit_layer.d")
        case ("Darwin", "arm64"):
            # LunarG's macOS `.zip` is an `InstallVulkan.app` installer bundle (the same Qt installer
            # Windows uses), not a ready-to-use SDK tree. We extract the bundle, then invoke its CLI
            # non-interactively to drop the actual SDK payload into a sibling prefix. LunarG didn't
            # publish a 1.4.321.1 macOS asset (the Linux / Windows pin's patch-level); the prior
            # 1.4.321.0 does, and is inlined here rather than factored into a separate constant.
            url = "https://sdk.lunarg.com/sdk/download/1.4.321.0/mac/vulkansdk-macos-1.4.321.0.zip"
            installer_dir = get_cache_home() / "vulkan-macos-1.4.321.0-installer"
            prefix = get_cache_home() / "vulkan-macos-1.4.321.0"

            download_dep(url, installer_dir, strip=1)
            if not (prefix / "macOS").exists():
                installer_bin = installer_dir / "Contents" / "MacOS" / "vulkansdk-macOS-1.4.321.0"
                # Python's `zipfile` doesn't preserve the Unix execute bit, so the extracted installer
                # comes out with mode 0644 and `exec` trips `PermissionError: [Errno 13]`. `chmod +x`
                # here is idempotent and scoped to this single binary.
                installer_bin.chmod(0o755)
                subprocess.check_call(
                    [
                        str(installer_bin),
                        "--root",
                        str(prefix),
                        "--accept-licenses",
                        "--default-answer",
                        "--confirm-command",
                        "install",
                    ]
                )
            sdk = prefix / "macOS"
            os.environ["VULKAN_SDK"] = str(sdk)
            path_prepend("PATH", sdk / "bin")
            path_prepend("DYLD_LIBRARY_PATH", sdk / "lib")
            os.environ["VK_LAYER_PATH"] = str(sdk / "share" / "vulkan" / "explicit_layer.d")
            # LunarG's macOS SDK installer drops `libMoltenVK.dylib` flat inside `macOS/lib/` alongside
            # the desktop loader - no `MoltenVK/` subtree, no xcframework of dylibs (only `.a` statics in
            # the xcframework). `MOLTENVK_DIR` points at the directory that actually contains the dylib
            # so `find_file` in `quadrants/rhi/CMakeLists.txt` can locate it without path guessing.
            os.environ["MOLTENVK_DIR"] = str(sdk / "lib")
        case ("Windows", "AMD64"):
            url = (
                f"https://sdk.lunarg.com/sdk/download/{VULKAN_VERSION}/windows/VulkanSDK-{VULKAN_VERSION}-Installer.exe"
            )
            prefix = get_cache_home() / "vulkan-{VULKAN_VERSION}"
            download_dep(
                url,
                prefix,
                elevate=True,
                args=[
                    "--accept-licenses",
                    "--default-answer",
                    "--confirm-command",
                    "--root",
                    prefix,
                    "install",
                    "com.lunarg.vulkan.sdl2",
                    "com.lunarg.vulkan.glm",
                    "com.lunarg.vulkan.volk",
                    "com.lunarg.vulkan.vma",
                    # 'com.lunarg.vulkan.debug',
                ],
            )
            os.environ["VULKAN_SDK"] = str(prefix)
            os.environ["VK_SDK_PATH"] = str(prefix)
            os.environ["VK_LAYER_PATH"] = str(prefix / "Bin")
            path_prepend("PATH", prefix / "Bin")
        case default:
            return
