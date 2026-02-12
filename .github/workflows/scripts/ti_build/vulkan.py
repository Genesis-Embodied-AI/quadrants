# -*- coding: utf-8 -*-

# -- stdlib --
import os
import platform

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
        # case ("Darwin", "arm64"):
        # case ("Darwin", "x86_64"):
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
