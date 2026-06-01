"""Cross-framework interop utilities for Quadrants.

This module provides helpers for integrating Quadrants with other GPU frameworks (e.g. PyTorch MPS) at the
command-queue level, enabling zero-overhead shared-queue execution without explicit synchronisation.
"""

from quadrants.interop._torch_mps import get_mps_command_queue

__all__ = ["get_mps_command_queue"]
