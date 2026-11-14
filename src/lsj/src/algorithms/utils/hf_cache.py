"""Resolve the user's Hugging Face Hub cache without filesystem side effects.

Call before importing model dependencies; Hugging Face reads configuration at
import time. This module never changes environment variables or creates paths.
"""

import os
from pathlib import Path


def resolve_hub_cache() -> Path:
    """Follow Hugging Face's current/legacy cache environment precedence."""
    default_home = os.path.join(os.environ.get("XDG_CACHE_HOME", "~/.cache"), "huggingface")
    hf_home = os.path.expandvars(os.path.expanduser(os.environ.get("HF_HOME", default_home)))
    legacy_cache = os.environ.get("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))
    hub_cache = os.environ.get("HF_HUB_CACHE", legacy_cache)
    return Path(os.path.expandvars(os.path.expanduser(hub_cache)))
