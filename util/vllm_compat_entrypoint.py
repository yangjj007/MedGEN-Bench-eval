#!/usr/bin/env python3
"""Launch vLLM with the narrow Gemma3 config compatibility shim we need.

vLLM 0.8.5 reads ``Gemma3TextConfig.sliding_window_pattern``.  Transformers
4.57 keeps the same model value under the private compatibility field
``_sliding_window_pattern`` after loading the newer MedGemma 1.5 config.  The
property below restores the public read-only alias expected by that vLLM
release.  Remove this wrapper once vLLM is upgraded to a release that supports
the current Transformers field directly.
"""

from __future__ import annotations

import sys

from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig


if not hasattr(Gemma3TextConfig, "sliding_window_pattern"):
    Gemma3TextConfig.sliding_window_pattern = property(  # type: ignore[attr-defined]
        lambda config: getattr(config, "_sliding_window_pattern", 6)
    )

from vllm.entrypoints.cli.main import main


if __name__ == "__main__":
    main()
