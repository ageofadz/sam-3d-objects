# Copyright (c) Meta Platforms, Inc. and affiliates.
import os

if not os.environ.get("LIDRA_SKIP_INIT"):
    from importlib import import_module
    import_module("sam3d_objects.init")