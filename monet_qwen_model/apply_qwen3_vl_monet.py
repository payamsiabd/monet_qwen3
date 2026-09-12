import importlib.util, sys, pathlib, os
patch_path = pathlib.Path(__file__).with_name("modeling_qwen3_vl_monet.py")
spec  = importlib.util.spec_from_file_location(
    "transformers.models.qwen3_vl.modeling_qwen3_vl",
    patch_path,
)
patched_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patched_mod)

sys.modules["transformers.models.qwen3_vl.modeling_qwen3_vl"] = patched_mod

print("Replaced the original Qwen3-VL model with the Monet version.")
