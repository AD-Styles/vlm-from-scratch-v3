"""Check HF Space build status (one-shot)."""
from huggingface_hub import HfApi

api = HfApi()
info = api.space_info("AD-Styles/mini-llava-v3-demo")
runtime = getattr(info, "runtime", None)
stage = getattr(runtime, "stage", "unknown") if runtime else "unknown"
hardware = getattr(runtime, "hardware", "unknown") if runtime else "unknown"
print(f"Stage:    {stage}")
print(f"Hardware: {hardware}")
print(f"SDK:      {info.sdk}")
print(f"URL:      https://huggingface.co/spaces/AD-Styles/mini-llava-v3-demo")
