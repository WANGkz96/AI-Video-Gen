from __future__ import annotations

import argparse
from pathlib import Path


BLACKWELL_SDPA_MARKER = "AI-Video-Gen Blackwell SDPA fallback"


def patch_source(source: str, runner: Path) -> str:
    marker = "RTX PRO 6000 Blackwell workaround"
    if marker not in source:
        anchor = "def generate(args):\n"
        replacement = (
            "def generate(args):\n"
            "    # RTX PRO 6000 Blackwell workaround: native PyTorch kernels avoid\n"
            "    # cuDNN initialization failures in Whisper/VAE Conv1d/Conv3d.\n"
            "    if torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0):\n"
            "        torch.backends.cudnn.enabled = False\n"
        )
        if anchor not in source:
            raise RuntimeError(f"Cannot find generate() anchor in {runner}")
        source = source.replace(anchor, replacement, 1)

    orientation_marker = "Applying target orientation"
    if orientation_marker not in source:
        input_anchor = "    with open(input_json, 'r', encoding='utf-8') as f:\n        input_data = json.load(f)\n"
        input_replacement = (
            "    with open(input_json, 'r', encoding='utf-8') as f:\n"
            "        input_data = json.load(f)\n"
            "    target_orientation = str(input_data.get('target_orientation', '')).strip().lower()\n"
            "    if target_orientation == 'portrait' and width > height:\n"
            "        height, width = width, height\n"
            "        print(f'Applying target orientation: portrait ({width}x{height})')\n"
            "    elif target_orientation == 'landscape' and height > width:\n"
            "        height, width = width, height\n"
            "        print(f'Applying target orientation: landscape ({width}x{height})')\n"
        )
        if input_anchor not in source:
            raise RuntimeError(f"Cannot find input JSON anchor in {runner}")
        source = source.replace(input_anchor, input_replacement, 1)

    schedule_marker = "Using prompt schedule item"
    if schedule_marker not in source:
        prompt_anchor = "    prompt = input_data['prompt']\n"
        prompt_replacement = (
            "    prompt = input_data['prompt']\n"
            "    prompt_schedule = input_data.get('prompt_schedule')\n"
            "    if prompt_schedule is not None:\n"
            "        if not isinstance(prompt_schedule, list) or len(prompt_schedule) == 0:\n"
            "            raise ValueError(\"prompt_schedule must be a non-empty list when provided\")\n"
            "        prompt = prompt_schedule[0]\n"
        )
        if prompt_anchor not in source:
            raise RuntimeError(f"Cannot find prompt input anchor in {runner}")
        source = source.replace(prompt_anchor, prompt_replacement, 1)

        generate_anchor = "        output_tuple = pipe.generate_avc(\n"
        generate_replacement = (
            "        segment_prompt = prompt\n"
            "        if prompt_schedule is not None:\n"
            "            segment_prompt = prompt_schedule[min(segment_idx, len(prompt_schedule) - 1)]\n"
            "            if local_rank == 0:\n"
            "                print(f\"Using prompt schedule item {min(segment_idx, len(prompt_schedule) - 1) + 1}/{len(prompt_schedule)}\")\n\n"
            "        output_tuple = pipe.generate_avc(\n"
        )
        if generate_anchor not in source:
            raise RuntimeError(f"Cannot find generation anchor in {runner}")
        source = source.replace(generate_anchor, generate_replacement, 1)

        generation_call_index = source.index(generate_replacement)
        before_call = source[:generation_call_index]
        generation_call = source[generation_call_index:]
        prompt_call_anchor = "            prompt=prompt,\n"
        if prompt_call_anchor not in generation_call:
            raise RuntimeError(f"Cannot find generation prompt argument in {runner}")
        generation_call = generation_call.replace(
            prompt_call_anchor,
            "            prompt=segment_prompt,\n",
            1,
        )
        source = before_call + generation_call
    return source


def patch_avatar_attention_source(source: str, attention: Path) -> str:
    """Use PyTorch's tested SDPA kernel instead of flash-attn 2 on Blackwell.

    Avatar 1.5's INT8 config enables ``flashattn2``.  flash-attn 2.7.4 can
    intermittently corrupt the CUDA context on RTX PRO 6000 Blackwell while
    rendering a real Avatar scene.  PyTorch 2.7.1+cu128's native SDPA path is
    stable on that GPU and retains the regular flash-attn path elsewhere.
    """
    if BLACKWELL_SDPA_MARKER in source:
        return source

    import_anchor = "import torch\nimport torch.nn as nn\n"
    if import_anchor not in source:
        raise RuntimeError(f"Cannot find torch imports in {attention}")
    source = source.replace(
        import_anchor,
        "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n",
        1,
    )

    self_attention_branch = """        elif self.enable_flashattn2:
            from flash_attn import flash_attn_func
            q = rearrange(q, \"B H S D -> B S H D\")
            k = rearrange(k, \"B H S D -> B S H D\")
            v = rearrange(v, \"B H S D -> B S H D\")
            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=0.0,
                softmax_scale=self.scale,
            )
            x = rearrange(x, \"B S H D -> B H S D\")
"""
    self_attention_replacement = """        elif self.enable_flashattn2:
            # AI-Video-Gen Blackwell SDPA fallback: flash-attn 2.7.4 can
            # intermittently cause CUDA illegal-memory-access errors on SM120.
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12:
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=0.0, scale=self.scale
                )
            else:
                from flash_attn import flash_attn_func
                q = rearrange(q, \"B H S D -> B S H D\")
                k = rearrange(k, \"B H S D -> B S H D\")
                v = rearrange(v, \"B H S D -> B S H D\")
                x = flash_attn_func(
                    q,
                    k,
                    v,
                    dropout_p=0.0,
                    softmax_scale=self.scale,
                )
                x = rearrange(x, \"B S H D -> B H S D\")
"""
    if self_attention_branch not in source:
        raise RuntimeError(f"Cannot find Avatar self-attention flash-attn branch in {attention}")
    source = source.replace(self_attention_branch, self_attention_replacement, 1)

    cross_attention_branch = """        elif self.enable_flashattn2:
            from flash_attn import flash_attn_func
            q = rearrange(q, \"B H S D -> B S H D\")
            encoder_k = rearrange(encoder_k, \"B H S D -> B S H D\")
            encoder_v = rearrange(encoder_v, \"B H S D -> B S H D\")
            x = flash_attn_func(
                q,
                encoder_k,
                encoder_v,
                dropout_p=0.0,
                softmax_scale=self.scale,
            )
            x = rearrange(x, \"B S H D -> B H S D\")
"""
    cross_attention_replacement = """        elif self.enable_flashattn2:
            # AI-Video-Gen Blackwell SDPA fallback: flash-attn 2.7.4 can
            # intermittently cause CUDA illegal-memory-access errors on SM120.
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12:
                x = F.scaled_dot_product_attention(
                    q, encoder_k, encoder_v, dropout_p=0.0, scale=self.scale
                )
            else:
                from flash_attn import flash_attn_func
                q = rearrange(q, \"B H S D -> B S H D\")
                encoder_k = rearrange(encoder_k, \"B H S D -> B S H D\")
                encoder_v = rearrange(encoder_v, \"B H S D -> B S H D\")
                x = flash_attn_func(
                    q,
                    encoder_k,
                    encoder_v,
                    dropout_p=0.0,
                    softmax_scale=self.scale,
                )
                x = rearrange(x, \"B S H D -> B H S D\")
"""
    if cross_attention_branch not in source:
        raise RuntimeError(f"Cannot find Avatar cross-attention flash-attn branch in {attention}")
    return source.replace(cross_attention_branch, cross_attention_replacement, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    runner = repo / "run_demo_avatar_multi_audio_to_video.py"
    source = runner.read_text(encoding="utf-8")
    source = patch_source(source, runner)
    runner.write_text(source, encoding="utf-8")
    attention = repo / "longcat_video" / "modules" / "avatar" / "attention.py"
    attention_source = attention.read_text(encoding="utf-8")
    attention.write_text(
        patch_avatar_attention_source(attention_source, attention),
        encoding="utf-8",
    )
    print(f"Patched LongCat runtime: {runner}")


if __name__ == "__main__":
    main()
