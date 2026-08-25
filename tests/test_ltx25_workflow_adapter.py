from __future__ import annotations

from types import SimpleNamespace

from backend.app.adapters.comfyui import ComfyUiWorkflowAdapter
from backend.app.services.provisioning import COMFY_LTX25_MODEL_NAMES


def test_ltx25_workflow_selects_packet_model_pack() -> None:
    adapter = object.__new__(ComfyUiWorkflowAdapter)
    prompt = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "bf16.safetensors"}},
        "2": {
            "class_type": "CLIPLoader",
            "_meta": {"title": "Load CLIP - Text Encoder"},
            "inputs": {"clip_name": "bf16.safetensors"},
        },
        "3": {
            "class_type": "CLIPLoader",
            "_meta": {"title": "Load CLIP - Text Enhancer"},
            "inputs": {"clip_name": "bf16.safetensors"},
        },
        "4": {
            "class_type": "VAELoader",
            "_meta": {"title": "Load Video VAE"},
            "inputs": {"vae_name": "bf16.safetensors"},
        },
        "5": {
            "class_type": "VAELoader",
            "_meta": {"title": "Load Audio VAE"},
            "inputs": {"vae_name": "bf16.safetensors"},
        },
        "6": {
            "class_type": "LTXInputParameters",
            "inputs": {"value_2": "", "value_3": False, "value_5": 8, "ckpt_name": "old.safetensors"},
        },
    }

    adapter._set_ltx25_model_files(prompt)

    assert prompt["1"]["inputs"]["unet_name"] == COMFY_LTX25_MODEL_NAMES["transformer"]
    assert prompt["2"]["inputs"]["clip_name"] == COMFY_LTX25_MODEL_NAMES["text_encoder"]
    assert prompt["3"]["inputs"]["clip_name"] == COMFY_LTX25_MODEL_NAMES["text_enhancer"]
    assert prompt["4"]["inputs"]["vae_name"] == COMFY_LTX25_MODEL_NAMES["video_vae"]
    assert prompt["5"]["inputs"]["vae_name"] == COMFY_LTX25_MODEL_NAMES["audio_vae"]
    assert prompt["6"]["inputs"]["ckpt_name"] == COMFY_LTX25_MODEL_NAMES["transformer"]


def test_ltx25_workflow_sets_resolution_on_joint_input_node() -> None:
    adapter = object.__new__(ComfyUiWorkflowAdapter)
    adapter._settings = SimpleNamespace(
        comfyui_t2v_workflow="ltx25.json",
        comfyui_i2v_workflow="ltx25.json",
    )
    prompt = {
        "5514": {
            "class_type": "LTXVPreprocess",
            "inputs": {
                "width": 960,
                "height": 544,
            },
        }
    }

    adapter._set_ltx25_resolution(prompt, width=1280, height=720)

    assert prompt["5514"]["inputs"]["width"] == 1280
    assert prompt["5514"]["inputs"]["height"] == 720


def test_ltx25_workflow_accepts_official_duration_control_title() -> None:
    adapter = object.__new__(ComfyUiWorkflowAdapter)
    prompt = {
        "5512": {
            "class_type": "PrimitiveFloat",
            "_meta": {"title": "duration in seconds (determines frames #)"},
            "inputs": {"value": 5.0},
        }
    }

    adapter._set_primitive_number(
        prompt,
        ("Duration", "duration in seconds", "duration in seconds (determines frames #)"),
        8,
    )

    assert prompt["5512"]["inputs"]["value"] == 8.0
