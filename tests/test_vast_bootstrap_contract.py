from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_vast_uses_the_shared_ltx25_runtime() -> None:
    onstart = (ROOT / "scripts/onstart_vast_instance.sh").read_text(encoding="utf-8")
    deploy = (ROOT / "scripts/deploy_vast.sh").read_text(encoding="utf-8")
    bootstrap = (ROOT / "scripts/bootstrap_vast.sh").read_text(encoding="utf-8")

    assert 'GENERATOR_BACKEND="comfyui-ltx25"' in onstart
    assert 'LTX25_MEGAPIXELS="${LTX25_MEGAPIXELS:-0.9}"' in deploy
    assert 'COMFYUI_T2V_WORKFLOW="${COMFY_ROOT}/blueprints/video_ltx2_5_t2v.json"' in deploy
    assert 'COMFYUI_I2V_WORKFLOW="${COMFY_ROOT}/blueprints/video_ltx2_5_i2v.json"' in deploy
    assert 'scripts/provision_packet_comfyui.sh' in deploy
    assert 'scripts/download_comfy_ltx25_models.py' in deploy
    assert 'scripts/wait_for_comfyui_ready.py' in deploy
    assert 'download_comfy_ltx23_models.py' not in deploy
    assert 'download_comfy_ltx23_models.py' not in bootstrap


def test_vast_replaces_legacy_template_workflow_paths_before_api_launch() -> None:
    deploy = (ROOT / "scripts/deploy_vast.sh").read_text(encoding="utf-8")
    remote_server = (ROOT / "scripts/run_remote_server.sh").read_text(encoding="utf-8")

    assert 'export COMFYUI_T2V_WORKFLOW="${COMFY_ROOT}/blueprints/video_ltx2_5_t2v.json"' in deploy
    assert 'export COMFYUI_I2V_WORKFLOW="${COMFY_ROOT}/blueprints/video_ltx2_5_i2v.json"' in deploy
    assert 'COMFYUI_T2V_WORKFLOW="${COMFYUI_T2V_WORKFLOW}" \\' in deploy
    assert 'COMFYUI_I2V_WORKFLOW="${COMFYUI_I2V_WORKFLOW}" \\' in deploy
    assert 'COMFYUI_T2V_WORKFLOW="${COMFYUI_T2V_WORKFLOW}"' in remote_server
    assert 'COMFYUI_I2V_WORKFLOW="${COMFYUI_I2V_WORKFLOW}"' in remote_server


def test_vast_keeps_longcat_separate_and_sequential() -> None:
    deploy = (ROOT / "scripts/deploy_vast.sh").read_text(encoding="utf-8")

    assert 'AI_VIDEO_GEN_EXECUTION_PROFILE="standard"' in deploy
    assert 'AI_VIDEO_GEN_RELEASE_LONGCAT_WEIGHTS_AFTER_BRANCH="0"' in deploy
    assert 'LONGCAT_CONDA_ENV_DIR="${LONGCAT_ENV_DIR}"' in deploy
    assert 'scripts/provision_longcat_avatar.sh' in deploy
    assert 'AI_VIDEO_GEN_ENABLE_LONGCAT="${ENABLE_LONGCAT}"' in deploy
