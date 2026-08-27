from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_longcat_provisioning_supports_vast_templates_without_conda() -> None:
    script = (ROOT / "scripts" / "provision_longcat_avatar.sh").read_text(encoding="utf-8")

    assert 'if [ -x "${CONDA_BIN}" ]' in script
    assert "command -v uv" in script
    assert 'uv venv --python 3.10 --seed "${LONGCAT_CONDA_ENV_DIR}"' in script
    assert 'if ! "${PYTHON_BIN}" -m pip --version' in script
    assert 'uv pip install --python "${PYTHON_BIN}" pip' in script
    assert "torchvision|torchaudio|numpy|sympy|libsndfile1" in script
    assert 'git clone --depth 1 --no-tags' in script
    assert 'git -C "${LONGCAT_REPO_DIR}" fetch --depth 1 origin' in script


def test_packet_avatar_only_boot_skips_irrelevant_comfyui_runtime() -> None:
    script = (ROOT / "scripts" / "deploy_packet.sh").read_text(encoding="utf-8")

    assert 'if [ "${AI_VIDEO_GEN_ENABLE_LTX}" = "1" ]; then' in script
    assert 'Skipping ComfyUI provisioning: this Packet job has no LTX branch.' in script
