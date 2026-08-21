from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_longcat_provisioning_supports_vast_templates_without_conda() -> None:
    script = (ROOT / "scripts" / "provision_longcat_avatar.sh").read_text(encoding="utf-8")

    assert 'if [ -x "${CONDA_BIN}" ]' in script
    assert "command -v uv" in script
    assert 'uv venv --python 3.10 "${LONGCAT_CONDA_ENV_DIR}"' in script
