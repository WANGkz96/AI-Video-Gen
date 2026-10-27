from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_remote_api_uses_an_isolated_process_session():
    script = (ROOT / 'scripts/run_remote_server.sh').read_text()
    assert 'nohup setsid env PORT=' in script


def test_packet_caps_host_threads_and_selects_safe_allocator():
    script = (ROOT / 'scripts/deploy_packet.sh').read_text()
    assert 'AI_VIDEO_GEN_CPU_THREADS="${AI_VIDEO_GEN_CPU_THREADS:-6}"' in script
    for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        assert f'export {variable}="${{AI_VIDEO_GEN_CPU_THREADS}}"' in script
    assert 'export COMFYUI_SAFE_CUDA_MEMORY="${COMFYUI_SAFE_CUDA_MEMORY:-1}"' in script
    assert 'export LONGCAT_OFFLOAD_ENCODERS="${LONGCAT_OFFLOAD_ENCODERS:-1}"' in script
    assert 'AI_VIDEO_GEN_CPU_THREADS must be a positive integer.' in script


def test_shared_comfy_bootstrap_keeps_memory_switches_opt_in():
    script = (ROOT / 'scripts/start_comfyui.sh').read_text()
    assert 'if [ "${COMFYUI_SAFE_CUDA_MEMORY:-0}" = "1" ]; then' in script
    assert 'COMFY_MEMORY_ARGS=()' in script
    assert '"${COMFY_MEMORY_ARGS[@]}"' in script
    for flag in ('--disable-cuda-malloc', '--disable-dynamic-vram', '--disable-async-offload', '--disable-pinned-memory'):
        assert flag in script
    assert 'CUDA_LAUNCH_BLOCKING' not in script


def test_mixed_standard_packet_prepares_but_defers_comfy_process():
    deploy = (ROOT / 'scripts/deploy_packet.sh').read_text()
    provision = (ROOT / 'scripts/provision_packet_comfyui.sh').read_text()
    assert 'COMFYUI_DEFER_START=1' in deploy
    assert 'COMFYUI_DEFER_START="${COMFYUI_DEFER_START}"' in deploy
    assert '--comfy-start-script "${ROOT_DIR}/scripts/start_comfyui.sh"' in deploy
    assert 'if [ "${COMFYUI_DEFER_START:-0}" = "1" ]; then' in provision
    assert '"$(dirname "${BASH_SOURCE[0]}")/start_comfyui.sh"' in provision
