import json
import wave
from pathlib import Path

import pytest
from pydantic import ValidationError
from backend.app.models import DialogueSceneAudio, DialogueSceneGenerationRequest
from backend.app.adapters.longcat_avatar import LongCatAvatarAdapter


def test_audio_contract_keeps_legacy_multi_and_validates_single():
    assert DialogueSceneAudio(speaker1File='one.wav', speaker2File='two.wav').mode == 'multi'
    assert DialogueSceneAudio(mode='single', speaker1File='one.wav').speaker2File is None
    with pytest.raises(ValidationError):
        DialogueSceneAudio(speaker1File='one.wav')
    with pytest.raises(ValidationError):
        DialogueSceneAudio(mode='single', speaker1File='one.wav', speaker2File='two.wav')


def test_single_preparation_has_one_audio_and_no_spatial_assignment(tmp_path: Path):
    audio = tmp_path / 'narrator.wav'
    with wave.open(str(audio), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b'\x01\x00' * 2400)
    request = DialogueSceneGenerationRequest(
        jobId='job', videoId=1, projectId=1, runId='run', videoTitle='Test',
        variantKey='v01', variantLabel='Variant 1', sceneId='scene_01', sceneIndex=1,
        prompt='The narrator explains the discovery.', imagePath=tmp_path / 'frame.png',
        speaker1Path=audio, audioMode='single', width=1280, height=720, fps=25,
        durationSec=8, outputPath=tmp_path / 'scene' / 'output.mp4',
        avatarIdentity={'person1': 'A patient archivist'},
    )
    result = object.__new__(LongCatAvatarAdapter)._prepare_scene(request)
    data = json.loads(result['inputPath'].read_text(encoding='utf-8'))
    assert data['cond_audio'] == {'person1': audio.as_posix()}
    assert data['avatar_layout'] == {}
    assert data['avatar_identity'] == {'person1': 'A patient archivist'}
    assert data['audio_mode'] == 'single'
    assert 'two people' not in data['prompt']
    assert result['substitutedSilentTracks'] == []


def test_single_runner_prepares_one_embedding_and_no_masks(tmp_path):
    # Execute the actual preparation functions with CPU stubs; importing the
    # runner normally provisions GPU-only libraries from the LongCat checkout.
    import ast
    import math
    from types import SimpleNamespace
    import numpy as np

    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / 'scripts/run_longcat_avatar_batch.py').read_text(encoding='utf-8'))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {'_make_audio_embeddings', '_target_masks'}]
    for function in functions:
        function.returns = None
        for argument in function.args.args:
            argument.annotation = None
    calls = []
    written = []
    scope = {
        'math': math, 'np': np,
        'librosa': SimpleNamespace(load=lambda *a, **kw: (np.ones(1600), 16000)),
        'torch': SimpleNamespace(isnan=np.isnan),
        'sf': SimpleNamespace(write=lambda *args: written.append(args)),
        'generate_random_uid': lambda: 'fixture',
        'extract_vocal_from_speech': lambda source, *args: source,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<runner-functions>', 'exec'), scope)
    def embed(speech, **kwargs):
        calls.append((speech, kwargs))
        return np.ones((10, 2))
    runtime = SimpleNamespace(num_frames=93, save_fps=25, num_cond_frames=13, audio_stride=1, local_rank=0, audio_temp_dir=tmp_path, vocal_separator=None, pipe=SimpleNamespace(get_audio_embedding=embed))
    result = scope['_make_audio_embeddings'](runtime, {'audio_mode': 'single', 'cond_audio': {'person1': 'speech.wav'}}, 2)
    assert len(calls) == 1
    assert result[1] is None
    assert len(calls[0][0]) == math.ceil((93 / 25 + 80 / 25) * 16000)
    assert len(written) == 1
    assert scope['_target_masks'](None, {'audio_mode': 'single'}, 0) is None
