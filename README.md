# AI Video Generation Service

Сервис собирает batch JSON из внешней платформы, прогоняет сегменты через выбранный backend-адаптер и возвращает `zip` с `result.json`, входным `batch.json` и сгенерированными видео/метаданными. Первый рабочий backend здесь `mock-gen`: он не генерирует видео, а по кругу копирует файлы из [`mock-media`](./mock-media), чтобы можно было отстроить весь пайплайн без GPU.

## Что уже есть

- Python API на FastAPI с очередью фоновых задач и последовательной обработкой сегментов.
- Контракт batch-обработки по [`docs/video-generation-service-tech-card.md`](./docs/video-generation-service-tech-card.md).
- Расширяемый adapter layer: `mock-gen`, scaffold под `comfyui-workflow`, плюс planned backends под `ltx-video-2`, `ltx-video-2-distilled`, `hunyuan-video`, `wan-2.1`, `cogvideox`.
- SSE-стрим логов и прогресса.
- Ручной direct mode: можно отправить один сегмент как отдельную job.
- Vue UI для загрузки JSON, ручного запуска, просмотра логов и скачивания результатов.

## Быстрый старт

### Backend

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
copy .env.example .env
uvicorn backend.app.main:app --reload --port 3001
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

## Основные API

- `GET /api/health`
- `GET /api/backends`
- `POST /api/jobs`
- `POST /api/jobs/direct`
- `GET /api/jobs/{jobId}`
- `GET /api/jobs/{jobId}/logs`
- `GET /api/jobs/{jobId}/events`
- `GET /api/jobs/{jobId}/result`
- `GET /api/jobs/{jobId}/archive`
- `GET /api/jobs/{jobId}/files/{relativePath}`

## Mock Backend

`mock-gen`:

- принимает стандартный adapter request;
- берёт следующий `.mp4` из [`mock-media`](./mock-media) по имени;
- копирует его в выходную структуру job;
- пишет debug-информацию о том, какой исходник был использован.

## Real Models

- `cogvideox-5b`, `wan2.2-ti2v-5b`, `sana-video-2b`, `hunyuan-video-1.5`, `ltx-2-distilled` работают через Diffusers.
- `ltx-2.3` работает через официальный native runtime от Lightricks, а не через Diffusers.
- При `download-models` или `scripts/deploy_vast.sh` для `ltx-2.3` сервис:
  - скачивает `ltx-2.3-22b-dev.safetensors` в `models/ltx-2.3/`;
  - скачивает Gemma 3 assets в `models/gemma-3-12b-it-qat-q4_0-unquantized/`;
  - клонирует pinned revision официального `LTX-2` runtime в `data/runtime/ltx-2-official/`;
  - устанавливает `ltx-core` и `ltx-pipelines` в текущее виртуальное окружение.
- Для `ltx-2.3` нужен `HF_TOKEN` с одобренным доступом к gated repo `google/gemma-3-12b-it-qat-q4_0-unquantized`.

Для полного Vast deploy:

```bash
git clone https://github.com/WANGkz96/AI-Video-Gen.git
cd AI-Video-Gen
PORT=8090 MODELS=all ./scripts/deploy_vast.sh
```

## ComfyUI Foundation

В [`backend/app/adapters/comfyui.py`](./backend/app/adapters/comfyui.py) есть каркас под backend `comfyui-workflow`. Он пока не запускает workflow реально, но уже выделяет место под payload builder и будущий вызов внешнего ComfyUI API.

## Пример входного batch

Файл для локальной проверки лежит в [`examples/sample-batch.json`](./examples/sample-batch.json).
