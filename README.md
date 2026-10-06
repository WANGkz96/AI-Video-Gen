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

## Remote generation

Packet and Vast use the same pinned ComfyUI LTX 2.5 workflows and model pack.
Dialogue scenes use LongCat Avatar in a separate Python environment. Vast
RTX PRO 6000 instances process those branches sequentially.

The Vast template starts [`scripts/onstart_vast_instance.sh`](./scripts/onstart_vast_instance.sh),
which checks out the requested repository revision and calls
[`scripts/deploy_vast.sh`](./scripts/deploy_vast.sh). See
[`docs/vast-comfyui-template.md`](./docs/vast-comfyui-template.md) for its
configuration and readiness checks. LTX 2.3 variables from older Vast
templates are ignored by this startup path.

## Пример входного batch

Файл для локальной проверки лежит в [`examples/sample-batch.json`](./examples/sample-batch.json).
