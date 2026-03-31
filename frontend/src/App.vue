<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";

type BackendInfo = {
  key: string;
  label: string;
  description: string;
  status: "ready" | "planned" | "experimental";
  available: boolean;
  notes?: string | null;
};

type JobSnapshot = {
  jobId: string;
  status: string;
  backend: string;
  totalVideos: number;
  totalVariants: number;
  totalSegments: number;
  completedSegments: number;
  failedSegments: number;
  startedAt?: string | null;
  updatedAt: string;
  finishedAt?: string | null;
  archiveFile?: string | null;
};

type LogEntry = {
  ts: string;
  level: "info" | "warning" | "error";
  message: string;
};

type SegmentResult = {
  segmentId: string;
  videoFile?: string;
  durationSec?: number;
  width?: number;
  height?: number;
  fps?: number;
  status?: string;
  error?: string;
};

type ResultPayload = {
  schemaVersion: string;
  generatedAt: string;
  videos: Array<{
    videoId: number;
    projectId: number;
    runId: string;
    variants: Array<{
      key: string;
      segments: SegmentResult[];
    }>;
  }>;
  errors?: Array<{ error: string; segmentId?: string; variantKey?: string }>;
};

const health = ref<Record<string, unknown> | null>(null);
const backends = ref<BackendInfo[]>([]);
const selectedBatchBackend = ref("mock-gen");
const batchFile = ref<File | null>(null);
const selectedDirectBackend = ref("mock-gen");
const directForm = ref({
  title: "Manual generation",
  prompt: "A dynamic futuristic city with cinematic motion and clear subject focus.",
  negativePrompt: "blurry, low quality, glitch",
  continuityNote: "",
  shotGoal: "Single hero shot",
  spokenText: "",
  subtitleText: "",
  durationSec: 8,
  width: 720,
  height: 1280,
  fps: 24,
  globalVisualDirection: "Portrait frame, cinematic realism, warm lighting.",
  globalNegativePrompt: "text overlays, artifacts, deformation",
});
const activeJob = ref<JobSnapshot | null>(null);
const logs = ref<LogEntry[]>([]);
const result = ref<ResultPayload | null>(null);
const busy = ref(false);
const errorMessage = ref("");
let source: EventSource | null = null;

const progressPercent = computed(() => {
  if (!activeJob.value || activeJob.value.totalSegments === 0) {
    return 0;
  }
  return Math.min(
    100,
    Math.round(((activeJob.value.completedSegments + activeJob.value.failedSegments) / activeJob.value.totalSegments) * 100),
  );
});

const successfulSegments = computed(() => {
  if (!result.value) {
    return [];
  }
  return result.value.videos.flatMap((video) =>
    video.variants.flatMap((variant) =>
      variant.segments
        .filter((segment) => segment.videoFile && segment.status !== "failed")
        .map((segment) => ({
          ...segment,
          videoId: video.videoId,
          variantKey: variant.key,
        })),
    ),
  );
});

const archiveUrl = computed(() =>
  activeJob.value ? `/api/jobs/${activeJob.value.jobId}/archive` : "",
);

const resultUrl = computed(() =>
  activeJob.value ? `/api/jobs/${activeJob.value.jobId}/result` : "",
);

function metadataUrl(videoFile: string) {
  if (!activeJob.value) {
    return "#";
  }
  return `/api/jobs/${activeJob.value.jobId}/files/${videoFile.replace(/\.mp4$/i, ".json")}`;
}

function videoUrl(videoFile: string) {
  if (!activeJob.value) {
    return "#";
  }
  return `/api/jobs/${activeJob.value.jobId}/files/${videoFile}`;
}

function chooseBatchFile(event: Event) {
  const target = event.target as HTMLInputElement;
  batchFile.value = target.files?.[0] ?? null;
}

async function loadBootstrap() {
  const [healthResponse, backendResponse] = await Promise.all([
    fetch("/api/health"),
    fetch("/api/backends"),
  ]);
  health.value = await healthResponse.json();
  backends.value = await backendResponse.json();
}

async function startBatchJob() {
  if (!batchFile.value) {
    errorMessage.value = "Выберите batch.json.";
    return;
  }
  busy.value = true;
  errorMessage.value = "";
  try {
    const form = new FormData();
    form.append("batch", batchFile.value);
    form.append("backend", selectedBatchBackend.value);
    const response = await fetch("/api/jobs", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail ?? "Не удалось создать batch job.");
    }
    await activateJob(payload.jobId);
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  } finally {
    busy.value = false;
  }
}

async function startDirectJob() {
  busy.value = true;
  errorMessage.value = "";
  try {
    const response = await fetch("/api/jobs/direct", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        backend: selectedDirectBackend.value,
        ...directForm.value,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail ?? "Не удалось создать direct job.");
    }
    await activateJob(payload.jobId);
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  } finally {
    busy.value = false;
  }
}

async function activateJob(jobId: string) {
  result.value = null;
  await Promise.all([fetchJob(jobId), fetchLogs(jobId)]);
  connectEvents(jobId);
  if (activeJob.value && ["completed", "completed_with_errors"].includes(activeJob.value.status)) {
    await fetchResult(jobId);
  }
}

async function fetchJob(jobId: string) {
  const response = await fetch(`/api/jobs/${jobId}`);
  if (!response.ok) {
    throw new Error("Не удалось загрузить job.");
  }
  activeJob.value = await response.json();
}

async function fetchLogs(jobId: string) {
  const response = await fetch(`/api/jobs/${jobId}/logs`);
  if (!response.ok) {
    throw new Error("Не удалось загрузить логи.");
  }
  logs.value = await response.json();
}

async function fetchResult(jobId: string) {
  const response = await fetch(`/api/jobs/${jobId}/result`);
  if (!response.ok) {
    return;
  }
  result.value = await response.json();
}

function connectEvents(jobId: string) {
  source?.close();
  source = new EventSource(`/api/jobs/${jobId}/events`);
  source.onmessage = async (event) => {
    const payload = JSON.parse(event.data) as { type: string; data: unknown };
    if (payload.type === "snapshot") {
      activeJob.value = payload.data as JobSnapshot;
      if (["completed", "completed_with_errors"].includes(activeJob.value.status)) {
        await fetchResult(jobId);
      }
    }
    if (payload.type === "log") {
      logs.value = [...logs.value, payload.data as LogEntry];
    }
  };
  source.onerror = () => {
    source?.close();
  };
}

onMounted(() => {
  loadBootstrap().catch((error) => {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  });
});

onBeforeUnmount(() => {
  source?.close();
});
</script>

<template>
  <main class="page-shell">
    <section class="hero">
      <div>
        <p class="eyebrow">Batch + Direct Video Pipeline</p>
        <h1>AI Video Generation Service</h1>
        <p class="hero-copy">
          Один интерфейс для прогона batch JSON, ручного вызова сегмента и просмотра всех логов,
          ошибок и артефактов generation pipeline.
        </p>
      </div>
      <div class="hero-status">
        <span class="status-pill">{{ health?.status ?? "..." }}</span>
        <p>Default backend: <strong>{{ health?.defaultBackend ?? "mock-gen" }}</strong></p>
        <p>Queue: <strong>{{ health?.queuedJobs ?? 0 }}</strong></p>
      </div>
    </section>

    <section class="grid">
      <article class="panel">
        <div class="panel-head">
          <h2>Batch Job</h2>
          <span>Tech-card contract</span>
        </div>
        <label class="field">
          <span>Backend</span>
          <select v-model="selectedBatchBackend">
            <option v-for="backend in backends" :key="backend.key" :value="backend.key" :disabled="!backend.available">
              {{ backend.label }} · {{ backend.status }}
            </option>
          </select>
        </label>
        <label class="dropzone">
          <input type="file" accept=".json,application/json" @change="chooseBatchFile" />
          <strong>{{ batchFile?.name ?? "Перетащите или выберите batch.json" }}</strong>
          <span>Загрузка через multipart в `POST /api/jobs`</span>
        </label>
        <button class="cta" :disabled="busy" @click="startBatchJob">Запустить batch</button>
      </article>

      <article class="panel">
        <div class="panel-head">
          <h2>Direct Job</h2>
          <span>1 synthetic segment</span>
        </div>
        <div class="field-grid">
          <label class="field">
            <span>Backend</span>
            <select v-model="selectedDirectBackend">
              <option v-for="backend in backends" :key="backend.key" :value="backend.key" :disabled="!backend.available">
                {{ backend.label }} · {{ backend.status }}
              </option>
            </select>
          </label>
          <label class="field">
            <span>Title</span>
            <input v-model="directForm.title" />
          </label>
        </div>
        <label class="field">
          <span>Prompt</span>
          <textarea v-model="directForm.prompt" rows="3" />
        </label>
        <div class="field-grid">
          <label class="field">
            <span>Negative Prompt</span>
            <input v-model="directForm.negativePrompt" />
          </label>
          <label class="field">
            <span>Shot Goal</span>
            <input v-model="directForm.shotGoal" />
          </label>
        </div>
        <div class="field-grid field-grid--triple">
          <label class="field"><span>Duration</span><input v-model.number="directForm.durationSec" type="number" min="1" step="0.1" /></label>
          <label class="field"><span>Width</span><input v-model.number="directForm.width" type="number" min="64" step="1" /></label>
          <label class="field"><span>Height</span><input v-model.number="directForm.height" type="number" min="64" step="1" /></label>
        </div>
        <button class="cta cta--ink" :disabled="busy" @click="startDirectJob">Запустить direct job</button>
      </article>
    </section>

    <p v-if="errorMessage" class="error-banner">{{ errorMessage }}</p>

    <section class="grid grid--wide">
      <article class="panel">
        <div class="panel-head">
          <h2>Job Monitor</h2>
          <span>{{ activeJob?.jobId ?? "job not selected" }}</span>
        </div>
        <div v-if="activeJob" class="stats">
          <div class="stat"><span>Status</span><strong>{{ activeJob.status }}</strong></div>
          <div class="stat"><span>Backend</span><strong>{{ activeJob.backend }}</strong></div>
          <div class="stat"><span>Segments</span><strong>{{ activeJob.completedSegments }}/{{ activeJob.totalSegments }}</strong></div>
          <div class="stat"><span>Failed</span><strong>{{ activeJob.failedSegments }}</strong></div>
        </div>
        <div class="progress">
          <div class="progress-bar" :style="{ width: `${progressPercent}%` }"></div>
        </div>
        <div class="download-row" v-if="activeJob && ['completed', 'completed_with_errors'].includes(activeJob.status)">
          <a class="link-button" :href="archiveUrl" download>Download zip</a>
          <a class="link-button link-button--ghost" :href="resultUrl" target="_blank">Open result.json</a>
        </div>
        <div class="log-box">
          <div v-for="entry in logs" :key="`${entry.ts}-${entry.message}`" class="log-entry" :data-level="entry.level">
            <span>{{ new Date(entry.ts).toLocaleTimeString() }}</span>
            <strong>{{ entry.level }}</strong>
            <p>{{ entry.message }}</p>
          </div>
        </div>
      </article>

      <article class="panel">
        <div class="panel-head">
          <h2>Artifacts</h2>
          <span>{{ successfulSegments.length }} previewable segments</span>
        </div>
        <div class="artifact-grid" v-if="successfulSegments.length">
          <article v-for="segment in successfulSegments" :key="segment.segmentId" class="artifact-card">
            <video :src="videoUrl(segment.videoFile!)" controls preload="metadata"></video>
            <h3>{{ segment.segmentId }}</h3>
            <p>{{ segment.width }}×{{ segment.height }} · {{ segment.fps?.toFixed(2) }} fps · {{ segment.durationSec?.toFixed(2) }}s</p>
            <div class="artifact-actions">
              <a :href="videoUrl(segment.videoFile!)" download>video</a>
              <a :href="metadataUrl(segment.videoFile!)" target="_blank">metadata</a>
            </div>
          </article>
        </div>
        <div class="empty-state" v-else>
          <p>После завершения job здесь появятся preview и ссылки на каждый сегмент отдельно.</p>
        </div>
        <div class="empty-state" v-if="result?.errors?.length">
          <h3>Errors</h3>
          <p v-for="entry in result.errors" :key="`${entry.segmentId}-${entry.error}`">{{ entry.error }}</p>
        </div>
      </article>
    </section>
  </main>
</template>

<style scoped>
.page-shell { max-width: 1420px; margin: 0 auto; padding: 32px 20px 56px; }
.hero, .panel { backdrop-filter: blur(14px); box-shadow: var(--shadow); border: 1px solid var(--line); background: var(--surface); }
.hero { display: grid; grid-template-columns: 1.6fr 0.8fr; gap: 24px; border-radius: 32px; padding: 28px; margin-bottom: 24px; }
.eyebrow { margin: 0 0 12px; text-transform: uppercase; letter-spacing: 0.18em; color: var(--ink); font-size: 12px; }
h1 { margin: 0; font-size: clamp(2.4rem, 4vw, 4.8rem); line-height: 0.95; }
.hero-copy { max-width: 760px; color: var(--muted); font-size: 1.02rem; }
.hero-status { padding: 18px; border-radius: 24px; background: linear-gradient(180deg, rgba(255,255,255,.7), rgba(255,255,255,.35)); }
.hero-status p { margin: 8px 0 0; color: var(--muted); }
.status-pill { display: inline-flex; padding: 8px 14px; border-radius: 999px; background: var(--accent-soft); color: var(--accent); font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; font-size: 11px; }
.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 24px; margin-bottom: 24px; }
.grid--wide { grid-template-columns: 0.95fr 1.05fr; }
.panel { border-radius: 28px; padding: 24px; }
.panel-head { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; margin-bottom: 18px; }
.panel-head h2 { margin: 0; font-size: 1.4rem; }
.panel-head span { color: var(--muted); font-size: 0.9rem; }
.field, .dropzone { display: flex; flex-direction: column; gap: 8px; margin-bottom: 14px; }
.field span { color: var(--muted); font-size: 0.88rem; }
input, select, textarea { width: 100%; border: 1px solid var(--line); border-radius: 18px; padding: 14px 16px; background: var(--surface-strong); color: var(--text); }
textarea { resize: vertical; min-height: 108px; }
.dropzone { position: relative; padding: 22px; border: 1px dashed rgba(15,124,130,.35); border-radius: 22px; background: var(--ink-soft); }
.dropzone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
.dropzone strong { font-size: 1rem; }
.dropzone span { color: var(--muted); }
.field-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.field-grid--triple { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.cta, .link-button { display: inline-flex; align-items: center; justify-content: center; min-height: 52px; border-radius: 18px; border: none; background: var(--accent); color: white; font-weight: 700; text-decoration: none; }
.cta--ink { background: var(--ink); }
.cta:disabled { opacity: 0.6; cursor: not-allowed; }
.error-banner { margin: 0 0 24px; padding: 16px 18px; border-radius: 18px; background: rgba(198, 51, 56, 0.12); border: 1px solid rgba(198, 51, 56, 0.22); color: #8f2228; }
.stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 18px; }
.stat { padding: 16px; border-radius: 18px; background: rgba(255,255,255,.66); border: 1px solid var(--line); }
.stat span { display: block; color: var(--muted); font-size: 0.82rem; margin-bottom: 6px; }
.progress { height: 14px; border-radius: 999px; background: rgba(31,40,51,.08); overflow: hidden; margin-bottom: 18px; }
.progress-bar { height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--ink), var(--accent)); transition: width 180ms ease; }
.download-row, .artifact-actions { display: flex; gap: 12px; flex-wrap: wrap; }
.link-button--ghost { background: transparent; color: var(--ink); border: 1px solid rgba(15,124,130,.25); }
.log-box { display: flex; flex-direction: column; gap: 10px; max-height: 560px; overflow: auto; padding-right: 4px; }
.log-entry { display: grid; grid-template-columns: 92px 64px 1fr; gap: 12px; align-items: start; padding: 13px 14px; border-radius: 16px; background: rgba(255,255,255,.64); font-family: "IBM Plex Mono", monospace; font-size: 0.82rem; border: 1px solid var(--line); }
.log-entry p { margin: 0; }
.log-entry[data-level="error"] { border-color: rgba(198,51,56,.28); background: rgba(198,51,56,.08); }
.artifact-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 16px; }
.artifact-card { display: flex; flex-direction: column; gap: 10px; padding: 14px; border-radius: 20px; background: rgba(255,255,255,.66); border: 1px solid var(--line); }
.artifact-card video { width: 100%; aspect-ratio: 9 / 16; object-fit: cover; border-radius: 14px; background: #0f1115; }
.artifact-card h3, .artifact-card p, .empty-state h3, .empty-state p { margin: 0; }
.artifact-card p, .empty-state p { color: var(--muted); }
.empty-state { display: flex; flex-direction: column; gap: 10px; padding: 22px; border-radius: 20px; background: rgba(255,255,255,.54); border: 1px dashed var(--line); }
@media (max-width: 1080px) { .hero, .grid, .grid--wide, .field-grid, .field-grid--triple, .stats { grid-template-columns: 1fr; } }
</style>
