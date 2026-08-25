<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";

type BackendInfo = {
  key: string;
  label: string;
  description: string;
  status: "ready" | "planned" | "experimental";
  available: boolean;
  notes?: string | null;
  requiresDownload?: boolean;
};

type ProvisioningModel = {
  id: string;
  label: string;
  targetPath: string;
  status: string;
  bytesDownloaded?: number | null;
  totalBytes?: number | null;
  progressPercent?: number | null;
  error?: string | null;
};

type ProvisioningPayload = {
  ready: boolean;
  status: string;
  message: string;
  progressPercent: number;
  modelFilesReady: number;
  modelFilesTotal: number;
  models: ProvisioningModel[];
  current?: {
    id: string;
    label: string;
    index: number;
    total: number;
    bytesDownloaded?: number | null;
    totalBytes?: number | null;
    attempt?: number | null;
    maxAttempts?: number | null;
    targetPath: string;
  } | null;
  error?: string | null;
  comfyui?: {
    ready: boolean;
    apiUrl: string;
    error?: string | null;
  };
  workflows?: {
    ready: boolean;
    missing: string[];
  };
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

type HealthPayload = {
  status: string;
  ready: boolean;
  defaultBackend: string;
  queuedJobs: number;
  activeJobs: number;
  frontendBuilt: boolean;
  provisioning?: ProvisioningPayload;
};

const health = ref<HealthPayload | null>(null);
const provisioning = ref<ProvisioningPayload | null>(null);
const backends = ref<BackendInfo[]>([]);
const selectedBatchBackend = ref("comfyui-ltx25");
const batchFile = ref<File | null>(null);
const selectedDirectBackend = ref("comfyui-ltx25");
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
  fps: 25,
  seed: 42,
  globalVisualDirection: "Portrait frame, cinematic realism, warm lighting.",
  globalNegativePrompt: "text overlays, artifacts, deformation",
});
const activeJob = ref<JobSnapshot | null>(null);
const recentJobs = ref<JobSnapshot[]>([]);
const logs = ref<LogEntry[]>([]);
const result = ref<ResultPayload | null>(null);
const busy = ref(false);
const errorMessage = ref("");
const apiToken = ref("");
let source: EventSource | null = null;
let readinessTimer: number | undefined;

const appReady = computed(() => Boolean(provisioning.value?.ready || health.value?.ready));

const provisioningPercent = computed(() =>
  Math.max(0, Math.min(100, Math.round(provisioning.value?.progressPercent ?? 0))),
);

const currentDownloadText = computed(() => {
  const current = provisioning.value?.current;
  if (!current) {
    return provisioning.value?.message ?? "Checking startup state...";
  }
  const attempt = current.maxAttempts === 0
    ? `attempt ${current.attempt ?? 1}`
    : `attempt ${current.attempt ?? 1}/${current.maxAttempts ?? "?"}`;
  return `${current.label} (${current.index}/${current.total}, ${attempt})`;
});

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
  activeJob.value ? apiUrl(`/api/jobs/${activeJob.value.jobId}/archive`) : "",
);

const resultUrl = computed(() =>
  activeJob.value ? apiUrl(`/api/jobs/${activeJob.value.jobId}/result`) : "",
);

function isTerminalStatus(status: string) {
  return ["completed", "completed_with_errors", "failed"].includes(status);
}

function formatBytes(value?: number | null) {
  if (!value || value <= 0) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${size.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function metadataUrl(videoFile: string) {
  if (!activeJob.value) {
    return "#";
  }
  return apiUrl(`/api/jobs/${activeJob.value.jobId}/files/${videoFile.replace(/\.mp4$/i, ".json")}`);
}

function videoUrl(videoFile: string) {
  if (!activeJob.value) {
    return "#";
  }
  return apiUrl(`/api/jobs/${activeJob.value.jobId}/files/${videoFile}`);
}

function chooseBatchFile(event: Event) {
  const target = event.target as HTMLInputElement;
  batchFile.value = target.files?.[0] ?? null;
}

async function readJson<T>(response: Response, fallback: T): Promise<T> {
  const text = await response.text();
  if (!text.trim()) {
    if (response.ok) {
      return fallback;
    }
    throw new Error(`HTTP ${response.status}: empty response`);
  }
  try {
    return JSON.parse(text) as T;
  } catch (error) {
    if (response.ok) {
      return fallback;
    }
    throw new Error(`HTTP ${response.status}: invalid JSON response`);
  }
}

function initializeApiToken() {
  const params = new URLSearchParams(window.location.search);
  const tokenFromUrl = params.get("token")?.trim();
  const savedToken = window.sessionStorage.getItem("ai-video-gen.apiToken")?.trim();
  apiToken.value = tokenFromUrl || savedToken || "";
  if (tokenFromUrl) {
    window.sessionStorage.setItem("ai-video-gen.apiToken", tokenFromUrl);
    params.delete("token");
    const query = params.toString();
    window.history.replaceState(
      {},
      document.title,
      `${window.location.pathname}${query ? `?${query}` : ""}${window.location.hash}`,
    );
  }
}

function apiUrl(path: string) {
  if (!apiToken.value) {
    return path;
  }
  const url = new URL(path, window.location.origin);
  url.searchParams.set("token", apiToken.value);
  return `${url.pathname}${url.search}${url.hash}`;
}

function apiFetch(path: string, init: RequestInit = {}) {
  const headers = new Headers(init.headers);
  if (apiToken.value) {
    headers.set("Authorization", `Bearer ${apiToken.value}`);
  }
  return fetch(apiUrl(path), {
    ...init,
    headers,
  });
}

async function loadBootstrap() {
  const [healthResponse, provisioningResponse, backendResponse, jobsResponse] = await Promise.all([
    apiFetch("/api/health"),
    apiFetch("/api/provisioning"),
    apiFetch("/api/backends?includeUnavailable=true"),
    apiFetch("/api/jobs?limit=20"),
  ]);
  health.value = await readJson<HealthPayload | null>(healthResponse, health.value);
  provisioning.value = await readJson<ProvisioningPayload | null>(provisioningResponse, provisioning.value);
  backends.value = await readJson<BackendInfo[]>(backendResponse, backends.value);
  recentJobs.value = await readJson<JobSnapshot[]>(jobsResponse, recentJobs.value);
  const available = backends.value.filter((backend) => backend.available);
  const fallback = available.find((backend) => backend.key === health.value?.defaultBackend) ?? available[0];
  if (fallback) {
    selectedBatchBackend.value = fallback.key;
    selectedDirectBackend.value = fallback.key;
  }
  await restoreSelectedJob();
}

async function refreshReadiness() {
  try {
    const [healthResponse, provisioningResponse, backendResponse, jobsResponse] = await Promise.all([
      apiFetch("/api/health"),
      apiFetch("/api/provisioning"),
      apiFetch("/api/backends?includeUnavailable=true"),
      apiFetch("/api/jobs?limit=20"),
    ]);
    health.value = await readJson<HealthPayload | null>(healthResponse, health.value);
    provisioning.value = await readJson<ProvisioningPayload | null>(provisioningResponse, provisioning.value);
    backends.value = await readJson<BackendInfo[]>(backendResponse, backends.value);
    recentJobs.value = await readJson<JobSnapshot[]>(jobsResponse, recentJobs.value);
    await syncActiveJobFromRecent();
  } catch (error) {
    console.warn("Readiness refresh failed", error);
  }
}

async function syncActiveJobFromRecent() {
  if (!activeJob.value) {
    await restoreSelectedJob();
    return;
  }
  const freshSnapshot = recentJobs.value.find((job) => job.jobId === activeJob.value?.jobId);
  if (!freshSnapshot) {
    return;
  }
  const previousStatus = activeJob.value.status;
  const previousCompleted = activeJob.value.completedSegments;
  const previousFailed = activeJob.value.failedSegments;
  activeJob.value = freshSnapshot;
  if (isTerminalStatus(freshSnapshot.status)) {
    source?.close();
    await Promise.all([
      fetchLogs(freshSnapshot.jobId).catch((error) => console.warn("Log refresh failed", error)),
      fetchResult(freshSnapshot.jobId).catch((error) => console.warn("Result refresh failed", error)),
    ]);
    return;
  }
  if (
    previousStatus !== freshSnapshot.status
    || previousCompleted !== freshSnapshot.completedSegments
    || previousFailed !== freshSnapshot.failedSegments
  ) {
    await fetchLogs(freshSnapshot.jobId).catch((error) => console.warn("Log refresh failed", error));
  }
}

async function restoreSelectedJob() {
  if (activeJob.value) {
    return;
  }
  const savedJobId = window.localStorage.getItem("ai-video-gen.activeJobId");
  const runningJob = recentJobs.value.find((job) => ["queued", "running"].includes(job.status));
  const savedJob = savedJobId ? recentJobs.value.find((job) => job.jobId === savedJobId) : null;
  const target = runningJob ?? savedJob ?? recentJobs.value[0];
  if (!target) {
    return;
  }
  await activateJob(target.jobId);
}

async function startBatchJob() {
  if (!appReady.value) {
    errorMessage.value = "Сервис ещё не готов: дождитесь завершения скачивания весов и запуска ComfyUI.";
    return;
  }
  if (!batchFile.value) {
    errorMessage.value = "Выберите batch ZIP.";
    return;
  }
  busy.value = true;
  errorMessage.value = "";
  try {
    const form = new FormData();
    form.append("batch", batchFile.value);
    form.append("backend", selectedBatchBackend.value);
    const response = await apiFetch("/api/jobs", { method: "POST", body: form });
    const payload = await readJson<{ jobId?: string; detail?: string }>(response, {});
    if (!response.ok) {
      throw new Error(payload.detail ?? "Не удалось создать batch job.");
    }
    if (!payload.jobId) {
      throw new Error("Сервер создал пустой ответ без jobId.");
    }
    await activateJob(payload.jobId);
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  } finally {
    busy.value = false;
  }
}

async function startDirectJob() {
  if (!appReady.value) {
    errorMessage.value = "Сервис ещё не готов: дождитесь завершения скачивания весов и запуска ComfyUI.";
    return;
  }
  busy.value = true;
  errorMessage.value = "";
  try {
    const response = await apiFetch("/api/jobs/direct", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        backend: selectedDirectBackend.value,
        ...directForm.value,
        backendParams: {
          seed: directForm.value.seed,
        },
      }),
    });
    const payload = await readJson<{ jobId?: string; detail?: string }>(response, {});
    if (!response.ok) {
      throw new Error(payload.detail ?? "Не удалось создать direct job.");
    }
    if (!payload.jobId) {
      throw new Error("Сервер создал пустой ответ без jobId.");
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
  window.localStorage.setItem("ai-video-gen.activeJobId", jobId);
  await Promise.all([fetchJob(jobId), fetchLogs(jobId)]);
  if (activeJob.value && isTerminalStatus(activeJob.value.status)) {
    source?.close();
    await fetchResult(jobId);
    return;
  }
  connectEvents(jobId);
}

async function fetchJob(jobId: string) {
  const response = await apiFetch(`/api/jobs/${jobId}`);
  if (!response.ok) {
    throw new Error("Не удалось загрузить job.");
  }
  activeJob.value = await readJson<JobSnapshot | null>(response, activeJob.value);
}

async function fetchLogs(jobId: string) {
  const response = await apiFetch(`/api/jobs/${jobId}/logs`);
  if (!response.ok) {
    throw new Error("Не удалось загрузить логи.");
  }
  logs.value = await readJson<LogEntry[]>(response, logs.value);
}

async function fetchResult(jobId: string) {
  const response = await apiFetch(`/api/jobs/${jobId}/result`);
  if (!response.ok) {
    return;
  }
  result.value = await readJson<ResultPayload | null>(response, result.value);
}

function connectEvents(jobId: string) {
  source?.close();
  source = new EventSource(apiUrl(`/api/jobs/${jobId}/events`));
  source.onmessage = async (event) => {
    const payload = JSON.parse(event.data) as { type: string; data: unknown };
    if (payload.type === "snapshot") {
      activeJob.value = payload.data as JobSnapshot;
      window.localStorage.setItem("ai-video-gen.activeJobId", activeJob.value.jobId);
      if (isTerminalStatus(activeJob.value.status)) {
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
  initializeApiToken();
  loadBootstrap().catch((error) => {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  });
  readinessTimer = window.setInterval(() => {
    refreshReadiness();
  }, 2500);
});

onBeforeUnmount(() => {
  source?.close();
  if (readinessTimer) {
    window.clearInterval(readinessTimer);
  }
});
</script>

<template>
  <main class="page-shell">
    <section class="hero">
      <div>
        <p class="eyebrow">Batch + Direct Video Pipeline</p>
        <h1>AI Video Generation Service</h1>
        <p class="hero-copy">
          Импорт batch ZIP, прямой запуск сегмента, мониторинг очереди и скачивание результата.
        </p>
      </div>
      <div class="hero-status">
        <span class="status-pill" :data-ready="appReady">{{ provisioning?.status ?? health?.status ?? "loading" }}</span>
        <p>Backend: <strong>{{ health?.defaultBackend ?? "comfyui-ltx25" }}</strong></p>
        <p>Queue: <strong>{{ health?.queuedJobs ?? 0 }}</strong></p>
      </div>
    </section>

    <section v-if="!appReady" class="provisioning-panel">
      <div class="panel-head">
        <div>
          <h2>Подготовка LTX 2.5</h2>
          <p>{{ provisioning?.message ?? "Проверяю состояние запуска..." }}</p>
        </div>
        <strong>{{ provisioningPercent }}%</strong>
      </div>
      <div class="progress progress--large">
        <div class="progress-bar" :style="{ width: `${provisioningPercent}%` }"></div>
      </div>
      <div class="readiness-grid">
        <div class="readiness-card">
          <span>Текущий шаг</span>
          <strong>{{ currentDownloadText }}</strong>
          <p v-if="provisioning?.current">
            {{ formatBytes(provisioning.current.bytesDownloaded) }} / {{ formatBytes(provisioning.current.totalBytes) }}
          </p>
        </div>
        <div class="readiness-card">
          <span>ComfyUI</span>
          <strong>{{ provisioning?.comfyui?.ready ? "ready" : "waiting" }}</strong>
          <p>{{ provisioning?.comfyui?.error ?? provisioning?.comfyui?.apiUrl }}</p>
        </div>
        <div class="readiness-card">
          <span>Workflow</span>
          <strong>{{ provisioning?.workflows?.ready ? "ready" : "missing" }}</strong>
          <p>{{ provisioning?.workflows?.missing?.join(", ") || "T2V/I2V workflow files found" }}</p>
        </div>
      </div>
      <p v-if="provisioning?.error" class="error-banner">{{ provisioning.error }}</p>
      <div class="model-list">
        <article v-for="model in provisioning?.models ?? []" :key="model.id" class="model-row" :data-status="model.status">
          <div>
            <strong>{{ model.label }}</strong>
            <p>{{ model.targetPath }}</p>
          </div>
          <span>{{ model.status }}</span>
          <small>{{ formatBytes(model.bytesDownloaded) }}<template v-if="model.totalBytes"> / {{ formatBytes(model.totalBytes) }}</template></small>
        </article>
      </div>
    </section>

    <section v-if="appReady" class="grid">
        <article class="panel">
          <div class="panel-head">
            <h2>Batch Job</h2>
            <span>ZIP input</span>
          </div>
          <label class="field">
            <span>Backend</span>
            <select v-model="selectedBatchBackend">
              <option v-for="backend in backends" :key="backend.key" :value="backend.key" :disabled="!backend.available">
                {{ backend.label }} · {{ backend.available ? "ready" : backend.notes ?? backend.status }}
              </option>
            </select>
          </label>
          <label class="dropzone">
            <input type="file" accept=".zip,application/zip,.json,application/json" @change="chooseBatchFile" />
            <strong>{{ batchFile?.name ?? "Перетащите или выберите batch ZIP" }}</strong>
            <span>Файл отправляется через `POST /api/jobs`.</span>
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
                  {{ backend.label }} · {{ backend.available ? "ready" : backend.notes ?? backend.status }}
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
          <div class="field-grid">
            <label class="field"><span>FPS</span><input v-model.number="directForm.fps" type="number" min="1" step="1" /></label>
            <label class="field"><span>Seed</span><input v-model.number="directForm.seed" type="number" min="0" step="1" /></label>
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
          <div v-if="recentJobs.length" class="recent-jobs">
            <button
              v-for="job in recentJobs"
              :key="job.jobId"
              :class="{ active: job.jobId === activeJob?.jobId }"
              type="button"
              @click="activateJob(job.jobId)"
            >
              <strong>{{ job.status }}</strong>
              <span>{{ job.completedSegments }}/{{ job.totalSegments }}</span>
              <small>{{ job.jobId }}</small>
            </button>
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
            <p>После завершения job здесь появятся preview и ссылки на сегменты.</p>
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
.hero, .panel, .provisioning-panel { backdrop-filter: blur(14px); box-shadow: var(--shadow); border: 1px solid var(--line); background: var(--surface); }
.hero { display: grid; grid-template-columns: 1.6fr 0.8fr; gap: 24px; border-radius: 32px; padding: 28px; margin-bottom: 24px; }
.eyebrow { margin: 0 0 12px; text-transform: uppercase; letter-spacing: 0.18em; color: var(--ink); font-size: 12px; }
h1 { margin: 0; font-size: clamp(2.4rem, 4vw, 4.8rem); line-height: 0.95; }
.hero-copy { max-width: 760px; color: var(--muted); font-size: 1.02rem; }
.hero-status { padding: 18px; border-radius: 24px; background: linear-gradient(180deg, rgba(255,255,255,.7), rgba(255,255,255,.35)); }
.hero-status p { margin: 8px 0 0; color: var(--muted); }
.status-pill { display: inline-flex; padding: 8px 14px; border-radius: 999px; background: rgba(198,51,56,.12); color: #8f2228; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; font-size: 11px; }
.status-pill[data-ready="true"] { background: var(--ink-soft); color: var(--ink); }
.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 24px; margin-bottom: 24px; }
.grid--wide { grid-template-columns: 0.95fr 1.05fr; }
.panel, .provisioning-panel { border-radius: 28px; padding: 24px; }
.provisioning-panel { margin-bottom: 24px; }
.panel-head { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; margin-bottom: 18px; }
.panel-head h2, .panel-head p { margin: 0; }
.panel-head h2 { font-size: 1.4rem; }
.panel-head p, .panel-head span { color: var(--muted); font-size: 0.92rem; }
.panel-head strong { font-size: 2rem; color: var(--ink); }
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
.readiness-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin: 18px 0; }
.readiness-card { padding: 16px; border-radius: 18px; background: rgba(255,255,255,.66); border: 1px solid var(--line); min-width: 0; }
.readiness-card span { display: block; color: var(--muted); font-size: 0.82rem; margin-bottom: 6px; }
.readiness-card p { margin: 6px 0 0; color: var(--muted); overflow-wrap: anywhere; }
.model-list { display: flex; flex-direction: column; gap: 10px; }
.model-row { display: grid; grid-template-columns: 1fr 120px 160px; gap: 14px; align-items: center; padding: 13px 14px; border-radius: 16px; background: rgba(255,255,255,.64); border: 1px solid var(--line); }
.model-row p { margin: 4px 0 0; color: var(--muted); overflow-wrap: anywhere; font-family: "IBM Plex Mono", monospace; font-size: 0.78rem; }
.model-row span { justify-self: end; padding: 6px 10px; border-radius: 999px; background: rgba(31,40,51,.08); font-size: 0.78rem; text-transform: uppercase; font-weight: 700; }
.model-row[data-status="ready"] span { color: var(--ink); background: var(--ink-soft); }
.model-row[data-status="error"] span { color: #8f2228; background: rgba(198,51,56,.12); }
.model-row small { color: var(--muted); text-align: right; }
.stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 18px; }
.stat { padding: 16px; border-radius: 18px; background: rgba(255,255,255,.66); border: 1px solid var(--line); }
.stat span { display: block; color: var(--muted); font-size: 0.82rem; margin-bottom: 6px; }
.progress { height: 14px; border-radius: 999px; background: rgba(31,40,51,.08); overflow: hidden; margin-bottom: 18px; }
.progress--large { height: 18px; }
.progress-bar { height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--ink), var(--accent)); transition: width 180ms ease; }
.download-row, .artifact-actions { display: flex; gap: 12px; flex-wrap: wrap; }
.link-button--ghost { background: transparent; color: var(--ink); border: 1px solid rgba(15,124,130,.25); }
.recent-jobs { display: flex; flex-direction: column; gap: 8px; margin: 0 0 18px; }
.recent-jobs button { display: grid; grid-template-columns: 88px 64px 1fr; gap: 10px; align-items: center; width: 100%; padding: 10px 12px; border: 1px solid var(--line); border-radius: 14px; background: rgba(255,255,255,.54); color: var(--text); text-align: left; }
.recent-jobs button.active { border-color: rgba(15,124,130,.45); background: var(--ink-soft); }
.recent-jobs small { color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.log-box { display: flex; flex-direction: column; gap: 10px; max-height: 560px; overflow: auto; padding-right: 4px; }
.log-entry { display: grid; grid-template-columns: 92px 64px 1fr; gap: 12px; align-items: start; padding: 13px 14px; border-radius: 16px; background: rgba(255,255,255,.64); font-family: "IBM Plex Mono", monospace; font-size: 0.82rem; border: 1px solid var(--line); }
.log-entry p { margin: 0; overflow-wrap: anywhere; }
.log-entry[data-level="error"] { border-color: rgba(198,51,56,.28); background: rgba(198,51,56,.08); }
.artifact-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 16px; }
.artifact-card { display: flex; flex-direction: column; gap: 10px; padding: 14px; border-radius: 20px; background: rgba(255,255,255,.66); border: 1px solid var(--line); }
.artifact-card video { width: 100%; aspect-ratio: 9 / 16; object-fit: cover; border-radius: 14px; background: #0f1115; }
.artifact-card h3, .artifact-card p, .empty-state h3, .empty-state p { margin: 0; }
.artifact-card p, .empty-state p { color: var(--muted); }
.empty-state { display: flex; flex-direction: column; gap: 10px; padding: 22px; border-radius: 20px; background: rgba(255,255,255,.54); border: 1px dashed var(--line); }
@media (max-width: 1080px) { .hero, .grid, .grid--wide, .field-grid, .field-grid--triple, .stats, .readiness-grid, .model-row { grid-template-columns: 1fr; } .model-row span, .model-row small { justify-self: start; text-align: left; } }
</style>
