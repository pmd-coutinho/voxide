import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { open, save } from "@tauri-apps/plugin-dialog";
import "./styles.css";

type Theme = "system" | "light" | "dark";
type DictationMode = "dictate" | "prompt" | "rewrite" | "command" | "file";
type TranscriptionSound = "none" | "cue_1" | "cue_2" | "cue_3" | "cue_4" | "cue_5";
type TextInsertionMode = "standard" | "reliablePaste";
const builtInProviderIds = new Set(["openai", "anthropic", "xai", "groq", "cerebras", "google", "openrouter", "ollama", "lmstudio"]);
const providerSetupLabels: Record<string, string> = {
  openai: "Get API key", anthropic: "Get API key", xai: "Get API key", groq: "Get API key",
  cerebras: "Get API key", google: "Get API key", openrouter: "Get API key",
  ollama: "Setup guide", lmstudio: "Setup guide",
};

interface Settings {
  onboardingCompleted: boolean;
  onboardingStep: number;
  onboardingAiSkipped: boolean;
  onboardingPlaygroundValidated: boolean;
  theme: Theme;
  accentColor: string;
  language: string;
  appleSpeechLocale: string;
  primaryDictationHotkey: string;
  secondaryDictationHotkey?: string;
  promptModeHotkey?: string;
  promptModeSelectedPromptId?: string;
  promptShortcutAssignments: PromptShortcutAssignment[];
  commandModeHotkey?: string;
  rewriteModeHotkey?: string;
  cancelRecordingHotkey?: string;
  pasteLastTranscriptionHotkey?: string;
  hotkeyActivationMode: "toggle" | "hold" | "automatic";
  enableStreamingPreview: boolean;
  enableAiStreaming: boolean;
  showThinkingTokens: boolean;
  transcriptionPreviewCharLimit: number;
  transcriptionStartSound: TranscriptionSound;
  transcriptionSoundVolume: number;
  copyToClipboard: boolean;
  typeIntoActiveApplication: boolean;
  textInsertionMode: TextInsertionMode;
  removeFillerWordsEnabled: boolean;
  fillerWords: string[];
  autoConvertPunctuationEnabled: boolean;
  punctuationDictionaryPrefix: string;
  punctuationDictionaryRules: PunctuationRule[];
  literalDictationFormattingEnabled: boolean;
  gaavLowercaseFirstLetterEnabled: boolean;
  gaavRemoveTrailingPeriodEnabled: boolean;
  continuousDictationSpacingEnabled: boolean;
  contextAwareCapitalizationEnabled: boolean;
  notifyAiProcessingFailures: boolean;
  saveTranscriptionHistory: boolean;
  automaticDictionaryLearningEnabled: boolean;
  vocabularyBoostingEnabled: boolean;
  userTypingWpm: number;
  weekendsDontBreakStreak: boolean;
  audioHistoryEnabled: boolean;
  audioHistoryBudgetGb: number;
  selectedVoiceEngine: "whisper" | "parakeet" | "nemotron" | "appleSpeech" | "cloud";
  selectedModel: string;
  localModelPath?: string;
  selectedInputDevice?: string;
  cloudTranscriptionModel: string;
  selectedDictationPromptProfile?: string;
  selectedRewritePromptProfile?: string;
  selectedCommandPromptProfile?: string;
  dictationPromptRoutingScope: "allApps" | "selectedAppsOnly";
  editPromptRoutingScope: "allApps" | "selectedAppsOnly";
  aiEnhancementEnabled: boolean;
  selectedAiProvider: string;
  selectedRewriteAiProvider?: string;
  selectedCommandAiProvider?: string;
  modelReasoningConfigs: Record<string, ModelReasoningConfig>;
  commandModeConfirmBeforeExecute: boolean;
  overlayPosition: "top" | "bottom";
  overlayBottomOffset: number;
  overlaySize: "pill" | "small" | "medium" | "large";
  launchAtStartup: boolean;
  showMainWindowAtLoginLaunch: boolean;
  showInDock: boolean;
  shareAnonymousAnalytics: boolean;
  autoUpdateCheckEnabled: boolean;
  betaReleasesEnabled: boolean;
  lastUpdateCheckAt?: string;
  updatePromptSnoozedUntil?: string;
  snoozedUpdateVersion?: string;
  localApiEnabled: boolean;
  localApiPort: number;
}

interface PunctuationRule {
  aliases: string[];
  symbol: string;
}

interface ModelReasoningConfig {
  parameterName: string;
  parameterValue: string;
  isEnabled: boolean;
}

interface DictationEntry {
  id: string;
  text: string;
  rawText?: string;
  createdAt: string;
  durationMs?: number;
  mode: DictationMode;
  sourceApplication?: string;
  sourceWindowTitle?: string;
  audioFile?: string;
  audioModel?: string;
  wasAiProcessed: boolean;
  processingModel?: string;
  aiProcessingError?: string;
}

interface FileTranscriptionEntry {
  id: string;
  fileName: string;
  text: string;
  createdAt: string;
  durationMs?: number;
  processingTimeMs?: number;
  confidence?: number;
}

interface DictionaryEntry {
  id: string;
  spoken: string;
  replacement: string;
  createdAt: string;
}

interface PromptProfile {
  id: string;
  name: string;
  prompt: string;
  mode: DictationMode;
}

interface DictationPromptConfiguration {
  promptProfileId: string;
  providerId?: string;
  model?: string;
}

interface PromptTestState {
  active: boolean;
  profileId?: string;
  draftPrompt: string;
  processing: boolean;
  rawText: string;
  outputText: string;
  error?: string;
}

interface AppPromptBinding {
  id: string;
  application: string;
  mode: "dictate" | "rewrite" | "command";
  promptProfileId: string;
}

interface CustomWord {
  text: string;
  weight?: number;
  aliases: string[];
}

interface DictionaryImportResult {
  dictionary: DictionaryEntry[];
  customWords: CustomWord[];
}

interface DictionaryLearningSuggestion {
  heardText: string;
  correctedText: string;
  occurrences: number;
}

interface AppDatabase {
  settings: Settings;
  dictationHistory: DictationEntry[];
  fileTranscriptionHistory: FileTranscriptionEntry[];
  dictionary: DictionaryEntry[];
  customWords: CustomWord[];
  promptProfiles: PromptProfile[];
  dictationPromptConfigurations: DictationPromptConfiguration[];
  appPromptBindings: AppPromptBinding[];
  activeCommandChatId?: string;
}

interface UsageStats {
  todayDictations: number;
  todayWords: number;
  todayTimeSavedMinutes: number;
  totalDictations: number;
  totalWords: number;
  totalCharacters: number;
  totalTimeSavedMinutes: number;
  averageWordsPerDictation: number;
  aiProcessedCount: number;
  aiEnhancementRate: number;
  currentStreak: number;
  bestStreak: number;
  dailyActivity7: DailyUsage[];
  dailyActivity30: DailyUsage[];
  topApps: UsageTopApp[];
  peakHour?: number;
  longestTranscriptionWords: number;
  mostWordsInDay: number;
  mostTranscriptionsInDay: number;
  wordMilestones: UsageMilestone[];
  transcriptionMilestones: UsageMilestone[];
  streakMilestones: UsageMilestone[];
}

interface DailyUsage { date: string; words: number; transcriptions: number; }
interface UsageTopApp { app: string; count: number; }
interface UsageMilestone { target: number; achieved: boolean; label: string; }

interface VoiceModelStatus {
  id: string;
  installed: boolean;
  path: string;
}

interface ModelDownloadProgress {
  id: string;
  downloadedBytes: number;
  totalBytes?: number;
}

interface NativeTranscriptionResult {
  text: string;
  rawText: string;
  durationMs: number;
  audioFile?: string;
  audioModel?: string;
  wasAiProcessed: boolean;
  processingModel?: string;
  aiProcessingError?: string;
  sourceApplication?: string;
  sourceWindowTitle?: string;
}

interface NativeCaptureStarted {
  sampleRate: number;
  channels: number;
  sourceApplication?: string;
  sourceWindowTitle?: string;
}

interface CapturedSelection {
  text: string;
  sourceApplication?: string;
}

interface AiProviderProfile {
  id: string;
  name: string;
  apiStyle: "openAiCompatible" | "anthropic";
  baseUrl: string;
  model: string;
  enabled: boolean;
}

interface AiProviderView {
  profile: AiProviderProfile;
  hasApiKey: boolean;
}

interface OverlayUpdate {
  state: "recording" | "processing" | "complete" | "hidden";
  mode: string;
  text: string;
}

interface HotkeyEvent {
  action: "dictate" | "prompt" | "command" | "rewrite" | "cancel" | "pasteLast";
  phase: "pressed" | "released";
  promptProfileId?: string;
}

interface HotkeyBackendStatus {
  backend: "native" | "portal";
  state: "active" | "initializing" | "inactive" | "unavailable" | "denied" | "error";
  detail?: string;
}

interface PromptShortcutAssignment {
  promptProfileId: string;
  hotkey: string;
}

interface HotkeyConfiguration {
  primaryDictationHotkey: string;
  secondaryDictationHotkey?: string;
  promptModeHotkey?: string;
  promptModeSelectedPromptId?: string;
  promptShortcutAssignments: PromptShortcutAssignment[];
  commandModeHotkey?: string;
  rewriteModeHotkey?: string;
  cancelRecordingHotkey?: string;
  pasteLastTranscriptionHotkey?: string;
  hotkeyActivationMode: Settings["hotkeyActivationMode"];
}

interface LocalApiStatus {
  enabled: boolean;
  port?: number;
  url?: string;
}

interface AccessibilityPermissionStatus {
  supported: boolean;
  trusted?: boolean;
  guidance: string;
}

interface UpdateCheckResult {
  hasUpdate: boolean;
  latestVersion?: string;
  releaseUrl?: string;
}

interface UpdateAvailableEvent {
  latestVersion: string;
  releaseUrl: string;
}

interface ReleaseNote {
  version: string;
  title: string;
  notes: string;
  publishedAt?: string;
  releaseUrl?: string;
  isPrerelease: boolean;
}

interface RewriteState {
  selectedText: string;
  outputText: string;
  processing: boolean;
  draft: string;
  conversation: RewriteMessage[];
  sourceApplication?: string;
}

interface RewriteMessage {
  role: "user" | "assistant";
  content: string;
}

interface CommandPlan {
  kind: "command" | "answer";
  conversationId?: string;
  toolCallId?: string;
  answer?: string;
  thinking?: string;
  command?: string;
  purpose?: string;
  workingDirectory?: string;
  destructive: boolean;
}

interface CommandExecutionResult {
  success: boolean;
  command: string;
  output: string;
  error?: string;
  exitCode: number;
  executionTimeMs: number;
}

interface CommandState {
    draft: string;
    processing: boolean;
  streamingText?: string;
  streamingThinking?: string;
    chatId?: string;
  plan?: CommandPlan;
  result?: CommandExecutionResult;
  sourceApplication?: string;
}

interface CommandChatMessage {
  id: string;
  role: "user" | "assistant" | "tool";
  content: string;
  thinking?: string;
  createdAt: string;
}

interface CommandChat {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: CommandChatMessage[];
  sourceApplication?: string;
}

interface CommandStreamUpdate {
  conversationId: string;
  text?: string;
  thinking?: string;
}

interface FileTranscriptionProgress {
  completedChunks: number;
  totalChunks: number;
}

interface TrayAction {
  action: "toggleDictation" | "pasteLast" | "settings" | "dictionary";
}

const defaultStats: UsageStats = {
  todayDictations: 0, todayWords: 0, todayTimeSavedMinutes: 0,
  totalDictations: 0, totalWords: 0, totalCharacters: 0, totalTimeSavedMinutes: 0,
  averageWordsPerDictation: 0, aiProcessedCount: 0, aiEnhancementRate: 0,
  currentStreak: 0, bestStreak: 0, dailyActivity7: [], dailyActivity30: [], topApps: [],
  longestTranscriptionWords: 0, mostWordsInDay: 0, mostTranscriptionsInDay: 0,
  wordMilestones: [], transcriptionMilestones: [], streakMilestones: [],
};
let database: AppDatabase;
let stats = defaultStats;
let currentView = "welcome";
let recording = false;
let liveText = "";
let modelStatus: VoiceModelStatus | undefined;
let modelDownloadProgress: ModelDownloadProgress | undefined;
let audioDevices: string[] = [];
let providers: AiProviderView[] = [];
let editingProviderId: string | undefined;
let apiStatus: LocalApiStatus | undefined;
let accessibilityPermissionStatus: AccessibilityPermissionStatus | undefined;
let hotkeyBackendStatus: HotkeyBackendStatus | undefined;
let recentReleaseNotes: ReleaseNote[] = [];
let rewriteState: RewriteState = { selectedText: "", outputText: "", processing: false, draft: "", conversation: [] };
let commandState: CommandState = { draft: "", processing: false };
let commandChats: CommandChat[] = [];
let commandStreamRenderScheduled = false;
let dictationInstructionTarget: "command" | "prompt" | "rewrite" | undefined;
let dictationPromptProfileId: string | undefined;
let automaticHotkeyTarget: "dictate" | "prompt" | "command" | "rewrite" | undefined;
let automaticHotkeyStartedAt: number | undefined;
let fileTranscriptionActive = false;
let fileProgress: FileTranscriptionProgress | undefined;
let selectedFileForTranscription: string | undefined;
let statsActivityDays: 7 | 30 = 7;
let historySearchQuery = "";
let pendingDictationContext: Pick<DictationEntry, "sourceApplication" | "sourceWindowTitle"> = {};
let dictionaryLearningSuggestions: DictionaryLearningSuggestion[] = [];
let restoreMainWindowAfterRecording = false;
let promptTestState: PromptTestState = {
  active: false,
  draftPrompt: "",
  processing: false,
  rawText: "",
  outputText: "",
};

function resetPromptTestState(): void {
  promptTestState = { active: false, draftPrompt: "", processing: false, rawText: "", outputText: "" };
}

function canRunDictationPromptTest(promptProfileId?: string): boolean {
  const configuration = promptProfileId
    ? database.dictationPromptConfigurations.find((candidate) => candidate.promptProfileId === promptProfileId)
    : undefined;
  const provider = providers.find((candidate) => candidate.profile.id === (configuration?.providerId ?? database.settings.selectedAiProvider));
  const model = configuration?.model ?? provider?.profile.model ?? "";
  if (!provider?.profile.enabled || !model.trim()) return false;
  try {
    const host = new URL(provider.profile.baseUrl).hostname.toLowerCase();
    const isLocal = host === "localhost" || host === "::1" || host === "127.0.0.1" || host.startsWith("127.");
    return provider.hasApiKey || isLocal;
  } catch {
    return false;
  }
}

const app = document.querySelector<HTMLDivElement>("#app");

if (!app) {
  throw new Error("Voxide application root is missing");
}

const appRoot = app;
const currentWindow = getCurrentWindow();
const isOverlayWindow = currentWindow.label === "overlay";

const menuGroups: readonly (readonly [string, readonly (readonly [string, string, string])[]])[] = [
  ["Dictation", [
    ["welcome", "Getting Started", "⌂"],
    ["voice", "Voice Engine", "⌁"],
    ["enhancement", "AI Enhancement", "✦"],
    ["dictionary", "Custom Dictionary", "▤"],
  ]],
  ["Modes", [
    ["command", "Command Mode", "⌘"],
    ["rewrite", "Write & Rewrite", "✎"],
    ["file", "File Transcription", "▧"],
  ]],
  ["Activity", [
    ["history", "History", "◷"],
    ["stats", "Stats", "▥"],
  ]],
  ["Application", [
    ["feedback", "Feedback", "✉"],
    ["settings", "Settings", "⚙"],
  ]],
] as const;

function escapeHtml(value: string): string {
  return value.replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "'": "&#39;",
    '"': "&quot;",
  })[character] ?? character);
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(value));
}

function archiveTimestamp(value = new Date()): string {
  return value.toISOString().replace(/\.\d+Z$/, "Z").replace(/:/g, "-");
}

function backupTimestamp(value = new Date()): string {
  const pad = (component: number) => String(component).padStart(2, "0");
  return `${value.getFullYear()}-${pad(value.getMonth() + 1)}-${pad(value.getDate())}_${pad(value.getHours())}-${pad(value.getMinutes())}`;
}

function formatDuration(milliseconds?: number): string {
  if (!milliseconds) return "";
  return `${Math.round(milliseconds / 1000)} sec`;
}

function formatSavedTime(minutes: number): string {
  if (minutes < 1) return "< 1m";
  const wholeMinutes = Math.floor(minutes);
  if (wholeMinutes < 60) return `${wholeMinutes}m`;
  const hours = Math.floor(wholeMinutes / 60);
  const remainder = wholeMinutes % 60;
  return remainder ? `${hours}h ${remainder}m` : `${hours}h`;
}

function formatPeakHour(hour?: number): string {
  if (hour === undefined) return "N/A";
  const start = new Date(2000, 0, 1, hour);
  const end = new Date(2000, 0, 1, (hour + 1) % 24);
  const formatter = new Intl.DateTimeFormat(undefined, { hour: "numeric" });
  return `${formatter.format(start)}–${formatter.format(end)}`;
}

function activityDayLabel(date: string): string {
  return new Intl.DateTimeFormat(undefined, { weekday: "short" }).format(new Date(`${date}T12:00:00`));
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

let transcriptionAudioContext: AudioContext | undefined;

async function playTranscriptionCue(phase: "start" | "stop"): Promise<void> {
  if (isOverlayWindow || database.settings.transcriptionStartSound === "none") return;
  // Cue 1 is the reference cue with both a start and an end sound. The
  // other choices deliberately signal only recording start.
  if (phase === "stop" && database.settings.transcriptionStartSound !== "cue_1") return;
  const AudioContextConstructor = window.AudioContext
    ?? (window as Window & typeof globalThis & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!AudioContextConstructor) return;
  transcriptionAudioContext ??= new AudioContextConstructor();
  const context = transcriptionAudioContext;
  if (!context) return;
  if (context.state === "suspended") {
    await context.resume().catch(() => undefined);
  }
  if (context.state !== "running") return;
  const volume = Math.max(0, Math.min(1, database.settings.transcriptionSoundVolume));
  if (volume === 0) return;
  const sound = database.settings.transcriptionStartSound;
  const notes: Record<Exclude<TranscriptionSound, "none">, number[]> = {
    cue_1: phase === "start" ? [550, 780] : [740, 490],
    cue_2: [720],
    cue_3: [480, 660, 880],
    cue_4: [640, 960],
    cue_5: [390, 585, 780],
  };
  const now = context.currentTime;
  notes[sound].forEach((frequency, index) => {
    const start = now + index * 0.065;
    const end = start + 0.11;
    const oscillator = context.createOscillator();
    const gain = context.createGain();
    oscillator.type = sound === "cue_4" ? "triangle" : "sine";
    oscillator.frequency.setValueAtTime(frequency, start);
    gain.gain.setValueAtTime(0.0001, start);
    gain.gain.exponentialRampToValueAtTime(Math.max(0.0001, volume * 0.14), start + 0.012);
    gain.gain.exponentialRampToValueAtTime(0.0001, end);
    oscillator.connect(gain).connect(context.destination);
    oscillator.start(start);
    oscillator.stop(end + 0.01);
  });
}

function settingToggle(key: keyof Settings, title: string, description: string): string {
  const checked = database.settings[key] ? "checked" : "";
  return `<label class="setting-row"><span><strong>${title}</strong><small>${description}</small></span><input data-setting-toggle="${key}" type="checkbox" ${checked}></label>`;
}

function applyTheme(): void {
  if (database.settings.theme === "system") {
    delete document.documentElement.dataset.theme;
  } else {
    document.documentElement.dataset.theme = database.settings.theme;
  }
}

function renderShell(content: string): void {
  applyTheme();
  const nav = menuGroups.map(([group, items]) => `
    <div class="nav-group">
      <span class="nav-group-label">${group}</span>
      ${items.map(([id, label, icon]) => `
      <button class="nav-item ${currentView === id ? "selected" : ""}" data-nav="${id}">
        <span aria-hidden="true">${icon}</span>${label}
      </button>`).join("")}
    </div>`).join("");
  appRoot.innerHTML = `
    <main class="app-shell">
      <aside class="sidebar">
        <div class="brand"><span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i></span><span>Voxide</span></div>
        <nav aria-label="Voxide sections">${nav}</nav>
        <button class="hotkey-button ${recording ? "live" : ""}" data-action="toggle-recording"><span class="signal-bars" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i></span>${recording ? "Stop dictation" : "Start dictation"}<kbd>${database.settings.primaryDictationHotkey}</kbd></button>
      </aside>
      <section class="main-content">${content}</section>
    </main>`;
  bindCommonEvents();
}

function pageTitle(title: string, subtitle: string, actions = ""): string {
  return `<header class="page-header"><div><h1>${title}</h1><p>${subtitle}</p></div>${actions}</header>`;
}

function renderWelcome(): void {
  const ready = database.settings.onboardingCompleted;
  const step = database.settings.onboardingStep;
  renderShell(`
    ${pageTitle("Getting Started", "Talk anywhere. Voxide types for you.")}
    <div class="two-column">
      <section class="card prominent">
        <div class="card-title"><h2>Quick setup</h2><span class="status ${ready ? "done" : "pending"}">${ready ? "Ready" : "Needs attention"}</span></div>
        <ol class="checklist">
          <li class="${ready || step >= 1 ? "complete" : ""}"><b>Choose a voice engine</b><span>${escapeHtml(database.settings.selectedVoiceEngine === "cloud" ? database.settings.cloudTranscriptionModel : database.settings.selectedModel)}</span><button data-action="onboarding-voice">Configure</button></li>
          <li class="${ready || step >= 2 ? "complete" : ""}"><b>Grant microphone access</b><span>Required for dictation.</span><button data-action="request-mic">Test microphone</button></li>
          <li class="${ready || step >= 3 ? "complete" : ""}"><b>Set your global hotkey</b><span>${escapeHtml(database.settings.primaryDictationHotkey)}</span><button data-action="onboarding-settings">Configure</button></li>
          <li class="${ready || step >= 4 ? "complete" : ""}"><b>Set up text enhancement</b><span>Optional: local or cloud AI cleanup.</span><button data-action="onboarding-enhancement">Configure</button></li>
        </ol>
        <div class="button-row"><button class="primary" data-action="complete-onboarding" ${ready ? "disabled" : ""}>${ready ? "Setup complete" : "Finish setup"}</button>${ready ? `<button data-action="reset-onboarding">Run setup again</button>` : ""}</div>
      </section>
      <section class="card stats-card">
        <h2>Today</h2>
        <strong class="big-stat">${stats.todayDictations}</strong><span>dictations</span>
        <div class="stat-rule"></div>
        <strong>${stats.todayWords.toLocaleString()}</strong><span> words captured</span>
      </section>
    </div>
    <section class="card playground">
      <div class="card-title"><div><h2>Test playground</h2><p>Record a short dictation and review the live transcription.</p></div><span class="recording-state ${recording ? "live" : ""}">${recording ? "Recording" : "Ready"}</span></div>
      <textarea id="live-text" rows="7" placeholder="Your live transcription will appear here…">${escapeHtml(liveText)}</textarea>
      <div class="playground-footer"><span>${recording ? "Listening for speech…" : "Use the button or your global hotkey."}</span><button class="primary" data-action="toggle-recording">${recording ? "Stop recording" : "Start recording"}</button></div>
    </section>`);
}

function renderVoiceEngine(): void {
  const engines: [Settings["selectedVoiceEngine"], string, string, boolean][] = [
    ["whisper", "Whisper", "Local models with broad language support", true],
    ["parakeet", "Parakeet", "Apple Silicon-only runtime — portable implementation pending", false],
    ["nemotron", "Nemotron Speech", "Apple Silicon-only runtime — portable implementation pending", false],
    ["appleSpeech", "System speech", "Use the operating system speech service", true],
    ["cloud", "Compatible cloud API", "OpenAI-compatible transcription endpoint", true],
  ];
  const selectedEngine = database.settings.selectedVoiceEngine;
  const isWhisper = selectedEngine === "whisper";
  const isCloud = selectedEngine === "cloud";
  const isAppleSpeech = selectedEngine === "appleSpeech";
  const status = modelStatus?.installed
    ? `<span class="status done">Installed</span>`
    : `<span class="status pending">Not installed</span>`;
  const downloading = isWhisper && modelDownloadProgress?.id === database.settings.selectedModel ? modelDownloadProgress : undefined;
  const downloadDetail = downloading
    ? `${formatBytes(downloading.downloadedBytes)}${downloading.totalBytes ? ` / ${formatBytes(downloading.totalBytes)} (${Math.round((downloading.downloadedBytes / downloading.totalBytes) * 100)}%)` : " downloaded"}`
    : "";
  const canDeleteDownloadedModel = isWhisper && !database.settings.localModelPath && modelStatus?.installed;
  const selectedInputDeviceUnavailable = Boolean(
    database.settings.selectedInputDevice && !audioDevices.includes(database.settings.selectedInputDevice),
  );
  const deviceOptions = [`<option value="" ${!database.settings.selectedInputDevice || selectedInputDeviceUnavailable ? "selected" : ""}>System default${selectedInputDeviceUnavailable ? " (preferred device unavailable)" : ""}</option>`, ...audioDevices.map((device) => `<option value="${escapeHtml(device)}" ${database.settings.selectedInputDevice === device ? "selected" : ""}>${escapeHtml(device)}</option>`)].join("");
  const engineConfiguration = isWhisper
    ? `<label>Selected Whisper model<select id="selected-model"><option value="tiny" ${database.settings.selectedModel === "tiny" ? "selected" : ""}>Tiny — multilingual, fastest</option><option value="base" ${database.settings.selectedModel === "base" ? "selected" : ""}>Base — multilingual, default</option><option value="small" ${database.settings.selectedModel === "small" ? "selected" : ""}>Small — multilingual, higher accuracy</option><option value="medium" ${database.settings.selectedModel === "medium" ? "selected" : ""}>Medium — multilingual</option><option value="large-v3-turbo" ${database.settings.selectedModel === "large-v3-turbo" ? "selected" : ""}>Large v3 Turbo — multilingual</option><option value="large-v3" ${database.settings.selectedModel === "large-v3" ? "selected" : ""}>Large v3 — multilingual</option><optgroup label="Legacy English-only Whisper models"><option value="tiny.en" ${database.settings.selectedModel === "tiny.en" ? "selected" : ""}>Tiny English</option><option value="base.en" ${database.settings.selectedModel === "base.en" ? "selected" : ""}>Base English</option><option value="small.en" ${database.settings.selectedModel === "small.en" ? "selected" : ""}>Small English</option><option value="medium.en" ${database.settings.selectedModel === "medium.en" ? "selected" : ""}>Medium English</option></optgroup></select></label>
      <label>Custom local model path (optional)<input id="local-model-path" value="${escapeHtml(database.settings.localModelPath ?? "")}" placeholder="/path/to/ggml-model.bin"></label>`
    : isCloud
      ? `<label>Cloud transcription model<input id="cloud-transcription-model" value="${escapeHtml(database.settings.cloudTranscriptionModel)}" placeholder="gpt-4o-mini-transcribe"></label><p class="muted">Uses the enabled OpenAI-compatible AI provider and its stored API key.</p>`
      : isAppleSpeech
        ? `<label>Apple Speech locale<input id="apple-speech-locale" value="${escapeHtml(database.settings.appleSpeechLocale)}" placeholder="en-US"><small>Use a supported BCP-47 locale such as en-US, pt-PT, or es-ES.</small></label><p class="muted">macOS Speech Recognition transcribes final results using the operating-system service. macOS asks for Speech Recognition permission when it is first used. This engine is unavailable on Windows and Linux.</p>`
        : `<p class="muted">This engine is listed for migration compatibility but is not yet available in the portable runtime. Select Whisper, System speech on macOS, or a compatible cloud API.</p>`;
  const whisperActions = isWhisper
    ? `<button data-action="download-model" ${downloading ? "disabled" : ""}>${downloading ? "Downloading…" : "Download selected model"}</button>${canDeleteDownloadedModel ? `<button data-action="delete-model" ${downloading ? "disabled" : ""}>Remove downloaded model</button>` : ""}`
    : "";
  renderShell(`
    ${pageTitle("Voice Engine", "Choose the transcription runtime used for dictation.")}
    <section class="card"><div class="engine-grid">${engines.map(([id, name, description, available]) => `
      <button class="engine-choice ${database.settings.selectedVoiceEngine === id ? "active" : ""}" data-engine="${id}" ${available ? "" : "disabled"}><strong>${name}</strong><span>${description}</span><em>${available ? (database.settings.selectedVoiceEngine === id ? "Selected" : "Select") : "Not available"}</em></button>`).join("")}</div></section>
    <section class="card form-card"><div class="card-title"><h2>Model configuration</h2>${status}</div>
      ${engineConfiguration}
      ${isAppleSpeech ? "" : `<label>Recognition language<input id="language" value="${escapeHtml(database.settings.language)}" placeholder="en"></label>`}
      <label>Microphone input<select id="input-device">${deviceOptions}</select></label>
      <div class="button-row"><button class="primary" data-action="save-engine">Save voice engine</button>${whisperActions}</div>
      ${downloadDetail ? `<p class="muted">Download progress: ${escapeHtml(downloadDetail)}</p>` : ""}
      ${modelStatus ? `<small class="muted">${escapeHtml(modelStatus.path)}</small>` : ""}
    </section>`);
}

function renderEnhancement(): void {
  const promptTestAvailable = canRunDictationPromptTest(promptTestState.profileId);
  if (promptTestState.active && !promptTestAvailable && !recording) resetPromptTestState();
  const enabledProviders = providers.filter(({ profile }) => profile.enabled);
  const globalProvider = enabledProviders.find(({ profile }) => profile.id === database.settings.selectedAiProvider)?.profile ?? enabledProviders[0]?.profile;
  const selectedProvider = providers.find(({ profile }) => profile.id === editingProviderId)?.profile ?? globalProvider;
  const providerOptions = enabledProviders.map(({ profile }) => `<option value="${escapeHtml(profile.id)}" ${globalProvider?.id === profile.id ? "selected" : ""}>${escapeHtml(profile.name)}</option>`).join("");
  const modeProviderOptions = (selected: string | undefined) => [`<option value="" ${selected ? "" : "selected"}>Use global provider</option>`, ...enabledProviders.map(({ profile }) => `<option value="${escapeHtml(profile.id)}" ${selected === profile.id ? "selected" : ""}>${escapeHtml(profile.name)}</option>`)].join("");
  const configurationProviderOptions = providers.map(({ profile }) => `<option value="${escapeHtml(profile.id)}" ${selectedProvider?.id === profile.id ? "selected" : ""}>${escapeHtml(profile.name)}${profile.enabled ? "" : " (disabled)"}</option>`).join("");
  const providerKeySet = providers.find(({ profile }) => profile.id === selectedProvider?.id)?.hasApiKey ?? false;
  const selectedProviderIsBuiltIn = Boolean(selectedProvider && builtInProviderIds.has(selectedProvider.id));
  const canDeleteSelectedProvider = Boolean(selectedProvider && !builtInProviderIds.has(selectedProvider.id));
  const providerSetupLabel = selectedProvider ? providerSetupLabels[selectedProvider.id] : undefined;
  const reasoningKey = selectedProvider ? `${selectedProvider.id}:${selectedProvider.model}` : "";
  const storedReasoning = reasoningKey ? database.settings.modelReasoningConfigs[reasoningKey] : undefined;
  const model = selectedProvider?.model.toLowerCase() ?? "";
  const defaultReasoning = model.startsWith("gpt-5")
    ? { parameterName: "reasoning_effort", parameterValue: "low", isEnabled: true }
    : model.startsWith("o1") || model.startsWith("o3") || model.startsWith("o4")
      ? { parameterName: "reasoning_effort", parameterValue: "medium", isEnabled: true }
      : model.includes("gpt-oss") || model.startsWith("openai/")
        ? { parameterName: "reasoning_effort", parameterValue: "low", isEnabled: true }
        : model.includes("deepseek") && model.includes("reasoner")
          ? { parameterName: "enable_thinking", parameterValue: "true", isEnabled: true }
          : undefined;
  const reasoning = storedReasoning ?? defaultReasoning;
  const reasoningConfiguration = selectedProvider
    ? `<section class="card form-card"><div class="card-title"><div><h2>Model reasoning</h2><p>Optional request parameter for ${escapeHtml(selectedProvider.model || "this model")}. Saved per provider and model.</p></div>${storedReasoning ? `<button data-action="clear-reasoning-config">Use default</button>` : ""}</div>
      <label>Parameter<input id="reasoning-parameter-name" value="${escapeHtml(reasoning?.parameterName ?? "reasoning_effort")}" placeholder="reasoning_effort"></label>
      <label>Value<input id="reasoning-parameter-value" value="${escapeHtml(reasoning?.parameterValue ?? "low")}" placeholder="low"></label>
      <label class="setting-row"><span><strong>Enable parameter</strong><small>Use false to deliberately suppress a model's built-in Voxide default.</small></span><input id="reasoning-parameter-enabled" type="checkbox" ${reasoning?.isEnabled ?? false ? "checked" : ""}></label>
      <div class="button-row"><button class="primary" data-action="save-reasoning-config">Save model reasoning</button></div>
      <p class="muted">Use <code>reasoning_effort</code> for OpenAI-compatible reasoning models or <code>enable_thinking</code> with true/false for compatible DeepSeek-style endpoints. OpenAI Responses sends reasoning effort as <code>reasoning.effort</code>.</p>
    </section>`
    : "";
  const promptProfiles = database.promptProfiles.map((profile) => {
    const active = activePromptProfileId(profile.mode) === profile.id;
    const profileTestAvailable = canRunDictationPromptTest(profile.id);
    const test = profile.mode === "dictate"
      ? `<button data-action="start-prompt-test" data-prompt-id="${profile.id}" ${profileTestAvailable ? "" : "disabled"}>${promptTestState.active && promptTestState.profileId === profile.id ? "Testing" : "Test"}</button>`
      : "";
    return `<article class="profile"><div><strong>${escapeHtml(profile.name)}</strong><small>${profile.mode}${active ? " · active" : ""}</small></div><p>${escapeHtml(profile.prompt)}</p><div class="entry-actions"><button data-action="activate-prompt" data-prompt-id="${profile.id}" ${active ? "disabled" : ""}>${active ? "Active" : "Use for mode"}</button>${test}<button data-action="edit-prompt" data-prompt-id="${profile.id}">Edit</button><button data-action="delete-prompt" data-prompt-id="${profile.id}">Delete</button></div></article>`;
  }).join("");
  const dictationPromptProviderRouting = database.promptProfiles
    .filter((profile) => profile.mode === "dictate")
    .map((profile) => {
      const configuration = database.dictationPromptConfigurations.find((candidate) => candidate.promptProfileId === profile.id);
      const encodedId = encodeURIComponent(profile.id);
      const providerOptions = [
        `<option value="">Use global dictation provider</option>`,
        ...providers.filter((candidate) => candidate.profile.enabled).map((candidate) =>
          `<option value="${escapeHtml(candidate.profile.id)}" ${candidate.profile.id === configuration?.providerId ? "selected" : ""}>${escapeHtml(candidate.profile.name)}${candidate.profile.model ? ` · ${escapeHtml(candidate.profile.model)}` : ""}</option>`,
        ),
      ].join("");
      return `<article class="profile"><div><strong>${escapeHtml(profile.name)}</strong><small>Dictate provider/model route</small></div><label>Provider<select id="dictation-prompt-provider-${encodedId}">${providerOptions}</select></label><label>Model override<input id="dictation-prompt-model-${encodedId}" value="${escapeHtml(configuration?.model ?? "")}" placeholder="Use selected provider model"></label><div class="entry-actions"><button data-action="save-dictation-prompt-provider" data-prompt-id="${escapeHtml(profile.id)}">Save route</button></div></article>`;
    }).join("");
  const promptTest = promptTestState.active
    ? `<section class="card form-card prompt-test"><div class="card-title"><div><h2>Dictate prompt test</h2><p>Press ${escapeHtml(database.settings.primaryDictationHotkey)} or use the button to record. This uses the unsaved draft below and never types, copies, saves history, audio, or ordinary dictation analytics.</p></div><button data-action="stop-prompt-test" ${recording ? "disabled" : ""}>Disable test mode</button></div>
      <label>Draft prompt<textarea id="prompt-test-draft" rows="6">${escapeHtml(promptTestState.draftPrompt)}</textarea><small>Changes here are only used for testing. Save the profile separately if you want to keep them.</small></label>
      <div class="button-row"><button class="primary" data-action="toggle-recording">${recording ? "Stop test recording" : "Start test recording"}</button></div>
      ${promptTestState.processing ? `<p class="muted">Transcribing and post-processing with the draft prompt…</p>` : ""}
      ${promptTestState.error ? `<p class="history-warning">${escapeHtml(promptTestState.error)}</p>` : ""}
      <label>Raw transcription<textarea rows="4" readonly>${escapeHtml(promptTestState.rawText)}</textarea></label>
      <label>Post-processed output<textarea rows="6" readonly>${escapeHtml(promptTestState.outputText)}</textarea></label>
    </section>`
    : "";
  const appPromptBindings = database.appPromptBindings.map((binding) => {
    const profile = database.promptProfiles.find((candidate) => candidate.id === binding.promptProfileId);
    return `<article class="profile"><div><strong>${escapeHtml(binding.application)}</strong><small>${binding.mode} · ${escapeHtml(profile?.name ?? binding.promptProfileId)}</small></div><p>Uses this prompt whenever dictation starts in the matching application name.</p><div class="entry-actions"><button data-delete-app-prompt-binding="${binding.id}">Remove</button></div></article>`;
  }).join("");
  renderShell(`
    ${pageTitle("AI Enhancement", "Optionally clean up and format transcriptions after speech recognition.", `<button data-action="new-provider">Add provider</button>`)}
    <section class="card form-card">
      ${settingToggle("aiEnhancementEnabled", "Enable AI enhancement", "Run the selected prompt after every completed dictation.")}
      ${settingToggle("showThinkingTokens", "Show model reasoning", "Display provider-supplied reasoning in Command Mode. It is never sent to another provider or to the terminal.")}
      <label>Dictation provider<select id="ai-provider">${providerOptions}</select></label>
      <label>Rewrite provider<select id="rewrite-ai-provider">${modeProviderOptions(database.settings.selectedRewriteAiProvider)}</select></label>
      <label>Command provider<select id="command-ai-provider">${modeProviderOptions(database.settings.selectedCommandAiProvider)}</select></label>
      <label>Configure provider<select id="provider-configuration">${configurationProviderOptions}</select></label>
      <label>Dictation prompt routing<select id="dictation-prompt-routing-scope"><option value="allApps" ${database.settings.dictationPromptRoutingScope === "allApps" ? "selected" : ""}>Use active profile in all apps</option><option value="selectedAppsOnly" ${database.settings.dictationPromptRoutingScope === "selectedAppsOnly" ? "selected" : ""}>Use custom prompts only in selected apps</option></select></label>
      <label>Rewrite & Command prompt routing<select id="edit-prompt-routing-scope"><option value="allApps" ${database.settings.editPromptRoutingScope === "allApps" ? "selected" : ""}>Use active profiles in all apps</option><option value="selectedAppsOnly" ${database.settings.editPromptRoutingScope === "selectedAppsOnly" ? "selected" : ""}>Use custom prompts only in selected apps</option></select></label>
      ${selectedProvider ? `<label>Provider ID<input id="provider-id" value="${escapeHtml(selectedProvider.id)}" disabled><small>Provider IDs are generated when a custom provider is created and stay fixed so settings and secure credentials remain linked.</small></label><label>Provider name<input id="provider-name" value="${escapeHtml(selectedProvider.name)}" ${selectedProviderIsBuiltIn ? "disabled" : ""}></label><label>API style<select id="provider-api-style"><option value="openAiCompatible" ${selectedProvider.apiStyle === "openAiCompatible" ? "selected" : ""}>OpenAI-compatible</option><option value="anthropic" ${selectedProvider.apiStyle === "anthropic" ? "selected" : ""}>Anthropic Messages API</option></select><small>Select the protocol used by this endpoint; it controls authentication, request shape, streaming, and tools.</small></label><label>Base URL<input id="provider-base-url" value="${escapeHtml(selectedProvider.baseUrl)}"></label><label>Model<input id="provider-model" value="${escapeHtml(selectedProvider.model)}" placeholder="Model ID"></label><label class="setting-row"><span><strong>Enable provider</strong><small>Disabled providers are removed from the Dictation, Rewrite, and Command provider choices.</small></span><input id="provider-enabled" type="checkbox" ${selectedProvider.enabled ? "checked" : ""}></label><label>API key<input id="provider-api-key" type="password" placeholder="${providerKeySet ? "Stored securely — leave blank to keep" : "Enter API key"}"></label><p class="muted">${providerKeySet ? "An API key is stored in your operating system’s secure credential store." : "API keys are stored in your operating system’s secure credential store, never in application preferences."}</p><div class="button-row"><button class="primary" data-action="save-provider">Save provider</button><button data-action="fetch-provider-models" ${selectedProvider.enabled ? "" : "disabled"}>Fetch models</button>${providerSetupLabel ? `<button data-action="open-provider-website">${providerSetupLabel}</button>` : ""}${providerKeySet ? `<button data-action="clear-provider-api-key">Clear API key</button>` : ""}${canDeleteSelectedProvider ? `<button data-action="delete-provider">Delete custom provider</button>` : ""}</div>` : `<p class="muted">Loading provider configurations…</p>`}
    </section>
    ${reasoningConfiguration}
    <section class="card"><div class="card-title"><div><h2>Prompt profiles</h2><p>Choose one active profile for each mode; it supplies that mode’s instructions.</p></div><button data-action="new-prompt">New profile</button></div>
      <div class="profile-list">${promptProfiles}</div>
    </section>
    <section class="card"><div class="card-title"><div><h2>Dictate profile provider routing</h2><p>Optionally give each Dictate profile its own provider or model. Empty values inherit the global Dictation provider.</p></div></div><div class="profile-list">${dictationPromptProviderRouting}</div></section>
    ${promptTest}
    <section class="card"><div class="card-title"><div><h2>Application prompt overrides</h2><p>Route a foreground application to a specific prompt profile. Application names are captured in History after a dictation.</p></div><button data-action="new-app-prompt-binding">Add override</button></div>
      <div class="profile-list">${appPromptBindings || `<div class="empty">No application-specific prompt overrides yet.</div>`}</div>
    </section>`);
}

function renderDictionary(): void {
  const customWords = database.customWords.length
    ? database.customWords.map((word, index) => `<div class="table-row"><span>${escapeHtml(word.text)}</span><span>${escapeHtml(word.aliases.join(", ") || "Recognition hint")}</span><button data-delete-custom-word="${index}">Remove</button></div>`).join("")
    : `<div class="empty">No recognition vocabulary yet. Add names and terms that Whisper should favor.</div>`;
  const learningSuggestions = dictionaryLearningSuggestions.map((suggestion) => `<article class="profile"><div><strong>“${escapeHtml(suggestion.heardText)}” → “${escapeHtml(suggestion.correctedText)}”</strong><small>${suggestion.occurrences} observed correction${suggestion.occurrences === 1 ? "" : "s"}</small></div><p>Accepting adds this as a local replacement. Nothing is changed until you accept.</p><div class="entry-actions"><button class="primary" data-accept-dictionary-learning="${escapeHtml(suggestion.heardText)}" data-learning-replacement="${escapeHtml(suggestion.correctedText)}">Add correction</button><button data-dismiss-dictionary-learning="${escapeHtml(suggestion.heardText)}" data-learning-replacement="${escapeHtml(suggestion.correctedText)}">Not now</button></div></article>`).join("");
  renderShell(`
    ${pageTitle("Custom Dictionary", "Correct names, product terms, and repeated transcription mistakes automatically.", `<div class="button-row"><button data-action="import-dictionary">Import</button><button data-action="export-dictionary">Export</button><button data-action="new-dictionary-entry">Add term</button></div>`)}
    <section class="card form-card"><h2>Automatic correction learning</h2>${settingToggle("automaticDictionaryLearningEnabled", "Learn from saved-history corrections", "After the same small correction is made twice in saved history, offer it here for explicit review. No replacement is added automatically.")}${learningSuggestions || `<p class="muted">No reviewed correction suggestions are ready yet.</p>`}</section>
    <section class="card"><div class="dictionary-table"><div class="table-row heading"><span>Spoken phrase</span><span>Replace with</span><span></span></div>
      ${database.dictionary.length ? database.dictionary.map((entry) => `<div class="table-row"><span>${escapeHtml(entry.spoken)}</span><span>${escapeHtml(entry.replacement)}</span><button data-delete-dictionary="${entry.id}">Remove</button></div>`).join("") : `<div class="empty">No corrections yet. Add terms that your voice engine commonly gets wrong.</div>`}
    </div></section>
    <section class="card"><div class="card-title"><div><h2>Recognition vocabulary</h2><p>These terms can be supplied to supported speech engines as recognition hints; they are not post-processing replacements.</p></div><button data-action="new-custom-word">Add vocabulary</button></div>${settingToggle("vocabularyBoostingEnabled", "Use recognition vocabulary hints", "When enabled, Voxide supplies the terms above to supported local and system speech engines.")}<div class="dictionary-table"><div class="table-row heading"><span>Term</span><span>Aliases</span><span></span></div>${customWords}</div></section>`);
}

function renderModePage(mode: "command" | "rewrite"): void {
  const command = mode === "command";
  const hasSelection = rewriteState.selectedText.trim().length > 0;
  const commandRequiresApproval = Boolean(
    commandState.plan?.kind === "command"
      && commandState.plan.destructive
      && database.settings.commandModeConfirmBeforeExecute,
  );
  const commandExecutionPolicy = database.settings.commandModeConfirmBeforeExecute
    ? "Non-destructive commands continue automatically. Destructive commands require your review."
    : "Automatic execution is enabled for every planned command, including destructive actions.";
  const activeChat = commandChats.find((chat) => chat.id === commandState.chatId) ?? commandChats[0];
  const chatOptions = commandChats.map((chat) => `<option value="${escapeHtml(chat.id)}" ${activeChat?.id === chat.id ? "selected" : ""}>${escapeHtml(chat.title)}</option>`).join("");
  const chatHistory = activeChat?.messages.length
    ? activeChat.messages.map((message) => `<article class="history-entry command-message"><div><strong>${message.role === "tool" ? "Command result" : message.role === "user" ? "You" : "Voxide"}</strong><small>${formatDate(message.createdAt)}</small></div><pre class="command-preview">${escapeHtml(message.content)}</pre>${database.settings.showThinkingTokens && message.thinking ? `<details class="history-raw"><summary>Model reasoning</summary><pre class="command-preview">${escapeHtml(message.thinking)}</pre></details>` : ""}</article>`).join("")
    : `<div class="empty">This conversation is ready for your first request.</div>`;
  const rewriteContent = command ? `
      <div class="card-title"><div><h2>Command conversation</h2><p>Conversation context and completed command results are saved locally.</p></div><div class="button-row"><button data-action="new-command-chat" ${commandState.processing ? "disabled" : ""}>New chat</button><button data-action="open-command-chat" ${commandState.processing ? "disabled" : ""}>Open</button><button data-action="clear-command-chat" ${commandState.processing ? "disabled" : ""}>Clear</button><button data-action="delete-command-chat" ${commandState.processing ? "disabled" : ""}>Delete</button></div></div>
      ${settingToggle("commandModeConfirmBeforeExecute", "Confirm destructive commands", "${commandExecutionPolicy}")}
      ${settingToggle("enableAiStreaming", "Stream AI responses", "Show command replies as the provider generates them. Tool arguments remain hidden until a complete command is ready for review.")}
      <label>Conversation<select id="command-chat" ${commandState.processing ? "disabled" : ""}>${chatOptions}</select></label>
      <section class="history-list command-history">${chatHistory}</section>
      <h2>Command request</h2>
      <textarea id="mode-input" rows="5" placeholder="For example: List the files in my Downloads folder…">${escapeHtml(commandState.draft)}</textarea>
      <div class="button-row"><button class="primary" data-action="plan-command" ${commandState.processing ? "disabled" : ""}>${commandState.processing ? "Planning…" : "Plan action"}</button><button data-action="dictate-mode" data-mode="command">Dictate instruction</button></div>
      ${commandState.processing && (commandState.streamingText || (database.settings.showThinkingTokens && commandState.streamingThinking)) ? `<section class="rewrite-result"><h3>Voxide is responding…</h3>${database.settings.showThinkingTokens && commandState.streamingThinking ? `<details class="history-raw" open><summary>Model reasoning</summary><pre class="command-preview">${escapeHtml(commandState.streamingThinking)}</pre></details>` : ""}${commandState.streamingText ? `<pre class="command-preview">${escapeHtml(commandState.streamingText)}</pre>` : ""}</section>` : ""}
      ${database.settings.showThinkingTokens && commandState.plan?.thinking ? `<details class="history-raw"><summary>Model reasoning</summary><pre class="command-preview">${escapeHtml(commandState.plan.thinking)}</pre></details>` : ""}
      ${commandState.plan?.kind === "answer" ? `<section class="rewrite-result"><h3>Answer</h3><p>${escapeHtml(commandState.plan.answer ?? "")}</p></section>` : ""}
      ${commandRequiresApproval ? `<section class="rewrite-result"><div class="card-title"><div><h3>Review before execution · destructive action</h3><p>${escapeHtml(commandState.plan?.purpose ?? "No purpose supplied.")}</p></div></div><pre class="command-preview">${escapeHtml(commandState.plan?.command ?? "")}</pre>${commandState.plan?.workingDirectory ? `<p class="muted">Working directory: ${escapeHtml(commandState.plan.workingDirectory)}</p>` : ""}<div class="button-row"><button class="primary" data-action="approve-command" ${commandState.processing ? "disabled" : ""}>${commandState.processing ? "Running…" : "Run reviewed command"}</button><button data-action="cancel-command" ${commandState.processing ? "disabled" : ""}>Cancel</button></div></section>` : ""}
      ${commandState.result ? `<section class="rewrite-result"><h3>${commandState.result.success ? "Command completed" : "Command failed"} · ${commandState.result.executionTimeMs} ms</h3>${commandState.result.output ? `<pre class="command-preview">${escapeHtml(commandState.result.output)}</pre>` : ""}${commandState.result.error ? `<pre class="command-preview error-output">${escapeHtml(commandState.result.error)}</pre>` : ""}</section>` : ""}
      <p class="muted">${escapeHtml(commandExecutionPolicy)} Commands stop after 30 seconds.</p>` : `
      <div class="card-title"><div><h2>${hasSelection ? "Selected text" : "Write new text"}</h2><p>${hasSelection ? "The selected text will be replaced only after you review and approve the result." : "Capture text from another app or ask the configured provider to write something new."}</p></div><div class="button-row"><button data-action="capture-selection">Capture selected text</button>${rewriteState.conversation.length ? `<button data-action="new-rewrite">New conversation</button>` : ""}</div></div>
      ${hasSelection ? `<textarea id="selected-text" rows="5" readonly>${escapeHtml(rewriteState.selectedText)}</textarea>` : ""}
      <label>${rewriteState.conversation.length ? "Follow-up instruction" : "Instruction"}<textarea id="mode-input" rows="3" placeholder="${hasSelection ? "For example: Make this concise and friendly…" : "For example: Draft a friendly follow-up email…"}">${escapeHtml(rewriteState.draft)}</textarea></label>
      <div class="button-row"><button class="primary" data-action="run-rewrite" ${rewriteState.processing ? "disabled" : ""}>${rewriteState.processing ? "Rewriting…" : rewriteState.conversation.length ? "Apply follow-up" : hasSelection ? "Create rewrite" : "Write text"}</button><button data-action="dictate-mode" data-mode="rewrite">Dictate instruction</button></div>
      ${rewriteState.outputText ? `<section class="rewrite-result"><h3>Result</h3><textarea id="rewrite-output" rows="6" readonly>${escapeHtml(rewriteState.outputText)}</textarea><div class="button-row"><button class="primary" data-action="insert-rewrite">${hasSelection ? "Replace original" : "Insert into active app"}</button><button data-action="copy-rewrite">Copy result</button></div></section>` : ""}
      ${rewriteState.conversation.length ? `<section class="history-list command-history"><h3>Session conversation</h3>${rewriteState.conversation.slice(-6).map((message) => `<article class="history-entry command-message"><div><strong>${message.role === "user" ? "You" : "Voxide"}</strong></div><pre class="command-preview">${escapeHtml(message.content)}</pre></article>`).join("")}</section>` : ""}
      <p class="muted">Voxide keeps the result separate until you choose to insert it. Follow-ups remain only for this rewrite session. Use the Rewrite global shortcut to capture selection without first focusing this window.</p>`;
  renderShell(`
    ${pageTitle(command ? "Command Mode" : "Write & Rewrite", command ? "Use speech to perform approved desktop actions and automation." : "Dictate fresh text or transform selected text in the active application.")}
    <section class="card form-card">
      ${rewriteContent}
    </section>`);
}

function renderFileTranscription(): void {
  // FileTranscriptionHistoryStore retains up to 50 entries. Keep every saved
  // entry reachable from the portable UI rather than silently limiting the
  // list to the newest eight.
  const items = database.fileTranscriptionHistory.map((entry) => {
    const details = [formatDate(entry.createdAt), formatDuration(entry.durationMs)].filter(Boolean);
    if (entry.confidence !== undefined) details.push(`${Math.round(entry.confidence * 100)}% confidence`);
    if (entry.processingTimeMs) {
      details.push(`${formatDuration(entry.processingTimeMs)} processing`);
      if (entry.durationMs) details.push(`${(entry.durationMs / entry.processingTimeMs).toFixed(1)}× realtime`);
    }
    return `<article class="history-entry"><div><strong>${escapeHtml(entry.fileName)}</strong><small>${escapeHtml(details.join(" · "))}</small></div><p>${escapeHtml(entry.text)}</p><div class="entry-actions"><button data-copy-file="${entry.id}">Copy</button><button data-export-file="${entry.id}">Export text</button><button data-delete-file="${entry.id}">Delete</button></div></article>`;
  }).join("");
  const selectedFileName = selectedFileForTranscription?.split(/[\\/]/).pop() ?? "";
  const selectedFile = selectedFileForTranscription
    ? `<p><strong>Selected:</strong> ${escapeHtml(selectedFileName)}</p><div class="button-row"><button data-action="choose-file" ${fileTranscriptionActive ? "disabled" : ""}>Choose another file</button><button class="primary" data-action="transcribe-selected-file" ${fileTranscriptionActive ? "disabled" : ""}>${fileTranscriptionActive ? "Transcribing…" : "Transcribe"}</button></div>`
    : `<button class="primary" data-action="choose-file" ${fileTranscriptionActive ? "disabled" : ""}>Choose file</button>`;
  renderShell(`
    ${pageTitle("File Transcription", "Transcribe audio and video files, then keep the result in local history.")}
    <section class="card drop-zone"><h2>Transcribe an audio or video file</h2><p>Choose or drop WAV, MP3, M4A, OGG, FLAC, MP4, MOV, WebM, or another supported media file.</p>${selectedFile}<small>${fileTranscriptionActive && fileProgress ? `Processing chunk ${fileProgress.completedChunks} of ${fileProgress.totalChunks}.` : "Long files are processed in 20-minute chunks. Non-WAV containers require FFmpeg on this computer."}</small></section>
    <section class="card"><div class="card-title"><h2>Recent files</h2>${database.fileTranscriptionHistory.length ? `<button data-action="clear-file-history">Clear history</button>` : ""}</div>${items || `<div class="empty">No files have been transcribed yet.</div>`}</section>`);
}

function renderHistory(): void {
  const query = historySearchQuery.trim().toLocaleLowerCase();
  const visibleEntries = database.dictationHistory.filter((entry) => !query || [
    entry.text,
    entry.rawText ?? "",
    entry.sourceApplication ?? "",
    entry.mode,
  ].some((value) => value.toLocaleLowerCase().includes(query)));
  const entries = visibleEntries.map((entry) => {
    const playback = entry.audioFile ? `<audio class="history-audio" controls preload="metadata" src="${escapeHtml(convertFileSrc(entry.audioFile))}">Saved recording</audio>` : "";
    const audioExport = entry.audioFile ? `<button data-export-audio-entry="${entry.id}">Export audio ZIP</button>` : "";
    const source = entry.sourceApplication?.trim() || "Unknown app";
    const rawText = entry.rawText && entry.rawText !== entry.text
      ? `<details class="history-raw"><summary>Original transcription${entry.wasAiProcessed ? " · AI enhanced" : ""}</summary><p>${escapeHtml(entry.rawText)}</p></details>`
      : "";
    const aiStatus = entry.wasAiProcessed ? " · AI enhanced" : entry.aiProcessingError ? " · AI fallback" : "";
    const error = entry.aiProcessingError ? `<p class="history-warning">AI enhancement failed: ${escapeHtml(entry.aiProcessingError)}</p>` : "";
    const rawActions = entry.rawText && entry.rawText !== entry.text
      ? `<button data-copy-raw-entry="${entry.id}">Copy original</button><button data-copy-both-entry="${entry.id}">Copy both</button>`
      : "";
    const windowTitle = entry.sourceWindowTitle ? ` · ${escapeHtml(entry.sourceWindowTitle)}` : "";
    return `<article class="history-entry"><div><strong>${escapeHtml(source)}</strong><small>${formatDate(entry.createdAt)} · ${entry.mode} · ${formatDuration(entry.durationMs) || "duration unavailable"}${windowTitle}${aiStatus}</small></div><p>${escapeHtml(entry.text)}</p>${rawText}${error}${playback}<div class="entry-actions"><button data-copy-entry="${entry.id}">Copy final</button>${rawActions}<button data-edit-entry="${entry.id}">Edit</button><button data-export-entry="${entry.id}">Export text</button>${audioExport}<button data-delete-entry="${entry.id}">Delete</button></div></article>`;
  }).join("");
  renderShell(`
    ${pageTitle("History", "Your locally saved dictations.", database.dictationHistory.length ? `<button data-action="clear-history">Clear history</button>` : "")}
    <section class="card history-toolbar"><label>Search transcriptions<input id="history-search" value="${escapeHtml(historySearchQuery)}" placeholder="Text, original text, app, or mode"></label><small>${visibleEntries.length} of ${database.dictationHistory.length} entries</small></section>
    <section class="card history-list">${entries || `<div class="empty">${database.dictationHistory.length ? "No results. Try a different search term." : "No dictations yet. Start with the test playground."}</div>`}</section>`);
}

function renderStats(): void {
  const activity = statsActivityDays === 7 ? stats.dailyActivity7 : stats.dailyActivity30;
  const maxWords = Math.max(0, ...activity.map((item) => item.words));
  const activityBars = activity.map((item) => {
    const height = item.words && maxWords ? Math.max(3, Math.round(item.words / maxWords * 96)) : 3;
    const title = `${item.date}: ${item.words.toLocaleString()} word${item.words === 1 ? "" : "s"}, ${item.transcriptions} session${item.transcriptions === 1 ? "" : "s"}`;
    return `<div class="activity-bar" title="${escapeHtml(title)}"><i class="${item.words ? "active" : ""}" style="height:${height}px"></i>${statsActivityDays === 7 ? `<small>${escapeHtml(activityDayLabel(item.date))}</small>` : ""}</div>`;
  }).join("");
  const milestoneGroup = (title: string, milestones: UsageMilestone[]) => `<div class="milestone-row"><strong>${title}</strong>${milestones.map((milestone) => `<span class="${milestone.achieved ? "achieved" : ""}" title="${milestone.target.toLocaleString()}">${milestone.achieved ? "✓" : "○"} ${escapeHtml(milestone.label)}</span>`).join("")}</div>`;
  const activeDays = activity.filter((item) => item.words > 0).length;
  const periodWords = activity.reduce((sum, item) => sum + item.words, 0);
  const topApps = stats.topApps.length ? stats.topApps.map((app) => `${escapeHtml(app.app)} (${app.count})`).join(", ") : "No data yet";
  renderShell(`
    ${pageTitle("Stats", "Local dictation activity, streaks, and milestones.")}
    <section class="card prominent stats-today"><div><h2>Today</h2><p>${stats.todayWords ? "Every word counts — keep going." : stats.currentStreak ? "Keep the streak alive — say a few words." : "Ready when you are. Start dictating to save time."}</p></div><div><strong>${stats.todayWords.toLocaleString()}</strong><span>words</span></div><div><strong>${formatSavedTime(stats.todayTimeSavedMinutes)}</strong><span>saved</span></div><div><strong>${stats.todayDictations}</strong><span>sessions</span></div></section>
    <section class="stat-grid stats-grid-wide"><article class="card"><small>TIME SAVED</small><strong>${formatSavedTime(stats.totalTimeSavedMinutes)}</strong><span>at ${database.settings.userTypingWpm} WPM typing</span></article><article class="card"><small>TOTAL WORDS</small><strong>${stats.totalWords.toLocaleString()}</strong><span>${stats.todayWords ? `+${stats.todayWords.toLocaleString()} today` : "Start dictating"}</span></article><article class="card"><small>CURRENT STREAK</small><strong>${stats.currentStreak}</strong><span>Best: ${stats.bestStreak} days</span></article><article class="card"><small>TRANSCRIPTIONS</small><strong>${stats.totalDictations}</strong><span>Avg: ${stats.averageWordsPerDictation} words each</span></article></section>
    <section class="card stats-section"><div class="card-title"><div><h2>Activity</h2><p>Words captured across recent days.</p></div><div class="button-row"><button data-stats-days="7" ${statsActivityDays === 7 ? "disabled" : ""}>7 days</button><button data-stats-days="30" ${statsActivityDays === 30 ? "disabled" : ""}>30 days</button></div></div>${maxWords ? `<div class="activity-chart">${activityBars}</div><p class="muted">${periodWords.toLocaleString()} words across ${activeDays} active day${activeDays === 1 ? "" : "s"}.</p>` : `<div class="empty">No activity yet.</div>`}</section>
    <section class="card stats-section"><div class="card-title"><div><h2>Milestones</h2><p>Progress is calculated from local history only.</p></div><span class="status done">${[...stats.wordMilestones, ...stats.transcriptionMilestones, ...stats.streakMilestones].filter((milestone) => milestone.achieved).length}/${stats.wordMilestones.length + stats.transcriptionMilestones.length + stats.streakMilestones.length}</span></div><div class="milestones">${milestoneGroup("Words", stats.wordMilestones)}${milestoneGroup("Transcriptions", stats.transcriptionMilestones)}${milestoneGroup("Streak", stats.streakMilestones)}</div></section>
    <section class="stats-insights"><article class="card"><small>TOP APPS</small><strong>${topApps}</strong><span>Saved source applications</span></article><article class="card"><small>AI ENHANCED</small><strong>${stats.aiEnhancementRate}%</strong><span>${stats.aiProcessedCount} completed with AI</span></article><article class="card"><small>PEAK TIME</small><strong>${formatPeakHour(stats.peakHour)}</strong><span>Most common local hour</span></article><article class="card"><small>CHARACTERS</small><strong>${stats.totalCharacters.toLocaleString()}</strong><span>Final transcription text</span></article></section>
    <section class="stats-insights"><article class="card"><small>LONGEST TRANSCRIPTION</small><strong>${stats.longestTranscriptionWords.toLocaleString()} words</strong></article><article class="card"><small>MOST WORDS IN A DAY</small><strong>${stats.mostWordsInDay.toLocaleString()} words</strong></article><article class="card"><small>MOST SESSIONS IN A DAY</small><strong>${stats.mostTranscriptionsInDay}</strong></article></section>`);
}

function renderFeedback(): void {
  renderShell(`
    ${pageTitle("Send Feedback", "Report bugs and request features on GitHub.")}
    <section class="card prominent"><h2>Found a bug or have an idea?</h2><p>Voxide is developed in the open. Feedback lives in the public issue tracker, so nothing is sent anywhere until you file the issue yourself.</p></section>
    <section class="card form-card">
      <div class="button-row"><button class="primary" data-action="open-feedback-issue">Open a GitHub issue</button><button data-action="copy-debug-information">Copy debug information</button></div>
      <p class="muted">Debug information covers the Voxide version, operating system, architecture, current timestamp, and the latest 30 diagnostic log lines. Dictation, command, clipboard, file-path, and API-key content are never logged. Paste it into the issue when it is relevant.</p>
    </section>`);
}

function hotkeySetting(label: string, id: string, value: string | undefined, placeholder: string): string {
  return `<label>${escapeHtml(label)}<span class="shortcut-control"><input id="${id}" value="${escapeHtml(value ?? "")}" placeholder="${escapeHtml(placeholder)}"><button type="button" data-capture-hotkey="${id}">Record</button></span></label>`;
}

function renderSettings(): void {
  const punctuationRules = JSON.stringify(database.settings.punctuationDictionaryRules, null, 2);
  const savedAudioCount = database.dictationHistory.filter((entry) => entry.audioFile).length;
  const dictationPromptProfiles = database.promptProfiles.filter((profile) => profile.mode === "dictate");
  const promptModeProfileOptions = [
    `<option value="">Default dictation prompt</option>`,
    ...dictationPromptProfiles.map((profile) =>
      `<option value="${escapeHtml(profile.id)}" ${profile.id === database.settings.promptModeSelectedPromptId ? "selected" : ""}>${escapeHtml(profile.name)}</option>`,
    ),
  ].join("");
  const promptShortcutAssignments = database.settings.promptShortcutAssignments.map((assignment, index) => {
    const profileOptions = dictationPromptProfiles.map((profile) =>
      `<option value="${escapeHtml(profile.id)}" ${profile.id === assignment.promptProfileId ? "selected" : ""}>${escapeHtml(profile.name)}</option>`,
    ).join("");
    return `<div class="shortcut-assignment"><label>Prompt profile<select id="prompt-shortcut-profile-${index}">${profileOptions}</select></label>${hotkeySetting("Shortcut", `prompt-shortcut-hotkey-${index}`, assignment.hotkey, "Optional")}<button type="button" data-action="delete-prompt-shortcut" data-prompt-shortcut-index="${index}">Remove</button></div>`;
  }).join("");
  const accessibilityText = accessibilityPermissionStatus?.supported
    ? accessibilityPermissionStatus.trusted
      ? "Accessibility access is enabled."
      : "Accessibility access is not enabled."
    : "This desktop platform does not expose one universal accessibility-permission status.";
  const releaseNotes = recentReleaseNotes.map((note) => `<article class="profile"><div><strong>${escapeHtml(note.title)}</strong><small>${escapeHtml(note.version)}${note.isPrerelease ? " · prerelease" : ""}${note.publishedAt ? ` · ${escapeHtml(formatDate(note.publishedAt))}` : ""}</small></div><p>${escapeHtml(note.notes).replace(/\n/g, "<br>")}</p>${note.releaseUrl ? `<button data-action="open-release-note" data-release-url="${escapeHtml(note.releaseUrl)}">Open release</button>` : ""}</article>`).join("");
  renderShell(`
    ${pageTitle("Settings", "Configure Voxide desktop behavior and dictation output.")}
    <section class="card form-card"><h2>Global dictation</h2>
      ${hotkeyBackendNotice()}
      ${hotkeySetting("Primary shortcut", "primary-hotkey", database.settings.primaryDictationHotkey, "Alt+Space")}
      ${hotkeySetting("Secondary dictation shortcut", "secondary-hotkey", database.settings.secondaryDictationHotkey, "Optional")}
      <label>Activation mode<select id="activation-mode"><option value="toggle" ${database.settings.hotkeyActivationMode === "toggle" ? "selected" : ""}>Toggle recording</option><option value="hold" ${database.settings.hotkeyActivationMode === "hold" ? "selected" : ""}>Hold to record</option><option value="automatic" ${database.settings.hotkeyActivationMode === "automatic" ? "selected" : ""}>Automatic (tap or hold)</option></select></label>
      ${hotkeySetting("Prompt mode shortcut", "prompt-hotkey", database.settings.promptModeHotkey, "Optional")}
      <label>Prompt mode profile<select id="prompt-mode-profile">${promptModeProfileOptions}</select><small>Use this Dictate profile whenever the Prompt mode shortcut is pressed. Leave it at Default to use the normal built-in dictation prompt.</small></label>
      <div class="setting-subsection"><h3>Prompt profile shortcuts</h3><p class="muted">Run a specific Dictate prompt profile without changing your normal active profile.</p>${promptShortcutAssignments || `<p class="muted">No profile shortcuts configured.</p>`}<button type="button" data-action="add-prompt-shortcut" ${dictationPromptProfiles.length ? "" : "disabled"}>Add prompt shortcut</button></div>
      ${hotkeySetting("Command mode shortcut", "command-hotkey", database.settings.commandModeHotkey, "Optional")}
      ${hotkeySetting("Rewrite mode shortcut", "rewrite-hotkey", database.settings.rewriteModeHotkey, "Optional")}
      ${hotkeySetting("Cancel recording shortcut", "cancel-hotkey", database.settings.cancelRecordingHotkey, "Escape")}
      ${hotkeySetting("Paste last transcription shortcut", "paste-last-hotkey", database.settings.pasteLastTranscriptionHotkey, "Optional")}
      ${settingToggle("enableStreamingPreview", "Live preview", "Show partial transcription while you speak.")}
      <label>Transcription preview length (${database.settings.transcriptionPreviewCharLimit} characters)<input id="transcription-preview-char-limit" type="range" min="50" max="800" step="50" value="${database.settings.transcriptionPreviewCharLimit}"><small>Show this many recent characters in the recording overlay.</small></label>
      <div class="setting-subsection"><h3>Dictation overlay</h3><p class="muted">Keep the non-activating recording overlay centered on the active display’s usable area.</p><label>Position<select id="overlay-position"><option value="bottom" ${database.settings.overlayPosition === "bottom" ? "selected" : ""}>Bottom of screen</option><option value="top" ${database.settings.overlayPosition === "top" ? "selected" : ""}>Top of screen</option></select></label><label>Size<select id="overlay-size"><option value="pill" ${database.settings.overlaySize === "pill" ? "selected" : ""}>Pill</option><option value="small" ${database.settings.overlaySize === "small" ? "selected" : ""}>Small</option><option value="medium" ${database.settings.overlaySize === "medium" ? "selected" : ""}>Medium</option><option value="large" ${database.settings.overlaySize === "large" ? "selected" : ""}>Large</option></select></label><label>Screen edge offset (${Math.round(database.settings.overlayBottomOffset)} px)<input id="overlay-bottom-offset" type="range" min="10" max="500" step="5" value="${database.settings.overlayBottomOffset}"></label></div>
      <div class="setting-subsection"><h3>Recording cue</h3><p class="muted">Choose a start cue; Cue 1 also plays an end cue. Cues use the selected volume and leave your system volume unchanged.</p>
        <label>Sound<select id="transcription-start-sound"><option value="none" ${database.settings.transcriptionStartSound === "none" ? "selected" : ""}>None</option><option value="cue_1" ${database.settings.transcriptionStartSound === "cue_1" ? "selected" : ""}>Cue 1 (start + end)</option><option value="cue_2" ${database.settings.transcriptionStartSound === "cue_2" ? "selected" : ""}>Cue 2</option><option value="cue_3" ${database.settings.transcriptionStartSound === "cue_3" ? "selected" : ""}>Cue 3</option><option value="cue_4" ${database.settings.transcriptionStartSound === "cue_4" ? "selected" : ""}>Cue 4</option><option value="cue_5" ${database.settings.transcriptionStartSound === "cue_5" ? "selected" : ""}>Cue 5</option></select></label>
        <label>Volume<input id="transcription-sound-volume" type="range" min="0" max="1" step="0.05" value="${database.settings.transcriptionSoundVolume}"></label>
      </div>
      ${settingToggle("copyToClipboard", "Copy completed dictations", "Copy final text to the clipboard.")}
      ${settingToggle("typeIntoActiveApplication", "Type into active application", "Insert completed text where you were working.")}
      <label>Text insertion mode<select id="text-insertion-mode"><option value="standard" ${database.settings.textInsertionMode === "standard" ? "selected" : ""}>Clipboard-free insert</option><option value="reliablePaste" ${database.settings.textInsertionMode === "reliablePaste" ? "selected" : ""}>Clipboard paste</option></select><small>${database.settings.textInsertionMode === "reliablePaste" ? "Compatibility path: temporarily pastes through the clipboard, so clipboard-history apps may briefly record dictated text." : "Fastest path: direct insertion leaves the clipboard unchanged, with clipboard paste only if direct insertion is unavailable."}</small></label>
      <button class="primary" data-action="save-hotkey">Apply shortcut</button>
    </section>
    <section class="card form-card"><h2>Dictation formatting</h2>
      ${settingToggle("removeFillerWordsEnabled", "Remove filler words", "Remove configured sounds such as um, uh, and er before dictionary or AI processing.")}
      <label>Filler words (comma-separated)<input id="filler-words" value="${escapeHtml(database.settings.fillerWords.join(", "))}"></label>
      ${settingToggle("autoConvertPunctuationEnabled", "Explicit spoken punctuation", "Convert phrases only when prefixed, such as “literal comma”.")}
      <label>Punctuation prefix<input id="punctuation-prefix" value="${escapeHtml(database.settings.punctuationDictionaryPrefix)}" placeholder="literal"></label>
      <label>Punctuation rules (JSON)<textarea id="punctuation-rules" rows="9" spellcheck="false">${escapeHtml(punctuationRules)}</textarea></label>
      ${settingToggle("literalDictationFormattingEnabled", "Slash commands & @ formatting", "Convert “slash status”, “tag Ada”, and “at sign Ada” without AI.")}
      ${settingToggle("gaavLowercaseFirstLetterEnabled", "Lowercase first letter", "Start each final dictation with a lowercase letter.")}
      ${settingToggle("gaavRemoveTrailingPeriodEnabled", "Remove trailing period", "Drop a final period from completed dictations.")}
      ${settingToggle("continuousDictationSpacingEnabled", "Space between dictations", "Insert spacing and a trailing space so consecutive dictations chain without using the spacebar.")}
      ${settingToggle("contextAwareCapitalizationEnabled", "Smart capitalization", "Use the previous insertion in the same captured app/window to decide whether the next dictation begins uppercase or lowercase.")}
      ${settingToggle("notifyAiProcessingFailures", "Notify AI enhancement failures", "Type the cleaned raw transcription and show a native notification if AI cleanup fails.")}
      <button data-action="save-output-formatting">Save formatting</button>
    </section>
    <section class="card form-card"><h2>Privacy and application</h2>
      <label>Theme<select id="theme"><option value="system" ${database.settings.theme === "system" ? "selected" : ""}>System</option><option value="light" ${database.settings.theme === "light" ? "selected" : ""}>Light</option><option value="dark" ${database.settings.theme === "dark" ? "selected" : ""}>Dark</option></select></label>
      ${settingToggle("saveTranscriptionHistory", "Save transcription history", "Keep completed dictations and usage statistics locally. Audio history requires this setting.")}
      <label>Typing speed for time saved (WPM)<input id="user-typing-wpm" type="number" min="1" max="200" value="${database.settings.userTypingWpm}"></label>
      ${settingToggle("weekendsDontBreakStreak", "Weekends don't break streaks", "Skip Saturday and Sunday when calculating usage streaks.")}
      <button data-action="save-stats-preferences">Save stats preferences</button>
      ${settingToggle("audioHistoryEnabled", "Save audio history", "Keep recordings locally within the configured disk budget.")}
      <label>Audio history budget (GB)<input id="audio-history-budget" type="number" min="0.1" step="0.1" value="${database.settings.audioHistoryBudgetGb}"></label>
      <button data-action="save-audio-history">Save audio history budget</button>
      <div class="button-row"><button data-action="export-audio-history" ${savedAudioCount ? "" : "disabled"}>Export audio ZIP</button><button data-action="delete-saved-audio" ${savedAudioCount ? "" : "disabled"}>Delete saved audio</button></div>
      <p class="muted">${savedAudioCount ? `${savedAudioCount} recording${savedAudioCount === 1 ? "" : "s"} can be exported with a JSONL manifest.` : "No saved audio recordings are available."}</p>
      ${settingToggle("launchAtStartup", "Launch at startup", "Start Voxide when you sign in.")}
      ${settingToggle("showMainWindowAtLoginLaunch", "Show window after login", "When Voxide starts at login, open the main window instead of running in the tray.")}
      ${settingToggle("showInDock", "Show in taskbar or dock", "Keep the Voxide main window visible in the operating system switcher.")}
      ${settingToggle("shareAnonymousAnalytics", "Share anonymous analytics", "Send bounded aggregate usage metadata only; transcription text, commands, and API keys are never included.")}
      <div class="setting-subsection"><h3>Text input permission</h3><p class="muted">${escapeHtml(accessibilityText)} ${escapeHtml(accessibilityPermissionStatus?.guidance ?? "Checking permission status…")}</p><div class="button-row"><button data-action="open-accessibility-settings">Open input settings</button><button data-action="refresh-accessibility-status">Refresh status</button></div></div>
    </section>
    <section class="card form-card"><h2>Software updates</h2>
      ${settingToggle("autoUpdateCheckEnabled", "Check automatically", "Check GitHub Releases on launch and then at most once per hour.")}
      ${settingToggle("betaReleasesEnabled", "Include beta releases", "Offer prerelease builds as well as stable releases.")}
      <div class="button-row"><button data-action="check-for-updates">Check for updates now</button><button data-action="view-release-notes">View recent release notes</button></div>
      <p class="muted">${database.settings.lastUpdateCheckAt ? `Last checked ${escapeHtml(formatDate(database.settings.lastUpdateCheckAt))}.` : "No update check has completed yet."} Updates open the verified release page for installation.</p>
      ${releaseNotes ? `<div class="setting-subsection"><h3>Recent releases</h3>${releaseNotes}</div>` : ""}
    </section>
    <section class="card form-card"><h2>Backup and restore</h2>
      <p class="muted">Backups include settings, providers, prompt profiles, dictionaries, and local history. API keys remain in the operating system credential store and are not exported.</p>
      <div class="button-row"><button data-action="export-backup">Export backup</button><button data-action="import-backup">Restore backup</button></div>
    </section>
    <section class="card form-card"><h2>Local API</h2>
      <label class="setting-row"><span><strong>Enable loopback API</strong><small>Expose local history, dictionary, post-processing, and transcription endpoints at 127.0.0.1 only.</small></span><input id="local-api-enabled" type="checkbox" ${database.settings.localApiEnabled ? "checked" : ""}></label>
      <label>Port<input id="local-api-port" type="number" min="1" max="65535" value="${database.settings.localApiPort}"></label>
      <div class="button-row"><button class="primary" data-action="save-local-api">Apply local API</button></div>
      <p class="muted">${apiStatus?.enabled && apiStatus.url ? `Listening at ${escapeHtml(apiStatus.url)}.` : "The API is disabled."}</p>
    </section>`);
}

function render(): void {
  switch (currentView) {
    case "voice": renderVoiceEngine(); break;
    case "enhancement": renderEnhancement(); break;
    case "dictionary": renderDictionary(); break;
    case "command": renderModePage("command"); break;
    case "rewrite": renderModePage("rewrite"); break;
    case "file": renderFileTranscription(); break;
    case "history": renderHistory(); break;
    case "stats": renderStats(); break;
    case "feedback": renderFeedback(); break;
    case "settings": renderSettings(); break;
    default: renderWelcome();
  }
}

async function refreshStats(): Promise<void> {
  stats = await invoke<UsageStats>("usage_stats");
}

async function refreshModelStatus(): Promise<void> {
  modelStatus = await invoke<VoiceModelStatus>("voice_model_status");
}

async function refreshAudioDevices(): Promise<void> {
  audioDevices = await invoke<string[]>("audio_input_devices");
}

async function refreshAudioDevicesWhenChanged(): Promise<void> {
  try {
    const updated = await invoke<string[]>("audio_input_devices");
    if (updated.length === audioDevices.length && updated.every((device, index) => device === audioDevices[index])) return;
    audioDevices = updated;
    if (currentView === "voice") render();
  } catch {
    // A transient audio-server failure should not disrupt an active dictation.
  }
}

async function refreshProviders(): Promise<void> {
  providers = await invoke<AiProviderView[]>("ai_providers");
}

async function refreshCommandChats(): Promise<void> {
  commandChats = await invoke<CommandChat[]>("command_chats");
  if (!commandState.chatId || !commandChats.some((chat) => chat.id === commandState.chatId)) {
    commandState.chatId = database.activeCommandChatId ?? commandChats[0]?.id;
  }
  commandState.sourceApplication = commandChats.find((chat) => chat.id === commandState.chatId)?.sourceApplication;
}

async function refreshLocalApiStatus(): Promise<void> {
  apiStatus = await invoke<LocalApiStatus>("local_api_status");
}

async function refreshAccessibilityPermissionStatus(): Promise<void> {
  accessibilityPermissionStatus = await invoke<AccessibilityPermissionStatus>("accessibility_permission_status");
}

async function refreshHotkeyBackendStatus(): Promise<void> {
  hotkeyBackendStatus = await invoke<HotkeyBackendStatus>("hotkey_backend_status");
}

function hotkeyBackendNotice(): string {
  if (hotkeyBackendStatus?.backend !== "portal") return "";
  const detail = hotkeyBackendStatus.detail ? ` ${hotkeyBackendStatus.detail}.` : "";
  const message = {
    active: `Global shortcuts run through the desktop portal on this Wayland session.${detail}`,
    initializing: `Waiting for the desktop portal to confirm global shortcut bindings.${detail} Your desktop environment may ask you to approve them.`,
    inactive: "Global shortcuts have not been bound through the desktop portal yet.",
    unavailable: `This Wayland desktop does not provide the global shortcuts portal, so system-wide shortcuts are unavailable.${detail} Bind keys in your compositor configuration to "voxide --trigger dictate" (or use the tray) instead.`,
    denied: `The desktop environment did not grant the global shortcut bindings.${detail} Re-apply the shortcuts to request them again, or bind keys in your compositor configuration to "voxide --trigger dictate".`,
    error: `Global shortcut binding through the desktop portal failed.${detail} Re-apply the shortcuts to retry.`,
  }[hotkeyBackendStatus.state] ?? "";
  if (!message) return "";
  const tone = hotkeyBackendStatus.state === "active" ? "muted" : "warning";
  return `<p class="${tone} hotkey-backend-status">${escapeHtml(message)}</p>`;
}

async function saveSettings(): Promise<void> {
  database.settings = await invoke<Settings>("save_settings", { settings: database.settings });
}

async function startRecording(promptProfileId?: string, fromAppUi = false): Promise<void> {
  if (recording) return;
  const isPromptTest = promptTestState.active
    && !dictationInstructionTarget
    && !promptProfileId
    && Boolean(promptTestState.profileId);
  const capturePromptProfileId = promptProfileId ?? (isPromptTest ? promptTestState.profileId : undefined);
  liveText = "";
  // Only a dictation started from the app's own UI hides the window (so the
  // text lands in the previously focused application) and restores it
  // afterwards. Hotkey- and tray-initiated dictations must never move the
  // user's focus — GTK's focus report is unreliable on Wayland, so it is not
  // consulted for those.
  const hideFocusedMainWindow = fromAppUi && !isOverlayWindow && !isPromptTest
    && await currentWindow.isFocused().catch(() => false);
  if (hideFocusedMainWindow) {
    await currentWindow.hide();
    // Let the OS restore the previously focused application before native
    // capture records its target context.
    await new Promise<void>((resolve) => window.setTimeout(resolve, 120));
  }
  try {
    const started = await invoke<NativeCaptureStarted>("start_native_dictation", { promptProfileId: capturePromptProfileId });
    pendingDictationContext = {
      sourceApplication: started.sourceApplication,
      sourceWindowTitle: started.sourceWindowTitle,
    };
    recording = true;
    restoreMainWindowAfterRecording = hideFocusedMainWindow;
    void playTranscriptionCue("start");
    render();
  } catch (error) {
    if (hideFocusedMainWindow) await currentWindow.show();
    showNotice(`Could not start native microphone capture: ${String(error)}`);
  }
}

async function restoreMainWindowAfterDictation(): Promise<void> {
  if (!restoreMainWindowAfterRecording) return;
  restoreMainWindowAfterRecording = false;
  await currentWindow.show();
  await currentWindow.setFocus();
}

async function stopRecording(): Promise<void> {
  if (!recording) return;
  recording = false;
  const completedTarget = dictationInstructionTarget;
  const isPromptTest = promptTestState.active && !completedTarget && !dictationPromptProfileId;
  if (isPromptTest) {
    promptTestState.processing = true;
    promptTestState.error = undefined;
    promptTestState.outputText = "";
    if (currentView === "enhancement") render();
  }
  let completedWithText = false;
  void playTranscriptionCue("stop");
  try {
    const result = await invoke<NativeTranscriptionResult>("stop_native_dictation", {
      promptMode: dictationInstructionTarget === "prompt",
      instructionMode: dictationInstructionTarget,
      promptTestMode: isPromptTest,
      promptTestPrompt: isPromptTest ? promptTestState.draftPrompt : undefined,
    });
    const hasTranscription = result.text.trim().length > 0;
    if (isPromptTest) {
      liveText = result.rawText;
      promptTestState.rawText = result.rawText;
      promptTestState.processing = false;
      if (result.aiProcessingError) {
        promptTestState.error = result.aiProcessingError;
        promptTestState.outputText = "";
      } else {
        promptTestState.outputText = result.text;
      }
    } else if (hasTranscription && database.settings.saveTranscriptionHistory) {
      liveText = result.text;
      completedWithText = hasTranscription;
      const entry = await invoke<DictationEntry>("save_dictation", {
        text: result.text,
        rawText: result.rawText,
        durationMs: result.durationMs,
        mode: dictationInstructionTarget ?? "dictate",
        audioFile: result.audioFile,
        audioModel: result.audioModel,
        sourceApplication: result.sourceApplication ?? pendingDictationContext.sourceApplication,
        sourceWindowTitle: result.sourceWindowTitle ?? pendingDictationContext.sourceWindowTitle,
        wasAiProcessed: result.wasAiProcessed,
        processingModel: result.processingModel,
        aiProcessingError: result.aiProcessingError,
      });
      database.dictationHistory.unshift(entry);
    } else {
      liveText = result.text;
      completedWithText = hasTranscription;
    }
    if (!isPromptTest && hasTranscription && database.settings.copyToClipboard) {
      try {
        await invoke("copy_completed_dictation", { text: liveText });
      } catch (error) {
        showNotice(`Dictation completed, but could not copy it: ${String(error)}`);
      }
    }
    if (!isPromptTest && hasTranscription) await refreshStats();
    if (!isPromptTest && hasTranscription && completedTarget === "rewrite") {
      rewriteState.draft = result.text;
      rewriteState.sourceApplication = result.sourceApplication ?? pendingDictationContext.sourceApplication;
      currentView = "rewrite";
    } else if (!isPromptTest && hasTranscription && completedTarget === "command") {
      commandState.draft = result.text;
      commandState.sourceApplication = result.sourceApplication ?? pendingDictationContext.sourceApplication;
      currentView = "command";
    }
  } catch (error) {
    if (isPromptTest) {
      promptTestState.processing = false;
      promptTestState.error = String(error);
    } else {
      showNotice(`Dictation could not be transcribed: ${String(error)}`);
    }
  }
  dictationInstructionTarget = undefined;
  dictationPromptProfileId = undefined;
  automaticHotkeyTarget = undefined;
  automaticHotkeyStartedAt = undefined;
  pendingDictationContext = {};
  await restoreMainWindowAfterDictation();
  render();
  if (completedWithText && completedTarget === "rewrite") await runRewrite();
  if (completedWithText && completedTarget === "command") await planCommand();
}

async function toggleRecording(fromAppUi = false): Promise<void> {
  if (recording) await stopRecording(); else await startRecording(undefined, fromAppUi);
}

async function handleAutomaticDictationHotkey(
  target: "dictate" | "prompt" | "command" | "rewrite",
  event: HotkeyEvent,
  promptProfileId?: string,
): Promise<void> {
  if (event.phase === "pressed") {
    if (recording) {
      await stopRecording();
      return;
    }
    automaticHotkeyTarget = target;
    automaticHotkeyStartedAt = Date.now();
    await beginModeDictation(target, promptProfileId);
    return;
  }
  if (event.phase !== "released" || automaticHotkeyTarget !== target || !recording) return;
  const heldForMs = Date.now() - (automaticHotkeyStartedAt ?? Date.now());
  automaticHotkeyTarget = undefined;
  automaticHotkeyStartedAt = undefined;
  if (heldForMs >= 350) await stopRecording();
}

async function beginModeDictation(
  target: "dictate" | "prompt" | "command" | "rewrite",
  promptProfileId?: string,
): Promise<void> {
  if (target === "rewrite") {
    try {
      const selection = await invoke<CapturedSelection>("capture_selected_text");
      rewriteState.selectedText = selection.text;
      rewriteState.sourceApplication = selection.sourceApplication;
      rewriteState.outputText = "";
      rewriteState.draft = "";
      rewriteState.conversation = [];
    } catch {
      // No selection starts write mode, matching the reference behavior.
      rewriteState.selectedText = "";
      rewriteState.sourceApplication = undefined;
      rewriteState.outputText = "";
      rewriteState.draft = "";
      rewriteState.conversation = [];
    }
  }
  dictationInstructionTarget = target === "dictate" ? undefined : target;
  dictationPromptProfileId = target === "prompt" ? promptProfileId : undefined;
  await startRecording(dictationPromptProfileId);
}

async function handleModeDictationHotkey(
  target: "prompt" | "command" | "rewrite",
  event: HotkeyEvent,
  promptProfileId?: string,
): Promise<void> {
  if (database.settings.hotkeyActivationMode === "hold") {
    if (event.phase === "pressed" && !recording) await beginModeDictation(target, promptProfileId);
    if (event.phase === "released" && recording && dictationInstructionTarget === target) await stopRecording();
    return;
  }
  if (database.settings.hotkeyActivationMode === "automatic") {
    await handleAutomaticDictationHotkey(target, event, promptProfileId);
    return;
  }
  if (event.phase !== "pressed") return;
  if (recording && dictationInstructionTarget === target) await stopRecording();
  else if (!recording) await beginModeDictation(target, promptProfileId);
}

async function handleGlobalHotkey(event: HotkeyEvent): Promise<void> {
  switch (event.action) {
    case "dictate":
      if (database.settings.hotkeyActivationMode === "hold") {
        if (event.phase === "pressed") await startRecording();
        if (event.phase === "released") await stopRecording();
      } else if (database.settings.hotkeyActivationMode === "automatic") {
        await handleAutomaticDictationHotkey("dictate", event);
      } else if (event.phase === "pressed") {
        await toggleRecording();
      }
      break;
    case "cancel":
      if (event.phase === "pressed") {
        const cancelled = await invoke<boolean>("cancel_native_dictation");
        if (cancelled) {
          recording = false;
          liveText = "";
          dictationInstructionTarget = undefined;
          dictationPromptProfileId = undefined;
          automaticHotkeyTarget = undefined;
          automaticHotkeyStartedAt = undefined;
          pendingDictationContext = {};
          await restoreMainWindowAfterDictation();
          render();
        }
      }
      break;
    case "pasteLast":
      if (event.phase === "pressed") {
        await invoke("paste_last_transcription");
      }
      break;
    case "rewrite":
      await handleModeDictationHotkey("rewrite", event);
      break;
    case "prompt":
      await handleModeDictationHotkey("prompt", event, event.promptProfileId ?? database.settings.promptModeSelectedPromptId);
      break;
    case "command":
      await handleModeDictationHotkey("command", event);
      break;
    default:
      break;
  }
}

async function handleTrayAction(event: TrayAction): Promise<void> {
  switch (event.action) {
    case "toggleDictation": await toggleRecording(); break;
    case "pasteLast":
      try {
        await invoke("paste_last_transcription");
      } catch (error) {
        showNotice(`Could not paste the last transcription: ${String(error)}`);
      }
      break;
    case "settings": currentView = "settings"; render(); break;
    case "dictionary": currentView = "dictionary"; render(); break;
    default: break;
  }
}

function readInput(id: string): HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement | null {
  return document.querySelector<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>(`#${id}`);
}

function optionalHotkey(id: string): string | undefined {
  return readInput(id)?.value.trim() || undefined;
}

let stopHotkeyCapture: (() => void) | undefined;

function shortcutFromKeyboardEvent(event: KeyboardEvent): string | undefined {
  const modifierCodes = new Set([
    "AltLeft", "AltRight", "ControlLeft", "ControlRight", "MetaLeft", "MetaRight", "ShiftLeft", "ShiftRight",
  ]);
  if (modifierCodes.has(event.code) || event.code === "Unidentified") return undefined;
  const key = event.code === "Space" ? "Space" : event.code;
  const modifiers = [
    event.ctrlKey ? "Ctrl" : "",
    event.altKey ? "Alt" : "",
    event.shiftKey ? "Shift" : "",
    event.metaKey ? "Super" : "",
  ].filter(Boolean);
  return [...modifiers, key].join("+");
}

function beginHotkeyCapture(inputId: string): void {
  stopHotkeyCapture?.();
  showNotice("Press the key combination you want to use.");
  const capture = (event: KeyboardEvent) => {
    const shortcut = shortcutFromKeyboardEvent(event);
    event.preventDefault();
    event.stopPropagation();
    if (!shortcut) {
      showNotice("Press a key together with any modifiers you want to record.");
      return;
    }
    const input = document.querySelector<HTMLInputElement>(`#${inputId}`);
    if (input) input.value = shortcut;
    stopHotkeyCapture?.();
    showNotice(`Recorded ${shortcut}. Select Apply shortcut to save it.`);
  };
  stopHotkeyCapture = () => {
    window.removeEventListener("keydown", capture, true);
    stopHotkeyCapture = undefined;
  };
  window.addEventListener("keydown", capture, true);
}

async function addDictionaryEntry(): Promise<void> {
  const spoken = window.prompt("Spoken phrase or aliases (comma-separated)");
  const triggers = spoken?.split(",").map((trigger) => trigger.trim()).filter(Boolean) ?? [];
  if (!triggers.length) return;
  const replacement = window.prompt("Replace with");
  if (!replacement?.trim()) return;
  const createdAt = new Date().toISOString();
  const replacementText = replacement.trim();
  const knownTriggers = new Set(database.dictionary.map((entry) => entry.spoken.toLocaleLowerCase()));
  triggers.forEach((trigger, index) => {
    if (knownTriggers.has(trigger.toLocaleLowerCase())) return;
    database.dictionary.push({ id: `dictionary-${Date.now()}-${index}`, spoken: trigger, replacement: replacementText, createdAt });
  });
  database.dictionary = await invoke<DictionaryEntry[]>("save_dictionary", { dictionary: database.dictionary });
  render();
}

async function addCustomWord(): Promise<void> {
  const text = window.prompt("Recognition vocabulary term");
  if (!text?.trim()) return;
  const aliases = window.prompt("Optional aliases, separated by commas", "") ?? "";
  database.customWords.push({
    text: text.trim(),
    aliases: aliases.split(",").map((alias) => alias.trim()).filter(Boolean),
  });
  database.customWords = await invoke<CustomWord[]>("save_custom_words", { words: database.customWords });
  render();
}

async function exportDictionary(): Promise<void> {
  const destination = await save({ title: "Export Voxide dictionary", defaultPath: "voxide-dictionary.json", filters: [{ name: "Voxide dictionary", extensions: ["json"] }] });
  if (!destination) return;
  try {
    await invoke("export_dictionary", { destination });
    showNotice("Dictionary exported.");
  } catch (error) {
    showNotice(`Could not export dictionary: ${String(error)}`);
  }
}

async function importDictionary(): Promise<void> {
  const source = await open({ title: "Import Voxide dictionary", multiple: false, filters: [{ name: "Voxide dictionary", extensions: ["json"] }] });
  if (!source || Array.isArray(source)) return;
  if (!window.confirm("Replace the current dictionary with this import?")) return;
  try {
    const imported = await invoke<DictionaryImportResult>("import_dictionary", { source });
    database.dictionary = imported.dictionary;
    database.customWords = imported.customWords;
    render();
    showNotice("Dictionary imported.");
  } catch (error) {
    showNotice(`Could not import dictionary: ${String(error)}`);
  }
}

async function addPromptProfile(): Promise<void> {
  const name = window.prompt("Profile name");
  if (!name?.trim()) return;
  const mode = window.prompt("Mode: dictate, rewrite, or command", "dictate")?.trim().toLowerCase();
  if (mode !== "dictate" && mode !== "rewrite" && mode !== "command") { showNotice("Prompt profiles must target dictate, rewrite, or command mode."); return; }
  const prompt = window.prompt("Prompt", mode === "rewrite" ? "Rewrite the requested text faithfully and output only the final rewritten text." : mode === "command" ? "Plan a single safe, explicit desktop action. Never execute it yourself." : "Clean up spoken text while preserving the speaker's meaning.");
  if (!prompt?.trim()) return;
  database.promptProfiles.push({ id: `profile-${Date.now()}`, name: name.trim(), prompt: prompt.trim(), mode });
  database.promptProfiles = await invoke<PromptProfile[]>("save_prompt_profiles", { profiles: database.promptProfiles });
  render();
}

async function addAppPromptBinding(): Promise<void> {
  const application = window.prompt("Application name (exactly as shown in History)");
  if (!application?.trim()) return;
  const mode = window.prompt("Mode: dictate, rewrite, or command", "dictate")?.trim().toLowerCase();
  if (mode !== "dictate" && mode !== "rewrite" && mode !== "command") { showNotice("Choose dictate, rewrite, or command."); return; }
  const candidates = database.promptProfiles.filter((profile) => profile.mode === mode);
  const defaultProfile = activePromptProfileId(mode) ?? candidates[0]?.id;
  const profileId = window.prompt(`Prompt profile ID (${candidates.map((profile) => `${profile.id}: ${profile.name}`).join(", ")})`, defaultProfile);
  if (!profileId?.trim() || !candidates.some((profile) => profile.id === profileId.trim())) { showNotice("Choose a prompt profile ID for that mode."); return; }
  const binding: AppPromptBinding = { id: `app-prompt-${Date.now()}`, application: application.trim(), mode, promptProfileId: profileId.trim() };
  try {
    database.appPromptBindings = await invoke<AppPromptBinding[]>("save_app_prompt_bindings", { bindings: [...database.appPromptBindings, binding] });
    render();
    showNotice("Application prompt override saved.");
  } catch (error) {
    showNotice(`Could not save application prompt override: ${String(error)}`);
  }
}

async function deleteAppPromptBinding(id: string): Promise<void> {
  const bindings = database.appPromptBindings.filter((binding) => binding.id !== id);
  try {
    database.appPromptBindings = await invoke<AppPromptBinding[]>("save_app_prompt_bindings", { bindings });
    render();
  } catch (error) {
    showNotice(`Could not remove application prompt override: ${String(error)}`);
  }
}

async function editPromptProfile(id: string): Promise<void> {
  const profile = database.promptProfiles.find((candidate) => candidate.id === id);
  if (!profile) return;
  const name = window.prompt("Profile name", profile.name);
  if (!name?.trim()) return;
  const prompt = window.prompt("Prompt", profile.prompt);
  if (!prompt?.trim()) return;
  database.promptProfiles = database.promptProfiles.map((candidate) => candidate.id === id ? { ...candidate, name: name.trim(), prompt: prompt.trim() } : candidate);
  database.promptProfiles = await invoke<PromptProfile[]>("save_prompt_profiles", { profiles: database.promptProfiles });
  database.appPromptBindings = await invoke<AppPromptBinding[]>("app_prompt_bindings");
  render();
}

async function deletePromptProfile(id: string): Promise<void> {
  if (database.promptProfiles.length === 1) { showNotice("Keep at least one prompt profile."); return; }
  database.promptProfiles = database.promptProfiles.filter((profile) => profile.id !== id);
  if (promptTestState.profileId === id) resetPromptTestState();
  database.promptProfiles = await invoke<PromptProfile[]>("save_prompt_profiles", { profiles: database.promptProfiles });
  database.appPromptBindings = await invoke<AppPromptBinding[]>("app_prompt_bindings");
  render();
}

function activePromptProfileId(mode: DictationMode): string | undefined {
  switch (mode) {
    case "dictate": return database.settings.selectedDictationPromptProfile;
    case "rewrite": return database.settings.selectedRewritePromptProfile;
    case "command": return database.settings.selectedCommandPromptProfile;
    default: return undefined;
  }
}

function promptForMode(mode: "rewrite" | "command", fallback: string, sourceApplication?: string): string {
  const applicationProfileId = sourceApplication && database.appPromptBindings.find((binding) =>
    binding.mode === mode && binding.application.toLowerCase() === sourceApplication.toLowerCase(),
  )?.promptProfileId;
  const scope = database.settings.editPromptRoutingScope;
  if (!applicationProfileId && scope === "selectedAppsOnly") return fallback;
  const activeProfileId = applicationProfileId || activePromptProfileId(mode);
  return database.promptProfiles.find((profile) => profile.mode === mode && profile.id === activeProfileId)?.prompt
    || database.promptProfiles.find((profile) => profile.mode === mode)?.prompt
    || fallback;
}

async function captureSelectionForRewrite(): Promise<void> {
  try {
    const selection = await invoke<CapturedSelection>("capture_selected_text");
    rewriteState.selectedText = selection.text;
    rewriteState.sourceApplication = selection.sourceApplication;
    rewriteState.outputText = "";
    rewriteState.draft = "";
    rewriteState.conversation = [];
    await currentWindow.show();
    await currentWindow.setFocus();
    render();
    showNotice("Selected text captured. Describe the rewrite you want.");
  } catch (error) {
    await currentWindow.show();
    await currentWindow.setFocus();
    showNotice(`Could not capture selected text: ${String(error)}`);
  }
}

function rewriteConversationRequest(instruction: string, selectedText: string): { request: string; messages: RewriteMessage[] } {
  const history = rewriteState.conversation.slice(-10);
  const isFirstTurn = history.length === 0;
  const turn = selectedText
    ? isFirstTurn
      ? `User's instruction: ${instruction}\n\nSelected context:\n${selectedText}\n\nApply the instruction to the selected context. Output only the rewritten text.`
      : `Follow-up instruction: ${instruction}\n\nApply this to the previous result. Output only the updated text.`
    : isFirstTurn
      ? instruction
      : `Follow-up instruction: ${instruction}\n\nApply this to the previous result. Output only the updated text.`;
  const messages = [...history, { role: "user" as const, content: turn }];
  const request = messages.map((message) => `${message.role === "user" ? "User" : "Assistant"}: ${message.content}`).join("\n\n");
  return { request, messages };
}

async function runRewrite(): Promise<void> {
  const instruction = readInput("mode-input")?.value.trim();
  if (!instruction) { showNotice("Enter an instruction for the rewrite."); return; }
  rewriteState.draft = instruction;
  rewriteState.processing = true;
  rewriteState.outputText = "";
  render();
  const selectedText = rewriteState.selectedText.trim();
  const fallbackPrompt = selectedText
    ? "Rewrite the selected text exactly as instructed. Preserve facts unless the instruction asks to change them. Output only the rewritten text, without commentary, headings, or quotation marks."
    : "Write the requested text. Output only the requested text, without commentary, headings, or quotation marks.";
  const systemPrompt = promptForMode("rewrite", fallbackPrompt, rewriteState.sourceApplication);
  const { request, messages } = rewriteConversationRequest(instruction, selectedText);
  try {
    rewriteState.outputText = await invoke<string>("enhance_text", { text: request, systemPrompt, providerId: database.settings.selectedRewriteAiProvider ?? database.settings.selectedAiProvider });
    const completedConversation: RewriteMessage[] = [
      ...messages,
      { role: "assistant", content: rewriteState.outputText },
    ];
    rewriteState.conversation = completedConversation.slice(-12);
    rewriteState.draft = "";
  } catch (error) {
    showNotice(`Rewrite failed: ${String(error)}`);
  } finally {
    rewriteState.processing = false;
    render();
  }
}

async function insertRewrite(): Promise<void> {
  if (!rewriteState.outputText.trim()) return;
  try {
    await invoke("replace_selected_text", { text: rewriteState.outputText });
    showNotice(rewriteState.selectedText.trim() ? "Selected text replaced." : "Text inserted into the active application.");
    rewriteState = { selectedText: "", outputText: "", processing: false, draft: "", conversation: [] };
  } catch (error) {
    showNotice(`Could not insert rewrite: ${String(error)}`);
  }
}

async function dictateModeInstruction(mode: "command" | "rewrite"): Promise<void> {
  if (recording) return;
  dictationInstructionTarget = mode;
  await startRecording(undefined, true);
  if (recording) showNotice("Speak the instruction, then stop dictation.");
}

async function planCommand(): Promise<void> {
  const request = readInput("mode-input")?.value.trim();
  if (!request) { showNotice("Describe the action or question first."); return; }
  commandState.draft = request;
  commandState.processing = true;
  commandState.streamingText = undefined;
  commandState.streamingThinking = undefined;
  commandState.plan = undefined;
  commandState.result = undefined;
  render();
  try {
    commandState.plan = await invoke<CommandPlan>("plan_command", {
      request,
      chatId: commandState.chatId,
      sourceApplication: commandState.sourceApplication,
    });
    commandState.chatId = commandState.plan.conversationId ?? commandState.chatId;
    await refreshCommandChats();
    if (shouldAutoExecuteCommand(commandState.plan)) await executeCommandPlan();
  } catch (error) {
    showNotice(`Could not plan the command: ${String(error)}`);
  } finally {
    commandState.processing = false;
    commandState.streamingText = undefined;
    commandState.streamingThinking = undefined;
    render();
  }
}

async function approveCommand(): Promise<void> {
  await executeCommandPlan();
}

function shouldAutoExecuteCommand(plan: CommandPlan | undefined): boolean {
  return plan?.kind === "command"
    && (!database.settings.commandModeConfirmBeforeExecute || !plan.destructive);
}

async function executeCommandPlan(): Promise<void> {
  if (!commandState.plan || commandState.plan.kind !== "command") return;
  commandState.processing = true;
  commandState.streamingText = undefined;
  commandState.streamingThinking = undefined;
  render();
  try {
    commandState.result = await invoke<CommandExecutionResult>("execute_approved_command", { plan: commandState.plan });
    await refreshCommandChats();
    if (commandState.plan.conversationId) {
      try {
        commandState.plan = await invoke<CommandPlan>("continue_command", {
          conversationId: commandState.plan.conversationId,
        });
        commandState.chatId = commandState.plan.conversationId ?? commandState.chatId;
        await refreshCommandChats();
        if (shouldAutoExecuteCommand(commandState.plan)) await executeCommandPlan();
      } catch (error) {
        commandState.plan = undefined;
        showNotice(`Command completed, but could not plan a follow-up: ${String(error)}`);
      }
    }
  } catch (error) {
    showNotice(`Could not run the command: ${String(error)}`);
  } finally {
    commandState.processing = false;
    commandState.streamingText = undefined;
    commandState.streamingThinking = undefined;
    render();
  }
}

async function chooseFileForTranscription(): Promise<void> {
  const selected = await open({
    title: "Choose audio or video file",
    multiple: false,
    filters: [{ name: "Audio and video", extensions: ["wav", "mp3", "m4a", "aac", "ogg", "oga", "opus", "flac", "wma", "aiff", "aif", "mp4", "m4v", "mov", "webm", "mkv", "avi"] }],
  });
  if (!selected || Array.isArray(selected)) return;
  selectFileForTranscription(selected);
}

function selectFileForTranscription(selected: string): void {
  selectedFileForTranscription = selected;
  render();
}

async function transcribeSelectedFile(selected: string): Promise<void> {
  if (fileTranscriptionActive) return;
  fileTranscriptionActive = true;
  fileProgress = undefined;
  render();
  try {
    const entry = await invoke<FileTranscriptionEntry | null>("transcribe_file", { filePath: selected });
    if (entry) {
      database.fileTranscriptionHistory.unshift(entry);
      showNotice(`${entry.fileName} was transcribed and saved locally.`);
    } else {
      showNotice("The file was processed, but no speech was recognized to save.");
    }
  } catch (error) {
    showNotice(`File transcription failed: ${String(error)}`);
  } finally {
    fileTranscriptionActive = false;
    fileProgress = undefined;
    render();
  }
}

function showNotice(message: string): void {
  const existing = document.querySelector<HTMLElement>(".notice");
  existing?.remove();
  const notice = document.createElement("div");
  notice.className = "notice";
  notice.textContent = message;
  document.body.append(notice);
  window.setTimeout(() => notice.remove(), 5000);
}

function renderOverlay(update: OverlayUpdate, audioLevel = 0): void {
  const status = update.state === "recording" ? "Listening" : update.state === "processing" ? "Processing" : "Complete";
  // Mutate the existing nodes instead of rebuilding the subtree: this runs
  // at audio-level rate while recording, and WebKitGTK never paints frames
  // for a subtree that is replaced wholesale on every update.
  let root = appRoot.querySelector<HTMLElement>(".overlay-root");
  if (!root) {
    appRoot.innerHTML = `<main class="overlay-root">
      <div class="overlay-waveform" aria-hidden="true">${"<i></i>".repeat(16)}</div>
      <div class="overlay-copy"><span></span><strong></strong></div>
    </main>`;
    root = appRoot.querySelector<HTMLElement>(".overlay-root");
    if (!root) return;
  }
  root.dataset.state = update.state;
  root.querySelectorAll<HTMLElement>(".overlay-waveform i").forEach((bar, index) => {
    const variation = 0.25 + ((index * 17) % 7) / 12;
    bar.style.height = `${Math.max(5, Math.round((audioLevel * 42 + 7) * variation))}px`;
  });
  const copyLabel = root.querySelector<HTMLElement>(".overlay-copy span");
  if (copyLabel) copyLabel.textContent = `${status} · ${update.mode}`;
  const copyText = root.querySelector<HTMLElement>(".overlay-copy strong");
  if (copyText) copyText.textContent = update.text || "Listening…";
}

let lastOverlayContentHeight = 0;

function syncOverlayHeightToContent(): void {
  const copy = document.querySelector<HTMLElement>(".overlay-copy");
  if (!copy) return;
  const rootPadding = 42;
  const waveformHeight = 48;
  const desired = Math.ceil((rootPadding + Math.max(waveformHeight, copy.scrollHeight)) * window.devicePixelRatio);
  if (Math.abs(desired - lastOverlayContentHeight) < 6) return;
  lastOverlayContentHeight = desired;
  void invoke("resize_overlay_to_content", { contentHeight: desired }).catch(() => undefined);
}

async function initializeOverlay(): Promise<void> {
  document.body.classList.add("overlay-window");
  let update: OverlayUpdate = { state: "hidden", mode: "dictate", text: "" };
  let audioLevel = 0;
  renderOverlay(update, audioLevel);
  await listen<OverlayUpdate>("overlay-update", async (event) => {
    update = event.payload;
    if (update.state === "hidden") {
      await currentWindow.hide();
      return;
    }
    renderOverlay(update, audioLevel);
    syncOverlayHeightToContent();
    await currentWindow.show();
  });
  await listen<number>("dictation-audio-level", (event) => {
    // Shape the normalized level for display: the exponent keeps small
    // noises low while speech still peaks, and the asymmetric envelope
    // rises quickly but falls smoothly so the bars read as a voice, not
    // as flicker.
    const shaped = Math.pow(Math.max(0, Math.min(1, event.payload)), 1.6);
    audioLevel += (shaped - audioLevel) * (shaped > audioLevel ? 0.5 : 0.22);
    if (update.state === "recording") renderOverlay(update, audioLevel);
  });
}

function bindCommonEvents(): void {
  document.querySelectorAll<HTMLElement>("[data-nav]").forEach((element) => element.addEventListener("click", () => {
    const nextView = element.dataset.nav ?? "welcome";
    if (promptTestState.active && nextView !== "enhancement") {
      if (recording) {
        showNotice("Stop the prompt test recording before leaving AI Enhancement.");
        return;
      }
      resetPromptTestState();
    }
    currentView = nextView;
    if (currentView === "dictionary") {
      void refreshDictionaryLearningSuggestions().catch(() => { dictionaryLearningSuggestions = []; }).then(render);
    } else {
      render();
    }
  }));
  document.querySelectorAll<HTMLElement>("[data-action]").forEach((element) => element.addEventListener("click", () => { void handleAction(element); }));
  document.querySelectorAll<HTMLElement>("[data-capture-hotkey]").forEach((element) => element.addEventListener("click", () => {
    const inputId = element.dataset.captureHotkey;
    if (inputId) beginHotkeyCapture(inputId);
  }));
  document.querySelectorAll<HTMLInputElement>("[data-setting-toggle]").forEach((element) => element.addEventListener("change", () => {
    const key = element.dataset.settingToggle as keyof Settings;
    database.settings[key] = element.checked as never;
    if (key === "automaticDictionaryLearningEnabled" && !element.checked) dictionaryLearningSuggestions = [];
    if (key === "commandModeConfirmBeforeExecute") {
      void saveSettings().then(render);
    } else {
      void saveSettings();
    }
  }));
  document.querySelectorAll<HTMLElement>("[data-stats-days]").forEach((element) => element.addEventListener("click", () => {
    const days = Number(element.dataset.statsDays);
    if (days === 7 || days === 30) {
      statsActivityDays = days;
      render();
    }
  }));
  document.querySelectorAll<HTMLElement>("[data-engine]").forEach((element) => element.addEventListener("click", () => {
    database.settings.selectedVoiceEngine = element.dataset.engine as Settings["selectedVoiceEngine"];
    void saveSettings().then(async () => { await refreshModelStatus(); render(); });
  }));
  document.querySelectorAll<HTMLElement>("[data-delete-entry]").forEach((element) => element.addEventListener("click", () => {
    void deleteHistoryEntry(element.dataset.deleteEntry ?? "");
  }));
  document.querySelectorAll<HTMLElement>("[data-edit-entry]").forEach((element) => element.addEventListener("click", () => {
    void editHistoryEntry(element.dataset.editEntry ?? "");
  }));
  document.querySelectorAll<HTMLElement>("[data-copy-file]").forEach((element) => element.addEventListener("click", () => {
    const entry = database.fileTranscriptionHistory.find((candidate) => candidate.id === element.dataset.copyFile);
    if (entry) void copyTextToClipboard(entry.text, "File transcription");
  }));
  document.querySelectorAll<HTMLElement>("[data-delete-file]").forEach((element) => element.addEventListener("click", () => {
    void deleteFileHistoryEntry(element.dataset.deleteFile ?? "");
  }));
  document.querySelectorAll<HTMLElement>("[data-export-file]").forEach((element) => element.addEventListener("click", () => {
    void exportFileHistoryEntry(element.dataset.exportFile ?? "");
  }));
  document.querySelectorAll<HTMLElement>("[data-copy-entry]").forEach((element) => element.addEventListener("click", () => {
    const entry = database.dictationHistory.find((candidate) => candidate.id === element.dataset.copyEntry);
    if (entry) void copyTextToClipboard(entry.text, "Final transcription");
  }));
  document.querySelectorAll<HTMLElement>("[data-copy-raw-entry]").forEach((element) => element.addEventListener("click", () => {
    const entry = database.dictationHistory.find((candidate) => candidate.id === element.dataset.copyRawEntry);
    if (entry) void copyTextToClipboard(entry.rawText ?? entry.text, "Original transcription");
  }));
  document.querySelectorAll<HTMLElement>("[data-copy-both-entry]").forEach((element) => element.addEventListener("click", () => {
    const entry = database.dictationHistory.find((candidate) => candidate.id === element.dataset.copyBothEntry);
    if (entry) void copyTextToClipboard(`${entry.rawText ?? entry.text}\n\n${entry.text}`, "Original and final transcription");
  }));
  document.querySelectorAll<HTMLElement>("[data-export-entry]").forEach((element) => element.addEventListener("click", () => {
    void exportDictationEntry(element.dataset.exportEntry ?? "");
  }));
  document.querySelectorAll<HTMLElement>("[data-export-audio-entry]").forEach((element) => element.addEventListener("click", () => {
    void exportDictationAudioPair(element.dataset.exportAudioEntry ?? "");
  }));
  document.querySelectorAll<HTMLElement>("[data-delete-dictionary]").forEach((element) => element.addEventListener("click", () => {
    const id = element.dataset.deleteDictionary;
    database.dictionary = database.dictionary.filter((entry) => entry.id !== id);
    void invoke<DictionaryEntry[]>("save_dictionary", { dictionary: database.dictionary }).then(render);
  }));
  document.querySelectorAll<HTMLElement>("[data-delete-custom-word]").forEach((element) => element.addEventListener("click", () => {
    const index = Number(element.dataset.deleteCustomWord);
    if (!Number.isInteger(index)) return;
    database.customWords = database.customWords.filter((_, candidate) => candidate !== index);
    void invoke<CustomWord[]>("save_custom_words", { words: database.customWords }).then((words) => {
      database.customWords = words;
      render();
    });
  }));
  document.querySelectorAll<HTMLElement>("[data-delete-app-prompt-binding]").forEach((element) => element.addEventListener("click", () => {
    void deleteAppPromptBinding(element.dataset.deleteAppPromptBinding ?? "");
  }));
  document.querySelectorAll<HTMLElement>("[data-accept-dictionary-learning]").forEach((element) => element.addEventListener("click", () => {
    void acceptDictionaryLearningSuggestion(element.dataset.acceptDictionaryLearning ?? "", element.dataset.learningReplacement ?? "");
  }));
  document.querySelectorAll<HTMLElement>("[data-dismiss-dictionary-learning]").forEach((element) => element.addEventListener("click", () => {
    void dismissDictionaryLearningSuggestion(element.dataset.dismissDictionaryLearning ?? "", element.dataset.learningReplacement ?? "");
  }));
  const providerSelect = document.querySelector<HTMLSelectElement>("#ai-provider");
  providerSelect?.addEventListener("change", () => {
    database.settings.selectedAiProvider = providerSelect.value;
    editingProviderId = providerSelect.value;
    void saveSettings().then(render);
  });
  const rewriteProviderSelect = document.querySelector<HTMLSelectElement>("#rewrite-ai-provider");
  rewriteProviderSelect?.addEventListener("change", () => {
    database.settings.selectedRewriteAiProvider = rewriteProviderSelect.value || undefined;
    void saveSettings().then(render);
  });
  const commandProviderSelect = document.querySelector<HTMLSelectElement>("#command-ai-provider");
  commandProviderSelect?.addEventListener("change", () => {
    database.settings.selectedCommandAiProvider = commandProviderSelect.value || undefined;
    void saveSettings().then(render);
  });
  const providerConfigurationSelect = document.querySelector<HTMLSelectElement>("#provider-configuration");
  providerConfigurationSelect?.addEventListener("change", () => {
    editingProviderId = providerConfigurationSelect.value;
    render();
  });
  const dictationPromptRoutingScope = document.querySelector<HTMLSelectElement>("#dictation-prompt-routing-scope");
  dictationPromptRoutingScope?.addEventListener("change", () => {
    database.settings.dictationPromptRoutingScope = dictationPromptRoutingScope.value as Settings["dictationPromptRoutingScope"];
    void saveSettings().then(render);
  });
  const editPromptRoutingScope = document.querySelector<HTMLSelectElement>("#edit-prompt-routing-scope");
  editPromptRoutingScope?.addEventListener("change", () => {
    database.settings.editPromptRoutingScope = editPromptRoutingScope.value as Settings["editPromptRoutingScope"];
    void saveSettings().then(render);
  });
  const transcriptionStartSound = document.querySelector<HTMLSelectElement>("#transcription-start-sound");
  transcriptionStartSound?.addEventListener("change", () => {
    database.settings.transcriptionStartSound = transcriptionStartSound.value as TranscriptionSound;
    void saveSettings().then(async () => {
      await playTranscriptionCue("start");
      render();
    });
  });
  const transcriptionSoundVolume = document.querySelector<HTMLInputElement>("#transcription-sound-volume");
  transcriptionSoundVolume?.addEventListener("change", () => {
    database.settings.transcriptionSoundVolume = Number(transcriptionSoundVolume.value);
    void saveSettings().then(() => playTranscriptionCue("start"));
  });
  const transcriptionPreviewCharLimit = document.querySelector<HTMLInputElement>("#transcription-preview-char-limit");
  transcriptionPreviewCharLimit?.addEventListener("change", () => {
    database.settings.transcriptionPreviewCharLimit = Number(transcriptionPreviewCharLimit.value);
    void saveSettings().then(render);
  });
  const textInsertionMode = document.querySelector<HTMLSelectElement>("#text-insertion-mode");
  textInsertionMode?.addEventListener("change", () => {
    database.settings.textInsertionMode = textInsertionMode.value as TextInsertionMode;
    void saveSettings().then(render);
  });
  const overlayPosition = document.querySelector<HTMLSelectElement>("#overlay-position");
  overlayPosition?.addEventListener("change", () => {
    database.settings.overlayPosition = overlayPosition.value as Settings["overlayPosition"];
    void saveSettings().then(render);
  });
  const overlaySize = document.querySelector<HTMLSelectElement>("#overlay-size");
  overlaySize?.addEventListener("change", () => {
    database.settings.overlaySize = overlaySize.value as Settings["overlaySize"];
    void saveSettings().then(render);
  });
  const overlayBottomOffset = document.querySelector<HTMLInputElement>("#overlay-bottom-offset");
  overlayBottomOffset?.addEventListener("change", () => {
    database.settings.overlayBottomOffset = Number(overlayBottomOffset.value);
    void saveSettings().then(render);
  });
  const promptTestDraft = document.querySelector<HTMLTextAreaElement>("#prompt-test-draft");
  promptTestDraft?.addEventListener("input", () => {
    promptTestState.draftPrompt = promptTestDraft.value;
  });
  const historySearch = document.querySelector<HTMLInputElement>("#history-search");
  historySearch?.addEventListener("input", () => {
    const cursor = historySearch.selectionStart;
    historySearchQuery = historySearch.value;
    render();
    const nextSearch = document.querySelector<HTMLInputElement>("#history-search");
    nextSearch?.focus();
    if (cursor !== null) nextSearch?.setSelectionRange(cursor, cursor);
  });
}

async function deleteHistoryEntry(id: string): Promise<void> {
  await invoke("delete_dictation", { id });
  database.dictationHistory = database.dictationHistory.filter((entry) => entry.id !== id);
  await refreshStats();
  render();
}

async function editHistoryEntry(id: string): Promise<void> {
  const entry = database.dictationHistory.find((candidate) => candidate.id === id);
  if (!entry) return;
  const text = window.prompt("Final transcription", entry.text);
  if (text === null) return;
  if (!text.trim()) { showNotice("A transcription cannot be empty."); return; }
  let rawText = entry.rawText;
  if (entry.rawText !== undefined) {
    const editedRawText = window.prompt("Original transcription", entry.rawText);
    if (editedRawText === null) return;
    rawText = editedRawText;
  }
  try {
    const updated = await invoke<DictationEntry>("update_dictation", { id, text, rawText });
    database.dictationHistory = database.dictationHistory.map((candidate) => candidate.id === id ? updated : candidate);
    await refreshStats();
    await refreshDictionaryLearningSuggestions();
    render();
    showNotice(dictionaryLearningSuggestions.length ? "Dictation updated. A correction suggestion is ready in Custom Dictionary." : "Dictation updated.");
  } catch (error) {
    showNotice(`Could not update dictation: ${String(error)}`);
  }
}

async function refreshDictionaryLearningSuggestions(): Promise<void> {
  const suggestions = await invoke<DictionaryLearningSuggestion[]>("dictionary_learning_suggestions");
  if (suggestions.length || !dictionaryLearningSuggestions.length) dictionaryLearningSuggestions = suggestions;
}

async function acceptDictionaryLearningSuggestion(heardText: string, correctedText: string): Promise<void> {
  try {
    database.dictionary = await invoke<DictionaryEntry[]>("accept_dictionary_learning_suggestion", { heardText, correctedText });
    dictionaryLearningSuggestions = dictionaryLearningSuggestions.filter((suggestion) => suggestion.heardText !== heardText || suggestion.correctedText !== correctedText);
    render();
    showNotice("Local dictionary correction added.");
  } catch (error) {
    showNotice(`Could not add the correction: ${String(error)}`);
  }
}

async function dismissDictionaryLearningSuggestion(heardText: string, correctedText: string): Promise<void> {
  try {
    await invoke("dismiss_dictionary_learning_suggestion", { heardText, correctedText });
    dictionaryLearningSuggestions = dictionaryLearningSuggestions.filter((suggestion) => suggestion.heardText !== heardText || suggestion.correctedText !== correctedText);
    render();
    showNotice("Dictionary suggestion dismissed for seven days.");
  } catch (error) {
    showNotice(`Could not dismiss the correction: ${String(error)}`);
  }
}

async function openFeedbackIssue(): Promise<void> {
  try {
    await invoke("open_feedback_issue");
  } catch (error) {
    showNotice(`Could not open the issue tracker: ${String(error)}`);
  }
}

async function copyDebugInformation(): Promise<void> {
  try {
    const information = await invoke<string>("feedback_debug_information");
    await invoke("copy_text_to_clipboard", { text: information });
    showNotice("Debug information copied. Paste it into your GitHub issue.");
  } catch (error) {
    showNotice(`Could not copy debug information: ${String(error)}`);
  }
}

async function deleteFileHistoryEntry(id: string): Promise<void> {
  await invoke("delete_file_transcription", { id });
  database.fileTranscriptionHistory = database.fileTranscriptionHistory.filter((entry) => entry.id !== id);
  render();
}

async function exportFileHistoryEntry(id: string): Promise<void> {
  const entry = database.fileTranscriptionHistory.find((candidate) => candidate.id === id);
  if (!entry) return;
  const format = window.prompt("Export format: text or json", "text")?.trim().toLowerCase();
  if (format !== "text" && format !== "json") {
    if (format) showNotice("Choose either text or json for the export format.");
    return;
  }
  const destination = await save({
    title: "Export transcription",
    defaultPath: `${entry.fileName}_transcript.${format === "json" ? "json" : "txt"}`,
    filters: format === "json"
      ? [{ name: "JSON", extensions: ["json"] }]
      : [{ name: "Text", extensions: ["txt"] }],
  });
  if (!destination) return;
  try {
    await invoke("export_file_transcription", { id, destination, format });
    showNotice(`File transcription exported as ${format.toUpperCase()}.`);
  } catch (error) {
    showNotice(`Could not export transcription: ${String(error)}`);
  }
}

async function exportDictationEntry(id: string): Promise<void> {
  const entry = database.dictationHistory.find((candidate) => candidate.id === id);
  if (!entry) return;
  const destination = await save({ title: "Export dictation", defaultPath: "voxide-dictation.txt", filters: [{ name: "Text", extensions: ["txt"] }] });
  if (!destination) return;
  try {
    await invoke("export_dictation", { id, destination });
    showNotice("Dictation exported.");
  } catch (error) {
    showNotice(`Could not export dictation: ${String(error)}`);
  }
}

async function exportAudioHistory(): Promise<void> {
  const destination = await save({
    title: "Export Voxide audio history",
    defaultPath: `Voxide_Audio_${archiveTimestamp()}.zip`,
    filters: [{ name: "ZIP archive", extensions: ["zip"] }],
  });
  if (!destination) return;
  try {
    const count = await invoke<number>("export_audio_history", { destination });
    showNotice(`Exported ${count} saved recording${count === 1 ? "" : "s"} with manifest.jsonl.`);
  } catch (error) {
    showNotice(`Could not export saved audio: ${String(error)}`);
  }
}

function newProviderId(): string {
  if (typeof globalThis.crypto?.randomUUID === "function") return globalThis.crypto.randomUUID();
  // Tauri's supported webviews provide randomUUID(). Keep a collision-resistant
  // fallback for older embedded webviews rather than exposing a mutable ID field.
  return `provider-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

async function exportDictationAudioPair(id: string): Promise<void> {
  const entry = database.dictationHistory.find((candidate) => candidate.id === id);
  if (!entry?.audioFile) return;
  const destination = await save({
    title: "Export dictation audio pair",
    defaultPath: `Voxide_Pair_${archiveTimestamp(new Date(entry.createdAt))}_${id.slice(-8)}.zip`,
    filters: [{ name: "ZIP archive", extensions: ["zip"] }],
  });
  if (!destination) return;
  try {
    await invoke<number>("export_dictation_audio_pair", { id, destination });
    showNotice("Dictation audio pair exported with manifest.jsonl.");
  } catch (error) {
    showNotice(`Could not export dictation audio: ${String(error)}`);
  }
}

async function deleteSavedAudio(): Promise<void> {
  if (!window.confirm("Delete every saved WAV recording? Dictation text history will remain.")) return;
  try {
    const count = await invoke<number>("delete_saved_audio");
    database.dictationHistory = database.dictationHistory.map((entry) => ({ ...entry, audioFile: undefined, audioModel: undefined }));
    render();
    showNotice(`Deleted ${count} saved recording${count === 1 ? "" : "s"}.`);
  } catch (error) {
    showNotice(`Could not delete saved audio: ${String(error)}`);
  }
}

async function copyTextToClipboard(text: string, label = "Text"): Promise<void> {
  try {
    await invoke("copy_text_to_clipboard", { text });
    showNotice(`${label} copied to the clipboard.`);
  } catch (error) {
    showNotice(`Could not copy ${label.toLowerCase()}: ${String(error)}`);
  }
}

async function exportBackup(): Promise<void> {
  const destination = await save({ title: "Export Voxide backup", defaultPath: `Voxide_Backup_${backupTimestamp()}.json`, filters: [{ name: "Voxide backup", extensions: ["json"] }] });
  if (!destination) return;
  try {
    await invoke("export_backup", { destination });
    showNotice("Voxide backup exported. API keys were not included.");
  } catch (error) {
    showNotice(`Could not export backup: ${String(error)}`);
  }
}

async function importBackup(): Promise<void> {
  const source = await open({ title: "Restore Voxide backup", multiple: false, filters: [{ name: "Voxide backup", extensions: ["json"] }] });
  if (!source || Array.isArray(source)) return;
  if (!window.confirm("Restore this backup? This replaces current Voxide settings and local history.")) return;
  try {
    database = await invoke<AppDatabase>("import_backup", { source });
    await refreshStats();
    await refreshModelStatus();
    await refreshAudioDevices().catch(() => { audioDevices = []; });
    await refreshProviders();
    await refreshLocalApiStatus();
    render();
    showNotice("Backup restored. Re-enter provider API keys if this computer does not already store them.");
  } catch (error) {
    showNotice(`Could not restore backup: ${String(error)}`);
  }
}

async function presentAvailableUpdate(update: UpdateCheckResult, canSnooze: boolean): Promise<void> {
  if (!update.hasUpdate || !update.latestVersion || !update.releaseUrl) {
    showNotice("Voxide is up to date.");
    return;
  }
  const install = window.confirm(
    `Voxide ${update.latestVersion} is available. Open its official GitHub release page to download and install the update?`,
  );
  if (install) {
    try {
      await invoke("open_update_release", { releaseUrl: update.releaseUrl });
    } catch (error) {
      showNotice(`Could not open the update release page: ${String(error)}`);
    }
  } else if (canSnooze) {
    await invoke("snooze_update_prompt", { version: update.latestVersion });
    database.settings.updatePromptSnoozedUntil = new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString();
    database.settings.snoozedUpdateVersion = update.latestVersion;
    showNotice(`Update reminder snoozed for ${update.latestVersion} for 24 hours.`);
  }
}

async function handleAction(element: HTMLElement): Promise<void> {
  switch (element.dataset.action) {
    case "toggle-recording": await toggleRecording(true); break;
    case "onboarding-voice":
      database.settings = await invoke<Settings>("set_onboarding_step", { step: 1 });
      currentView = "voice";
      render();
      break;
    case "onboarding-settings":
      database.settings = await invoke<Settings>("set_onboarding_step", { step: 3 });
      currentView = "settings";
      render();
      break;
    case "onboarding-enhancement":
      database.settings = await invoke<Settings>("set_onboarding_step", { step: 4 });
      currentView = "enhancement";
      render();
      break;
    case "request-mic":
      await startRecording(undefined, true);
      if (recording) {
        database.settings = await invoke<Settings>("set_onboarding_step", { step: 2 });
        showNotice("Native microphone capture is active. Speak a short phrase, then stop dictation.");
      }
      break;
    case "complete-onboarding":
      try {
        database.settings = await invoke<Settings>("complete_onboarding");
        showNotice("Setup is complete. Voxide is ready for dictation.");
        render();
      } catch (error) {
        showNotice(`Setup needs a usable voice engine: ${String(error)}`);
      }
      break;
    case "reset-onboarding":
      database.settings = await invoke<Settings>("reset_onboarding");
      showNotice("Setup has been restarted.");
      render();
      break;
    case "save-engine": {
      const model = readInput("selected-model")?.value.trim();
      const language = readInput("language")?.value.trim();
      const appleSpeechLocale = readInput("apple-speech-locale")?.value.trim();
      const localModelPath = readInput("local-model-path")?.value.trim();
      const selectedInputDevice = readInput("input-device")?.value.trim();
      const cloudTranscriptionModel = readInput("cloud-transcription-model")?.value.trim();
      if (model) database.settings.selectedModel = model;
      if (language) database.settings.language = language;
      if (appleSpeechLocale) database.settings.appleSpeechLocale = appleSpeechLocale;
      database.settings.localModelPath = localModelPath || undefined;
      database.settings.selectedInputDevice = selectedInputDevice || undefined;
      if (cloudTranscriptionModel) database.settings.cloudTranscriptionModel = cloudTranscriptionModel;
      await saveSettings(); showNotice("Voice engine settings saved."); render(); break;
    }
    case "save-output-formatting": {
      const fillerWords = readInput("filler-words")?.value
        .split(",")
        .map((word) => word.trim())
        .filter(Boolean);
      const punctuationPrefix = readInput("punctuation-prefix")?.value.trim();
      const rulesInput = readInput("punctuation-rules")?.value.trim();
      if (fillerWords) database.settings.fillerWords = fillerWords;
      if (punctuationPrefix) database.settings.punctuationDictionaryPrefix = punctuationPrefix;
      if (rulesInput) {
        try {
          const rules = JSON.parse(rulesInput) as PunctuationRule[];
          if (!Array.isArray(rules) || rules.some((rule) => !rule || typeof rule.symbol !== "string" || !Array.isArray(rule.aliases) || rule.aliases.some((alias) => typeof alias !== "string"))) {
            throw new Error("Each rule must contain a string symbol and an aliases string array.");
          }
          database.settings.punctuationDictionaryRules = rules;
        } catch (error) {
          showNotice(`Formatting rules are not valid JSON: ${String(error)}`);
          break;
        }
      }
      await saveSettings(); showNotice("Dictation formatting saved."); render(); break;
    }
    case "download-model": {
      const model = readInput("selected-model")?.value.trim();
      if (!model) { showNotice("Select a Whisper model first."); break; }
      showNotice(`Downloading ${model}. This may take a few minutes.`);
      modelDownloadProgress = { id: model, downloadedBytes: 0 };
      render();
      try {
        modelStatus = await invoke<VoiceModelStatus>("download_whisper_model", { modelId: model });
        database.settings.selectedVoiceEngine = "whisper";
        database.settings.selectedModel = model;
        database.settings.localModelPath = undefined;
        showNotice(`${model} is ready for local dictation.`);
      } catch (error) {
        showNotice(`Could not download ${model}: ${String(error)}`);
      } finally {
        modelDownloadProgress = undefined;
        render();
      }
      break;
    }
    case "delete-model": {
      const model = database.settings.selectedModel;
      if (!window.confirm(`Remove the downloaded ${model} Whisper model? You can download it again later.`)) break;
      try {
        modelStatus = await invoke<VoiceModelStatus>("delete_whisper_model", { modelId: model });
        showNotice(`${model} was removed.`);
        render();
      } catch (error) {
        showNotice(`Could not remove ${model}: ${String(error)}`);
      }
      break;
    }
    case "add-prompt-shortcut": {
      const profile = database.promptProfiles.find((candidate) => candidate.mode === "dictate");
      if (!profile) { showNotice("Create a Dictate prompt profile first."); break; }
      database.settings.promptShortcutAssignments.push({ promptProfileId: profile.id, hotkey: "" });
      render();
      break;
    }
    case "delete-prompt-shortcut": {
      const index = Number(element.dataset.promptShortcutIndex);
      if (!Number.isInteger(index)) break;
      database.settings.promptShortcutAssignments = database.settings.promptShortcutAssignments.filter((_, candidate) => candidate !== index);
      render();
      break;
    }
    case "save-hotkey": {
      const hotkey = readInput("primary-hotkey")?.value.trim();
      const activation = readInput("activation-mode")?.value as Settings["hotkeyActivationMode"];
      const promptModeSelectedPromptId = readInput("prompt-mode-profile")?.value || undefined;
      if (!hotkey) { showNotice("A global shortcut is required."); break; }
      const promptShortcutAssignments = database.settings.promptShortcutAssignments
        .map((assignment, index) => ({
          promptProfileId: readInput(`prompt-shortcut-profile-${index}`)?.value || assignment.promptProfileId,
          hotkey: optionalHotkey(`prompt-shortcut-hotkey-${index}`) || "",
        }))
        .filter((assignment) => assignment.hotkey);
      const configuration: HotkeyConfiguration = {
        primaryDictationHotkey: hotkey,
        secondaryDictationHotkey: optionalHotkey("secondary-hotkey"),
        promptModeHotkey: optionalHotkey("prompt-hotkey"),
        promptModeSelectedPromptId,
        promptShortcutAssignments,
        commandModeHotkey: optionalHotkey("command-hotkey"),
        rewriteModeHotkey: optionalHotkey("rewrite-hotkey"),
        cancelRecordingHotkey: optionalHotkey("cancel-hotkey"),
        pasteLastTranscriptionHotkey: optionalHotkey("paste-last-hotkey"),
        hotkeyActivationMode: activation,
      };
      database.settings = await invoke<Settings>("configure_hotkeys", { configuration });
      const theme = readInput("theme")?.value as Theme;
      if (theme) database.settings.theme = theme;
      await saveSettings(); showNotice("Global dictation shortcuts applied."); render(); break;
    }
    case "save-local-api": {
      const enabled = (readInput("local-api-enabled") as HTMLInputElement | null)?.checked ?? false;
      const port = Number(readInput("local-api-port")?.value);
      if (!Number.isInteger(port) || port < 1 || port > 65535) { showNotice("Choose a local API port between 1 and 65535."); break; }
      apiStatus = await invoke<LocalApiStatus>("configure_local_api", { configuration: { enabled, port } });
      database.settings.localApiEnabled = enabled;
      database.settings.localApiPort = port;
      showNotice(enabled ? `Local API listening at ${apiStatus.url}.` : "Local API stopped.");
      render();
      break;
    }
    case "open-accessibility-settings": {
      try {
        await invoke("open_accessibility_settings");
        showNotice("Input accessibility settings opened. Return here and refresh after granting access.");
      } catch (error) {
        showNotice(`Could not open input settings: ${String(error)}`);
      }
      break;
    }
    case "refresh-accessibility-status": {
      await refreshAccessibilityPermissionStatus();
      render();
      break;
    }
    case "save-audio-history": {
      const budget = Number(readInput("audio-history-budget")?.value);
      if (!Number.isFinite(budget) || budget <= 0) { showNotice("Choose a positive audio history budget."); break; }
      database.settings.audioHistoryBudgetGb = Math.max(0.1, budget);
      await saveSettings();
      showNotice("Audio history budget saved. Older recordings were pruned if needed.");
      break;
    }
    case "open-feedback-issue": await openFeedbackIssue(); break;
    case "copy-debug-information": await copyDebugInformation(); break;
    case "save-stats-preferences": {
      const typingWpm = Number(readInput("user-typing-wpm")?.value);
      if (!Number.isInteger(typingWpm) || typingWpm < 1 || typingWpm > 200) { showNotice("Choose a typing speed between 1 and 200 WPM."); break; }
      database.settings.userTypingWpm = typingWpm;
      await saveSettings();
      await refreshStats();
      showNotice("Stats preferences saved.");
      render();
      break;
    }
    case "export-audio-history": await exportAudioHistory(); break;
    case "delete-saved-audio": await deleteSavedAudio(); break;
    case "check-for-updates": {
      try {
        const update = await invoke<UpdateCheckResult>("check_for_updates");
        database.settings.lastUpdateCheckAt = new Date().toISOString();
        await presentAvailableUpdate(update, false);
        if (currentView === "settings") render();
      } catch (error) {
        showNotice(`Update check failed: ${String(error)}`);
      }
      break;
    }
    case "view-release-notes": {
      try {
        recentReleaseNotes = await invoke<ReleaseNote[]>("recent_release_notes");
        render();
      } catch (error) {
        showNotice(`Could not load release notes: ${String(error)}`);
      }
      break;
    }
    case "open-release-note": {
      const releaseUrl = element.dataset.releaseUrl;
      if (!releaseUrl) break;
      try {
        await invoke("open_update_release", { releaseUrl });
      } catch (error) {
        showNotice(`Could not open the release page: ${String(error)}`);
      }
      break;
    }
    case "export-backup": await exportBackup(); break;
    case "import-backup": await importBackup(); break;
    case "new-dictionary-entry": await addDictionaryEntry(); break;
    case "export-dictionary": await exportDictionary(); break;
    case "import-dictionary": await importDictionary(); break;
    case "new-custom-word": await addCustomWord(); break;
    case "new-prompt": await addPromptProfile(); break;
    case "new-app-prompt-binding": await addAppPromptBinding(); break;
    case "start-prompt-test": {
      const profile = database.promptProfiles.find((candidate) => candidate.id === element.dataset.promptId && candidate.mode === "dictate");
      if (!profile) { showNotice("Prompt test mode is available only for Dictate prompt profiles."); break; }
      if (!canRunDictationPromptTest(profile.id)) { showNotice("Configure this Dictate profile's provider, model, and API key before testing prompts."); break; }
      promptTestState = { active: true, profileId: profile.id, draftPrompt: profile.prompt, processing: false, rawText: "", outputText: "" };
      render();
      break;
    }
    case "stop-prompt-test":
      if (!recording) {
        resetPromptTestState();
        render();
      }
      break;
    case "save-dictation-prompt-provider": {
      const profileId = element.dataset.promptId;
      if (!profileId) break;
      const encodedId = encodeURIComponent(profileId);
      const providerId = readInput(`dictation-prompt-provider-${encodedId}`)?.value || undefined;
      const model = readInput(`dictation-prompt-model-${encodedId}`)?.value.trim() || undefined;
      const next = database.dictationPromptConfigurations.filter((configuration) => configuration.promptProfileId !== profileId);
      if (providerId || model) next.push({ promptProfileId: profileId, providerId, model });
      try {
        database.dictationPromptConfigurations = await invoke<DictationPromptConfiguration[]>("save_dictation_prompt_configurations", { configurations: next });
        render();
        showNotice("Dictate profile provider route saved.");
      } catch (error) {
        showNotice(`Could not save Dictate profile provider route: ${String(error)}`);
      }
      break;
    }
    case "edit-prompt": await editPromptProfile(element.dataset.promptId ?? ""); break;
    case "delete-prompt": await deletePromptProfile(element.dataset.promptId ?? ""); break;
    case "activate-prompt": {
      database.settings = await invoke<Settings>("set_active_prompt_profile", { profileId: element.dataset.promptId ?? "" });
      render();
      showNotice("Prompt profile activated for its mode.");
      break;
    }
    case "new-provider": {
      const name = window.prompt("Provider name");
      if (!name?.trim()) break;
      const styleInput = window.prompt("API style (openai or anthropic)", "openai")?.trim().toLowerCase();
      if (!styleInput) break;
      if (styleInput !== "openai" && styleInput !== "openai-compatible" && styleInput !== "anthropic") {
        showNotice("API style must be openai or anthropic.");
        break;
      }
      const apiStyle: AiProviderProfile["apiStyle"] = styleInput === "anthropic" ? "anthropic" : "openAiCompatible";
      const baseUrl = window.prompt(apiStyle === "anthropic" ? "Anthropic Messages API base URL" : "OpenAI-compatible base URL", "https://");
      if (!baseUrl?.trim()) break;
      const id = newProviderId();
      providers.push({ profile: { id, name: name.trim(), apiStyle, baseUrl: baseUrl.trim(), model: "", enabled: true }, hasApiKey: false });
      await invoke<AiProviderProfile[]>("save_ai_providers", { providers: providers.map(({ profile }) => profile) });
      database.settings.selectedAiProvider = id;
      editingProviderId = id;
      await saveSettings();
      await refreshProviders();
      render();
      break;
    }
    case "save-provider": {
      const selectedId = editingProviderId ?? database.settings.selectedAiProvider;
      const current = providers.find(({ profile }) => profile.id === selectedId)?.profile;
      if (!current) { showNotice("Choose a provider first."); break; }
      const name = readInput("provider-name")?.value.trim();
      const apiStyle = readInput("provider-api-style")?.value as AiProviderProfile["apiStyle"];
      const baseUrl = readInput("provider-base-url")?.value.trim();
      const model = readInput("provider-model")?.value.trim();
      const apiKey = readInput("provider-api-key")?.value.trim();
      const enabled = (readInput("provider-enabled") as HTMLInputElement | null)?.checked ?? true;
      if (!name || !baseUrl) { showNotice("Provider name and base URL are required."); break; }
      if (apiStyle !== "openAiCompatible" && apiStyle !== "anthropic") { showNotice("Choose a valid provider API style."); break; }
      const edited: AiProviderProfile = { ...current, name, apiStyle, baseUrl, model: model ?? "", enabled };
      providers = providers.map((provider) => provider.profile.id === current.id ? { ...provider, profile: edited } : provider);
      if (!enabled && database.settings.selectedAiProvider === current.id) {
        const fallback = providers.find(({ profile }) => profile.enabled)?.profile.id;
        if (!fallback) { showNotice("Keep at least one AI provider enabled."); break; }
        database.settings.selectedAiProvider = fallback;
      }
      if (!enabled && database.settings.selectedRewriteAiProvider === current.id) database.settings.selectedRewriteAiProvider = undefined;
      if (!enabled && database.settings.selectedCommandAiProvider === current.id) database.settings.selectedCommandAiProvider = undefined;
      await invoke<AiProviderProfile[]>("save_ai_providers", { providers: providers.map(({ profile }) => profile) });
      if (apiKey) await invoke("set_provider_api_key", { providerId: current.id, apiKey });
      editingProviderId = current.id;
      await saveSettings();
      await refreshProviders();
      showNotice("AI provider saved.");
      render();
      break;
    }
    case "clear-provider-api-key": {
      const providerId = editingProviderId ?? database.settings.selectedAiProvider;
      const profile = providers.find((candidate) => candidate.profile.id === providerId)?.profile;
      if (!profile || !window.confirm(`Remove the stored API key for “${profile.name}”?`)) break;
      await invoke("set_provider_api_key", { providerId, apiKey: "" });
      await refreshProviders();
      showNotice("Stored API key removed.");
      render();
      break;
    }
    case "open-provider-website": {
      const providerId = editingProviderId ?? database.settings.selectedAiProvider;
      try {
        await invoke("open_provider_website", { providerId });
      } catch (error) {
        showNotice(`Could not open provider setup: ${String(error)}`);
      }
      break;
    }
    case "delete-provider": {
      const providerId = editingProviderId ?? database.settings.selectedAiProvider;
      if (builtInProviderIds.has(providerId)) { showNotice("Built-in providers cannot be removed."); break; }
      const profile = providers.find((candidate) => candidate.profile.id === providerId)?.profile;
      if (!profile || !window.confirm(`Delete the custom provider “${profile.name}”? Its stored API key will also be removed.`)) break;
      providers = providers.filter((candidate) => candidate.profile.id !== providerId);
      await invoke<AiProviderProfile[]>("save_ai_providers", { providers: providers.map(({ profile }) => profile) });
      await invoke("set_provider_api_key", { providerId, apiKey: "" });
      const fallback = providers.find(({ profile }) => profile.enabled)?.profile.id;
      if (!fallback) { showNotice("Voxide must retain an AI provider."); break; }
      if (database.settings.selectedAiProvider === providerId) database.settings.selectedAiProvider = fallback;
      if (database.settings.selectedRewriteAiProvider === providerId) database.settings.selectedRewriteAiProvider = undefined;
      if (database.settings.selectedCommandAiProvider === providerId) database.settings.selectedCommandAiProvider = undefined;
      editingProviderId = database.settings.selectedAiProvider;
      await saveSettings();
      await refreshProviders();
      showNotice("Custom provider deleted.");
      render();
      break;
    }
    case "save-reasoning-config": {
      const profile = providers.find(({ profile }) => profile.id === (editingProviderId ?? database.settings.selectedAiProvider))?.profile;
      const parameterName = readInput("reasoning-parameter-name")?.value.trim();
      const parameterValue = readInput("reasoning-parameter-value")?.value.trim();
      const enabled = (readInput("reasoning-parameter-enabled") as HTMLInputElement | null)?.checked ?? false;
      if (!profile || !profile.model.trim()) { showNotice("Save a provider model before configuring reasoning."); break; }
      if (!parameterName) { showNotice("Enter the request parameter name."); break; }
      database.settings.modelReasoningConfigs[`${profile.id}:${profile.model}`] = {
        parameterName,
        parameterValue: parameterValue ?? "",
        isEnabled: enabled,
      };
      await saveSettings();
      showNotice("Model reasoning configuration saved.");
      render();
      break;
    }
    case "clear-reasoning-config": {
      const profile = providers.find(({ profile }) => profile.id === (editingProviderId ?? database.settings.selectedAiProvider))?.profile;
      if (!profile) break;
      delete database.settings.modelReasoningConfigs[`${profile.id}:${profile.model}`];
      await saveSettings();
      showNotice("Model reasoning now uses the Voxide default.");
      render();
      break;
    }
    case "fetch-provider-models": {
      const providerId = editingProviderId ?? database.settings.selectedAiProvider;
      const models = await invoke<string[]>("fetch_ai_provider_models", { providerId });
      const selected = window.prompt(`Available models (${models.length})`, models.slice(0, 30).join("\n"));
      if (selected?.trim()) {
        const modelInput = readInput("provider-model");
        if (modelInput) modelInput.value = selected.trim();
      }
      break;
    }
    case "clear-history":
      if (window.confirm("Clear all saved dictation history?")) {
        await invoke("clear_dictation_history"); database.dictationHistory = []; await refreshStats(); render();
      }
      break;
    case "clear-file-history":
      if (window.confirm("Clear all saved file transcription history?")) {
        await invoke("clear_file_transcription_history");
        database.fileTranscriptionHistory = [];
        render();
      }
      break;
    case "capture-selection": await captureSelectionForRewrite(); break;
    case "new-rewrite":
      rewriteState.outputText = "";
      rewriteState.draft = "";
      rewriteState.conversation = [];
      render();
      showNotice("Rewrite conversation cleared. The selected text remains available for a fresh request.");
      break;
    case "dictate-mode": await dictateModeInstruction(element.dataset.mode as "command" | "rewrite"); break;
    case "run-rewrite": await runRewrite(); break;
    case "insert-rewrite": await insertRewrite(); break;
    case "copy-rewrite":
      await copyTextToClipboard(rewriteState.outputText, "Rewritten text");
      break;
    case "plan-command": await planCommand(); break;
    case "approve-command": await approveCommand(); break;
    case "cancel-command": {
      const conversationId = commandState.plan?.conversationId ?? commandState.chatId;
      await invoke("cancel_command_plan", {
        conversationId,
        toolCallId: commandState.plan?.toolCallId,
      });
      commandState.plan = undefined;
      commandState.result = undefined;
      await refreshCommandChats();
      render();
      showNotice("Planned command cancelled.");
      break;
    }
    case "new-command-chat": {
      const chat = await invoke<CommandChat>("create_command_chat");
      commandState = { draft: "", processing: false, chatId: chat.id, sourceApplication: chat.sourceApplication };
      await refreshCommandChats();
      render();
      break;
    }
    case "open-command-chat": {
      const chatId = readInput("command-chat")?.value;
      if (!chatId) { showNotice("Choose a conversation to open."); break; }
      const chat = await invoke<CommandChat>("select_command_chat", { chatId });
      commandState = { draft: "", processing: false, chatId: chat.id, sourceApplication: chat.sourceApplication };
      await refreshCommandChats();
      render();
      break;
    }
    case "clear-command-chat": {
      if (!commandState.chatId || !window.confirm("Clear this command conversation?")) break;
      const chat = await invoke<CommandChat>("clear_command_chat", { chatId: commandState.chatId });
      commandState = { draft: "", processing: false, chatId: chat.id, sourceApplication: chat.sourceApplication };
      await refreshCommandChats();
      render();
      break;
    }
    case "delete-command-chat": {
      if (!commandState.chatId || !window.confirm("Delete this command conversation?")) break;
      const chat = await invoke<CommandChat>("delete_command_chat", { chatId: commandState.chatId });
      commandState = { draft: "", processing: false, chatId: chat.id, sourceApplication: chat.sourceApplication };
      await refreshCommandChats();
      render();
      break;
    }
    case "choose-file": await chooseFileForTranscription(); break;
    case "transcribe-selected-file":
      if (selectedFileForTranscription) await transcribeSelectedFile(selectedFileForTranscription);
      break;
    default: break;
  }
}

async function initialize(): Promise<void> {
  database = await invoke<AppDatabase>("bootstrap");
  commandState.chatId = database.activeCommandChatId;
  await refreshStats();
  await refreshModelStatus();
  await refreshAudioDevices().catch(() => { audioDevices = []; });
  await refreshProviders();
  await refreshCommandChats();
  await refreshLocalApiStatus();
  await refreshAccessibilityPermissionStatus();
  await refreshHotkeyBackendStatus().catch(() => { hotkeyBackendStatus = undefined; });
  await listen<HotkeyBackendStatus>("voxide-hotkey-backend", (event) => {
    hotkeyBackendStatus = event.payload;
    if (currentView === "settings") render();
  });
  await listen<OverlayUpdate>("overlay-update", (event) => {
    if (recording && event.payload.state === "recording" && event.payload.text !== "Listening…") {
      liveText = event.payload.text;
      if (currentView === "welcome") render();
    }
  });
  await listen<HotkeyEvent>("voxide-hotkey", (event) => { void handleGlobalHotkey(event.payload); });
  await listen<TrayAction>("voxide-tray-action", (event) => { void handleTrayAction(event.payload); });
  await listen<ModelDownloadProgress>("model-download-progress", (event) => {
    modelDownloadProgress = event.payload;
    if (currentView === "voice") render();
  });
  await listen<FileTranscriptionProgress>("file-transcription-progress", (event) => {
    fileProgress = event.payload;
    if (currentView === "file") render();
  });
  await listen<CommandStreamUpdate>("command-stream", (event) => {
    if (!commandState.processing || event.payload.conversationId !== commandState.chatId) return;
    if (event.payload.text) commandState.streamingText = `${commandState.streamingText ?? ""}${event.payload.text}`;
    if (event.payload.thinking) commandState.streamingThinking = `${commandState.streamingThinking ?? ""}${event.payload.thinking}`;
    if (currentView !== "command" || commandStreamRenderScheduled) return;
    commandStreamRenderScheduled = true;
    window.requestAnimationFrame(() => {
      commandStreamRenderScheduled = false;
      if (currentView === "command" && commandState.processing) render();
    });
  });
  await currentWindow.onDragDropEvent((event) => {
    if (event.payload.type !== "drop" || currentView !== "file" || fileTranscriptionActive) return;
    const [filePath] = event.payload.paths;
    if (filePath) selectFileForTranscription(filePath);
  });
  await listen<UpdateAvailableEvent>("voxide-update-available", (event) => {
    void presentAvailableUpdate({
      hasUpdate: true,
      latestVersion: event.payload.latestVersion,
      releaseUrl: event.payload.releaseUrl,
    }, true);
  });
  window.setInterval(() => { void refreshAudioDevicesWhenChanged(); }, 5_000);
  render();
}

const startup = isOverlayWindow ? initializeOverlay() : initialize();

void startup
  .catch((error) => {
    appRoot.innerHTML = `<main class="fatal"><h1>Voxide could not start</h1><p>${escapeHtml(String(error))}</p></main>`;
  })
  .finally(() => {
    // The main window stays hidden until this first render so launch never
    // flashes an empty webview.
    if (!isOverlayWindow) void invoke("frontend_ready").catch(() => undefined);
  });
