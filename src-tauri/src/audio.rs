use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, Sample, SampleFormat, Stream, I24, U24,
};

pub const WHISPER_SAMPLE_RATE: u32 = 16_000;
pub const MINIMUM_TRANSCRIPTION_SAMPLES: usize = WHISPER_SAMPLE_RATE as usize;
pub type LevelCallback = Arc<dyn Fn(f32) + Send + Sync + 'static>;

/// File transcription in the reference app skips fragments shorter than one
/// second after conversion to 16 kHz mono. Sending those fragments to an ASR
/// backend is both unhelpful and can produce provider-specific failures.
pub fn has_minimum_transcription_samples(samples: &[f32]) -> bool {
    samples.len() >= MINIMUM_TRANSCRIPTION_SAMPLES
}

/// Live dictation and local API audio retain a genuine short utterance by
/// appending silence, matching Voxide's one-second ASR input minimum.
pub fn pad_short_transcription_samples(samples: &mut Vec<f32>) {
    if samples.len() < MINIMUM_TRANSCRIPTION_SAMPLES {
        samples.resize(MINIMUM_TRANSCRIPTION_SAMPLES, 0.0);
    }
}

pub struct CapturedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    pub duration_ms: u64,
}

/// Content-free health data for one microphone capture. These counters make a
/// full raw-ring condition explicit instead of silently losing an input block.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CaptureHealth {
    pub callback_blocks: u64,
    pub input_samples: u64,
    pub accepted_samples: u64,
    pub dropped_samples: u64,
    pub overflow_blocks: u64,
    pub ring_high_water_samples: usize,
    pub canonical_samples: u64,
    pub stream_errors: u64,
    pub discontinuities: u64,
    pub latest_capture_delay_ns: u64,
}

#[derive(Clone, Copy)]
struct RawSample {
    value: f32,
    packet_sequence: u64,
    capture_delay_ns: u64,
}

const RAW_RING_MINIMUM_SAMPLES: usize = 16_384;
const RAW_RING_BUFFER_SECONDS: usize = 2;
const CAPTURE_WORKER_IDLE_SLEEP: Duration = Duration::from_millis(2);

struct CaptureCounters {
    callback_blocks: AtomicU64,
    input_samples: AtomicU64,
    accepted_samples: AtomicU64,
    dropped_samples: AtomicU64,
    overflow_blocks: AtomicU64,
    queued_samples: AtomicUsize,
    ring_high_water_samples: AtomicUsize,
    canonical_samples: AtomicU64,
    stream_errors: AtomicU64,
    next_packet_sequence: AtomicU64,
    discontinuities: AtomicU64,
    latest_capture_delay_ns: AtomicU64,
}

impl CaptureCounters {
    fn health(&self) -> CaptureHealth {
        CaptureHealth {
            callback_blocks: self.callback_blocks.load(Ordering::Relaxed),
            input_samples: self.input_samples.load(Ordering::Relaxed),
            accepted_samples: self.accepted_samples.load(Ordering::Relaxed),
            dropped_samples: self.dropped_samples.load(Ordering::Relaxed),
            overflow_blocks: self.overflow_blocks.load(Ordering::Relaxed),
            ring_high_water_samples: self.ring_high_water_samples.load(Ordering::Relaxed),
            canonical_samples: self.canonical_samples.load(Ordering::Relaxed),
            stream_errors: self.stream_errors.load(Ordering::Relaxed),
            discontinuities: self.discontinuities.load(Ordering::Relaxed),
            latest_capture_delay_ns: self.latest_capture_delay_ns.load(Ordering::Relaxed),
        }
    }
}

impl Default for CaptureCounters {
    fn default() -> Self {
        Self {
            callback_blocks: AtomicU64::new(0),
            input_samples: AtomicU64::new(0),
            accepted_samples: AtomicU64::new(0),
            dropped_samples: AtomicU64::new(0),
            overflow_blocks: AtomicU64::new(0),
            queued_samples: AtomicUsize::new(0),
            ring_high_water_samples: AtomicUsize::new(0),
            canonical_samples: AtomicU64::new(0),
            stream_errors: AtomicU64::new(0),
            next_packet_sequence: AtomicU64::new(0),
            discontinuities: AtomicU64::new(0),
            latest_capture_delay_ns: AtomicU64::new(0),
        }
    }
}

pub struct AudioCapture {
    // The callback is the sole ring producer. It performs only sample
    // conversion, bounded writes, and atomic counter updates.
    stream: Option<Stream>,
    canonical_samples: Arc<Mutex<Vec<f32>>>,
    device_sample_rate: u32,
    device_channels: u16,
    stop_worker: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    counters: Arc<CaptureCounters>,
}

/// A microphone target resolved ahead of the latency-sensitive record path:
/// the device handle, its negotiated config, and (on Linux) the sound-server
/// routing decision. Cheap to clone, so a prewarmed instance can be cached and
/// reused across recordings. `AudioCapture::start_prepared` turns it into a
/// running capture with only a stream build + play — no device enumeration,
/// config negotiation, or `pactl` fork on the hotkey path.
#[derive(Clone)]
pub struct PreparedInput {
    device: cpal::Device,
    supported_config: cpal::SupportedStreamConfig,
    #[cfg(target_os = "linux")]
    routing: pulse::Routing,
}

impl AudioCapture {
    /// Everything the record hotkey should NOT have to do on the critical path:
    /// enumerate devices, negotiate the stream config, and (on Linux) fork
    /// `pactl` to resolve sound-server routing. The result is cheap to clone
    /// and cache, so a prewarmed instance can be reused across recordings.
    pub fn prepare(requested_device: Option<&str>) -> Result<PreparedInput, String> {
        let host = cpal::default_host();
        let requested = requested_device.filter(|name| !name.trim().is_empty());
        // Decide routing (may fork `pactl`) before touching the environment,
        // then apply it under the lock so a concurrent prewarm/record cannot
        // interleave env-var writes with the config query and stream open.
        #[cfg(target_os = "linux")]
        let routing = pulse::resolve_routing(requested);
        #[cfg(target_os = "linux")]
        let _routing_guard = pulse::routing_lock();
        #[cfg(target_os = "linux")]
        pulse::apply_routing(&routing);
        #[cfg(target_os = "linux")]
        let requested = routing.fallback_label();
        #[cfg(target_os = "linux")]
        let requested = requested.as_deref();
        let preferred_device = if let Some(requested) = requested {
            host.input_devices()
                .map_err(|error| format!("Could not enumerate microphone devices: {error}"))?
                .find(|device| device_label(device).as_deref() == Some(requested))
        } else {
            None
        };
        let device = preferred_device.or_else(|| host.default_input_device()).ok_or_else(|| {
            match requested {
                Some(requested) => format!(
                    "The selected microphone '{requested}' is unavailable and there is no system default input device."
                ),
                None => "No microphone input device is available".into(),
            }
        })?;
        let supported_config = device
            .default_input_config()
            .map_err(|error| format!("Could not read microphone configuration: {error}"))?;
        match supported_config.sample_format() {
            SampleFormat::I8
            | SampleFormat::I16
            | SampleFormat::I24
            | SampleFormat::I32
            | SampleFormat::I64
            | SampleFormat::U8
            | SampleFormat::U16
            | SampleFormat::U24
            | SampleFormat::U32
            | SampleFormat::U64
            | SampleFormat::F32
            | SampleFormat::F64 => {}
            format => {
                return Err(format!(
                    "Microphone sample format {format} is not supported by this desktop audio backend"
                ));
            }
        }
        Ok(PreparedInput {
            device,
            supported_config,
            #[cfg(target_os = "linux")]
            routing,
        })
    }

    /// Opens the microphone from an already-`prepare`d target. This is the only
    /// step that touches hardware, so it is all the record hotkey must run on
    /// the latency-sensitive path.
    pub fn start_prepared(
        prepared: &PreparedInput,
        on_level: Option<LevelCallback>,
    ) -> Result<Self, String> {
        Self::start_prepared_appending(prepared, on_level, Arc::new(Mutex::new(Vec::new())))
    }

    /// Clones the shared 16 kHz mono timeline this capture appends to, so a
    /// mid-recording rebuild can continue the same recording (see
    /// `start_prepared_appending`).
    pub fn canonical_handle(&self) -> Arc<Mutex<Vec<f32>>> {
        Arc::clone(&self.canonical_samples)
    }

    /// Opens the microphone appending into an EXISTING canonical timeline
    /// instead of a fresh one. Used to rebuild capture mid-recording after a
    /// device error without losing the audio captured so far: the caller drops
    /// the failed capture (flushing its tail into the timeline) and hands its
    /// `canonical_handle` here so the new stream continues the same recording.
    /// The rebuilt capture gets fresh health counters, so a stale error count
    /// does not immediately re-trip the recovery path.
    pub fn start_prepared_appending(
        prepared: &PreparedInput,
        on_level: Option<LevelCallback>,
        canonical_samples: Arc<Mutex<Vec<f32>>>,
    ) -> Result<Self, String> {
        // Re-assert routing right before opening: another `prepare` may have
        // run since this target was resolved, and the pipewire/pulse ALSA
        // plugins read these env vars when the stream is opened below. Held
        // under the lock through play() so the env cannot shift mid-open.
        #[cfg(target_os = "linux")]
        let _routing_guard = pulse::routing_lock();
        #[cfg(target_os = "linux")]
        pulse::apply_routing(&prepared.routing);
        let device = prepared.device.clone();
        let supported_config = prepared.supported_config.clone();
        let sample_rate = supported_config.sample_rate();
        let channels = supported_config.channels();
        let ring_capacity = (sample_rate as usize)
            .saturating_mul(channels.max(1) as usize)
            .saturating_mul(RAW_RING_BUFFER_SECONDS)
            .max(RAW_RING_MINIMUM_SAMPLES);
        let (mut producer, consumer) = rtrb::RingBuffer::<RawSample>::new(ring_capacity);
        let counters = Arc::new(CaptureCounters::default());
        let stop_worker = Arc::new(AtomicBool::new(false));
        let worker = spawn_capture_worker(
            consumer,
            Arc::clone(&canonical_samples),
            Arc::clone(&counters),
            Arc::clone(&stop_worker),
            sample_rate,
            channels,
            on_level,
        );
        let callback_counters = Arc::clone(&counters);
        let error_counters = Arc::clone(&counters);
        let config = supported_config.config();

        macro_rules! build_stream {
            ($sample:ty) => {
                device.build_input_stream(
                    &config,
                    move |data: &[$sample], info| {
                        let timestamp = info.timestamp();
                        let capture_delay_ns = timestamp
                            .callback
                            .duration_since(&timestamp.capture)
                            .map(|delay| delay.as_nanos().min(u64::MAX as u128) as u64)
                            .unwrap_or_default();
                        append_samples(data, &mut producer, &callback_counters, capture_delay_ns)
                    },
                    move |error| {
                        error_counters.stream_errors.fetch_add(1, Ordering::Relaxed);
                        eprintln!("Voxide audio capture error: {error}");
                    },
                    None,
                )
            };
        }
        let stream_result = match supported_config.sample_format() {
            SampleFormat::I8 => build_stream!(i8),
            SampleFormat::I16 => build_stream!(i16),
            SampleFormat::I24 => build_stream!(I24),
            SampleFormat::I32 => build_stream!(i32),
            SampleFormat::I64 => build_stream!(i64),
            SampleFormat::U8 => build_stream!(u8),
            SampleFormat::U16 => build_stream!(u16),
            SampleFormat::U24 => build_stream!(U24),
            SampleFormat::U32 => build_stream!(u32),
            SampleFormat::U64 => build_stream!(u64),
            SampleFormat::F32 => build_stream!(f32),
            SampleFormat::F64 => build_stream!(f64),
            _ => unreachable!("sample format was validated before capture worker startup"),
        };
        let stream = match stream_result {
            Ok(stream) => stream,
            Err(error) => {
                stop_worker.store(true, Ordering::Release);
                let _ = worker.join();
                return Err(format!("Could not start microphone capture: {error}"));
            }
        };
        if let Err(error) = stream.play() {
            drop(stream);
            stop_worker.store(true, Ordering::Release);
            let _ = worker.join();
            return Err(format!("Could not activate microphone capture: {error}"));
        }
        Ok(Self {
            stream: Some(stream),
            canonical_samples,
            device_sample_rate: sample_rate,
            device_channels: channels,
            stop_worker,
            worker: Some(worker),
            counters,
        })
    }

    pub fn finish_with_health(mut self) -> Result<(CapturedAudio, CaptureHealth), String> {
        self.stop_and_join()?;
        let samples = self.snapshot_samples()?;
        let captured = CapturedAudio {
            samples,
            sample_rate: WHISPER_SAMPLE_RATE,
            channels: 1,
            duration_ms: self.canonical_duration_ms(),
        };
        Ok((captured, self.health()))
    }

    pub fn snapshot_recent(&self, maximum_duration: Duration) -> Result<CapturedAudio, String> {
        let maximum_samples =
            (maximum_duration.as_secs_f64() * WHISPER_SAMPLE_RATE as f64).ceil() as usize;
        let samples = self
            .canonical_samples
            .lock()
            .map_err(|_| "Audio capture buffer lock was poisoned".to_string())?;
        let samples = if maximum_samples == 0 || samples.len() <= maximum_samples {
            samples.clone()
        } else {
            samples[samples.len() - maximum_samples..].to_vec()
        };
        Ok(CapturedAudio {
            duration_ms: samples.len() as u64 * 1_000 / WHISPER_SAMPLE_RATE as u64,
            samples,
            sample_rate: WHISPER_SAMPLE_RATE,
            channels: 1,
        })
    }

    /// Returns the complete capture so an incremental recognizer can keep a
    /// stable absolute sample cursor. Unlike `finish`, this does not stop the
    /// microphone stream.
    pub fn snapshot_all(&self) -> Result<CapturedAudio, String> {
        let samples = self.snapshot_samples()?;
        Ok(CapturedAudio {
            samples,
            sample_rate: WHISPER_SAMPLE_RATE,
            channels: 1,
            duration_ms: self.canonical_duration_ms(),
        })
    }

    fn snapshot_samples(&self) -> Result<Vec<f32>, String> {
        self.canonical_samples
            .lock()
            .map_err(|_| "Audio capture buffer lock was poisoned".to_string())
            .map(|samples| samples.clone())
    }

    pub fn sample_rate(&self) -> u32 {
        self.device_sample_rate
    }

    pub fn channels(&self) -> u16 {
        self.device_channels
    }

    pub fn health(&self) -> CaptureHealth {
        self.counters.health()
    }

    fn canonical_duration_ms(&self) -> u64 {
        self.counters
            .canonical_samples
            .load(Ordering::Relaxed)
            .saturating_mul(1_000)
            / WHISPER_SAMPLE_RATE as u64
    }

    fn stop_and_join(&mut self) -> Result<(), String> {
        // Drop the CPAL stream first so it cannot enqueue audio for a session
        // after its exact stop boundary. The worker then drains packets already
        // accepted into the ring before it exits.
        self.stream.take();
        self.stop_worker.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            worker
                .join()
                .map_err(|_| "Audio capture worker stopped unexpectedly".to_string())?;
        }
        Ok(())
    }
}

impl Drop for AudioCapture {
    fn drop(&mut self) {
        let _ = self.stop_and_join();
    }
}

pub fn input_device_names() -> Result<Vec<String>, String> {
    // On PipeWire/PulseAudio desktops the ALSA backend only exposes raw PCM
    // plugin names ("default", "sysdefault:CARD=…"), not the devices users
    // recognize. Prefer the sound server's source list when it is available.
    #[cfg(target_os = "linux")]
    if let Some(sources) = pulse::sources() {
        if !sources.is_empty() {
            let mut names = sources
                .into_iter()
                .map(|source| source.description)
                .collect::<Vec<_>>();
            names.sort();
            names.dedup();
            return Ok(names);
        }
    }
    let host = cpal::default_host();
    let mut names = host
        .input_devices()
        .map_err(|error| format!("Could not enumerate microphone devices: {error}"))?
        .filter_map(|device| device_label(&device))
        .collect::<Vec<_>>();
    names.sort();
    names.dedup();
    Ok(names)
}

/// Friendly microphone selection on Linux sound servers (PipeWire or
/// PulseAudio) via `pactl`. Capture still runs through the ALSA "default"
/// device; its pipewire/pulse plugin routes to the requested source through
/// the PIPEWIRE_NODE / PULSE_SOURCE environment variables.
#[cfg(target_os = "linux")]
mod pulse {
    use std::{
        process::Command,
        sync::{Mutex, OnceLock},
        time::{Duration, Instant},
    };

    const SOURCE_CACHE_TTL: Duration = Duration::from_secs(5);

    struct SourceCache {
        refreshed_at: Instant,
        sources: Option<Vec<Source>>,
    }

    static SOURCE_CACHE: OnceLock<Mutex<Option<SourceCache>>> = OnceLock::new();

    #[derive(Clone)]
    pub struct Source {
        pub name: String,
        pub description: String,
    }

    pub fn sources() -> Option<Vec<Source>> {
        let cache = SOURCE_CACHE.get_or_init(|| Mutex::new(None));
        if let Ok(cached) = cache.lock() {
            if let Some(cached) = cached.as_ref() {
                if cached.refreshed_at.elapsed() < SOURCE_CACHE_TTL {
                    return cached.sources.clone();
                }
            }
        }
        let sources = sources_uncached();
        if let Ok(mut cached) = cache.lock() {
            *cached = Some(SourceCache {
                refreshed_at: Instant::now(),
                sources: sources.clone(),
            });
        }
        sources
    }

    fn sources_uncached() -> Option<Vec<Source>> {
        let output = Command::new("pactl")
            .args(["--format=json", "list", "sources"])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        parse_sources(&output.stdout)
    }

    fn parse_sources(json: &[u8]) -> Option<Vec<Source>> {
        let parsed: serde_json::Value = serde_json::from_slice(json).ok()?;
        let sources = parsed
            .as_array()?
            .iter()
            .filter_map(|source| {
                let name = source.get("name")?.as_str()?;
                // Monitor sources capture other applications' output, not a
                // microphone; keep the picker to real inputs.
                if name.ends_with(".monitor") {
                    return None;
                }
                let description = source.get("description")?.as_str()?;
                Some(Source {
                    name: name.into(),
                    description: description.into(),
                })
            })
            .collect();
        Some(sources)
    }

    /// The sound-server routing decision for a requested microphone, resolved
    /// once (possibly forking `pactl`) so it can be cached and re-applied
    /// cheaply. Applying it only sets/clears the env vars the ALSA
    /// pipewire/pulse plugins read when the capture stream is opened.
    #[derive(Clone, Debug)]
    pub enum Routing {
        /// Route the ALSA default device to this sound-server source name.
        Source(String),
        /// No routing: record from the system default. `fallback` carries a raw
        /// ALSA device name for CPAL to match itself when no sound server is
        /// present to route on our behalf.
        Default { fallback: Option<String> },
    }

    impl Routing {
        /// The device name CPAL should still try to match itself (only set when
        /// no sound server is available to route on our behalf).
        pub fn fallback_label(&self) -> Option<String> {
            match self {
                Routing::Source(_) => None,
                Routing::Default { fallback } => fallback.clone(),
            }
        }
    }

    /// Serializes environment-based routing so a background prewarm and a
    /// record action cannot interleave PIPEWIRE_NODE/PULSE_SOURCE writes with
    /// each other's config query and stream open.
    static ROUTING_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    pub fn routing_lock() -> std::sync::MutexGuard<'static, ()> {
        ROUTING_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Decides how the requested microphone should be routed, without touching
    /// the environment. May fork `pactl` (through the cached `sources`).
    pub fn resolve_routing(requested: Option<&str>) -> Routing {
        match sources().filter(|sources| !sources.is_empty()) {
            Some(sources) => decide_routing(&sources, requested),
            // No sound server: keep the raw ALSA device-name matching.
            None => Routing::Default {
                fallback: requested.map(str::to_string),
            },
        }
    }

    /// The pure source-list decision, split out so it is testable without a
    /// sound server. Matches the requested device by its human-readable
    /// description (the value the picker persists).
    fn decide_routing(sources: &[Source], requested: Option<&str>) -> Routing {
        match requested.and_then(|requested| {
            sources
                .iter()
                .find(|source| source.description == requested)
        }) {
            Some(source) => Routing::Source(source.name.clone()),
            // Unset selection (or a source that is currently unavailable,
            // e.g. headphones powered off) records from the system default.
            None => Routing::Default { fallback: None },
        }
    }

    /// Applies a previously resolved routing decision to the environment.
    pub fn apply_routing(routing: &Routing) {
        match routing {
            Routing::Source(name) => {
                std::env::set_var("PIPEWIRE_NODE", name);
                std::env::set_var("PULSE_SOURCE", name);
            }
            Routing::Default { .. } => clear_routing(),
        }
    }

    fn clear_routing() {
        std::env::remove_var("PIPEWIRE_NODE");
        std::env::remove_var("PULSE_SOURCE");
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn parses_sources_and_skips_monitors() {
            let json = br#"[
                {"name":"bluez_input.AA:BB:CC:11:22:33","description":"Bluetooth Headset"},
                {"name":"alsa_output.pci.HiFi__sink.monitor","description":"Monitor of Speakers"},
                {"name":"alsa_input.pci.HiFi__Mic1__source","description":"Digital Microphone"}
            ]"#;
            let sources = super::parse_sources(json).expect("sources should parse");
            assert_eq!(sources.len(), 2);
            assert_eq!(sources[0].description, "Bluetooth Headset");
            assert_eq!(sources[1].name, "alsa_input.pci.HiFi__Mic1__source");
        }

        fn fixture_sources() -> Vec<super::Source> {
            vec![
                super::Source {
                    name: "bluez_input.AA:BB:CC:11:22:33".into(),
                    description: "Bluetooth Headset".into(),
                },
                super::Source {
                    name: "alsa_input.pci.HiFi__Mic1__source".into(),
                    description: "Digital Microphone".into(),
                },
            ]
        }

        #[test]
        fn routing_selects_the_source_matching_the_requested_description() {
            let sources = fixture_sources();
            match super::decide_routing(&sources, Some("Digital Microphone")) {
                super::Routing::Source(name) => {
                    assert_eq!(name, "alsa_input.pci.HiFi__Mic1__source")
                }
                super::Routing::Default { .. } => panic!("expected a routed source"),
            }
        }

        #[test]
        fn routing_falls_back_to_system_default_when_the_request_is_unmatched_or_empty() {
            let sources = fixture_sources();
            // A description no source advertises (e.g. a powered-off mic) and an
            // unset selection both record from the system default with no raw
            // ALSA fallback, because a sound server is present to route.
            for requested in [Some("Unplugged Mic"), None] {
                match super::decide_routing(&sources, requested) {
                    super::Routing::Default { fallback: None } => {}
                    other => panic!("expected system-default routing, got {other:?}"),
                }
            }
        }
    }
}

fn device_label(device: &cpal::Device) -> Option<String> {
    device
        .description()
        .ok()
        .map(|description| description.to_string())
}

fn append_samples<T>(
    data: &[T],
    producer: &mut rtrb::Producer<RawSample>,
    counters: &CaptureCounters,
    capture_delay_ns: u64,
) where
    T: Sample,
    f32: FromSample<T>,
{
    // One sequence number per callback block, assigned whether or not the block
    // is accepted. A block dropped whole under backpressure therefore leaves a
    // hole in the sequence that the consumer detects, instead of a silent
    // partial drop that would smear time across the missing samples.
    let packet_sequence = counters
        .next_packet_sequence
        .fetch_add(1, Ordering::Relaxed);
    counters.callback_blocks.fetch_add(1, Ordering::Relaxed);
    counters
        .input_samples
        .fetch_add(data.len() as u64, Ordering::Relaxed);
    if data.is_empty() {
        return;
    }
    // All-or-nothing: reserve the whole block's worth of slots or drop the
    // entire block. A partial write would carry this block's sequence number
    // with fewer samples than the source, hiding the loss from the consumer's
    // discontinuity check and compressing time across the missing samples.
    match producer.write_chunk_uninit(data.len()) {
        Ok(chunk) => {
            let written = chunk.fill_from_iter(data.iter().copied().map(|value| RawSample {
                value: f32::from_sample(value),
                packet_sequence,
                capture_delay_ns,
            }));
            counters
                .accepted_samples
                .fetch_add(written as u64, Ordering::Relaxed);
            let queued = counters
                .queued_samples
                .fetch_add(written, Ordering::Relaxed)
                + written;
            counters
                .ring_high_water_samples
                .fetch_max(queued, Ordering::Relaxed);
        }
        Err(_) => {
            counters
                .dropped_samples
                .fetch_add(data.len() as u64, Ordering::Relaxed);
            counters.overflow_blocks.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Resamples one drained batch into the canonical 16 kHz mono timeline,
/// updating the level meter and canonical-sample counter. The resampler state
/// persists across calls so batching never introduces packet-boundary clicks.
fn flush_capture_packet(
    packet: &[f32],
    resampler: &mut StatefulMonoResampler,
    canonical: &mut Vec<f32>,
    canonical_samples: &Mutex<Vec<f32>>,
    counters: &CaptureCounters,
    level_tracker: &mut LevelTracker,
    on_level: Option<&LevelCallback>,
) {
    canonical.clear();
    resampler.push_interleaved(packet, canonical);
    if canonical.is_empty() {
        return;
    }
    if let Some(on_level) = on_level {
        on_level(level_tracker.observe(canonical));
    }
    if let Ok(mut timeline) = canonical_samples.lock() {
        timeline.extend_from_slice(canonical);
        counters
            .canonical_samples
            .fetch_add(canonical.len() as u64, Ordering::Relaxed);
    }
}

fn spawn_capture_worker(
    mut consumer: rtrb::Consumer<RawSample>,
    canonical_samples: Arc<Mutex<Vec<f32>>>,
    counters: Arc<CaptureCounters>,
    stop: Arc<AtomicBool>,
    sample_rate: u32,
    channels: u16,
    on_level: Option<LevelCallback>,
) -> JoinHandle<()> {
    thread::Builder::new()
        .name("voxide-audio-capture".into())
        .spawn(move || {
            let mut packet = Vec::with_capacity(4_096);
            let mut canonical = Vec::with_capacity(2_048);
            let mut resampler = StatefulMonoResampler::new(sample_rate, channels);
            let mut level_tracker = LevelTracker::default();
            let mut previous_packet_sequence = None;
            loop {
                packet.clear();
                while packet.len() < packet.capacity() {
                    match consumer.pop() {
                        Ok(sample) => {
                            let is_gap = matches!(
                                previous_packet_sequence,
                                Some(previous) if sample.packet_sequence > previous + 1
                            );
                            previous_packet_sequence = Some(sample.packet_sequence);
                            counters
                                .latest_capture_delay_ns
                                .store(sample.capture_delay_ns, Ordering::Relaxed);
                            counters.queued_samples.fetch_sub(1, Ordering::Relaxed);
                            if is_gap {
                                // A block was dropped under backpressure. Flush
                                // the audio captured up to the hole, then reset
                                // the resampler so the samples after it re-seed
                                // the interpolator instead of bridging the gap.
                                counters.discontinuities.fetch_add(1, Ordering::Relaxed);
                                if !packet.is_empty() {
                                    flush_capture_packet(
                                        &packet,
                                        &mut resampler,
                                        &mut canonical,
                                        &canonical_samples,
                                        &counters,
                                        &mut level_tracker,
                                        on_level.as_ref(),
                                    );
                                    packet.clear();
                                }
                                resampler.reset();
                            }
                            packet.push(sample.value);
                        }
                        Err(_) => break,
                    }
                }
                if !packet.is_empty() {
                    flush_capture_packet(
                        &packet,
                        &mut resampler,
                        &mut canonical,
                        &canonical_samples,
                        &counters,
                        &mut level_tracker,
                        on_level.as_ref(),
                    );
                    continue;
                }
                if stop.load(Ordering::Acquire) {
                    canonical.clear();
                    resampler.finish(&mut canonical);
                    if !canonical.is_empty() {
                        if let Some(on_level) = on_level.as_ref() {
                            on_level(level_tracker.observe(&canonical));
                        }
                        if let Ok(mut timeline) = canonical_samples.lock() {
                            timeline.extend_from_slice(&canonical);
                            counters
                                .canonical_samples
                                .fetch_add(canonical.len() as u64, Ordering::Relaxed);
                        }
                    }
                    break;
                }
                thread::sleep(CAPTURE_WORKER_IDLE_SLEEP);
            }
        })
        .expect("could not start audio capture worker")
}

/// Stateful downmixing and linear resampling. The output cursor is retained
/// across every raw-ring drain, which prevents packet-boundary clicks, gaps,
/// and sample-count drift at rates such as 44.1 kHz.
struct StatefulMonoResampler {
    channels: usize,
    ratio: f64,
    pending_frame: Vec<f32>,
    previous: Option<f32>,
    source_index: u64,
    next_output_position: f64,
}

impl StatefulMonoResampler {
    fn new(sample_rate: u32, channels: u16) -> Self {
        Self {
            channels: channels.max(1) as usize,
            ratio: sample_rate.max(1) as f64 / WHISPER_SAMPLE_RATE as f64,
            pending_frame: Vec::with_capacity(channels.max(1) as usize),
            previous: None,
            source_index: 0,
            next_output_position: 0.0,
        }
    }

    fn push_interleaved(&mut self, samples: &[f32], destination: &mut Vec<f32>) {
        for sample in samples {
            self.pending_frame.push(*sample);
            if self.pending_frame.len() != self.channels {
                continue;
            }
            let mono = self.pending_frame.iter().sum::<f32>() / self.channels as f32;
            self.pending_frame.clear();
            self.push_mono(mono, destination);
        }
    }

    fn push_mono(&mut self, current: f32, destination: &mut Vec<f32>) {
        let current_index = self.source_index;
        if let Some(previous) = self.previous {
            while self.next_output_position <= current_index as f64 {
                let fraction = (self.next_output_position - (current_index - 1) as f64) as f32;
                destination.push(previous + (current - previous) * fraction.clamp(0.0, 1.0));
                self.next_output_position += self.ratio;
            }
        } else if self.next_output_position == 0.0 {
            destination.push(current);
            self.next_output_position += self.ratio;
        }
        self.previous = Some(current);
        self.source_index += 1;
    }

    /// Drops all resampling state back to a fresh start while keeping the
    /// channel/ratio configuration. Called at a capture discontinuity (a block
    /// dropped under backpressure) so the next sample re-seeds the interpolator
    /// (`previous == None`) instead of interpolating across the missing audio.
    /// Only the live-capture path resets; the contiguous file-decode path never
    /// does.
    fn reset(&mut self) {
        self.pending_frame.clear();
        self.previous = None;
        self.source_index = 0;
        self.next_output_position = 0.0;
    }

    /// Completes a finite source by holding its final sample for any output
    /// positions still inside the source duration. Live capture calls this
    /// only after the stream has stopped and its raw ring has drained; file
    /// decoding calls it at end-of-file. That keeps both paths sample-for-
    /// sample equivalent without extrapolating beyond the recording boundary.
    fn finish(&mut self, destination: &mut Vec<f32>) {
        self.pending_frame.clear();
        let Some(last) = self.previous else {
            return;
        };
        while self.next_output_position < self.source_index as f64 {
            destination.push(last);
            self.next_output_position += self.ratio;
        }
    }
}

#[derive(Default)]
struct LevelTracker {
    peak: f32,
    floor: f32,
}

impl LevelTracker {
    fn observe(&mut self, samples: &[f32]) -> f32 {
        let rms = (samples.iter().map(|sample| sample * sample).sum::<f32>()
            / samples.len().max(1) as f32)
            .sqrt();
        if self.peak == 0.0 {
            self.peak = 0.02;
            self.floor = 0.002;
        }
        self.floor = (self.floor * 1.01 + 1e-5).min(rms).max(1e-4);
        self.peak = (self.peak * 0.995).max(rms).max(self.floor * 3.0);
        ((rms - self.floor) / (self.peak - self.floor).max(1e-4)).clamp(0.0, 1.0)
    }
}

pub fn mono_resample_for_whisper(audio: CapturedAudio) -> Result<Vec<f32>, String> {
    if audio.samples.is_empty() {
        return Err("No audio was captured".into());
    }
    if audio.channels == 0 || audio.sample_rate == 0 {
        return Err("The captured audio has an invalid format".into());
    }

    if audio.sample_rate == WHISPER_SAMPLE_RATE && audio.channels == 1 {
        return Ok(audio.samples);
    }

    let mut resampler = StatefulMonoResampler::new(audio.sample_rate, audio.channels);
    let mut output = Vec::new();
    resampler.push_interleaved(&audio.samples, &mut output);
    resampler.finish(&mut output);
    if output.is_empty() {
        return Err("The recording is too short to transcribe".into());
    }
    Ok(output)
}

pub fn wav_bytes_from_16khz_mono(samples: &[f32]) -> Result<Vec<u8>, String> {
    if samples.is_empty() {
        return Err("No audio was captured".into());
    }
    let data_size = samples
        .len()
        .checked_mul(2)
        .ok_or("The captured audio is too large to encode")?;
    let riff_size = 36usize
        .checked_add(data_size)
        .ok_or("The captured audio is too large to encode")?;
    let data_size =
        u32::try_from(data_size).map_err(|_| "The captured audio is too large to encode")?;
    let riff_size =
        u32::try_from(riff_size).map_err(|_| "The captured audio is too large to encode")?;
    let mut wav = Vec::with_capacity(44 + samples.len() * 2);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&riff_size.to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16_u32.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&WHISPER_SAMPLE_RATE.to_le_bytes());
    wav.extend_from_slice(&(WHISPER_SAMPLE_RATE * 2).to_le_bytes());
    wav.extend_from_slice(&2_u16.to_le_bytes());
    wav.extend_from_slice(&16_u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());
    for sample in samples {
        wav.extend_from_slice(&((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16).to_le_bytes());
    }
    Ok(wav)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic synthetic utterance with a long internal pause. It is
    /// intentionally not speech content: capture regression tests need a
    /// consent-free waveform that still exposes timing, silence, and packet
    /// boundary bugs.
    fn pause_heavy_fixture(sample_rate: u32, channels: u16) -> Vec<f32> {
        let frames = sample_rate as usize * 3;
        let mut samples = Vec::with_capacity(frames * channels as usize);
        for frame in 0..frames {
            let seconds = frame as f32 / sample_rate as f32;
            let speech = !(0.8..1.6).contains(&seconds);
            let sample = if speech {
                let envelope = if seconds < 0.08 {
                    seconds / 0.08
                } else if seconds > 2.92 {
                    (3.0 - seconds) / 0.08
                } else {
                    1.0
                };
                envelope
                    * ((seconds * std::f32::consts::TAU * 173.0).sin() * 0.25
                        + (seconds * std::f32::consts::TAU * 311.0).sin() * 0.1)
            } else {
                0.0
            };
            for _ in 0..channels {
                samples.push(sample);
            }
        }
        samples
    }

    fn resample_packetized(input: &[f32], sample_rate: u32, channels: u16) -> Vec<f32> {
        let mut resampler = StatefulMonoResampler::new(sample_rate, channels);
        let mut output = Vec::new();
        let packet_sizes = [1, 31, 257, 1_021, 4_093, 79];
        let mut offset = 0;
        let mut packet_index = 0;
        while offset < input.len() {
            let end = (offset + packet_sizes[packet_index % packet_sizes.len()]).min(input.len());
            resampler.push_interleaved(&input[offset..end], &mut output);
            offset = end;
            packet_index += 1;
        }
        resampler.finish(&mut output);
        output
    }

    #[test]
    fn file_transcription_requires_at_least_one_second_of_16khz_audio() {
        assert!(!has_minimum_transcription_samples(&vec![0.0; 15_999]));
        assert!(has_minimum_transcription_samples(&vec![0.0; 16_000]));
    }

    #[test]
    fn short_live_audio_is_padded_to_the_transcription_minimum() {
        let mut samples = vec![0.25; 800];
        pad_short_transcription_samples(&mut samples);
        assert_eq!(samples.len(), MINIMUM_TRANSCRIPTION_SAMPLES);
        assert_eq!(samples[799], 0.25);
        assert_eq!(samples[800], 0.0);
    }

    #[test]
    fn downmixes_and_resamples_stereo_audio() {
        let samples = mono_resample_for_whisper(CapturedAudio {
            samples: vec![0.4, 0.6, 0.2, 0.8],
            sample_rate: 8_000,
            channels: 2,
            duration_ms: 1,
        })
        .expect("audio should resample");
        assert_eq!(samples.len(), 4);
        assert!((samples[0] - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn encodes_whisper_samples_as_a_valid_wav_header() {
        let wav = wav_bytes_from_16khz_mono(&[0.0, 0.5]).expect("WAV should encode");
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[36..40], b"data");
        assert_eq!(wav.len(), 48);
    }

    #[test]
    fn stateful_resampling_preserves_length_across_common_input_rates() {
        for sample_rate in [8_000, 16_000, 32_000, 44_100, 48_000, 96_000] {
            let input = (0..sample_rate)
                .map(|index| (index % 127) as f32 / 127.0)
                .collect::<Vec<_>>();
            let mut resampler = StatefulMonoResampler::new(sample_rate, 1);
            let mut output = Vec::new();
            resampler.push_interleaved(&input, &mut output);
            assert!(
                output.len().abs_diff(WHISPER_SAMPLE_RATE as usize) <= 1,
                "{sample_rate} Hz produced {} samples",
                output.len()
            );
        }
    }

    #[test]
    fn stateful_resampling_is_packet_boundary_deterministic() {
        let input = (0..44_100 * 2)
            .map(|index| ((index * 17 % 997) as f32 / 997.0) * 2.0 - 1.0)
            .collect::<Vec<_>>();
        let mut whole_resampler = StatefulMonoResampler::new(44_100, 1);
        let mut whole = Vec::new();
        whole_resampler.push_interleaved(&input, &mut whole);

        let mut packet_resampler = StatefulMonoResampler::new(44_100, 1);
        let mut packetized = Vec::new();
        let packet_sizes = [1, 17, 63, 128, 509, 3, 2_047];
        let mut offset = 0;
        let mut packet_index = 0;
        while offset < input.len() {
            let end = (offset + packet_sizes[packet_index % packet_sizes.len()]).min(input.len());
            packet_resampler.push_interleaved(&input[offset..end], &mut packetized);
            offset = end;
            packet_index += 1;
        }
        assert_eq!(whole, packetized);
    }

    #[test]
    fn pause_heavy_fixture_stays_synchronized_at_common_device_rates() {
        for (sample_rate, channels) in [(44_100, 1), (44_100, 2), (48_000, 1), (48_000, 2)] {
            let input = pause_heavy_fixture(sample_rate, channels);
            let mut whole_resampler = StatefulMonoResampler::new(sample_rate, channels);
            let mut whole = Vec::new();
            whole_resampler.push_interleaved(&input, &mut whole);
            whole_resampler.finish(&mut whole);
            let packetized = resample_packetized(&input, sample_rate, channels);

            assert_eq!(
                packetized, whole,
                "{sample_rate} Hz/{channels} channel fixture"
            );
            assert_eq!(whole.len(), WHISPER_SAMPLE_RATE as usize * 3);
            assert!(
                whole[WHISPER_SAMPLE_RATE as usize..WHISPER_SAMPLE_RATE as usize + 8_000]
                    .iter()
                    .all(|sample| sample.abs() < 1e-6),
                "the internal silence must remain silence after resampling"
            );
        }
    }

    #[test]
    fn file_audio_and_packetized_capture_share_the_canonical_timeline() {
        for (sample_rate, channels) in [(44_100, 1), (44_100, 2), (48_000, 1), (48_000, 2)] {
            let input = pause_heavy_fixture(sample_rate, channels);
            let file_audio = mono_resample_for_whisper(CapturedAudio {
                samples: input.clone(),
                sample_rate,
                channels,
                duration_ms: 3_000,
            })
            .expect("fixture should convert");
            assert_eq!(
                file_audio,
                resample_packetized(&input, sample_rate, channels),
                "{sample_rate} Hz/{channels} channel fixture"
            );
        }
    }

    #[test]
    fn downmixing_retains_incomplete_frames_until_the_next_packet() {
        let mut resampler = StatefulMonoResampler::new(16_000, 2);
        let mut output = Vec::new();
        resampler.push_interleaved(&[0.2, 0.6, 0.1], &mut output);
        assert_eq!(output, vec![0.4]);
        resampler.push_interleaved(&[0.9], &mut output);
        assert_eq!(output, vec![0.4, 0.5]);
    }

    #[test]
    fn full_raw_ring_is_counted_instead_of_silently_dropped() {
        let (mut producer, _consumer) = rtrb::RingBuffer::<RawSample>::new(2);
        let counters = CaptureCounters::default();
        // The 3-sample block does not fit the 2-slot ring, so it is dropped
        // whole and fully counted — never partially written under this block's
        // sequence (which would smear time across the missing samples).
        append_samples(&[0.1_f32, 0.2, 0.3], &mut producer, &counters, 42);
        assert_eq!(counters.health().callback_blocks, 1);
        assert_eq!(counters.health().accepted_samples, 0);
        assert_eq!(counters.health().dropped_samples, 3);
        assert_eq!(counters.health().overflow_blocks, 1);
        assert_eq!(counters.health().ring_high_water_samples, 0);
        assert_eq!(counters.health().latest_capture_delay_ns, 0);
    }

    #[test]
    fn raw_ring_retains_packet_sequence_and_capture_delay_metadata() {
        let (mut producer, mut consumer) = rtrb::RingBuffer::<RawSample>::new(4);
        let counters = CaptureCounters::default();
        append_samples(&[0.25_f32, 0.5], &mut producer, &counters, 12);
        append_samples(&[-0.25_f32], &mut producer, &counters, 34);
        let first = consumer.pop().expect("first packet sample");
        let second = consumer.pop().expect("second packet sample");
        let third = consumer.pop().expect("next packet sample");
        assert_eq!(first.packet_sequence, 0);
        assert_eq!(second.packet_sequence, 0);
        assert_eq!(third.packet_sequence, 1);
        assert_eq!(first.capture_delay_ns, 12);
        assert_eq!(third.capture_delay_ns, 34);
        assert_eq!(third.value, -0.25);
    }

    #[test]
    fn overflowing_block_is_dropped_whole_and_leaves_a_sequence_gap() {
        let (mut producer, mut consumer) = rtrb::RingBuffer::<RawSample>::new(6);
        let counters = CaptureCounters::default();
        let block = [0.1_f32, 0.2, 0.3, 0.4];
        // Block 0 (seq 0) fits; block 1 (seq 1) needs four slots but only two
        // remain, so it is dropped whole rather than partially written.
        append_samples(&block, &mut producer, &counters, 0);
        append_samples(&block, &mut producer, &counters, 0);
        assert_eq!(counters.accepted_samples.load(Ordering::Relaxed), 4);
        assert_eq!(counters.dropped_samples.load(Ordering::Relaxed), 4);
        assert_eq!(counters.overflow_blocks.load(Ordering::Relaxed), 1);
        let mut sequences = Vec::new();
        while let Ok(sample) = consumer.pop() {
            sequences.push(sample.packet_sequence);
        }
        // Only block 0's samples reached the ring — no partial block-1 samples.
        assert_eq!(sequences, vec![0, 0, 0, 0]);
        // A later block is seq 2, so the consumer sees a 0 -> 2 jump: the
        // dropped block leaves a detectable hole, not a silent same-seq smear.
        append_samples(&block, &mut producer, &counters, 0);
        assert_eq!(
            consumer.pop().expect("third block sample").packet_sequence,
            2
        );
    }

    #[test]
    fn resampler_reset_matches_a_fresh_resampler() {
        let input = [0.25_f32, 0.5, 0.75, 1.0, 0.5, 0.0];
        let mut fresh = StatefulMonoResampler::new(44_100, 1);
        let mut fresh_out = Vec::new();
        fresh.push_interleaved(&input, &mut fresh_out);

        let mut reused = StatefulMonoResampler::new(44_100, 1);
        let mut scratch = Vec::new();
        reused.push_interleaved(&[0.9_f32, 0.8, 0.7], &mut scratch);
        scratch.clear();
        reused.reset();
        let mut reused_out = Vec::new();
        reused.push_interleaved(&input, &mut reused_out);

        // After reset the resampler behaves exactly like a fresh one — phase,
        // previous sample, source cursor, and pending frame are all cleared —
        // so no interpolation bridges the discontinuity.
        assert!(!fresh_out.is_empty());
        assert_eq!(fresh_out, reused_out);
    }
}
