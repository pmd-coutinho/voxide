use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
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

pub struct AudioCapture {
    _stream: Stream,
    samples: Arc<Mutex<Vec<f32>>>,
    sample_rate: u32,
    channels: u16,
    started_at: Instant,
}

impl AudioCapture {
    pub fn start(
        requested_device: Option<&str>,
        on_level: Option<LevelCallback>,
    ) -> Result<Self, String> {
        // Microphones differ wildly in raw gain (built-in DMIC arrays are far
        // quieter than headset mics), so a fixed scale leaves quiet inputs
        // showing a flat level meter. Track the session's noise floor and a
        // slowly-decaying speech peak, and report where the current RMS sits
        // between them: silence stays near zero and speech reaches the top on
        // any microphone.
        let on_level: Option<LevelCallback> = on_level.map(|callback| {
            let tracker = Mutex::new((0.02f32, 0.002f32));
            Arc::new(move |rms: f32| {
                let level = {
                    let Ok(mut tracker) = tracker.lock() else {
                        return;
                    };
                    let (peak, floor) = &mut *tracker;
                    *floor = (*floor * 1.01 + 1e-5).min(rms).max(1e-4);
                    *peak = (*peak * 0.995).max(rms).max(*floor * 3.0);
                    ((rms - *floor) / (*peak - *floor).max(1e-4)).clamp(0.0, 1.0)
                };
                callback(level);
            }) as LevelCallback
        });
        let host = cpal::default_host();
        let requested = requested_device.filter(|name| !name.trim().is_empty());
        #[cfg(target_os = "linux")]
        let requested = pulse::route_to_requested_source(requested);
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
        let sample_rate = supported_config.sample_rate();
        let channels = supported_config.channels();
        let samples = Arc::new(Mutex::new(Vec::new()));
        let callback_samples = Arc::clone(&samples);
        let error_callback = |error| eprintln!("Voxide audio capture error: {error}");
        let config = supported_config.config();

        macro_rules! build_stream {
            ($sample:ty) => {
                device.build_input_stream(
                    &config,
                    move |data: &[$sample], _| {
                        append_samples(data, &callback_samples, on_level.as_deref())
                    },
                    error_callback,
                    None,
                )
            };
        }
        let stream = match supported_config.sample_format() {
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
            format => {
                return Err(format!(
                "Microphone sample format {format} is not supported by this desktop audio backend"
            ))
            }
        }
        .map_err(|error| format!("Could not start microphone capture: {error}"))?;

        stream
            .play()
            .map_err(|error| format!("Could not activate microphone capture: {error}"))?;
        Ok(Self {
            _stream: stream,
            samples,
            sample_rate,
            channels,
            started_at: Instant::now(),
        })
    }

    pub fn finish(self) -> Result<CapturedAudio, String> {
        let samples = self.snapshot_samples()?;
        Ok(CapturedAudio {
            samples,
            sample_rate: self.sample_rate,
            channels: self.channels,
            duration_ms: self.started_at.elapsed().as_millis() as u64,
        })
    }

    pub fn snapshot_recent(&self, maximum_duration: Duration) -> Result<CapturedAudio, String> {
        let maximum_samples = (maximum_duration.as_secs_f64()
            * self.sample_rate as f64
            * self.channels as f64) as usize;
        let samples = self
            .samples
            .lock()
            .map_err(|_| "Audio capture buffer lock was poisoned".to_string())?;
        let samples = if maximum_samples == 0 || samples.len() <= maximum_samples {
            samples.clone()
        } else {
            samples[samples.len() - maximum_samples..].to_vec()
        };
        let frame_count = samples.len() as u64 / self.channels.max(1) as u64;
        Ok(CapturedAudio {
            samples,
            sample_rate: self.sample_rate,
            channels: self.channels,
            duration_ms: frame_count.saturating_mul(1_000) / self.sample_rate.max(1) as u64,
        })
    }

    fn snapshot_samples(&self) -> Result<Vec<f32>, String> {
        self.samples
            .lock()
            .map_err(|_| "Audio capture buffer lock was poisoned".to_string())
            .map(|samples| samples.clone())
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> u16 {
        self.channels
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

/// Performs the one-time device and sound-server discovery during startup so
/// the first record action only needs to configure and start its stream.
pub fn prewarm_input_devices() -> Result<usize, String> {
    input_device_names().map(|devices| devices.len())
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

    /// Points the ALSA default device at the requested sound-server source.
    /// Returns the device name CPAL should still try to match itself (only
    /// when no sound server is available to do the routing).
    pub fn route_to_requested_source(requested: Option<&str>) -> Option<String> {
        let Some(sources) = sources().filter(|sources| !sources.is_empty()) else {
            // No sound server: keep the raw ALSA device-name matching.
            clear_routing();
            return requested.map(str::to_string);
        };
        let selected = requested.and_then(|requested| {
            sources
                .iter()
                .find(|source| source.description == requested)
        });
        match selected {
            Some(source) => {
                // The pipewire ALSA plugin honors PIPEWIRE_NODE and the pulse
                // ALSA plugin honors PULSE_SOURCE; set both so the default
                // device records from the chosen source on either server.
                std::env::set_var("PIPEWIRE_NODE", &source.name);
                std::env::set_var("PULSE_SOURCE", &source.name);
            }
            // Unset selection (or a source that is currently unavailable,
            // e.g. headphones powered off) records from the system default.
            None => clear_routing(),
        }
        None
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
    destination: &Arc<Mutex<Vec<f32>>>,
    on_level: Option<&(dyn Fn(f32) + Send + Sync)>,
) where
    T: Sample,
    f32: FromSample<T>,
{
    if let Some(on_level) = on_level {
        let rms = (data
            .iter()
            .copied()
            .map(f32::from_sample)
            .map(|sample| sample * sample)
            .sum::<f32>()
            / data.len().max(1) as f32)
            .sqrt();
        on_level(rms);
    }
    if let Ok(mut samples) = destination.try_lock() {
        samples.extend(data.iter().copied().map(f32::from_sample));
    }
}

pub fn mono_resample_for_whisper(audio: CapturedAudio) -> Result<Vec<f32>, String> {
    if audio.samples.is_empty() {
        return Err("No audio was captured".into());
    }
    if audio.channels == 0 || audio.sample_rate == 0 {
        return Err("The captured audio has an invalid format".into());
    }

    let channel_count = audio.channels as usize;
    let mono: Vec<f32> = audio
        .samples
        .chunks(channel_count)
        .map(|frame| frame.iter().copied().sum::<f32>() / frame.len() as f32)
        .collect();

    if audio.sample_rate == WHISPER_SAMPLE_RATE {
        return Ok(mono);
    }

    let output_length =
        ((mono.len() as u64 * WHISPER_SAMPLE_RATE as u64) / audio.sample_rate as u64) as usize;
    if output_length == 0 {
        return Err("The recording is too short to transcribe".into());
    }

    let ratio = audio.sample_rate as f64 / WHISPER_SAMPLE_RATE as f64;
    Ok((0..output_length)
        .map(|index| {
            let source_position = index as f64 * ratio;
            let before = source_position.floor() as usize;
            let after = (before + 1).min(mono.len() - 1);
            let fraction = (source_position - before as f64) as f32;
            mono[before] + (mono[after] - mono[before]) * fraction
        })
        .collect())
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
}
