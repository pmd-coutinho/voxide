//! Log-mel front end for Cohere Transcribe.
//!
//! Cohere's encoder is NVIDIA's Parakeet encoder, so the features it expects are
//! NeMo-shaped rather than Whisper-shaped. Every step below is transcribed from
//! `transformers.models.cohere_asr.feature_extraction_cohere_asr` (5.14.1) and
//! differs from Whisper's front end in at least one way that would not fail
//! loudly — a wrong front end yields fluent, confident, wrong transcripts. See
//! `docs/COHERE_ENGINE.md` for the full derivation.
//!
//! Deliberate divergence from the reference: **dither is not applied.** The
//! reference adds `1e-5 * randn(n)` from a PyTorch generator seeded with the
//! sample count, which cannot be reproduced outside PyTorch. It is a
//! training-time regulariser at an amplitude four orders of magnitude below
//! speech, and the fixture this module is tested against was generated with
//! `dither = 0.0` so the comparison is exact rather than approximate.

// Nothing calls this yet: the front end is verified against the reference
// extractor, but the two ONNX sessions that consume its output are still to be
// written (see `docs/COHERE_ENGINE.md`, "Remaining work"). Landing it verified
// and unused is deliberate — the alternative is carrying an unverified front end
// alongside the session work, and this is the component whose errors are silent.
#![allow(dead_code)]

use std::sync::Arc;

use rustfft::{num_complex::Complex32, Fft, FftPlanner};

/// Audio rate the model was trained at. Callers resample before this.
pub const SAMPLE_RATE: u32 = 16_000;
/// Mel bins per frame — the encoder's `num_mel_bins`.
pub const MEL_BINS: usize = 128;

const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const HOP_LENGTH: usize = 160;
/// Applied to the waveform before the STFT, not to the spectrum.
const PREEMPHASIS: f32 = 0.97;
/// `2^-24`, not the more usual `1e-10`. Added inside the natural log.
const LOG_GUARD: f32 = 5.960_464_5e-8;
/// Guards the division in per-bin normalisation.
const NORM_EPSILON: f32 = 1e-5;
/// Past this, audio is split at its quietest point rather than transcribed whole.
const MAX_CLIP_SECONDS: f32 = 35.0;
/// How far back from a chunk boundary to look for a quiet point.
const OVERLAP_SECONDS: f32 = 5.0;
/// Granularity of that search, and the shortest segment worth searching.
const MIN_ENERGY_WINDOW: usize = 1600;

/// Reusable mel filterbank, window and FFT plan.
///
/// Building the filterbank is not free, and a dictation front end runs per
/// utterance, so this is constructed once and shared.
pub struct CohereFbank {
    /// `[MEL_BINS][N_FFT / 2 + 1]`, Slaney-scaled and Slaney-normalised.
    mel_filters: Vec<Vec<f32>>,
    /// A symmetric 400-point Hann window, zero-padded and centred in `N_FFT`.
    window: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
}

impl Default for CohereFbank {
    fn default() -> Self {
        Self::new()
    }
}

impl CohereFbank {
    pub fn new() -> Self {
        Self {
            mel_filters: slaney_mel_filters(),
            window: centred_hann_window(),
            fft: FftPlanner::new().plan_fft_forward(N_FFT),
        }
    }

    /// Splits `samples` the way the reference does and returns one feature block
    /// per chunk, each `[frames][MEL_BINS]`.
    ///
    /// Normalisation is per chunk because it is per chunk in the reference: each
    /// block is fed to the encoder separately, so its statistics are its own.
    pub fn features(&self, samples: &[f32]) -> Vec<Vec<Vec<f32>>> {
        split_into_chunks(samples)
            .into_iter()
            .map(|chunk| self.features_for_chunk(chunk))
            .collect()
    }

    /// The single-chunk path: preemphasis, STFT, mel, log, per-bin CMVN.
    ///
    /// Returns every frame the STFT produced, with any beyond
    /// [`valid_frame_count`] zeroed — see that function for why the two differ.
    pub fn features_for_chunk(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        let emphasised = preemphasise(samples);
        let mut frames = self.log_mel_frames(&emphasised);
        normalise_per_feature(&mut frames, valid_frame_count(samples.len()));
        frames
    }

    fn log_mel_frames(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        // `center=True` with `pad_mode="constant"`: N_FFT/2 zeros each side, so
        // the first frame is centred on sample 0.
        let pad = N_FFT / 2;
        let mut padded = vec![0.0f32; samples.len() + 2 * pad];
        padded[pad..pad + samples.len()].copy_from_slice(samples);

        let frame_count = 1 + padded.len().saturating_sub(N_FFT) / HOP_LENGTH;
        let mut buffer = vec![Complex32::new(0.0, 0.0); N_FFT];
        let mut frames = Vec::with_capacity(frame_count);

        for index in 0..frame_count {
            let start = index * HOP_LENGTH;
            for (slot, (sample, weight)) in buffer
                .iter_mut()
                .zip(padded[start..start + N_FFT].iter().zip(&self.window))
            {
                *slot = Complex32::new(sample * weight, 0.0);
            }
            self.fft.process(&mut buffer);

            // Power spectrum: magnitude, then squared. Only the non-redundant
            // half of a real transform is used.
            let power: Vec<f32> = buffer[..N_FFT / 2 + 1]
                .iter()
                .map(|bin| bin.re * bin.re + bin.im * bin.im)
                .collect();

            frames.push(
                self.mel_filters
                    .iter()
                    .map(|filter| {
                        let energy: f32 = filter
                            .iter()
                            .zip(&power)
                            .map(|(weight, value)| weight * value)
                            .sum();
                        (energy + LOG_GUARD).ln()
                    })
                    .collect(),
            );
        }
        frames
    }
}

/// `[x[0], x[1] - 0.97*x[0], …]`. First sample passes through unchanged.
fn preemphasise(samples: &[f32]) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(samples.len());
    out.push(samples[0]);
    for pair in samples.windows(2) {
        out.push(pair[1] - PREEMPHASIS * pair[0]);
    }
    out
}

/// How many leading frames the model treats as real signal.
///
/// The reference computes this as
/// `(audio_len + 2 * (n_fft / 2) - n_fft) / hop`, which reduces to
/// `audio_len / hop` — **one fewer than the STFT actually produces**, because
/// centred padding buys an extra trailing frame that the model's attention mask
/// then discards. Getting this wrong is invisible frame by frame but shifts the
/// statistics of every bin, so it has to be exact.
fn valid_frame_count(sample_count: usize) -> usize {
    (sample_count + 2 * (N_FFT / 2)).saturating_sub(N_FFT) / HOP_LENGTH
}

/// Zero-mean, unit-variance per mel bin across time, over the valid frames only.
///
/// Two details that are easy to miss and both silently degrade accuracy: the
/// variance denominator is `valid - 1` (sample, not population), and the
/// statistics come from the valid frames while the scaling is applied to all of
/// them — after which the invalid tail is zeroed, exactly as the reference's
/// attention mask does.
fn normalise_per_feature(frames: &mut [Vec<f32>], valid: usize) {
    let valid = valid.min(frames.len());
    if valid == 0 {
        for frame in frames.iter_mut() {
            frame.fill(0.0);
        }
        return;
    }
    for bin in 0..MEL_BINS {
        let sum: f32 = frames[..valid].iter().map(|frame| frame[bin]).sum();
        let mean = sum / valid as f32;
        let deviation: f32 = frames[..valid]
            .iter()
            .map(|frame| (frame[bin] - mean).powi(2))
            .sum();
        // One valid frame has no sample variance; the reference divides by zero
        // and yields NaN, so fall back to a zero standard deviation instead.
        let variance = if valid > 1 {
            deviation / (valid - 1) as f32
        } else {
            0.0
        };
        let std = variance.sqrt();
        for frame in frames.iter_mut() {
            frame[bin] = (frame[bin] - mean) / (std + NORM_EPSILON);
        }
    }
    for frame in frames[valid..].iter_mut() {
        frame.fill(0.0);
    }
}

/// Splits audio longer than 35 s at its quietest point within the last 5 s of
/// each chunk, so a cut lands in a pause rather than mid-word.
fn split_into_chunks(samples: &[f32]) -> Vec<&[f32]> {
    let chunk_size = (MAX_CLIP_SECONDS * SAMPLE_RATE as f32).round().max(1.0) as usize;
    let boundary_context = (OVERLAP_SECONDS * SAMPLE_RATE as f32).round().max(1.0) as usize;
    if samples.len() <= chunk_size {
        return vec![samples];
    }
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < samples.len() {
        if start + chunk_size >= samples.len() {
            chunks.push(&samples[start..]);
            break;
        }
        let search_start = start.max(start + chunk_size - boundary_context);
        let search_end = (start + chunk_size).min(samples.len());
        let split = if search_end <= search_start {
            start + chunk_size
        } else {
            quietest_offset(samples, search_start, search_end)
        }
        .clamp(start + 1, samples.len());
        chunks.push(&samples[start..split]);
        start = split;
    }
    chunks
}

/// Offset of the lowest-RMS window in `[start, end)`, stepping a window at a
/// time. Falls back to the midpoint when the span is too short to search.
fn quietest_offset(samples: &[f32], start: usize, end: usize) -> usize {
    let span = end - start;
    if span <= MIN_ENERGY_WINDOW {
        return (start + end) / 2;
    }
    let mut quietest = start;
    let mut lowest = f32::INFINITY;
    let mut offset = 0usize;
    while offset + MIN_ENERGY_WINDOW <= span {
        let window = &samples[start + offset..start + offset + MIN_ENERGY_WINDOW];
        let mean_square =
            window.iter().map(|value| value * value).sum::<f32>() / MIN_ENERGY_WINDOW as f32;
        let energy = mean_square.sqrt();
        if energy < lowest {
            lowest = energy;
            quietest = start + offset;
        }
        offset += MIN_ENERGY_WINDOW;
    }
    quietest
}

/// A 400-point symmetric Hann window centred inside `N_FFT` zeros.
///
/// Symmetric — `periodic=false` — where Whisper uses a periodic window, and
/// centred because `torch.stft` pads `(n_fft - win_length) / 2` on the left when
/// the window is shorter than the transform.
fn centred_hann_window() -> Vec<f32> {
    let mut window = vec![0.0f32; N_FFT];
    let offset = (N_FFT - WIN_LENGTH) / 2;
    let denominator = (WIN_LENGTH - 1) as f32;
    for index in 0..WIN_LENGTH {
        let phase = 2.0 * std::f32::consts::PI * index as f32 / denominator;
        window[offset + index] = 0.5 - 0.5 * phase.cos();
    }
    window
}

/// Slaney mel scale, as used by `librosa` with `htk=False`.
fn hz_to_mel(hz: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
    let logstep = (6.4f64).ln() / 27.0;
    if hz >= MIN_LOG_HZ {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / logstep
    } else {
        hz / F_SP
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
    let logstep = (6.4f64).ln() / 27.0;
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HZ * (logstep * (mel - MIN_LOG_MEL)).exp()
    } else {
        F_SP * mel
    }
}

/// `librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, fmin=0, fmax=8000,
/// norm="slaney")`, computed in f64 and stored as f32.
fn slaney_mel_filters() -> Vec<Vec<f32>> {
    let bins = N_FFT / 2 + 1;
    let nyquist = SAMPLE_RATE as f64 / 2.0;
    let fft_freqs: Vec<f64> = (0..bins)
        .map(|index| nyquist * index as f64 / (bins - 1) as f64)
        .collect();

    // n_mels + 2 edges, evenly spaced on the mel scale.
    let min_mel = hz_to_mel(0.0);
    let max_mel = hz_to_mel(nyquist);
    let edges: Vec<f64> = (0..MEL_BINS + 2)
        .map(|index| {
            let mel = min_mel + (max_mel - min_mel) * index as f64 / (MEL_BINS + 1) as f64;
            mel_to_hz(mel)
        })
        .collect();

    (0..MEL_BINS)
        .map(|mel| {
            let lower_width = edges[mel + 1] - edges[mel];
            let upper_width = edges[mel + 2] - edges[mel + 1];
            // Slaney normalisation: each filter is scaled by the reciprocal of
            // its width, so filters carry equal energy rather than equal peak.
            let norm = 2.0 / (edges[mel + 2] - edges[mel]);
            fft_freqs
                .iter()
                .map(|&freq| {
                    let rising = (freq - edges[mel]) / lower_width;
                    let falling = (edges[mel + 2] - freq) / upper_width;
                    (rising.min(falling).max(0.0) * norm) as f32
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rebuilds the exact waveform `scripts/../gen-fixture.py` fed to the
    /// reference extractor: a 200→3000 Hz chirp under a 3 Hz envelope. Closed
    /// form, so the fixture needs to carry only the expected output.
    fn reference_waveform(samples: usize, seconds: f64) -> Vec<f32> {
        (0..samples)
            .map(|n| {
                let t = n as f64 / SAMPLE_RATE as f64;
                let freq = 200.0 + (3000.0 - 200.0) * (t / seconds);
                let envelope = 0.5 * (1.0 + (2.0 * std::f64::consts::PI * 3.0 * t).sin());
                (envelope * (2.0 * std::f64::consts::PI * freq * t).sin()) as f32
            })
            .collect()
    }

    struct Reference {
        frames: usize,
        mel_bins: usize,
        features: Vec<f32>,
        waveform: Vec<f32>,
    }

    /// The fixture is the output of the real `CohereAsrFeatureExtractor`, so this
    /// test compares against transformers rather than against itself.
    fn load_reference() -> Reference {
        use base64::{engine::general_purpose::STANDARD, Engine as _};
        let raw = include_str!("../fixtures/cohere-fbank-reference.json");
        let parsed: serde_json::Value =
            serde_json::from_str(raw).expect("the fixture is valid JSON");
        let decoded = STANDARD
            .decode(
                parsed["features_f32_le_base64"]
                    .as_str()
                    .expect("the fixture carries base64 features"),
            )
            .expect("the fixture's features decode");
        let features = decoded
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect();
        let samples = parsed["samples"].as_u64().expect("samples") as usize;
        let seconds = parsed["seconds"].as_f64().expect("seconds");
        Reference {
            frames: parsed["frames"].as_u64().expect("frames") as usize,
            mel_bins: parsed["mel_bins"].as_u64().expect("mel_bins") as usize,
            features,
            waveform: reference_waveform(samples, seconds),
        }
    }

    #[test]
    fn matches_the_reference_extractor_element_for_element() {
        let reference = load_reference();
        assert_eq!(reference.mel_bins, MEL_BINS);

        let produced = CohereFbank::new().features_for_chunk(&reference.waveform);
        assert_eq!(
            produced.len(),
            reference.frames,
            "frame count differs from the reference"
        );

        let mut worst = 0.0f32;
        let mut worst_at = (0usize, 0usize);
        for (index, frame) in produced.iter().enumerate() {
            assert_eq!(frame.len(), MEL_BINS);
            for (bin, &value) in frame.iter().enumerate() {
                let expected = reference.features[index * MEL_BINS + bin];
                let error = (value - expected).abs();
                if error > worst {
                    worst = error;
                    worst_at = (index, bin);
                }
            }
        }
        // The STFT and the mel matmul are float-order sensitive, so this is a
        // tolerance rather than equality — but a front end that is actually wrong
        // (periodic window, HTK mel, magnitude instead of power, N instead of
        // N-1) misses by whole units, not by thousandths.
        println!("worst deviation from the reference: {worst:.8}");
        assert!(
            worst < 2e-3,
            "worst deviation {worst:.6} at frame {} bin {} exceeds tolerance",
            worst_at.0,
            worst_at.1
        );
    }

    #[test]
    #[ignore = "diagnostic"]
    fn diagnose_reference_deviation() {
        let reference = load_reference();
        let produced = CohereFbank::new().features_for_chunk(&reference.waveform);
        println!(
            "frames: produced {} reference {}",
            produced.len(),
            reference.frames
        );
        for (index, frame) in produced.iter().enumerate() {
            let mut worst = 0.0f32;
            let mut at = 0usize;
            for (bin, &value) in frame.iter().enumerate() {
                let e = (value - reference.features[index * MEL_BINS + bin]).abs();
                if e > worst {
                    worst = e;
                    at = bin;
                }
            }
            if worst > 1e-3 || index >= produced.len() - 2 || index < 2 {
                println!("  frame {index:>3}: worst {worst:.6} at bin {at}");
            }
        }
    }

    /// The second fixture is real speech rather than a synthetic sweep, so a match
    /// cannot be an artefact of one signal's spectral shape. Its input samples are
    /// embedded because speech has no closed form.
    #[test]
    fn matches_the_reference_extractor_on_real_speech() {
        use base64::{engine::general_purpose::STANDARD, Engine as _};
        let raw = include_str!("../fixtures/cohere-fbank-speech-reference.json");
        let parsed: serde_json::Value = serde_json::from_str(raw).expect("valid JSON");
        let decode = |key: &str| -> Vec<f32> {
            STANDARD
                .decode(parsed[key].as_str().expect("base64 field"))
                .expect("decodes")
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        };
        let samples = decode("samples_f32_le_base64");
        let expected = decode("features_f32_le_base64");
        let frames = parsed["frames"].as_u64().expect("frames") as usize;

        let produced = CohereFbank::new().features_for_chunk(&samples);
        assert_eq!(produced.len(), frames);
        let mut worst = 0.0f32;
        for (index, frame) in produced.iter().enumerate() {
            for (bin, &value) in frame.iter().enumerate() {
                worst = worst.max((value - expected[index * MEL_BINS + bin]).abs());
            }
        }
        println!("worst deviation on real speech: {worst:.8}");
        assert!(worst < 2e-3, "worst deviation {worst:.6} exceeds tolerance");
    }

    #[test]
    fn the_frame_count_follows_centred_stft_padding() {
        // 0.5 s at 16 kHz, padded by N_FFT/2 each side: 1 + (8000 + 512 - 512)/160.
        let produced = CohereFbank::new().features_for_chunk(&vec![0.0; 8000]);
        assert_eq!(produced.len(), 51);
    }

    #[test]
    fn per_bin_normalisation_centres_each_bin_not_each_frame() {
        // Normalising across bins instead of across time is an easy transposition
        // to make and leaves plausible-looking features.
        let fbank = CohereFbank::new();
        let frames = fbank.features_for_chunk(&reference_waveform(8000, 0.5));
        let valid = valid_frame_count(8000);
        for bin in 0..MEL_BINS {
            let mean: f32 =
                frames[..valid].iter().map(|frame| frame[bin]).sum::<f32>() / valid as f32;
            assert!(mean.abs() < 1e-3, "bin {bin} has mean {mean}");
        }
    }

    #[test]
    fn the_window_is_symmetric_and_centred() {
        let window = centred_hann_window();
        assert_eq!(window.len(), N_FFT);
        let offset = (N_FFT - WIN_LENGTH) / 2;
        // Zero outside the 400-point span, and symmetric within it — a periodic
        // window would not start and end at exactly zero.
        assert!(window[..offset].iter().all(|&value| value == 0.0));
        assert!(window[offset + WIN_LENGTH..]
            .iter()
            .all(|&value| value == 0.0));
        assert!(window[offset].abs() < 1e-7, "{}", window[offset]);
        assert!(
            window[offset + WIN_LENGTH - 1].abs() < 1e-7,
            "{}",
            window[offset + WIN_LENGTH - 1]
        );
        for index in 0..WIN_LENGTH / 2 {
            let left = window[offset + index];
            let right = window[offset + WIN_LENGTH - 1 - index];
            assert!((left - right).abs() < 1e-6, "asymmetric at {index}");
        }
    }

    #[test]
    fn preemphasis_leaves_the_first_sample_and_differences_the_rest() {
        let out = preemphasise(&[1.0, 2.0, 3.0]);
        assert_eq!(out[0], 1.0);
        assert!((out[1] - (2.0 - 0.97)).abs() < 1e-6, "{}", out[1]);
        assert!((out[2] - (3.0 - 0.97 * 2.0)).abs() < 1e-6, "{}", out[2]);
        assert!(preemphasise(&[]).is_empty());
    }

    #[test]
    fn the_mel_scale_is_slaney_rather_than_htk() {
        // Slaney is linear below 1 kHz at 3 mel per 200 Hz; HTK would give
        // ~999.99 mel at 1 kHz instead of exactly 15.
        assert!((hz_to_mel(1000.0) - 15.0).abs() < 1e-9);
        assert!((hz_to_mel(200.0) - 3.0).abs() < 1e-9);
        assert!((mel_to_hz(hz_to_mel(4321.0)) - 4321.0).abs() < 1e-6);
        assert!((mel_to_hz(hz_to_mel(123.0)) - 123.0).abs() < 1e-9);
    }

    #[test]
    fn short_audio_is_one_chunk_and_long_audio_splits_at_a_quiet_point() {
        let short = vec![0.1f32; SAMPLE_RATE as usize * 10];
        assert_eq!(split_into_chunks(&short).len(), 1);

        // 40 s of tone with a deliberate silence inside the search window that
        // precedes the 35 s boundary; the split must land on it.
        let mut long = vec![0.5f32; SAMPLE_RATE as usize * 40];
        let silence_at = SAMPLE_RATE as usize * 32;
        for sample in &mut long[silence_at..silence_at + MIN_ENERGY_WINDOW] {
            *sample = 0.0;
        }
        let chunks = split_into_chunks(&long);
        assert_eq!(chunks.len(), 2, "40 s should split once");
        assert_eq!(
            chunks[0].len(),
            silence_at,
            "the cut should land on the silent window"
        );
    }

    #[test]
    fn a_single_frame_does_not_produce_nan() {
        // N-1 would divide by zero; the reference yields NaN here and we do not.
        let fbank = CohereFbank::new();
        let frames = fbank.features_for_chunk(&[0.0; 16]);
        for frame in &frames {
            assert!(frame.iter().all(|value| value.is_finite()), "{frame:?}");
        }
    }
}
