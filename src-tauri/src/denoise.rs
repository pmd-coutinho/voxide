//! GTCRN speech enhancement, applied ahead of recognition.
//!
//! A ~520 KB ONNX model (48k parameters) that suppresses background noise and
//! speaker bleed-through frame by frame. Cheap enough to run on the CPU in front
//! of any engine, which is the point: it helps a bad microphone regardless of
//! whether the recognizer is Whisper on CPU or Parakeet on a GPU.
//!
//! **Opt-in on purpose.** A denoiser that strips phonemes along with noise makes
//! transcription *worse*, and that failure does not show up as an error — it
//! shows up as words quietly going missing. Nothing here is enabled by default,
//! and [`Enhancer::enhance`] returns the input untouched rather than a
//! half-processed signal if anything goes wrong.
//!
//! ## Shape
//!
//! The model is streaming: one STFT frame in, one enhanced frame out, plus three
//! recurrent caches threaded from each call to the next.
//!
//! ```text
//! mix          f32 [1, 257, 1, 2]        # one frame, 257 bins, (re, im)
//! conv_cache   f32 [2, 1, 16, 16, 33]
//! tra_cache    f32 [2, 3, 1, 1, 16]
//! inter_cache  f32 [2, 1, 33, 16]
//!   ->
//! enh          f32 [1, 257, 1, 2]
//! *_cache_out  (same shapes, fed back next frame)
//! ```
//!
//! Every dimension is static, so there is no shape negotiation — but the caches
//! carry state across frames, which means a frame must never be reordered or
//! skipped, and a fresh utterance must start from zeroed caches.

// Nothing calls this yet. The enhancer is verified — it measurably raises SNR
// against a known clean signal — but wiring it into the capture path needs a
// settings toggle and a UI affordance, and it must stay opt-in: a denoiser that
// strips phonemes along with noise degrades transcription silently. Landing it
// verified and unused is the same trade as `cohere_fbank`.
#![allow(dead_code)]

use std::{path::Path, sync::Mutex};

use ort::{session::Session, value::Tensor};
use rustfft::{num_complex::Complex32, num_traits::Zero, Fft, FftPlanner};
use std::sync::Arc;

/// Rate the model was trained at. Callers resample first.
pub const SAMPLE_RATE: u32 = 16_000;

const N_FFT: usize = 512;
const HOP: usize = 256;
const BINS: usize = N_FFT / 2 + 1;

// Recurrent cache element counts, flattened.
const CONV_CACHE: [usize; 5] = [2, 1, 16, 16, 33];
const TRA_CACHE: [usize; 5] = [2, 3, 1, 1, 16];
const INTER_CACHE: [usize; 4] = [2, 1, 33, 16];

fn product(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Loaded GTCRN model plus the FFT plans it needs.
pub struct Enhancer {
    /// `Mutex` because `ort` sessions are not `Sync` for inference and a
    /// dictation may touch this from more than one task.
    session: Mutex<Session>,
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
    /// sqrt-Hann. Used for both analysis and synthesis: squared it is a periodic
    /// Hann, which sums to exactly 1 at 50% overlap, so overlap-add reconstructs
    /// the signal without an amplitude correction pass.
    window: Vec<f32>,
}

impl Enhancer {
    /// Loads `gtcrn_simple.onnx`.
    pub fn load(model_path: &Path) -> Result<Self, String> {
        if !model_path.is_file() {
            return Err(format!(
                "The speech enhancement model is not installed: {}",
                model_path.display()
            ));
        }
        let session = Session::builder()
            .map_err(|error| format!("Could not create an ONNX session builder: {error}"))?
            .commit_from_file(model_path)
            .map_err(|error| {
                format!(
                    "Could not load the speech enhancement model {}: {error}",
                    model_path.display()
                )
            })?;
        let mut planner = FftPlanner::new();
        Ok(Self {
            session: Mutex::new(session),
            forward: planner.plan_fft_forward(N_FFT),
            inverse: planner.plan_fft_inverse(N_FFT),
            window: sqrt_hann_window(),
        })
    }

    /// Enhances `samples` (16 kHz mono) and returns a signal of the same length.
    ///
    /// Errors are deliberately not propagated: a failure mid-stream would
    /// otherwise replace a usable recording with a partly-processed one. The
    /// caller gets the original audio and a log line instead.
    pub fn enhance_or_passthrough(&self, samples: &[f32]) -> Vec<f32> {
        match self.enhance(samples) {
            Ok(enhanced) => enhanced,
            Err(error) => {
                crate::debug_log::append(&format!(
                    "speech enhancement skipped ({} samples): {error}",
                    samples.len()
                ));
                samples.to_vec()
            }
        }
    }

    pub fn enhance(&self, samples: &[f32]) -> Result<Vec<f32>, String> {
        if samples.len() < N_FFT {
            // Too short for even one frame; nothing to do and nothing to fail.
            return Ok(samples.to_vec());
        }
        let mut session = self
            .session
            .lock()
            .map_err(|_| "The speech enhancement lock was poisoned".to_string())?;

        // Caches start zeroed: each utterance is an independent stream, and
        // carrying state across utterances would leak one recording into the next.
        let mut conv = vec![0.0f32; product(&CONV_CACHE)];
        let mut tra = vec![0.0f32; product(&TRA_CACHE)];
        let mut inter = vec![0.0f32; product(&INTER_CACHE)];

        let frames = 1 + (samples.len() - N_FFT) / HOP;
        let mut output = vec![0.0f32; samples.len()];
        let mut buffer = vec![Complex32::zero(); N_FFT];

        for index in 0..frames {
            let start = index * HOP;
            for (slot, (sample, weight)) in buffer
                .iter_mut()
                .zip(samples[start..start + N_FFT].iter().zip(&self.window))
            {
                *slot = Complex32::new(sample * weight, 0.0);
            }
            self.forward.process(&mut buffer);

            // Interleaved (re, im) over the non-redundant half.
            let mut mix = Vec::with_capacity(BINS * 2);
            for bin in &buffer[..BINS] {
                mix.push(bin.re);
                mix.push(bin.im);
            }

            let outputs = session
                .run(ort::inputs![
                    "mix" => Tensor::from_array(([1usize, BINS, 1, 2], mix)).map_err(mapper)?,
                    "conv_cache" => Tensor::from_array((CONV_CACHE, conv.clone())).map_err(mapper)?,
                    "tra_cache" => Tensor::from_array((TRA_CACHE, tra.clone())).map_err(mapper)?,
                    "inter_cache" => Tensor::from_array((INTER_CACHE, inter.clone())).map_err(mapper)?,
                ])
                .map_err(|error| format!("Speech enhancement inference failed: {error}"))?;

            let enhanced = outputs["enh"]
                .try_extract_tensor::<f32>()
                .map_err(mapper)?
                .1
                .to_vec();
            if enhanced.len() != BINS * 2 {
                return Err(format!(
                    "The enhancement model returned {} values, expected {}",
                    enhanced.len(),
                    BINS * 2
                ));
            }
            conv = outputs["conv_cache_out"]
                .try_extract_tensor::<f32>()
                .map_err(mapper)?
                .1
                .to_vec();
            tra = outputs["tra_cache_out"]
                .try_extract_tensor::<f32>()
                .map_err(mapper)?
                .1
                .to_vec();
            inter = outputs["inter_cache_out"]
                .try_extract_tensor::<f32>()
                .map_err(mapper)?
                .1
                .to_vec();
            drop(outputs);

            // Rebuild the full spectrum: the model returns the non-redundant
            // half, and the inverse transform needs the conjugate mirror.
            for bin in 0..BINS {
                buffer[bin] = Complex32::new(enhanced[bin * 2], enhanced[bin * 2 + 1]);
            }
            for bin in 1..BINS - 1 {
                buffer[N_FFT - bin] = buffer[bin].conj();
            }
            self.inverse.process(&mut buffer);

            // rustfft does not normalise, so divide by N; then the synthesis
            // window, then overlap-add.
            let scale = 1.0 / N_FFT as f32;
            for (offset, (value, weight)) in buffer.iter().zip(&self.window).enumerate() {
                if let Some(slot) = output.get_mut(start + offset) {
                    *slot += value.re * scale * weight;
                }
            }
        }

        // The tail past the last full frame was never covered by a window, so it
        // is copied through rather than left as silence.
        let covered = (frames - 1) * HOP + N_FFT;
        if covered < samples.len() {
            output[covered..].copy_from_slice(&samples[covered..]);
        }
        Ok(output)
    }
}

fn mapper<E: std::fmt::Display>(error: E) -> String {
    format!("Speech enhancement tensor error: {error}")
}

/// Square root of a periodic Hann window.
///
/// Periodic rather than symmetric because periodic Hann is what sums to unity at
/// 50% overlap; using it for analysis *and* synthesis means the two square to a
/// Hann and overlap-add is exactly unity gain.
fn sqrt_hann_window() -> Vec<f32> {
    (0..N_FFT)
        .map(|index| {
            let phase = 2.0 * std::f32::consts::PI * index as f32 / N_FFT as f32;
            (0.5 - 0.5 * phase.cos()).max(0.0).sqrt()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_path() -> std::path::PathBuf {
        std::env::var("VOXIDE_GTCRN_MODEL")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                directories::ProjectDirs::from("dev", "pmdcoutinho", "Voxide")
                    .map(|d| d.data_local_dir().join("models").join("gtcrn_simple.onnx"))
                    .unwrap_or_default()
            })
    }

    /// A 300 Hz tone under a 4 Hz envelope: periodic, so a denoiser should keep
    /// it, and narrowband, so broadband noise is clearly separable from it.
    fn clean_signal(samples: usize) -> Vec<f32> {
        (0..samples)
            .map(|n| {
                let t = n as f64 / SAMPLE_RATE as f64;
                let envelope = 0.5 * (1.0 + (2.0 * std::f64::consts::PI * 4.0 * t).sin());
                (0.4 * envelope * (2.0 * std::f64::consts::PI * 300.0 * t).sin()) as f32
            })
            .collect()
    }

    /// Deterministic pseudo-noise. A plain LCG rather than a real RNG so the test
    /// is reproducible and needs no dependency.
    fn noise(samples: usize, amplitude: f32) -> Vec<f32> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        (0..samples)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let unit = ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
                unit * amplitude
            })
            .collect()
    }

    fn snr_db(clean: &[f32], measured: &[f32]) -> f32 {
        let signal: f32 = clean.iter().map(|v| v * v).sum();
        let error: f32 = clean
            .iter()
            .zip(measured)
            .map(|(c, m)| (c - m) * (c - m))
            .sum();
        10.0 * (signal / error.max(f32::MIN_POSITIVE)).log10()
    }

    #[test]
    fn the_window_reconstructs_at_unity_gain() {
        // sqrt-Hann used for analysis and synthesis squares to a Hann, which sums
        // to 1 at 50% overlap. If this drifts, enhanced audio comes back quieter
        // or louder than it went in, which looks like the model misbehaving.
        let window = sqrt_hann_window();
        assert_eq!(window.len(), N_FFT);
        for offset in N_FFT..(N_FFT * 3) {
            let mut sum = 0.0f32;
            let mut start = 0usize;
            while start + N_FFT <= N_FFT * 4 {
                if offset >= start && offset - start < N_FFT {
                    let w = window[offset - start];
                    sum += w * w;
                }
                start += HOP;
            }
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "overlap-add gain {sum} at offset {offset}"
            );
        }
    }

    #[test]
    fn short_input_passes_through_unchanged() {
        // Below one frame there is nothing to transform; the caller must still get
        // its audio back rather than an error or silence.
        let path = model_path();
        if !path.is_file() {
            return;
        }
        let enhancer = Enhancer::load(&path).expect("model loads");
        let short = clean_signal(100);
        assert_eq!(enhancer.enhance(&short).expect("passthrough"), short);
    }

    #[test]
    fn a_missing_model_is_reported_rather_than_panicking() {
        let error = match Enhancer::load(Path::new("/nonexistent/gtcrn.onnx")) {
            Err(error) => error,
            Ok(_) => panic!("loading a nonexistent model must fail"),
        };
        assert!(error.contains("not installed"), "{error}");
    }

    /// The real check: enhancement must raise SNR against the known clean signal.
    /// This is the oracle for the whole STFT/iSTFT convention too — if the window,
    /// hop, scaling or conjugate mirror were wrong, the output would be garbled
    /// and SNR would fall rather than rise.
    #[test]
    fn enhancement_improves_signal_to_noise_ratio() {
        let path = model_path();
        if !path.is_file() {
            eprintln!("skipping: no model at {}", path.display());
            return;
        }
        let enhancer = Enhancer::load(&path).expect("model loads");
        let samples = SAMPLE_RATE as usize; // one second
        let clean = clean_signal(samples);
        let noise = noise(samples, 0.15);
        let noisy: Vec<f32> = clean.iter().zip(&noise).map(|(c, n)| c + n).collect();

        let before = snr_db(&clean, &noisy);
        let enhanced = enhancer.enhance(&noisy).expect("enhancement runs");
        assert_eq!(enhanced.len(), noisy.len());
        let after = snr_db(&clean, &enhanced);

        println!(
            "SNR before {before:.2} dB -> after {after:.2} dB (gain {:.2} dB)",
            after - before
        );
        assert!(
            after > before,
            "enhancement lowered SNR: {before:.2} dB -> {after:.2} dB"
        );
    }
}
