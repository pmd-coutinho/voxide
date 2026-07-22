use std::{fs::File, io::ErrorKind, path::Path, process::Command};

use crate::audio::CapturedAudio;
use symphonia::{
    core::{
        audio::SampleBuffer, codecs::DecoderOptions, errors::Error as SymphoniaError,
        formats::FormatOptions, io::MediaSourceStream, meta::MetadataOptions, probe::Hint,
    },
    default::{get_codecs, get_probe},
};

pub const TRANSCRIPTION_CHUNK_SECONDS: f64 = 20.0 * 60.0;

pub fn file_duration_ms(path: &Path) -> Result<u64, String> {
    ensure_supported(path)?;
    if is_wav(path) {
        let reader = hound::WavReader::open(path)
            .map_err(|error| format!("Could not read WAV metadata: {error}"))?;
        let spec = reader.spec();
        if spec.sample_rate == 0 {
            return Err("The WAV file has an invalid sample rate".into());
        }
        return Ok((reader.duration() as u64 * 1000) / spec.sample_rate as u64);
    }

    let ffprobe_problem = match Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(path)
        .output()
    {
        Ok(output) if output.status.success() => {
            match String::from_utf8_lossy(&output.stdout)
                .trim()
                .parse::<f64>()
            {
                Ok(seconds) if seconds.is_finite() && seconds > 0.0 => {
                    return Ok((seconds * 1000.0).round() as u64);
                }
                _ => "FFprobe did not return a valid media duration".to_string(),
            }
        }
        Ok(output) => format!(
            "FFprobe could not read this media file: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ),
        Err(error) => format!("FFprobe is unavailable ({error})"),
    };
    symphonia_file_duration_ms(path).map_err(|symphonia_error| {
        format!(
            "Could not inspect this media file: {ffprobe_problem}. The built-in audio decoder also could not read it ({symphonia_error}). Install FFmpeg to transcribe this media."
        )
    })
}

pub fn decode_audio_segment(
    path: &Path,
    start_seconds: f64,
    duration_seconds: f64,
) -> Result<CapturedAudio, String> {
    ensure_supported(path)?;
    if is_wav(path) {
        return decode_wav_segment(path, start_seconds, duration_seconds);
    }

    let ffmpeg_problem = match Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-ss", &start_seconds.to_string()])
        .args(["-t", &duration_seconds.to_string(), "-i"])
        .arg(path)
        .args([
            "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1",
        ])
        .output()
    {
        Ok(output) if output.status.success() => {
            let samples = output
                .stdout
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                .map(|sample| if sample.is_finite() { sample } else { 0.0 })
                .collect::<Vec<_>>();
            if samples.is_empty() {
                "FFmpeg returned no audio for the selected media segment".to_string()
            } else {
                return Ok(CapturedAudio {
                    duration_ms: (samples.len() as u64 * 1000) / 16_000,
                    samples,
                    sample_rate: 16_000,
                    channels: 1,
                });
            }
        }
        Ok(output) => format!(
            "FFmpeg could not decode this media file: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ),
        Err(error) => format!("FFmpeg is unavailable ({error})"),
    };
    decode_symphonia_segment(path, start_seconds, duration_seconds).map_err(|symphonia_error| {
        format!(
            "Could not decode this media file: {ffmpeg_problem}. The built-in audio decoder also could not read it ({symphonia_error}). Install FFmpeg to transcribe this media."
        )
    })
}

fn symphonia_file_duration_ms(path: &Path) -> Result<u64, String> {
    let mut format = symphonia_format(path)?;
    let track = format
        .default_track()
        .ok_or("The media file does not contain a supported audio track")?;
    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or("The media audio track has no sample rate")?;
    let declared_frames = track.codec_params.n_frames;
    let frames = declared_frames.unwrap_or_else(|| {
        let mut packet_frames = 0_u64;
        loop {
            match format.next_packet() {
                Ok(packet) if packet.track_id() == track_id => {
                    packet_frames = packet_frames.saturating_add(packet.dur());
                }
                Ok(_) => {}
                Err(SymphoniaError::IoError(error)) if error.kind() == ErrorKind::UnexpectedEof => {
                    break;
                }
                Err(_) => break,
            }
        }
        packet_frames
    });
    if sample_rate == 0 || frames == 0 {
        return Err("The media file has no readable audio duration".into());
    }
    Ok((frames.saturating_mul(1_000) / sample_rate as u64).max(1))
}

fn decode_symphonia_segment(
    path: &Path,
    start_seconds: f64,
    duration_seconds: f64,
) -> Result<CapturedAudio, String> {
    let mut format = symphonia_format(path)?;
    let track = format
        .default_track()
        .ok_or("The media file does not contain a supported audio track")?;
    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or("The media audio track has no sample rate")?;
    let channels = track
        .codec_params
        .channels
        .ok_or("The media audio track has no channel layout")?
        .count() as u16;
    if sample_rate == 0 || channels == 0 {
        return Err("The media audio track has an invalid format".into());
    }
    let mut decoder = get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|error| {
            format!("The built-in decoder cannot initialize this audio track: {error}")
        })?;
    let start_frame = (start_seconds.max(0.0) * sample_rate as f64).floor() as u64;
    let requested_frames = (duration_seconds.max(0.0) * sample_rate as f64).ceil() as u64;
    let end_frame = start_frame.saturating_add(requested_frames);
    let mut decoded_frames = 0_u64;
    let mut samples = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(error)) if error.kind() == ErrorKind::UnexpectedEof => {
                break
            }
            Err(SymphoniaError::ResetRequired) => {
                return Err("The media stream requires an unsupported decoder reset".into())
            }
            Err(error) => return Err(format!("Could not read the media audio stream: {error}")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(audio) => {
                let mut sample_buffer =
                    SampleBuffer::<f32>::new(audio.capacity() as u64, *audio.spec());
                sample_buffer.copy_interleaved_ref(audio);
                let decoded_samples = sample_buffer.samples();
                let frame_count = decoded_samples.len() as u64 / channels as u64;
                let next_frame = decoded_frames.saturating_add(frame_count);
                let selection_start = start_frame.max(decoded_frames).min(next_frame);
                let selection_end = end_frame.max(decoded_frames).min(next_frame);
                if selection_end > selection_start {
                    let sample_start =
                        (selection_start - decoded_frames).saturating_mul(channels as u64) as usize;
                    let sample_end =
                        (selection_end - decoded_frames).saturating_mul(channels as u64) as usize;
                    samples.extend_from_slice(&decoded_samples[sample_start..sample_end]);
                }
                decoded_frames = next_frame;
                if decoded_frames >= end_frame {
                    break;
                }
            }
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(error) => return Err(format!("Could not decode the media audio stream: {error}")),
        }
    }
    if samples.is_empty() {
        return Err("The selected media segment contains no audio".into());
    }
    Ok(CapturedAudio {
        duration_ms: ((samples.len() as u64 / channels as u64) * 1_000) / sample_rate as u64,
        samples,
        sample_rate,
        channels,
    })
}

fn symphonia_format(
    path: &Path,
) -> Result<Box<dyn symphonia::core::formats::FormatReader>, String> {
    let file = File::open(path).map_err(|error| format!("Could not open media: {error}"))?;
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|extension| extension.to_str()) {
        hint.with_extension(extension);
    }
    get_probe()
        .format(
            &hint,
            MediaSourceStream::new(Box::new(file), Default::default()),
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map(|probed| probed.format)
        .map_err(|error| format!("The built-in decoder could not identify this media: {error}"))
}

fn decode_wav_segment(
    path: &Path,
    start_seconds: f64,
    duration_seconds: f64,
) -> Result<CapturedAudio, String> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|error| format!("Could not open WAV audio: {error}"))?;
    let spec = reader.spec();
    if spec.sample_rate == 0 || spec.channels == 0 {
        return Err("The WAV file has an invalid format".into());
    }
    let start_frame = (start_seconds.max(0.0) * spec.sample_rate as f64) as u32;
    let frame_count = (duration_seconds.max(0.0) * spec.sample_rate as f64).ceil() as usize;
    reader
        .seek(start_frame)
        .map_err(|error| format!("Could not seek within WAV audio: {error}"))?;
    let sample_count = frame_count.saturating_mul(spec.channels as usize);
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .take(sample_count)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("Could not decode WAV samples: {error}"))?,
        hound::SampleFormat::Int => {
            let scale = (1_i64 << (spec.bits_per_sample.saturating_sub(1))).max(1) as f32;
            reader
                .samples::<i32>()
                .take(sample_count)
                .map(|sample| sample.map(|sample| sample as f32 / scale))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| format!("Could not decode WAV samples: {error}"))?
        }
    };
    if samples.is_empty() {
        return Err("The selected media segment contains no audio".into());
    }
    Ok(CapturedAudio {
        duration_ms: ((samples.len() as u64 / spec.channels as u64) * 1000)
            / spec.sample_rate as u64,
        samples,
        sample_rate: spec.sample_rate,
        channels: spec.channels,
    })
}

fn ensure_supported(path: &Path) -> Result<(), String> {
    if !path.is_file() {
        return Err("The selected media file no longer exists or is not readable".into());
    }
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if supported_extensions().contains(&extension.as_str()) {
        Ok(())
    } else {
        Err(format!(
            "Format .{extension} is not supported. Choose WAV, MP3, M4A, OGG, FLAC, MP4, MOV, WebM, or another FFmpeg-readable audio/video file."
        ))
    }
}

fn is_wav(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("wav"))
}

fn supported_extensions() -> &'static [&'static str] {
    &[
        "wav", "mp3", "m4a", "aac", "ogg", "oga", "opus", "flac", "wma", "aiff", "aif", "mp4",
        "m4v", "mov", "webm", "mkv", "avi",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::{engine::general_purpose::STANDARD, Engine as _};

    fn spoken_fixture_path() -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "voxide-spoken-fixture-{}-{}.wav",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let encoded = include_str!("../fixtures/spoken-a-espeak-ng-8khz.wav.b64").trim();
        std::fs::write(
            &path,
            STANDARD
                .decode(encoded)
                .expect("checked-in WAV fixture should be valid base64"),
        )
        .expect("spoken fixture should be written");
        path
    }

    #[test]
    fn supports_common_audio_and_video_extensions() {
        assert!(supported_extensions().contains(&"wav"));
        assert!(supported_extensions().contains(&"mp3"));
        assert!(supported_extensions().contains(&"mp4"));
    }

    #[test]
    fn reads_a_wav_segment_without_ffmpeg() {
        let path =
            std::env::temp_dir().join(format!("voxide-media-test-{}.wav", std::process::id()));
        let mut writer = hound::WavWriter::create(
            &path,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .expect("temporary WAV should be created");
        for _ in 0..16_000 {
            writer.write_sample(1_000_i16).expect("sample should write");
        }
        writer.finalize().expect("WAV should finalize");

        assert_eq!(
            file_duration_ms(&path).expect("duration should read"),
            1_000
        );
        let audio = decode_audio_segment(&path, 0.25, 0.5).expect("segment should decode");
        assert_eq!(audio.sample_rate, 16_000);
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.samples.len(), 8_000);
        assert_eq!(
            symphonia_file_duration_ms(&path).expect("built-in duration should read"),
            1_000
        );
        let portable_audio =
            decode_symphonia_segment(&path, 0.25, 0.5).expect("built-in segment should decode");
        assert_eq!(portable_audio.sample_rate, 16_000);
        assert_eq!(portable_audio.channels, 1);
        assert_eq!(portable_audio.samples.len(), 8_000);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn reads_checked_in_spoken_audio_fixture_without_ffmpeg() {
        let path = spoken_fixture_path();
        let audio = decode_audio_segment(&path, 0.0, 1.0)
            .expect("checked-in spoken WAV fixture should decode");

        assert_eq!(audio.sample_rate, 8_000);
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.samples.len(), 554);
        assert_eq!(audio.duration_ms, 69);
        assert!(
            audio.samples.iter().any(|sample| sample.abs() > 0.1),
            "spoken fixture must contain audible, non-silent samples"
        );
        let _ = std::fs::remove_file(path);
    }
}
