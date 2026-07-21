//! macOS Speech.framework integration.
//!
//! The rest of the application uses a final-result interface so this module
//! deliberately uses `SFSpeechURLRecognitionRequest`: capture is still owned
//! by the portable CPAL layer, then macOS recognizes a temporary WAV file.

#[cfg(target_os = "macos")]
mod platform {
    use std::{
        fs,
        path::{Path, PathBuf},
        sync::mpsc,
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    use block2::RcBlock;
    use objc2::{rc::autoreleasepool, AnyThread};
    use objc2_foundation::{NSArray, NSError, NSLocale, NSString, NSURL};
    use objc2_speech::{
        SFSpeechRecognitionResult, SFSpeechRecognizer, SFSpeechRecognizerAuthorizationStatus,
        SFSpeechURLRecognitionRequest,
    };

    use crate::audio;

    const AUTHORIZATION_TIMEOUT: Duration = Duration::from_secs(30);
    // Speech.framework documents a one-minute audio-request limit. Leave a
    // little room for a final callback before returning control to the app.
    const RECOGNITION_TIMEOUT: Duration = Duration::from_secs(70);

    pub fn is_supported() -> bool {
        true
    }

    pub fn transcribe_samples(
        samples: &[f32],
        language: &str,
        contextual_words: &[String],
    ) -> Result<String, String> {
        let wav = audio::wav_bytes_from_16khz_mono(samples)?;
        let path = temporary_wav_path();
        fs::write(&path, wav)
            .map_err(|error| format!("Could not prepare audio for macOS Speech: {error}"))?;
        let result = transcribe_file(&path, language, contextual_words);
        let _ = fs::remove_file(path);
        result
    }

    pub fn transcribe_file(
        path: &Path,
        language: &str,
        contextual_words: &[String],
    ) -> Result<String, String> {
        if !path.is_file() {
            return Err("The audio file for macOS Speech is no longer available".into());
        }
        autoreleasepool(|_| recognize_file(path, language, contextual_words))
    }

    fn temporary_wav_path() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or_default();
        std::env::temp_dir().join(format!(
            "voxide-apple-speech-{}-{nonce}.wav",
            std::process::id()
        ))
    }

    fn ensure_authorized() -> Result<(), String> {
        let status = unsafe { SFSpeechRecognizer::authorizationStatus() };
        match status {
            SFSpeechRecognizerAuthorizationStatus::Authorized => Ok(()),
            SFSpeechRecognizerAuthorizationStatus::NotDetermined => {
                let (sender, receiver) = mpsc::sync_channel(1);
                let callback = RcBlock::new(
                    move |result: SFSpeechRecognizerAuthorizationStatus| {
                    let _ = sender.send(result);
                    },
                );
                unsafe { SFSpeechRecognizer::requestAuthorization(&callback) };
                match receiver.recv_timeout(AUTHORIZATION_TIMEOUT) {
                    Ok(SFSpeechRecognizerAuthorizationStatus::Authorized) => Ok(()),
                    Ok(SFSpeechRecognizerAuthorizationStatus::Denied) => Err(
                        "macOS Speech permission was denied. Enable Speech Recognition for Voxide in System Settings.".into(),
                    ),
                    Ok(SFSpeechRecognizerAuthorizationStatus::Restricted) => Err(
                        "macOS Speech Recognition is restricted on this device.".into(),
                    ),
                    Ok(_) => Err("macOS Speech authorization did not complete.".into()),
                    Err(_) => Err("Timed out waiting for macOS Speech authorization.".into()),
                }
            }
            SFSpeechRecognizerAuthorizationStatus::Denied => Err(
                "macOS Speech permission was denied. Enable Speech Recognition for Voxide in System Settings.".into(),
            ),
            SFSpeechRecognizerAuthorizationStatus::Restricted => {
                Err("macOS Speech Recognition is restricted on this device.".into())
            }
            _ => Err("macOS Speech authorization has an unknown status.".into()),
        }
    }

    fn recognize_file(
        path: &Path,
        language: &str,
        contextual_words: &[String],
    ) -> Result<String, String> {
        ensure_authorized()?;
        let recognizer = recognizer_for_language(language)?;
        if !unsafe { recognizer.isAvailable() } {
            return Err(
                "macOS Speech Recognition is temporarily unavailable for this language.".into(),
            );
        }

        let path_string = path.to_string_lossy();
        let url = NSURL::fileURLWithPath(&NSString::from_str(&path_string));
        let request = unsafe {
            SFSpeechURLRecognitionRequest::initWithURL(SFSpeechURLRecognitionRequest::alloc(), &url)
        };
        unsafe {
            request.setShouldReportPartialResults(false);
            if recognizer.supportsOnDeviceRecognition() {
                request.setRequiresOnDeviceRecognition(true);
            }
        }
        let words = contextual_words
            .iter()
            .map(|word| word.trim())
            .filter(|word| !word.is_empty())
            .take(100)
            .map(NSString::from_str)
            .collect::<Vec<_>>();
        if !words.is_empty() {
            let words = NSArray::from_retained_slice(&words);
            unsafe { request.setContextualStrings(&words) };
        }

        let (sender, receiver) = mpsc::sync_channel(1);
        let callback = RcBlock::new(
            move |result: *mut SFSpeechRecognitionResult, error: *mut NSError| {
                if let Some(error) = unsafe { error.as_ref() } {
                    let _ = sender.send(Err(format!(
                        "macOS Speech failed: {}",
                        error.localizedDescription()
                    )));
                    return;
                }
                let Some(result) = (unsafe { result.as_ref() }) else {
                    return;
                };
                if unsafe { result.isFinal() } {
                    let text = unsafe { result.bestTranscription().formattedString() }.to_string();
                    let _ = sender.send(Ok(text));
                }
            },
        );
        let task =
            unsafe { recognizer.recognitionTaskWithRequest_resultHandler(&request, &callback) };
        match receiver.recv_timeout(RECOGNITION_TIMEOUT) {
            Ok(result) => result.and_then(|text| {
                if text.trim().is_empty() {
                    Err("macOS Speech did not detect any speech in this recording.".into())
                } else {
                    Ok(text)
                }
            }),
            Err(_) => {
                unsafe { task.cancel() };
                Err("macOS Speech timed out before returning a final transcription.".into())
            }
        }
    }

    fn recognizer_for_language(
        language: &str,
    ) -> Result<objc2::rc::Retained<SFSpeechRecognizer>, String> {
        let language = language.trim();
        if language.is_empty() || language.eq_ignore_ascii_case("auto") {
            return Ok(unsafe { SFSpeechRecognizer::new() });
        }
        let locale = NSLocale::localeWithLocaleIdentifier(&NSString::from_str(language));
        unsafe { SFSpeechRecognizer::initWithLocale(SFSpeechRecognizer::alloc(), &locale) }
            .ok_or_else(|| format!("macOS Speech does not support the language '{language}'."))
    }
}

#[cfg(target_os = "macos")]
pub use platform::*;

#[cfg(not(target_os = "macos"))]
pub fn is_supported() -> bool {
    false
}

#[cfg(not(target_os = "macos"))]
pub fn transcribe_samples(
    _samples: &[f32],
    _language: &str,
    _contextual_words: &[String],
) -> Result<String, String> {
    Err("Apple Speech is available only on macOS. Select Whisper or a compatible cloud provider on this platform.".into())
}
