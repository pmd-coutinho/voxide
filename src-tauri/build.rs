use std::{env, path::PathBuf};

fn main() {
    tauri_build::build();

    // whisper-rs-sys' CUDA feature links CUDA libraries by name but only
    // searches system toolkit locations. Voxide supports a user-local CUDA
    // toolkit (including the one staged beside the Python venv), so make the
    // chosen toolkit visible to rust-lld and embed its runtime search path in
    // locally built CUDA binaries.
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDAToolkit_ROOT");
    println!("cargo:rerun-if-env-changed=PARAKEET_CUDA_LIB_DIRS");
    println!("cargo:rerun-if-env-changed=SHERPA_ONNX_LIB_DIR");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let parakeet_enabled = env::var_os("CARGO_FEATURE_PARAKEET").is_some();
    let cuda_home = env::var_os("CUDA_HOME")
        .or_else(|| env::var_os("CUDAToolkit_ROOT"))
        .map(PathBuf::from);
    let Some(cuda_home) = cuda_home else {
        println!(
            "cargo:warning=CUDA feature enabled without CUDA_HOME; relying on system linker paths"
        );
        return;
    };
    let library_directory = [cuda_home.join("lib64"), cuda_home.join("lib")]
        .into_iter()
        .find(|path| path.is_dir());
    let Some(library_directory) = library_directory else {
        println!(
            "cargo:warning=CUDA toolkit has no lib64 or lib directory: {}",
            cuda_home.display()
        );
        return;
    };
    println!(
        "cargo:rustc-link-search=native={}",
        library_directory.display()
    );
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath,{}",
            library_directory.display()
        );
        // Parakeet's official ONNX Runtime release is built against CUDA 12
        // and cuDNN 9. These can be supplied by NVIDIA's pip runtime wheels
        // on a machine whose Whisper build uses a different CUDA toolkit.
        // DT_RPATH (rather than RUNPATH) intentionally applies to the CUDA
        // provider's indirect dependencies as well.
        if parakeet_enabled {
            // sherpa-onnx copies this runtime beside Cargo binaries, but its
            // dependency build script cannot add an rpath to Voxide's final
            // link. Keep the explicit source location too, so a no-bundle
            // release binary and desktop trigger can start without an env var.
            let runtime = env::var_os("SHERPA_ONNX_LIB_DIR")
                .map(PathBuf::from)
                .filter(|path| path.is_dir())
                .expect("Parakeet CUDA builds require SHERPA_ONNX_LIB_DIR to point to the official GPU runtime's lib directory; run scripts/setup-parakeet-cuda-runtime.sh first");
            println!(
                "cargo:rustc-link-arg=-Wl,--disable-new-dtags,-rpath,{}",
                runtime.display()
            );
            let directories = env::var_os("PARAKEET_CUDA_LIB_DIRS")
                .map(|paths| env::split_paths(&paths).filter(|path| path.is_dir()).collect::<Vec<_>>())
                .filter(|paths| !paths.is_empty())
                .expect("Parakeet CUDA builds require PARAKEET_CUDA_LIB_DIRS with the CUDA 12/cuDNN 9 library directory; run scripts/setup-parakeet-cuda-runtime.sh first");
            for directory in directories {
                println!(
                    "cargo:rustc-link-arg=-Wl,--disable-new-dtags,-rpath,{}",
                    directory.display()
                );
            }
        }
    }
}
