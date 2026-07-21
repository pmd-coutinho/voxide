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
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
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
    }
}
