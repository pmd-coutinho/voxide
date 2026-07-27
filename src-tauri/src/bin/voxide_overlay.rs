//! The dictation overlay as a `wlr-layer-shell` surface, in its own process.
//!
//! One blocker, precisely located: this uses `delegate_compositor!`,
//! `delegate_output!`, `delegate_shm!` and `delegate_layer!`, which existed up to
//! smithay-client-toolkit 0.19 but were **removed by 0.21.1**. That version ships
//! only `delegate_dispatch!`, `delegate_registry!` and `registry_handlers!`, so
//! the per-protocol wiring has to be rewritten against 0.21's pattern — read its
//! `examples/simple_layer.rs`.
//!
//! Everything else here is believed correct and is the part worth keeping: layer
//! creation and anchoring, margins, the logical-pixel sizing that the webview
//! overlay got wrong, premultiplied ARGB8888 shm buffers, damage and frame
//! callbacks, and the meter drawing. It is gated behind the
//! `overlay-layer-shell` feature and a `required-features` bin target, so the
//! default build, the CUDA build and CI are all unaffected — verified.
//!
//! ## Why a separate binary
//!
//! Tauri links `webkit2gtk-4.1`, which is GTK 3, and GTK 3 cannot share a process
//! with GTK 4 — so `gtk4-layer-shell` is unusable from inside the app. A layer
//! surface is still the right primitive for an overlay: it cannot take keyboard
//! focus, the compositor positions it against the output instead of the app
//! guessing, and there are no webview DPI semantics to get wrong. Both of those
//! were real problems for the webview overlay this replaces.
//!
//! Smithay's client toolkit with a shared-memory buffer, rather than GTK 4 or
//! wgpu: the overlay is a level meter, so hand-drawing it costs less than
//! carrying a second toolkit or a GPU stack through packaging.
//!
//! ## Scope
//!
//! This is the surface and the level meter. Text rendering needs a font stack and
//! is not here yet; the IPC that will drive `--level` from the running app is the
//! next piece (the trigger socket in `crate::trigger` is the pattern to follow).
//! Run it standalone to see it:
//!
//! ```text
//! voxide-overlay --level 0.6 --seconds 3
//! ```

use std::time::{Duration, Instant};

use smithay_client_toolkit::reexports::client::{
    globals::registry_queue_init,
    protocol::{wl_output, wl_shm, wl_surface},
    Connection, QueueHandle,
};
use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState, FrameCallbackData},
    delegate_dispatch2, delegate_registry,
    output::{OutputHandler, OutputState},
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    shell::{
        wlr_layer::{
            Anchor, Layer, LayerShell, LayerShellHandler, LayerSurface, LayerSurfaceConfigure,
        },
        WaylandSurface,
    },
    shm::{slot::SlotPool, Shm, ShmHandler},
};

/// Logical size of the overlay. Logical, not physical: the compositor scales it,
/// which is the bug class that made the webview overlay wrong on scaled outputs.
const WIDTH: u32 = 420;
const HEIGHT: u32 = 96;
/// Distance from the bottom edge, matching where system OSDs sit.
const BOTTOM_MARGIN: i32 = 64;

const BARS: usize = 24;

fn main() {
    let mut level = 0.5f32;
    let mut seconds = 3u64;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--level" => {
                level = arguments
                    .next()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(level)
            }
            "--seconds" => {
                seconds = arguments
                    .next()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(seconds)
            }
            other => eprintln!("voxide-overlay: ignoring unknown argument {other}"),
        }
    }

    if let Err(error) = run(level.clamp(0.0, 1.0), Duration::from_secs(seconds)) {
        eprintln!("voxide-overlay: {error}");
        std::process::exit(1);
    }
}

fn run(level: f32, lifetime: Duration) -> Result<(), String> {
    let connection =
        Connection::connect_to_env().map_err(|error| format!("no Wayland display: {error}"))?;
    let (globals, mut queue) = registry_queue_init(&connection)
        .map_err(|error| format!("could not read the Wayland registry: {error}"))?;
    let handle = queue.handle();

    let compositor = CompositorState::bind(&globals, &handle)
        .map_err(|error| format!("wl_compositor is unavailable: {error}"))?;
    let layer_shell = LayerShell::bind(&globals, &handle)
        .map_err(|error| format!("this compositor does not implement wlr-layer-shell: {error}"))?;
    let shm =
        Shm::bind(&globals, &handle).map_err(|error| format!("wl_shm is unavailable: {error}"))?;

    let surface = compositor.create_surface(&handle);
    // Overlay layer so it sits above windows; no keyboard interactivity at all,
    // which is what stops it stealing focus mid-dictation.
    let layer =
        layer_shell.create_layer_surface(&handle, surface, Layer::Overlay, Some("voxide"), None);
    layer.set_anchor(Anchor::BOTTOM);
    layer.set_margin(0, 0, BOTTOM_MARGIN, 0);
    layer.set_size(WIDTH, HEIGHT);
    layer.commit();

    let pool = SlotPool::new(WIDTH as usize * HEIGHT as usize * 4, &shm)
        .map_err(|error| format!("could not allocate a shared-memory pool: {error}"))?;

    let mut state = Overlay {
        registry: RegistryState::new(&globals),
        outputs: OutputState::new(&globals, &handle),
        shm,
        pool,
        layer,
        width: WIDTH,
        height: HEIGHT,
        configured: false,
        closed: false,
        level,
    };

    let deadline = Instant::now() + lifetime;
    while !state.closed && Instant::now() < deadline {
        queue
            .blocking_dispatch(&mut state)
            .map_err(|error| format!("Wayland dispatch failed: {error}"))?;
    }
    Ok(())
}

struct Overlay {
    registry: RegistryState,
    outputs: OutputState,
    shm: Shm,
    pool: SlotPool,
    layer: LayerSurface,
    width: u32,
    height: u32,
    configured: bool,
    closed: bool,
    level: f32,
}

impl Overlay {
    /// Paints the meter into a fresh shm buffer and attaches it.
    ///
    /// Premultiplied ARGB8888, because that is what `wl_shm` expects and getting
    /// it wrong shows up as a wrongly-tinted or inverted surface rather than an
    /// error.
    fn draw(&mut self, handle: &QueueHandle<Self>) {
        let (buffer, canvas) = match self.pool.create_buffer(
            self.width as i32,
            self.height as i32,
            self.width as i32 * 4,
            wl_shm::Format::Argb8888,
        ) {
            Ok(pair) => pair,
            Err(error) => {
                eprintln!("voxide-overlay: buffer allocation failed: {error}");
                return;
            }
        };

        // Voxide's iron background, at the same opacity the webview overlay used.
        let background = [0x19u8, 0x11u8, 0x13u8, 0xF0u8]; // B, G, R, A
        for pixel in canvas.chunks_exact_mut(4) {
            pixel.copy_from_slice(&background);
        }

        let inset = 18i32;
        let bar_area = self.width as i32 - inset * 2;
        let bar_width = (bar_area / BARS as i32).max(1) - 3;
        let lit = (self.level * BARS as f32).round() as usize;
        for index in 0..BARS {
            // Copper for lit bars, a dim iron line for the rest.
            let colour = if index < lit {
                [0x3Cu8, 0x60u8, 0xC2u8, 0xFFu8]
            } else {
                [0x2Au8, 0x24u8, 0x22u8, 0xFFu8]
            };
            // Taller towards the middle so the meter reads as a waveform.
            let position = index as f32 / (BARS - 1) as f32;
            let shape = 1.0 - (position - 0.5).abs() * 1.4;
            let height = ((self.height as f32 - 40.0) * shape).max(6.0) as i32;
            let x0 = inset + index as i32 * (bar_area / BARS as i32);
            let y0 = (self.height as i32 - height) / 2;
            for y in y0..(y0 + height) {
                for x in x0..(x0 + bar_width) {
                    let offset = ((y * self.width as i32 + x) * 4) as usize;
                    if let Some(pixel) = canvas.get_mut(offset..offset + 4) {
                        pixel.copy_from_slice(&colour);
                    }
                }
            }
        }

        let surface = self.layer.wl_surface();
        surface.damage_buffer(0, 0, self.width as i32, self.height as i32);
        // 0.21 wants the callback's userdata wrapped so SCTK can route it back
        // to `CompositorHandler::frame`.
        surface.frame(handle, FrameCallbackData(surface.clone()));
        if let Err(error) = buffer.attach_to(surface) {
            eprintln!("voxide-overlay: could not attach the buffer: {error}");
            return;
        }
        surface.commit();
    }
}

impl CompositorHandler for Overlay {
    fn scale_factor_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: i32,
    ) {
        // Sizes above are logical, so the compositor's scale needs no response.
    }

    fn transform_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: wl_output::Transform,
    ) {
    }

    fn frame(
        &mut self,
        _: &Connection,
        handle: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: u32,
    ) {
        if self.configured {
            self.draw(handle);
        }
    }

    fn surface_enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: &wl_output::WlOutput,
    ) {
    }

    fn surface_leave(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: &wl_output::WlOutput,
    ) {
    }
}

impl LayerShellHandler for Overlay {
    fn closed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &LayerSurface) {
        self.closed = true;
    }

    fn configure(
        &mut self,
        _: &Connection,
        handle: &QueueHandle<Self>,
        _: &LayerSurface,
        configure: LayerSurfaceConfigure,
        _: u32,
    ) {
        // A zero from the compositor means "your choice", so the request stands.
        if configure.new_size.0 != 0 {
            self.width = configure.new_size.0;
        }
        if configure.new_size.1 != 0 {
            self.height = configure.new_size.1;
        }
        self.configured = true;
        self.draw(handle);
    }
}

impl OutputHandler for Overlay {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.outputs
    }
    fn new_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn update_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn output_destroyed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
}

impl ShmHandler for Overlay {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm
    }
}

impl ProvidesRegistryState for Overlay {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry
    }
    registry_handlers![OutputState];
}

// One macro for every protocol SCTK handles on our behalf: 0.21 replaced the
// per-protocol `delegate_compositor!`/`delegate_layer!`/... family with this.
delegate_dispatch2!(Overlay);
delegate_registry!(Overlay);
