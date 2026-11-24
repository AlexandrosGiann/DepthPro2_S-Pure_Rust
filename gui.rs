#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

use depthpro2_s::DepthPro2S;
use eframe::egui;
use image::{DynamicImage, ImageDecoder, ImageReader, RgbImage};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc};

const TITLE: &str = "Depth Pro 2 S";
const WEIGHTS: &str = "depthpro2-s.safetensors";
const MAX_SIDE: u32 = 640;

enum JobResult {
    Weights(Result<(PathBuf, Arc<DepthPro2S>), String>),
    Depth(Result<egui::ColorImage, String>),
}

struct DepthApp {
    model: Option<(PathBuf, Arc<DepthPro2S>)>,
    photo: Option<Arc<RgbImage>>,
    photo_texture: Option<egui::TextureHandle>,
    depth_texture: Option<egui::TextureHandle>,
    pending: Option<mpsc::Receiver<JobResult>>,
    status: String,
    error: bool,
}

fn main() {
    let weights = std::env::args_os().nth(1).map(PathBuf::from).or_else(default_weights);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([960.0, 630.0])
            .with_min_inner_size([380.0, 460.0]),
        ..Default::default()
    };
    if let Err(error) = eframe::run_native(
        TITLE,
        options,
        Box::new(move |cc| Ok(Box::new(DepthApp::new(&cc.egui_ctx, weights)))),
    ) {
        rfd::MessageDialog::new()
            .set_title(TITLE)
            .set_description(error.to_string())
            .set_level(rfd::MessageLevel::Error)
            .show();
    }
}

fn default_weights() -> Option<PathBuf> {
    let cwd = std::env::current_dir().ok();
    let exe = std::env::current_exe().ok();
    cwd.as_deref()
        .into_iter()
        .chain(exe.as_deref().and_then(Path::parent))
        .map(|dir| dir.join(WEIGHTS))
        .find(|file| file.is_file())
}

fn prepare_photo(path: &Path) -> image::ImageResult<RgbImage> {
    let mut decoder = ImageReader::open(path)?.with_guessed_format()?.into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut photo = DynamicImage::from_decoder(decoder)?;
    photo.apply_orientation(orientation);
    if photo.width().max(photo.height()) > MAX_SIDE {
        photo = photo.resize(MAX_SIDE, MAX_SIDE, image::imageops::FilterType::Triangle);
    }
    Ok(photo.into_rgb8())
}

fn preview(values: &[f32], width: usize, height: usize) -> egui::ColorImage {
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f32::total_cmp);
    let low = sorted[sorted.len() * 2 / 100];
    let high = sorted[sorted.len() * 98 / 100];
    let gray: Vec<_> = values.iter().map(|&x| {
        if high > low {
            (255.0 * (x - low) / (high - low)) as u8
        } else if x > 0.0 {
            128
        } else {
            0
        }
    }).collect();
    egui::ColorImage::from_gray([width, height], &gray)
}

fn show_preview(ui: &mut egui::Ui, title: &str, image: Option<&egui::TextureHandle>) {
    ui.heading(title);
    if let Some(image) = image {
        let size = egui::vec2(ui.available_width(), 380.0);
        ui.add(egui::Image::new(image).max_size(size));
    } else {
        ui.add_space(6.0);
        ui.label("Preview appears here");
    }
}

impl DepthApp {
    fn new(ctx: &egui::Context, weights: Option<PathBuf>) -> Self {
        let mut app = Self {
            model: None,
            photo: None,
            photo_texture: None,
            depth_texture: None,
            pending: None,
            status: "Choose the official Depth Pro 2 S weights.".into(),
            error: false,
        };
        if let Some(path) = weights {
            app.load_weights(ctx, path);
        }
        app
    }

    fn start_job(
        &mut self,
        ctx: &egui::Context,
        status: &str,
        job: impl FnOnce() -> JobResult + Send + 'static,
    ) {
        let (tx, rx) = mpsc::channel();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let _ = tx.send(job());
            ctx.request_repaint();
        });
        self.pending = Some(rx);
        self.status = status.into();
        self.error = false;
    }

    fn load_weights(&mut self, ctx: &egui::Context, path: PathBuf) {
        self.start_job(ctx, "Loading model weights…", move || {
            let result = std::fs::read(&path)
                .map_err(|err| format!("Could not read {} ({err})", path.display()))
                .and_then(|bytes| DepthPro2S::from_safetensors(&bytes))
                .map(|model| (path, Arc::new(model)));
            JobResult::Weights(result)
        });
    }

    fn open_photo(&mut self, ctx: &egui::Context, path: &Path) {
        match prepare_photo(path) {
            Ok(photo) => {
                let size = [photo.width() as usize, photo.height() as usize];
                let texture = egui::ColorImage::from_rgb(size, photo.as_raw());
                self.photo_texture = Some(ctx.load_texture("photo", texture, Default::default()));
                self.photo = Some(Arc::new(photo));
                self.depth_texture = None;
                self.status = format!("Photo ready · {} × {} px", size[0], size[1]);
                self.error = false;
            }
            Err(err) => self.show_error(format!("Could not open photo ({err})")),
        }
    }

    fn calculate(&mut self, ctx: &egui::Context) {
        if let (Some((_, model)), Some(photo)) = (&self.model, &self.photo) {
            let (model, photo) = (Arc::clone(model), Arc::clone(photo));
            self.depth_texture = None;
            self.start_job(ctx, "Calculating depth…", move || {
                let (w, h) = (photo.width() as usize, photo.height() as usize);
                let result = model.predict(photo.as_raw(), w, h, None)
                    .map(|map| preview(&map.inverse_depth, w, h));
                JobResult::Depth(result)
            });
        }
    }

    fn show_error(&mut self, error: String) {
        self.status = error;
        self.error = true;
    }

    fn finish_job(&mut self, ctx: &egui::Context) {
        let Some(pending) = &self.pending else { return };
        let result = match pending.try_recv() {
            Ok(result) => result,
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => {
                self.pending = None;
                self.show_error("The background operation stopped unexpectedly.".into());
                return;
            }
        };
        self.pending = None;
        match result {
            JobResult::Weights(Ok((path, model))) => {
                self.model = Some((path, model));
                self.depth_texture = None;
                self.status = if self.photo.is_some() {
                    "Model and photo ready.".into()
                } else {
                    "Model ready. Choose a photo.".into()
                };
            }
            JobResult::Depth(Ok(image)) => {
                self.depth_texture = Some(ctx.load_texture("depth", image, Default::default()));
                self.status = "Depth map ready. Brighter areas are closer.".into();
            }
            JobResult::Weights(Err(error)) | JobResult::Depth(Err(error)) => self.show_error(error),
        }
    }
}

impl eframe::App for DepthApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.finish_job(ctx);
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.spacing_mut().item_spacing = egui::vec2(8.0, 9.0);
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading(TITLE);
                ui.horizontal_wrapped(|ui| {
                    let idle = self.pending.is_none();
                    if ui.add_enabled(idle, egui::Button::new("Choose weights…")).clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("SafeTensors weights", &["safetensors"]).pick_file() {
                            self.load_weights(ctx, path);
                        }
                    }
                    if ui.add_enabled(idle, egui::Button::new("Open photo…")).clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Photos", &["jpg", "jpeg", "png", "webp"]).pick_file() {
                            self.open_photo(ctx, &path);
                        }
                    }
                    let can_calculate = idle && self.model.is_some() && self.photo.is_some();
                    if ui.add_enabled(can_calculate, egui::Button::new("Calculate depth")).clicked() {
                        self.calculate(ctx);
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    if self.pending.is_some() {
                        ui.spinner();
                    }
                    if self.error {
                        ui.colored_label(ui.visuals().error_fg_color, &self.status);
                    } else {
                        ui.label(&self.status);
                    }
                });
                ui.separator();
                if ui.available_width() >= 650.0 {
                    ui.columns(2, |columns| {
                        show_preview(&mut columns[0], "Photo", self.photo_texture.as_ref());
                        show_preview(&mut columns[1], "Depth map", self.depth_texture.as_ref());
                    });
                } else {
                    show_preview(ui, "Photo", self.photo_texture.as_ref());
                    show_preview(ui, "Depth map", self.depth_texture.as_ref());
                }
            });
        });
    }
}
