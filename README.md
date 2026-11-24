# Depth Pro 2 S in Rust

An unofficial Rust implementation of Apple's Depth Pro 2 S for CPU inference. It provides a desktop app for photos and a library and command-line tool for photos and short clips. The inference engine uses the `matrixmultiply` and `serde_json` crates.

## Desktop app

![Depth Pro 2 S desktop app showing a triceratops image beside its generated inverse depth map](Screenshot%202026-11-24%20235744.png)

From the project directory, run:

```sh
cargo run --release --locked --features gui --bin depthpro2-gui
```

On Windows, you can also double-click `target/release/depthpro2-gui.exe` after building it. Save the [official weights](https://huggingface.co/apple/DepthPro2-S/resolve/main/model.safetensors) as `depthpro2-s.safetensors` in the project directory or next to the executable to load them on startup. If you keep the weights elsewhere, use **Choose weights**.

Click **Open photo**, select a JPEG, PNG, or WebP image, then click **Calculate depth**. The app shows the photo and its inverse depth map side by side in a wide window. Brighter areas indicate closer points. The image is processed locally, and the app scales it down when its longest side exceeds 640 pixels.

## Command line

You need Rust with Cargo and the **97.3 MB** [official F32 SafeTensors weights](https://huggingface.co/apple/DepthPro2-S/resolve/main/model.safetensors). Save the weights as `depthpro2-s.safetensors` next to `Cargo.toml`.

Use FFmpeg to convert a photo to packed RGB8. Run these commands from the project directory and replace `640 480` with the actual image dimensions.

```sh
ffmpeg -i photo.jpg -frames:v 1 -pix_fmt rgb24 -f rawvideo photo.rgb
cargo run --release --locked -- depthpro2-s.safetensors photo.rgb 640 480 depth.f32
```

The `.rgb` file can contain one frame or a sequence of frames with the same dimensions. The `.f32` output contains little-endian float32 values in row-major order, with frames in input order. Pass the focal length in pixels as the final argument to get depth in meters along the optical axis. Omit it to get normalized inverse depth.

## Library

Load the weights with `DepthPro2S::from_safetensors(&bytes)` from the `depthpro2_s` crate. Use `predict` for one frame or `predict_clip` to process multiple frames together. Pass `Some(focal_px)` for depth in meters or `None` for normalized inverse depth.

## Credits and model license

This port implements the encoder, decoder, and depth head in Rust and uses Apple's original checkpoint. Apple's separate focal-length estimator and CLI streaming modes are outside the scope of this port. [Apple source code](https://github.com/apple-aiml-research/ml-depth-pro-2) · [DINOv3 in timm](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/eva.py)

Apple Machine Learning Research Model is licensed under the Apple Machine Learning Research Model License Agreement. The weights are available for non-commercial scientific research and academic development. [Apple model license](https://github.com/apple-aiml-research/ml-depth-pro-2/blob/main/LICENSE_MODEL)
