use serde_json::Value;
use std::collections::HashMap;
use std::f32::consts::{FRAC_1_SQRT_2, TAU};

type Result<T> = std::result::Result<T, String>;
const DIM: usize = 384;
const HEAD: usize = 64;
const PREFIX: usize = 5;

struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

struct Image {
    h: usize,
    w: usize,
    c: usize,
    data: Vec<f32>,
}

pub struct Prediction {
    pub width: usize,
    pub height: usize,
    pub inverse_depth: Vec<f32>,
    pub depth_meters: Option<Vec<f32>>,
}

pub struct DepthPro2S {
    weights: HashMap<String, Tensor>,
}

fn index(value: &Value) -> Result<usize> {
    value
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| "Invalid SafeTensors integer".to_owned())
}

fn weights(bytes: &[u8]) -> Result<HashMap<String, Tensor>> {
    let header_len = bytes
        .get(..8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n <= 100_000_000)
        .ok_or_else(|| "Invalid SafeTensors header length".to_owned())?;
    let (header, data) = bytes
        .get(8..)
        .and_then(|b| b.split_at_checked(header_len))
        .ok_or_else(|| "Truncated SafeTensors file".to_owned())?;
    let header: Value = serde_json::from_slice(header).map_err(|error| error.to_string())?;
    let mut result = HashMap::new();
    for (name, entry) in header
        .as_object()
        .ok_or("Expected SafeTensors header object")?
    {
        if name == "__metadata__" {
            continue;
        }
        let shape = entry["shape"]
            .as_array()
            .ok_or("Missing tensor shape")?
            .iter()
            .map(index)
            .collect::<Result<Vec<_>>>()?;
        let size = shape
            .iter()
            .try_fold(4usize, |n, &d| n.checked_mul(d))
            .ok_or("SafeTensors shape too large")?;
        let offsets = entry["data_offsets"]
            .as_array()
            .filter(|v| v.len() == 2)
            .ok_or("Invalid tensor offsets")?;
        let (lo, hi) = (index(&offsets[0])?, index(&offsets[1])?);
        let raw = data
            .get(lo..hi)
            .filter(|raw| raw.len() == size && entry["dtype"] == "F32")
            .ok_or_else(|| format!("Invalid F32 tensor: {name}"))?;
        let values = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        result.insert(
            name.clone(),
            Tensor {
                shape,
                data: values,
            },
        );
    }
    Ok(result)
}

// x is [rows, input], weights are [output, input].
fn matmul(x: &[f32], weights: &[f32], input: usize, output: usize, result: &mut [f32]) {
    let rows = x.len() / input;
    assert_eq!(x.len(), rows * input);
    assert_eq!(weights.len(), output * input);
    assert_eq!(result.len(), rows * output);
    // The assertions bound all pointer accesses. The mutable output cannot alias either input.
    unsafe {
        matrixmultiply::sgemm(
            rows,
            input,
            output,
            1.0,
            x.as_ptr(),
            input as isize,
            1,
            weights.as_ptr(),
            1,
            input as isize,
            0.0,
            result.as_mut_ptr(),
            output as isize,
            1,
        );
    }
}

fn add_bias(data: &mut [f32], bias: &[f32]) {
    for row in data.chunks_exact_mut(bias.len()) {
        for (value, shift) in row.iter_mut().zip(bias) {
            *value += shift;
        }
    }
}

fn norm(x: &[f32], scale: &[f32], bias: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(DIM) {
        let mean = row.iter().sum::<f32>() / DIM as f32;
        let variance = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / DIM as f32;
        let inv = (variance + 1e-5).sqrt().recip();
        out.extend((0..DIM).map(|i| (row[i] - mean) * inv * scale[i] + bias[i]));
    }
    out
}

fn gelu(x: f32) -> f32 {
    let z = x.abs() * FRAC_1_SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * z);
    let tail = ((((1.061_405_429 * t - 1.453_152_027) * t + 1.421_413_741) * t - 0.284_496_736)
        * t
        + 0.254_829_592)
        * t
        * (-z * z).exp();
    0.5 * x * if x < 0.0 { tail } else { 2.0 - tail }
}

fn rotary(height: usize, width: usize) -> Vec<(f32, f32)> {
    let mut table = Vec::with_capacity(height * width * HEAD / 2);
    let extent = height.max(width) as f32;
    for y in 0..height {
        for x in 0..width {
            for i in 0..HEAD / 2 {
                let coordinate = if i < HEAD / 4 { y } else { x };
                let period = 100f32.powf((i % (HEAD / 4)) as f32 / (HEAD / 4) as f32);
                let angle = TAU * (2.0 * (coordinate as f32 + 0.5) / extent - 1.0) / period;
                table.push(angle.sin_cos());
            }
        }
    }
    table
}

fn attention(qkv: &mut [f32], rope: &[(f32, f32)]) -> Vec<f32> {
    let rows = qkv.len() / (3 * DIM);
    let mut out = vec![0f32; rows * DIM];
    let mut scores = vec![0f32; rows];
    let patches = rope.len() / (HEAD / 2);
    for (row, values) in qkv.chunks_exact_mut(3 * DIM).enumerate().skip(PREFIX) {
        let pos = ((row - PREFIX) % patches) * (HEAD / 2);
        for head in values[..2 * DIM].chunks_exact_mut(HEAD) {
            for i in 0..HEAD / 2 {
                let (a, b) = (head[i], head[i + HEAD / 2]);
                let (sin, cos) = rope[pos + i];
                head[i] = a * cos - b * sin;
                head[i + HEAD / 2] = b * cos + a * sin;
            }
        }
    }
    for row in 0..rows {
        for head in (0..DIM).step_by(HEAD) {
            let start = row * 3 * DIM + head;
            let query = &qkv[start..start + HEAD];
            let mut max = f32::NEG_INFINITY;
            for (key, score) in qkv.chunks_exact(3 * DIM).zip(&mut scores) {
                *score = query
                    .iter()
                    .zip(&key[DIM + head..DIM + head + HEAD])
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    / 8.0;
                max = max.max(*score);
            }
            let mut total = 0.0;
            for score in &mut scores {
                *score = (*score - max).exp();
                total += *score;
            }
            let output = &mut out[row * DIM + head..row * DIM + head + HEAD];
            for (values, score) in qkv.chunks_exact(3 * DIM).zip(&scores) {
                let factor = score / total;
                for (pixel, value) in output
                    .iter_mut()
                    .zip(&values[2 * DIM + head..2 * DIM + head + HEAD])
                {
                    *pixel += factor * value;
                }
            }
        }
    }
    out
}

fn relu(values: &mut [f32]) {
    for value in values {
        *value = value.max(0.0);
    }
}

impl DepthPro2S {
    pub fn from_safetensors(bytes: &[u8]) -> Result<Self> {
        let model = Self {
            weights: weights(bytes)?,
        };
        model.tensor("encoder.cls_token", &[1, 1, DIM])?;
        model.tensor("encoder.reg_token", &[1, PREFIX - 1, DIM])?;
        Ok(model)
    }

    fn tensor(&self, name: &str, shape: &[usize]) -> Result<&[f32]> {
        self.weights
            .get(name)
            .filter(|tensor| tensor.shape == shape)
            .map(|tensor| tensor.data.as_slice())
            .ok_or_else(|| format!("Missing or incorrectly shaped tensor: {name} {shape:?}"))
    }

    fn linear(
        &self,
        x: &[f32],
        name: &str,
        input: usize,
        output: usize,
        bias: bool,
    ) -> Result<Vec<f32>> {
        let w = self.tensor(&format!("{name}.weight"), &[output, input])?;
        let mut result = vec![0.0; x.len() / input * output];
        matmul(x, w, input, output, &mut result);
        if bias {
            add_bias(
                &mut result,
                self.tensor(&format!("{name}.bias"), &[output])?,
            );
        }
        Ok(result)
    }

    fn conv(
        &self,
        image: &Image,
        name: &str,
        channels: usize,
        kernel: usize,
        stride: usize,
        pad: usize,
        bias: bool,
    ) -> Result<Image> {
        let Image { h, w, c, data } = image;
        let weight = self.tensor(&format!("{name}.weight"), &[channels, *c, kernel, kernel])?;
        let height = (h + 2 * pad - kernel) / stride + 1;
        let width = (w + 2 * pad - kernel) / stride + 1;
        let mut out = vec![0.0; height * width * channels];
        if kernel == 1 && stride == 1 && pad == 0 {
            matmul(data, weight, *c, channels, &mut out);
        } else {
            let patch_size = c * kernel * kernel;
            // Limit the temporary im2col matrix for large images.
            let mut patches = vec![0.0; 64 * patch_size];
            for (group, output) in out.chunks_mut(64 * channels).enumerate() {
                let count = output.len() / channels;
                for (local, patch) in patches.chunks_exact_mut(patch_size).take(count).enumerate() {
                    let location = group * 64 + local;
                    let (y, x) = (location / width, location % width);
                    for ky in 0..kernel {
                        for kx in 0..kernel {
                            let yy = (y * stride + ky).checked_sub(pad).filter(|yy| yy < h);
                            let xx = (x * stride + kx).checked_sub(pad).filter(|xx| xx < w);
                            let pixel = yy.zip(xx).map(|(yy, xx)| (yy * w + xx) * c);
                            for i in 0..*c {
                                patch[(i * kernel + ky) * kernel + kx] =
                                    pixel.map_or(0.0, |p| data[p + i]);
                            }
                        }
                    }
                }
                matmul(
                    &patches[..count * patch_size],
                    weight,
                    patch_size,
                    channels,
                    output,
                );
            }
        }
        if bias {
            add_bias(&mut out, self.tensor(&format!("{name}.bias"), &[channels])?);
        }
        Ok(Image {
            h: height,
            w: width,
            c: channels,
            data: out,
        })
    }

    fn deconv(&self, image: &Image, name: &str, scale: usize, bias: bool) -> Result<Image> {
        let Image { h, w, c, data } = image;
        let taps = scale * scale;
        let outputs = c * taps;
        let weight = self.tensor(&format!("{name}.weight"), &[*c, *c, scale, scale])?;
        let mut transposed = vec![0.0; weight.len()];
        for i in 0..*c {
            for o in 0..outputs {
                transposed[o * c + i] = weight[i * outputs + o];
            }
        }
        let mut projected = vec![0.0; h * w * outputs];
        matmul(data, &transposed, *c, outputs, &mut projected);
        let width = w * scale;
        let mut out = vec![0.0; projected.len()];
        for (pixel, row) in projected.chunks_exact(outputs).enumerate() {
            for dy in 0..scale {
                for dx in 0..scale {
                    let y = pixel / w * scale + dy;
                    let x = pixel % w * scale + dx;
                    let tap = dy * scale + dx;
                    let start = (y * width + x) * c;
                    for o in 0..*c {
                        out[start + o] = row[o * taps + tap];
                    }
                }
            }
        }
        if bias {
            add_bias(&mut out, self.tensor(&format!("{name}.bias"), &[*c])?);
        }
        Ok(Image {
            h: h * scale,
            w: width,
            c: *c,
            data: out,
        })
    }

    fn block(&self, x: &[f32], index: usize, rope: &[(f32, f32)]) -> Result<Vec<f32>> {
        let name = format!("encoder.blocks.{index}");
        let normalize = |x: &[f32], part: &str| -> Result<Vec<f32>> {
            Ok(norm(
                x,
                self.tensor(&format!("{name}.{part}.weight"), &[DIM])?,
                self.tensor(&format!("{name}.{part}.bias"), &[DIM])?,
            ))
        };
        let input = normalize(x, "norm1")?;
        let mut qkv = self.linear(&input, &format!("{name}.attn.qkv"), DIM, 3 * DIM, false)?;
        let attended = self.linear(
            &attention(&mut qkv, rope),
            &format!("{name}.attn.proj"),
            DIM,
            DIM,
            true,
        )?;
        let scale1 = self.tensor(&format!("{name}.gamma_1"), &[DIM])?;
        let mut middle: Vec<_> = x
            .iter()
            .zip(attended)
            .enumerate()
            .map(|(i, (x, a))| x + a * scale1[i % DIM])
            .collect();
        let mut hidden = self.linear(
            &normalize(&middle, "norm2")?,
            &format!("{name}.mlp.fc1"),
            DIM,
            4 * DIM,
            true,
        )?;
        hidden.iter_mut().for_each(|x| *x = gelu(*x));
        let projected = self.linear(&hidden, &format!("{name}.mlp.fc2"), 4 * DIM, DIM, true)?;
        let scale2 = self.tensor(&format!("{name}.gamma_2"), &[DIM])?;
        for (i, (x, value)) in middle.iter_mut().zip(projected).enumerate() {
            *x += value * scale2[i % DIM];
        }
        Ok(middle)
    }

    fn encode(&self, images: &[Image]) -> Result<Vec<Vec<Image>>> {
        let mut tokens = Vec::with_capacity(images.len());
        let mut features: Vec<Vec<Image>> = images.iter().map(|_| Vec::with_capacity(4)).collect();
        let (h, w) = (images[0].h / 16, images[0].w / 16);
        let prefix_size = PREFIX * DIM;
        let patch_size = h * w * DIM;
        let rope = rotary(h, w);
        for image in images {
            let patch = self.conv(image, "encoder.patch_embed.proj", DIM, 16, 16, 0, true)?;
            let mut token = Vec::with_capacity(prefix_size + patch_size);
            token.extend(self.tensor("encoder.cls_token", &[1, 1, DIM])?);
            token.extend(self.tensor("encoder.reg_token", &[1, PREFIX - 1, DIM])?);
            token.extend(patch.data);
            tokens.push(token);
        }
        for index in 0..12 {
            if matches!(index, 5 | 7 | 9 | 11) {
                let mut joint = vec![0.0; prefix_size];
                for token in &tokens {
                    for (x, y) in joint.iter_mut().zip(token) {
                        *x += y;
                    }
                }
                for x in &mut joint {
                    *x /= tokens.len() as f32;
                }
                for token in &tokens {
                    joint.extend_from_slice(&token[prefix_size..]);
                }
                let output = self.block(&joint, index, &rope)?;
                for (token, patch) in tokens
                    .iter_mut()
                    .zip(output[prefix_size..].chunks_exact(patch_size))
                {
                    token[prefix_size..].copy_from_slice(patch);
                }
            } else {
                for token in &mut tokens {
                    *token = self.block(token, index, &rope)?;
                }
            }
            if matches!(index, 2 | 5 | 8 | 11) {
                for (token, levels) in tokens.iter().zip(&mut features) {
                    levels.push(Image {
                        h,
                        w,
                        c: DIM,
                        data: token[prefix_size..].to_vec(),
                    });
                }
            }
        }
        Ok(features)
    }

    fn residual(&self, image: Image, name: &str) -> Result<Image> {
        let mut activated = Image {
            data: image.data.clone(),
            ..image
        };
        relu(&mut activated.data);
        let mut out = self.conv(&activated, &format!("{name}.residual.1"), 64, 3, 1, 1, true)?;
        relu(&mut out.data);
        out = self.conv(&out, &format!("{name}.residual.3"), 64, 3, 1, 1, true)?;
        for (x, skip) in out.data.iter_mut().zip(image.data) {
            *x += skip;
        }
        Ok(out)
    }

    fn decode(&self, features: &[Image]) -> Result<Image> {
        let channels = [48, 96, 192, 384];
        let mut result: Option<Image> = None;
        for i in (0..4).rev() {
            let name = format!("encoder.reassemble.{i}");
            let mut level = self.conv(
                &features[i],
                &format!("{name}.0"),
                channels[i],
                1,
                1,
                0,
                true,
            )?;
            if i < 2 {
                level = self.deconv(&level, &format!("{name}.1"), 4 >> i, true)?;
            }
            if i == 3 {
                level = self.conv(&level, &format!("{name}.1"), channels[i], 3, 2, 1, true)?;
            }
            level = self.conv(&level, &format!("decoder.convs.{i}"), 64, 3, 1, 1, false)?;
            let fusion = format!("decoder.fusions.{i}");
            let input = if let Some(mut previous) = result {
                let skip = self.residual(level, &format!("{fusion}.resnet1"))?;
                for (x, skip) in previous.data.iter_mut().zip(skip.data) {
                    *x += skip;
                }
                previous
            } else {
                level
            };
            let refined = self.residual(input, &format!("{fusion}.resnet2"))?;
            let up = self.deconv(&refined, &format!("{fusion}.deconv"), 2, false)?;
            result = Some(self.conv(&up, &format!("{fusion}.out_conv"), 64, 1, 1, 0, true)?);
        }
        let result = self.conv(&result.unwrap(), "head.0", 32, 3, 1, 1, true)?;
        let result = self.deconv(&result, "head.1", 2, true)?;
        let mut result = self.conv(&result, "head.2", 32, 3, 1, 1, true)?;
        relu(&mut result.data);
        let mut result = self.conv(&result, "head.4", 1, 1, 1, 0, true)?;
        relu(&mut result.data);
        Ok(result)
    }

    pub fn predict(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        focal_px: Option<f32>,
    ) -> Result<Prediction> {
        Ok(self
            .predict_clip(&[rgb], width, height, focal_px)?
            .remove(0))
    }

    pub fn predict_clip(
        &self,
        frames: &[&[u8]],
        width: usize,
        height: usize,
        focal_px: Option<f32>,
    ) -> Result<Vec<Prediction>> {
        let size = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(3))
            .ok_or("Image too large")?;
        if size == 0 || frames.is_empty() || frames.iter().any(|rgb| rgb.len() != size) {
            return Err(
                "Expected one or more packed RGB8 frames of matching positive dimensions".into(),
            );
        }
        if focal_px.is_some_and(|f| !f.is_finite() || f <= 0.0) {
            return Err("Focal length in pixels must be positive and finite".into());
        }
        let w = width
            .checked_next_multiple_of(32)
            .ok_or("Image too large")?;
        let h = height
            .checked_next_multiple_of(32)
            .ok_or("Image too large")?;
        let padded = w
            .checked_mul(h)
            .and_then(|n| n.checked_mul(3))
            .ok_or("Image too large")?;
        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];
        let mut images = Vec::with_capacity(frames.len());
        for rgb in frames {
            let mut data = Vec::with_capacity(padded);
            for y in 0..h {
                for x in 0..w {
                    let base = (y.min(height - 1) * width + x.min(width - 1)) * 3;
                    for c in 0..3 {
                        data.push((f32::from(rgb[base + c]) / 255.0 - mean[c]) / std[c]);
                    }
                }
            }
            images.push(Image { h, w, c: 3, data });
        }
        self.encode(&images)?
            .iter()
            .map(|features| {
                let map = self.decode(features)?;
                let mut inverse_depth = Vec::with_capacity(size / 3);
                for row in map.data.chunks_exact(w).take(height) {
                    inverse_depth.extend_from_slice(&row[..width]);
                }
                let depth_meters = focal_px.map(|f| {
                    inverse_depth
                        .iter()
                        .map(|d| f / (width as f32 * d))
                        .collect()
                });
                Ok(Prediction {
                    width,
                    height,
                    inverse_depth,
                    depth_meters,
                })
            })
            .collect()
    }
}
