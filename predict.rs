use depthpro2_s::DepthPro2S;
use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if !(5..=6).contains(&args.len()) {
        return Err(
            "Usage: depthpro2-s WEIGHTS INPUT.rgb WIDTH HEIGHT OUTPUT.f32 [FOCAL_PX]".into(),
        );
    }
    let width: usize = args[2].parse()?;
    let height: usize = args[3].parse()?;
    let frame_bytes = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(3))
        .ok_or("Image too large")?;
    let focal = args.get(5).map(|value| value.parse::<f32>()).transpose()?;
    let input = fs::read(&args[1])?;
    if input.is_empty() || frame_bytes == 0 || input.len() % frame_bytes != 0 {
        return Err("INPUT.rgb must contain whole packed RGB8 frames".into());
    }
    let frames: Vec<_> = input.chunks_exact(frame_bytes).collect();
    let model = DepthPro2S::from_safetensors(&fs::read(&args[0])?)?;
    let maps = model.predict_clip(&frames, width, height, focal)?;
    let mut writer = BufWriter::new(File::create(&args[4])?);
    for map in maps {
        for value in map.depth_meters.unwrap_or(map.inverse_depth) {
            writer.write_all(&value.to_le_bytes())?;
        }
    }
    writer.flush()?;
    Ok(())
}
