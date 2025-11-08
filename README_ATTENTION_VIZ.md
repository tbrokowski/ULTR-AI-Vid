# Visualize Top-3 Attention Frames

Shows the 3 ultrasound frames with **highest** attention scores from the attention_pool_extra3 model.

## Usage

```bash
python visualize_attention_minimal.py
```

That's it! Uses the first video found by default. Output: `top3_attention_frames.png`

Or specify your own video:

```bash
python visualize_attention_minimal.py --video /path/to/video.mp4
```

## Output

A simple image showing the 3 frames side-by-side with their attention scores.
