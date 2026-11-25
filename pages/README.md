# Video prediction on the sphere: results pages

Every picture the Sphere Encoder sees becomes a point on a sphere, so a video becomes a path on that sphere. These pages test one question: shown only the first frame of a video, can a small model guess where the later frames go?

In one line: the models learn how far a video moves, but not which way, so on new videos they still lose to simply freezing the first frame.

## The pages

| Page | What it shows |
|---|---|
| [index.html](index.html) | The overview. The clips we generated, how well the encoder can redraw them, searching the sphere for each frame, how videos move on the sphere, and the first frame-prediction runs. Start here. |
| [fn106m.html](fn106m.html) | FrameNet, the one-arrow predictor, grown from 10 M to 106 M parameters. Its architecture, its single objective, and 10 M vs 106 M results. The bigger model memorises the training clips but is no better on new ones. |
| [rotnet.html](rotnet.html) | RotNet, which splits the guess into "how far" and "which way". Its two heads, its two-part objective, results with and without a frame 2 hint, the 200k-step run, and what happens when it is given the true direction. |
| [compare.html](compare.html) | FrameNet and RotNet side by side, with animations of both architectures, the error split into distance and direction, every clip on a flat map of the sphere, and both 200k-step runs. |

Supporting folders: `media/` holds the clips (true video, redraw, guesses), `anim/`, `anim106/` and `anim_rn/` hold the animations, `anim_src/` holds the manim scenes that render them, and `live/` holds the RotNet 200k training log.

## Viewing locally

The pages load their videos and animations by relative path, so serve the folder rather than opening the files directly:

```bash
cd pages
python -m http.server 8000 --bind 127.0.0.1
```

Then open http://127.0.0.1:8000/index.html. Equations render with KaTeX from cdnjs, so the first load needs internet access.

## Headline numbers (16 new clips, angle error on the sphere, lower is better)

| Model | Error | How far off | Which way off |
|---|---|---|---|
| Freeze the first frame | 13.6° | | |
| FrameNet 10 M | 21.3° | 5.8° | 82° |
| FrameNet 106 M | 21.9° | 6.0° | 82° |
| RotNet 10 M | 20.4° | 4.5° | 82° |
| RotNet after 200k steps | 21.0° | 4.8° | 84° |
| RotNet with the true direction | 4.5° | 4.5° | 0° |

A direction 90° off is a blind guess. Beyond 60° off, any move is worse than freezing.
