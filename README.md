# CAST project webpage

Anonymous project site for **Anchoring Critic-Guided Fine-Tuning of Generative Robot Policies**. Open `index.html` locally or serve this directory with `python -m http.server 8000`.

The page uses the current ICRA manuscript in `../imitation/_ICRA_2027__CAST/` for method descriptions and result tables. Figures and robot stills were copied from that manuscript and `../imitation/icra_video/`. The five method animations were extracted from `icra_video/cast_video_slides.pptx`. `assets/videos/cast_method_preview.mp4` is a silent, two-minute rendering of slides 1–6 with those animations. The three benchmark videos are copied from `icra_video/task_demos/` and are recorded policy rollouts.

The robot photo sequences are **teleoperated demonstration frames**, not CAST evaluation videos. The supplied `icra_video` folder contains no autonomous real-robot MP4s; its `V01` and `V02` slots are explicitly reserved. Replace or add clips only when correctly attributed footage is available. The manuscript main text and supplement differ on physical evaluation counts, so this page reports the shared success percentages without trial counts. The main paper PDF is not present in the source folder and is not linked here yet.

The site is served from the repository root on GitHub Pages.

## Distribution-fitting animation

The hero and method section use `assets/animations/cast_distribution.mp4` as a smooth, muted loop. The same 27-second sequence is available as `cast_distribution.gif`. Pause/replay controls and the four method-stage buttons are implemented in `animation.js`; autoplay pauses outside the viewport and respects reduced-motion preferences.

The animation computes four CAST updates for a synthetic two-dimensional residual actor, using same-noise base pairs, normalized and clipped critic gradients, both restoring terms, and a capped total target displacement. Each round runs 12 gradient-descent steps toward fixed targets. The fitting error is measured from those actual steps. This is an explanation of the update, not experimental robot evidence. The frozen synthetic critic and smoothed density display are documented in `assets/animations/cast_distribution_trace.json`.

Rebuild with Python, NumPy, Pillow, and FFmpeg:

```bash
python scripts/render_cast_transport.py
# Or set FFMPEG=/path/to/ffmpeg; imageio-ffmpeg is also supported.
python scripts/render_cast_transport.py --preview
```

The renderer checks target-step bounds and decreasing regression loss. It saves both media formats, preview frames, and the numerical trace.
