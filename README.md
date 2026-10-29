# CAST project webpage

Anonymous project site for **Anchoring Critic-Guided Fine-Tuning of Generative Robot Policies**. Open `index.html` locally or serve this directory with `python -m http.server 8000`.

The page uses the current ICRA manuscript in `../imitation/_ICRA_2027__CAST/` for method descriptions and result tables. Figures and robot stills were copied from that manuscript and `../imitation/icra_video/`. The five method animations were extracted from `icra_video/cast_video_slides.pptx`. `assets/videos/cast_method_preview.mp4` is a silent, two-minute rendering of slides 1–6 with those animations. The three benchmark videos are copied from `icra_video/task_demos/` and are recorded policy rollouts.

The robot photo sequences are **teleoperated demonstration frames**, not CAST evaluation videos. The supplied `icra_video` folder contains no autonomous real-robot MP4s; its `V01` and `V02` slots are explicitly reserved. Replace or add clips only when correctly attributed footage is available. The manuscript main text and supplement differ on physical evaluation counts, so this page reports the shared success percentages without trial counts. The main paper PDF is not present in the source folder and is not linked here yet.

The site is served from the repository root on GitHub Pages.
