</h1><div id="toc">
  <ul align="center" style="list-style: none; padding: 0; margin: 0;">
    <summary>
      <h1 style="margin-bottom: 0.0em;">
        BLANKET: Anonymizing Faces in Infant Video Recordings
      </h1>
    </summary>
  </ul>
</div>
</h1><div id="toc">
  <ul align="center" style="list-style: none; padding: 0; margin: 0;">
    <summary>
      <h2 style="margin-bottom: 0.2em;">
        ICDL 2025
      </h2>
    </summary>
  </ul>
</div>

<div align="center">

![BLANKET video](resources/BLANKET_video.gif)


[![ICDL 2025](https://img.shields.io/badge/Accepted%20to-ICDL%202025-blue)](https://icdl2025.fel.cvut.cz) &nbsp;&nbsp;&nbsp;
[![Paper and Supplementary](https://img.shields.io/badge/Paper%20+%20Supplementary-arXiv-red)](resources/BLANKET.pdf) &nbsp;&nbsp;&nbsp;
[![License](https://img.shields.io/badge/License-GPL%203.0-green.svg)](LICENSE)

<span style="color:tomato; font-weight:bold">MP: I Used GPL-3.0 license which I use for all my code. Let me know if you want to use different one.</span>
</div>

## 📋 Overview

BLANKET is a two-stage pipeline for **seamless**, **expression-preserving** face anonymization in infant videos. It replaces identities with synthetic baby faces while maintaining gaze, head pose, and emotional expression, enabling ethical data sharing and robust downstream analytics.

Key contributions:
- **Two-stage design**: diffusion-based inpainting + temporally-consistent swap  
- **Attribute preservation**: expression, gaze, head orientation, eye/mouth openness  
- **High downstream performance**: ~90% detection AP, ~97% pose estimation retention  
- **Outperforms SOTA**: beats DeepPrivacy2 on de-identification, perceptual metrics and downsteam task performance

## 📢 News

- **May 2025**: Paper accepted to ICDL 2025! 🎉

## 🚀 Installation
<span style="color:red; font-weight:bold">WARNING: This section will need edits after the code is prepared.</span>

```bash
git clone https://github.com/ctu-vras/blanket-infant-face-anonym.git
cd blanket-infant-face-anonym
pip install -r requirements.txt
```

**Requirements**
<span style="color:red; font-weight:bold">WARNING: This section will need edits after the code is prepared.</span>

* Python 3.9+
* PyTorch 1.12+
* [Stable Diffusion checkpoint](https://huggingface.co/SG161222/Realistic_Vision_V2.0)
* [FaceFusion](https://github.com/facefusion/facefusion)

## 🎬 Demo
<span style="color:red; font-weight:bold">WARNING: This section will need edits after the code is prepared.</span>

Run anonymization on a sample video:

```bash
python anonymize.py \
  --input_video demo/baby_video.mp4 \
  --output_video demo/anonymized.mp4 \
  --sd_checkpoint Realistic_Vision_V2.0 \
  --noise_level 0.8
```

Check `demo/` for example input/output pairs.

## 📦 Pre-trained Models
<span style="color:red; font-weight:bold">WARNING: This section will need edits after the code is prepared.</span>

Pre-trained landmarks and swap models will be available via Hugging Face soon.

## 📊 Evaluation

BLANKET outperforms DeepPrivacy V2 in all measured metrics.


| Metric                          | BLANKET       | DeepPrivacy2  |
| ------------------------------- | ------------- | ------------- |
| Identity cosine distance (↓)    | 0.11 ± 0.18   | 0.19 ± 0.26   |
| Emotion preservation (↑)        | 0.51 ± 0.13   | 0.27 ± 0.11   |
| Temporal landmark corr. (↑)     | 0.956 ± 0.064 | 0.860 ± 0.140 |
| Detection AP vs. orig. (↑)      | 90.7 mAP      | 81.5 mAP      |
| Pose AP vs. orig. (↑)           | 97.2 mAP      | 79.1 mAP      |

## 🗺️ Roadmap (remove upon publication)
<span style="color:magenta; font-weight:bold">WARNING: Remove this section after publication as we will have no roadmap by then.</span>

* [ ] Upload the code (installation instructions, demo)
* [ ] Make the code nice - iSort, Black, Copilot
* [ ] Upload pre-trained weights or link specific models for the demo
* [ ] Publish paper on ArXiv (and link it here) -- should be done after code is ready
* [x] Add visualization GIF
* [x] Add paper to repository to host it ourselves

## 🙏 Acknowledgments

Supported by GA CR 25-18113S, EC Digital Europe CEDMO 2.0 101158609, CTU SGS23/173/OHK3/3T/13.
Thanks to Adéla Šubrtová for early feedback.
Thanks to Max Familly Fun for the banner picture.

## 📝 Citation
<span style="color:orange; font-weight:bold">WARNING: This section will need edits after the paper is published or Arxiv.</span>

If you use BLANKET, please cite:

```bibtex
@INPROCEEDINGS{hadera2025BLANKET,
  author={Hadera, Ditmar and Cech, Jan and Purkrabek, Miroslav and Hoffmann, Matej},
  title={{BLANKET}: Anonymizing Faces in Infant Video Recordings},
  booktitle={2025 IEEE International Conference on Development and Learning (ICDL)}, 
  year={2025},
  volume={},
  number={},
  pages={1-8},
  keywords={?????},
  doi={????}}
```

