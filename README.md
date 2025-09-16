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

![BLANKET video](https://cmp.felk.cvut.cz/~cechj/video/icdl-2025/BLANKET_video.gif)

[click here for full video](https://cmp.felk.cvut.cz/~cechj/video/icdl-2025/BLANKET_video.mp4)

[![ICDL 2025](https://img.shields.io/badge/Accepted%20to-ICDL%202025-blue)](https://icdl2025.fel.cvut.cz) &nbsp;&nbsp;&nbsp;
[![Paper and Supplementary](https://img.shields.io/badge/Paper%20+%20Supplementary-arXiv-red)](resources/BLANKET.pdf) &nbsp;&nbsp;&nbsp;
[![License](https://img.shields.io/badge/License-GPL%203.0-green.svg)](LICENSE)
</div>

## 📋 Overview

BLANKET is a two-stage pipeline for **seamless**, **expression-preserving** face anonymization in infant videos. It replaces identities with synthetic baby faces while maintaining gaze, head pose, and emotional expression, enabling ethical data sharing and robust downstream analytics.

Key contributions:
- **Two-stage design**: diffusion-based inpainting + temporally-consistent swap  
- **Attribute preservation**: expression, gaze, head orientation, eye/mouth openness  
- **High downstream performance**: ~90% detection AP, ~97% pose estimation retention  
- **Outperforms SOTA**: beats DeepPrivacyV2 on de-identification, perceptual metrics and downstream task performance

> [!WARNING]  
> This code is experimental and not yet production-ready. Anonymization results may be imperfect and may miss detections. Full reliability will be achieved once the “missing detections” problem is solved (see Roadmap below).


## 📢 News

- **May 2025**: Paper accepted to ICDL 2025! 🎉

## 🚀 Installation
To install BLANKET and its dependencies:

```bash
git clone https://github.com/ctu-vras/blanket-infant-face-anonym.git
cd blanket-infant-face-anonym
pip install -r requirements.txt
pip install .
```

**Pre-trained models**

* YOLOv11L-face - [download from Ultralytics](https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov11l-face.pt)


## 🎮 Demo
To run the image anonymization demo:

1. Set your input and output folders in `blanket/configs/config.yaml`:
  - `input_folder`: path to your images (default: `../data/`)
  - `output_folder`: path for results (default: `outputs/`)

2. Run the demo script:

```bash
python run_image_anonymization.py
```

This will process all images in your input folder, apply face detection and three anonymization methods (black box, pixelation, gaussian blur), and save composite results to your output folder.

You can adjust detection and anonymization settings in `blanket/configs/config.yaml`.


## 📊 Evaluation

BLANKET outperforms DeepPrivacy V2 in all measured metrics.


| Metric                          | BLANKET       | DeepPrivacy2  |
| ------------------------------- | ------------- | ------------- |
| Identity cosine distance (↓)    | 0.11 ± 0.18   | 0.19 ± 0.26   |
| Emotion preservation (↑)        | 0.51 ± 0.13   | 0.27 ± 0.11   |
| Temporal landmark corr. (↑)     | 0.956 ± 0.064 | 0.860 ± 0.140 |
| Detection AP vs. orig. (↑)      | 90.7 mAP      | 81.5 mAP      |
| Pose AP vs. orig. (↑)           | 97.2 mAP      | 79.1 mAP      |


## 🗺️ Roadmap

We will continue refining BLANKET with a focus on quality and reliability:

- [ ] Implement Stable Diffusion–based inpainting 
- [ ] Implement FaceFusion in video and a video demo  
- [ ] Ensure robust anonymization in frames where faces are not detected  

## 🙏 Acknowledgments

Supported by GA CR 25-18113S, EC Digital Europe CEDMO 2.0 101158609, CTU SGS23/173/OHK3/3T/13.
Thanks to Adéla Šubrtová for early feedback.
Thanks to Max Familly Fun for the banner picture.

## 📝 Citation

If you use BLANKET, please cite:

```bibtex
@inproceedings{hadera2025BLANKET,
  author = {Hadera, Ditmar and Cech, Jan and Purkrabek, Miroslav and Hoffmann, Matej},
  booktitle = {2025 IEEE International Conference on Development and Learning (ICDL)},
  month = sep,
  pages = {1--8},
  title = {{BLANKET: Anonymizing Faces in Infant Video Recordings}},
  year = {2025}
}
```

