# RT-Focuser-V2

Official implementation of **RT-Focuser-V2: Progressive All-Scale Attention for Real-Time Image Deblurring**, submitted to IEEE ICTA 2026.

RT-Focuser-V2 is a compact multi-exit image deblurring network built from lightweight LD blocks, an Efficient Global Attention Module (E-GAM), and progressive restoration exits. A deployment selects one of four fixed exits before inference to obtain a deterministic quality-latency operating point.

## Results

GoPro benchmark:

| Mode | Exit | PSNR | SSIM | Params | GMACs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Accurate | Y4 | 31.03 | 0.9172 | 4.72M | 12.89 |
| Balanced | Y3 | 30.26 | 0.9007 | 1.80M | 4.49 |
| Fast | Y2 | 28.37 | 0.8743 | 1.27M | 4.07 |
| Fastest | Y1 | 25.67 | 0.8012 | 1.25M | 3.63 |

Cross-platform throughput at batch size 1:

| Mode | iPhone 15 | Apple M4 | Intel i7-13700H | RTX 3090 |
| --- | ---: | ---: | ---: | ---: |
| Fastest (Y1) | 259.06 FPS | 434.78 FPS | 54.45 FPS | 309.27 FPS |
| Fast (Y2) | 223.95 FPS | 384.61 FPS | 45.67 FPS | 278.95 FPS |
| Balanced (Y3) | 194.55 FPS | 322.58 FPS | 37.30 FPS | 253.84 FPS |
| Accurate (Y4) | 134.58 FPS | 212.76 FPS | 18.18 FPS | 183.48 FPS |

All latency measurements use a fixed-exit deployment graph, batch size 1, and a 256 x 256 RGB input. Core ML is used on Apple devices, OpenVINO on Intel CPU, and ONNX Runtime CUDA on RTX 3090.

## Installation

```bash
git clone https://github.com/ReaganWu/RT-Focuser-v2.git
cd RT-Focuser-v2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Inference

Place the ICTA 2026 checkpoint at `weights/rt-focuser-v2-gopro.pth`, then run:

```bash
python inference.py \
  --checkpoint weights/rt-focuser-v2-gopro.pth \
  --input examples/blur.png \
  --output results/restored.png \
  --exit 4
```

Use `--exit 1`, `2`, `3`, or `4` for the Fastest, Fast, Balanced, or Accurate operating mode.

## Evaluation

Arrange the GoPro dataset as follows:

```text
GOPRO_Large/
  train/<sequence>/blur_gamma/*.png
  train/<sequence>/sharp/*.png
  test/<sequence>/blur_gamma/*.png
  test/<sequence>/sharp/*.png
```

Evaluate a fixed exit:

```bash
python evaluate.py \
  --data-root /path/to/GOPRO_Large \
  --checkpoint weights/rt-focuser-v2-gopro.pth \
  --exit 4
```

## Training

```bash
python train.py \
  --data-root /path/to/GOPRO_Large \
  --output-dir runs/rt-focuser-v2 \
  --device cuda
```

The training objective applies equal supervision to all exits using L1 and FFT-L1 reconstruction terms, with progressive self-distillation from Y4 to earlier exits.

## Fixed-exit export

Export one ONNX graph at a time so unused later decoder stages are removed from the deployment graph:

```bash
python rt_focuser_v2.py \
  --checkpoint weights/rt-focuser-v2-gopro.pth \
  --exit 4 \
  --out-dir exports/Y4 \
  --skip-coreml
```

## Citation

```bibtex
@inproceedings{wu2026rtfocuserv2,
  title={RT-Focuser-V2: Progressive All-Scale Attention for Real-Time Image Deblurring},
  author={Wu, Zhuoyu and Ou, Wenhui and Tan, Pei-Sze and Zheng, Qiawei and Wang, Quanjun and Fang, Wenqi and Wang, Zheng and Phan, Raphael C.-W.},
  booktitle={2026 IEEE International Conference on Integrated Circuits, Technologies and Applications (ICTA)},
  year={2026}
}
```

## License

This project is released under the MIT License. The GoPro dataset is distributed separately under its own terms.
