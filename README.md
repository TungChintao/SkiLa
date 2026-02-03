<div align="center">
  <h1 style="display: inline-block; margin: 0;">Sketch-in-Latents: Eciliting Unified Reasoning in MLLMs
  </h1>
</div>


<h4 align="center"> 
Jintao Tong<sup>1</sup>,
Jiaqi Gu<sup>2</sup>, 
Yujing Lou<sup>2</sup>, 
Lubin Fan<sup>2✉</sup>, 
Yue Wu<sup>2</sup>,<br>
Jieping Ye<sup>2</sup>,
Ruixuan Li<sup>1✉</sup>,
Yixiong Zou<sup>1✉</sup>
<br><br> 
<sup>1</sup>Huazhong University of Science and Technology<br> <sup>2</sup>Alibaba Cloud Computing

</h4>

<div align="center">
	
[![arXiv](https://img.shields.io/badge/arXiv-2505.19536-AD1C18.svg?logo=arXiv)](https://arxiv.org/pdf/2505.19536)
[![HF](https://img.shields.io/badge/%F0%9F%A4%97%20Model-SkiLa_7B-yellow)](https://huggingface.co/JosephTong/SkiLa-7B)
</div>

## 🔥 News

* **`2026.02.03`** 🤗 The checkpoints of [SkiLa 7B](https://huggingface.co/JosephTong/SkiLa-7B) is released!
* **`2026.02.03`** 🚀 [Code](https://github.com/TungChintao/SkiLa) is released ！
* **`2025.12.16`** 📝 We release our latest work [Sketch-in-Latents (SkiLa)](https://arxiv.org/abs/2512.16584), a novel unified
reasoning MLLMs to flexibly and seamlessly interleave multi-step explicit textual thoughts and latent visual thoughts.

## 💡 Highlights
<p align='center'>
<img src='https://github.com/TungChintao/FlowCut/blob/main/methods.jpg' alt='mask' width='950px'>
</p>


> **TLDR:** We propose SkiLa (Sketch-in-Latents), a unified multimodal reasoning paradigm that enables MLLMs to autoregressively generate continuous visual embeddings as visual thoughts alongside text tokens. The model alternates between textual thinking and visual sketching during multi step reasoning, and uses a semantic reconstruction mechanism to keep the latent sketches grounded. 


## 🛠 Preparation

### 1. Code

```
git clone https://github.com/TungChintao/SkiLa.git
cd SkiLa

pip install -r requirements.txt
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
```

### 2. Training Data
Download Datasets of [Zebra-CoT](https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT)


## 🎯 Training

To run the training script, use the following command:

```Shell
bash scripts/train_skila.sh
```

## 📖 Evaluation
We adopt [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) to conduct the evaluation. You can get started as follows:

### 1. Install 

```Shell
cd VLMEvalKit
pip install -e.
```

### 2. Inference

```Shell
bash test.sh
```

See here [[QuickStar](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Quickstart.md) | [快速开始](https://github.com/open-compass/VLMEvalKit/blob/main/docs/zh-CN/Quickstart.md)] for more details about arguments.


## 🔑 License

- This project is released under the [Apache 2.0 license](https://github.com/TungChintao/SkiLa/blob/main/LICENSE).

## 📌 Citation

If you find this project useful in your research, please consider citing:

```bibtex
@article{tong2025sketch,
  title={Sketch-in-latents: Eliciting unified reasoning in mllms},
  author={Tong, Jintao and Gu, Jiaqi and Lou, Yujing and Fan, Lubin and Zou, Yixiong and Wu, Yue and Ye, Jieping and Li, Ruixuan},
  journal={arXiv preprint arXiv:2512.16584},
  year={2025}
}
```


## 👍 Acknowledgment
- We sincerely thank [Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune), [LVR](https://github.com/Gumpest/SparseVLMs), [Zebra-CoT](https://github.com/multimodal-reasoning-lab/Bagel-Zebra-CoT) and others for their contributions, which have provided valuable insights.
