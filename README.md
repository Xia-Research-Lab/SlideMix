<div align="center">

<img src="assets/slidemix.png" alt="SlideMix" width="100%">

# SlideMix

### *Enhancing Whole Slide Image Analysis via Multimodal Shuffling*

[![arXiv](https://img.shields.io/badge/arXiv-2609.00396-B31B1B.svg?style=for-the-badge)](https://arxiv.org/abs/2609.00396)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg?style=for-the-badge)](LICENSE)

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)

[![Open In Colab](https://img.shields.io/badge/Open_In_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/Xia-Research-Lab/SlideMix/blob/main/demo.ipynb)
[![Demo Data](https://img.shields.io/badge/Demo_Data-Release-111827?style=for-the-badge&logo=github)](https://github.com/Xia-Research-Lab/SlideMix/releases/tag/demo-data)

<p align="center">
  <a href="#-highlights">Highlights</a> |
  <a href="#-demo">Demo</a> |
  <a href="#️-installation">Installation</a> |
  <a href="#-data-preparation">Data Preparation</a> |
  <a href="#-usage">Usage</a> |
  <a href="#-citation">Citation</a>
</p>

</div>

## 📢 Updates

|     Date      | Update                                                                                                   |
| :-----------: | :------------------------------------------------------------------------------------------------------- |
| 🧪 **2026.11** | Demo notebook and [demo data](https://github.com/Xia-Research-Lab/SlideMix/releases/tag/demo-data) released. |
| 📄 **2026.10** | Preprint available on [arXiv](https://arxiv.org/abs/2609.00396) and code released.                       |

---

## ✨ Highlights

Official implementation of **SlideMix**, a data augmentation method for multiple instance learning (MIL) on whole slide images (WSIs). SlideMix creates new training slides by shuffling tile embeddings between two slides in place, guided by vision-language models, and assigns soft labels to the mixed slides.

---

## 🔬 Demo

> **Note:** The demo is still a work in progress and may change.

[`demo.ipynb`](demo.ipynb) visualizes how SlideMix mixes two slides: VARS, alignment, In-place Tile Shuffling, the curriculum stages and the soft labels.

The demo requires a GPU runtime (in Colab: Runtime > Change runtime type > GPU) and a Hugging Face account with access to [GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) and [CONCH](https://huggingface.co/MahmoodLab/conch). An API key is optional.

The demo set (two CAMELYON16 slides and their labels, 0.88 GB) is published as the [`demo-data` release](https://github.com/Xia-Research-Lab/SlideMix/releases/tag/demo-data) and is downloaded by the notebook. To download it manually:

```bash
wget https://github.com/Xia-Research-Lab/SlideMix/releases/download/demo-data/demo_set.zip
unzip demo_set.zip
```

---

## 🛠️ Installation

```bash
git clone https://github.com/Xia-Research-Lab/SlideMix.git
cd SlideMix

conda create -n slidemix python=3.12
conda activate slidemix
pip install -r requirements.txt
```

CONCH weights are downloaded from the Hugging Face Hub and require access to [MahmoodLab/conch](https://huggingface.co/MahmoodLab/conch). Set `HF_TOKEN`, or pass a local checkpoint with `--conch_checkpoint`.

---

## 📚 Data Preparation

SlideMix trains on tile embeddings stored as one `.safetensors` file per slide, with `features` `[N, D]` and `coords_yx` `[N, 2]` (level-0 pixel coordinates):

```text
<data_path>/<dataset_name>/tiles-embeddings/<slide_id>.safetensors         # GigaPath
<data_path>/<dataset_name>/tiles-embeddings_conch/<slide_id>.safetensors   # CONCH (for VARS and VLM labels)
```

Labels and fold assignments are read from `dataset_configs/<dataset_name>/`. Configurations are provided for PANDA, CAMELYON16, IMP-CRS-2024, TCGA-lung, TCGA-BLCA, TCGA-BRCA, TCGA-CESC, TCGA-GBM, TCGA-UCEC, TCGA-UCS and TCGA-UVM.

---

## 🚀 Usage

### 1. Retrieval corpus (optional)

```bash
python tools/slidemix_rag.py fetch --dataset_name CAMELYON16 --out rag/docs.jsonl
python tools/slidemix_rag.py build --docs rag/docs.jsonl --out_dir rag/index
```

### 2. ROI masks (VARS)

```bash
python tools/slidemix_var_selector.py --data_path $DATA --dataset_name CAMELYON16 \
    --mode gemini --wsi_dir $WSI --rag_index rag/index --out_dir slidemix/roi --device cuda
```

`--mode conch` grounds the region prompt with CONCH alone, without raw WSIs.

### 3. VLM soft-label scores

```bash
python tools/slidemix_vlm_scores.py --data_path $DATA --dataset_name CAMELYON16 \
    --rag_index rag/index --out_dir slidemix/vlm --device cuda
```

### 4. Training

Baseline:

```bash
python main.py --mode train --data_path $DATA --dataset_name CAMELYON16 \
    --model_name ABMIL --embedding_name GigaPath --fold 1 --devices 0
```

With SlideMix:

```bash
python main.py --mode train --data_path $DATA --dataset_name CAMELYON16 \
    --model_name ABMIL --embedding_name GigaPath --fold 1 --devices 0 \
    --slidemix --slidemix_roi_dir slidemix/roi --slidemix_vlm_dir slidemix/vlm
```

The best checkpoint is evaluated on the test split after training. Mixing is applied to the training split only.

### Main SlideMix Options

| Flag                       | Default             | Description                                                        |
| -------------------------- | ------------------- | ------------------------------------------------------------------ |
| `--slidemix_prob`          | `0.5`               | Probability that a training slide is mixed                         |
| `--slidemix_N`             | `16,8,4,2,1`        | Block granularity per curriculum stage                             |
| `--slidemix_TPCA`          | `1.0,0.8,0.6,0.4,0.2` | PCA distance threshold per stage                                 |
| `--slidemix_P`             | `0.1,0.2,0.3,0.4,0.5` | Shuffling ratio per stage                                        |
| `--slidemix_strategy`      | `hold`              | Behaviour when the loss does not improve: `hold`, `back`, `fixed`  |
| `--slidemix_warmup_epochs` | `20`                | Epochs without mixing before the curriculum starts                 |
| `--slidemix_label`         | `vlm`               | Soft-label method: `vlm`, `linear`, `random`                       |

Run `python main.py --help` for the full list.

---

## 📁 Project Structure

```text
SlideMix/
├── assets/                        # Banner and figures
├── dataset_configs/               # Task definitions and 5-fold splits
├── models/
│   ├── ABMIL.py                   # MIL backbone
│   ├── CONCH/                     # CONCH model code
│   └── clip/                      # BiomedCLIP model code
├── tools/
│   ├── slidemix_rag.py            # PubMed retrieval corpus and index
│   ├── slidemix_var_selector.py   # Per-slide ROI masks (VARS)
│   └── slidemix_vlm_scores.py     # Per-tile VLM class scores
├── utils/
│   ├── slidemix.py                # Alignment, PCA filter, ITS, curriculum, soft labels
│   ├── slidemix_callback.py       # Curriculum feedback callback
│   ├── slidemix_vlm.py            # CONCH text encoding and tile-level scoring
│   └── dataset.py, model.py, utils.py, CONSTS.py
├── main.py                        # Training / testing entry point
├── demo.ipynb                     # SlideMix visualization demo
├── requirements.txt               # Python dependencies
├── CITATION.cff                   # Citation metadata
├── LICENSE                        # Apache 2.0 license
└── README.md                      # This file
```

---

## 📄 Citation

If you find SlideMix useful, use any part of our code, or draw inspiration from our work, we would appreciate it if you could cite our paper:

```bibtex
@article{wong2026slidemix,
  title={SlideMix: Enhancing Whole Slide Image Analysis via Multimodal Shuffling},
  author={Wong, Chad and Chen, Sicheng and Zhang, Tianyi and Chai, Enhui and Jin, Yueming and Liu, Zeyu and Xia, Fei},
  journal={arXiv preprint arXiv:2609.00396},
  year={2026},
  doi={10.48550/arXiv.2609.00396},
  url={https://arxiv.org/abs/2609.00396}
}
```

---

## 📜 License

This project is licensed under the Apache 2.0 License. See [`LICENSE`](LICENSE) for details.

---

## 🙏 Acknowledgments

<table>
<tr>
<td align="center">
<a href="https://github.com/mahmoodlab/CONCH">
<img src="https://img.shields.io/badge/CONCH-VLM-purple?style=flat-square" alt="CONCH">
</a>
<br><sub>Pathology Vision-Language Model</sub>
</td>
<td align="center">
<a href="https://github.com/mlfoundations/open_clip">
<img src="https://img.shields.io/badge/Open-CLIP-blue?style=flat-square" alt="OpenCLIP">
</a>
<br><sub>CLIP Training Framework</sub>
</td>
<td align="center">
<a href="https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224">
<img src="https://img.shields.io/badge/Biomed-CLIP-green?style=flat-square" alt="BiomedCLIP">
</a>
<br><sub>Biomedical Vision-Language Model</sub>
</td>
<td align="center">
<a href="https://github.com/AMLab-Amsterdam/AttentionDeepMIL">
<img src="https://img.shields.io/badge/AB-MIL-yellow?style=flat-square" alt="ABMIL">
</a>
<br><sub>Attention-based MIL</sub>
</td>
</tr>
</table>

---

<div align="center">

**⭐ Star us on GitHub — it motivates us a lot!**

<p>
  <a href="https://github.com/Xia-Research-Lab/SlideMix/stargazers">
    <img src="https://img.shields.io/github/stars/Xia-Research-Lab/SlideMix?style=social" alt="Stars">
  </a>
  <a href="https://github.com/Xia-Research-Lab/SlideMix/network/members">
    <img src="https://img.shields.io/github/forks/Xia-Research-Lab/SlideMix?style=social" alt="Forks">
  </a>
  <a href="https://github.com/Xia-Research-Lab/SlideMix/watchers">
    <img src="https://img.shields.io/github/watchers/Xia-Research-Lab/SlideMix?style=social" alt="Watchers">
  </a>
</p>

</div>
