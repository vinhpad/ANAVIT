# ANAVIT: Enhancing Document-Level Relation Extraction with Anaphor Nodes and Visual Transformation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/IEEE-11309552-red.svg)](https://ieeexplore.ieee.org/document/11309552/)

Official implementation of the paper
**"ANAVIT: Enhancing Document-Level Relation Extraction with Anaphor Nodes and Visual Transformation"**
*An Duc Vinh Pham, Quynh-Trang Pham Thi, Thanh-Hai Dang* — KSE 2025.

---

## Abstract

ANAVIT is a framework for document-level relation extraction (DocRE) that combines graph-based anaphora resolution with visual transformation of the entity-pair relation matrix. It introduces two complementary components:

1. **Anaphor nodes** explicitly modeling pronoun-entity coreference relationships in a heterogeneous document graph, processed by Graph Attention Networks (GATs).
2. **Visual transformation** of the entity-pair relation matrix using 2D convolutions and multi-head self-attention to capture global spatial reasoning patterns across the document.

## Architecture

![ANAVIT Method Overview](anavit_method.png)

The ANAVIT pipeline consists of:

1. **Encoder Module** — SciBERT-based contextualized representations with long-input handling.
2. **GAT-based Entity Enhancement**
   - Anaphor detection via the spaCy POS tagger.
   - Heterogeneous document graph combining mention nodes and anaphor nodes.
   - Stacked Graph Attention layers for node interaction.
3. **Visual Transformation Module**
   - 2D convolutions (5×5 kernels) over the relation feature map.
   - Multi-head self-attention on the resulting spatial patterns.
4. **Adaptive Thresholding (ATLoss)** — learnable per entity-pair thresholds.

## Key Results

| Dataset | ANAVIT (F1) | ATLOP (F1) | Δ            |
|---------|-------------|------------|--------------|
| CDR     | **79.1**    | 69.4       | +9.7         |
| GDA     | **84.7**    | 83.9       | +0.8         |

See the [paper](ANAVIT_final.pdf) for the full ablation and comparison tables.

## Repository Structure

```
.
├── model.py            # ANAVIT DocRE model (encoder + GAT + ViT + ATLoss)
├── graph.py            # GAT-based attention layers over the document graph
├── vit.py              # Visual transformation module (Conv + MHSA)
├── losses.py           # Adaptive-Thresholding loss
├── long_seq.py         # Long-sequence handling for the encoder
├── prepro.py           # Pre-processing for CDR / GDA
├── evaluation.py       # Evaluation utilities
├── train_cdr.py        # CDR training entry point
├── train_gda.py        # GDA training entry point
├── scripts/            # Reproducibility shell scripts
│   ├── run_cdr.sh
│   └── run_gda.sh
├── dataset/
│   ├── cdr/            # CDR train/dev/test (filtered)
│   └── gda/            # GDA train/dev/test
├── requirements.txt
└── ANAVIT_final.pdf
```

## Installation

```bash
git clone https://github.com/vinhpad/anavit.git
cd anavit
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Tested with Python 3.8+, PyTorch 2.4 and CUDA 12.4 (DGL is installed from the matching wheel index in `requirements.txt`).

## Datasets

The benchmarks used in the paper are included under `dataset/`:

- **CDR** (Chemical-Disease Relations) — `dataset/cdr/{train,dev,test}_filter.data`
- **GDA** (Gene-Disease Associations) — `dataset/gda/{train,dev,test}.data`

Both datasets follow the format consumed by `prepro.read_cdr` / `prepro.read_gda`.

## Training

### CDR

```bash
bash scripts/run_cdr.sh
```

### GDA

```bash
bash scripts/run_gda.sh
```

## Citation

If you use this code or build on this work, please cite:

```bibtex
@INPROCEEDINGS{11309552,
  author={Pham, An Duc Vinh and Thi, Quynh-Trang Pham and Dang, Thanh-Hai},
  booktitle={2025 17th International Conference on Knowledge and System Engineering (KSE)}, 
  title={ANAVIT: Enhancing Document-Level Relation Extraction with Anaphor Nodes and Visual Transformation}, 
  year={2025},
  volume={},
  number={},
  pages={1-6},
  abstract={Current transformer-based models for document-level relation extraction effectively utilize techniques such as adaptive thresholding and localized context pooling. However, their performance is often limited by an inability to resolve anaphoric references or identify global relational patterns. This presents a significant challenge in domains like biomedicine, where understanding complex relationships requires analyzing information across multiple sentences. We introduce ANAVIT (Anaphor-Assisted Visual Transformation). Our method enhances DocRE through a dual-component architecture. First, it constructs a document graph incorporating dedicated anaphor nodes, employing Graph Attention Networks to explicitly model pronoun-entity coreferences. Second, it applies a visual transformation using convolutional neural networks andmulti-head attention to the resulting relation matrices, treating them as spatial patterns to capture global dependencies. We evaluated ANAVIT on the standard CDR and GDA biomedical datasets. Experiments show that our method achieves new state-of-theart performance on CDR with a 79.1 % F1 score, substantially outperforming the ATLOP baseline of 69.4 %. This performance gain is particularly notable in documents that feature complex coreference chains, confirming the efficacy of our graph-based enhancements for anaphora resolution. Our source code is available at: https://github.com/vinhpad/anavit},
  keywords={Knowledge engineering;Visualization;Current transformers;Biological system modeling;Source coding;Linguistics;Performance gain;Feature extraction;Graph neural networks;Standards;Document-Level Relation Extraction;Graph Neural Networks;Anaphor Resolution;visual transformation;Adaptive Thresholding},
  doi={10.1109/KSE68178.2025.11309552},
  ISSN={2694-4804},
  month={Nov},}
```

## Acknowledgments

This work builds on the [ATLOP](https://github.com/wzhouad/ATLOP) framework and integrates advances in graph neural networks and visual processing for NLP. We thank the biomedical NLP community for releasing the CDR and GDA benchmarks.

## License

This project is released under the MIT License — see [LICENSE](LICENSE) for details.

## Contact

- GitHub: <https://github.com/vinhpad/anavit>
- Issues and questions are welcome via the repository's issue tracker.
