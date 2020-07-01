<img src="docs/open_mmlab.png" align="right" width="30%">

# OpenUnReID

## Introduction
`OpenUnReID` is an open-source PyTorch-based codebase for both unsupervised learning (**USL**) and unsupervised domain adaptation (**UDA**) in the context of object re-ID tasks. It provides strong baselines and multiple state-of-the-art methods with highly refactored codes for both *pseudo-label-based* and *domain-translation-based* frameworks. It works with **Python >=3.5** and **PyTorch >=1.1**.

We are actively updating this repo, and more methods will be supported soon. Contributions are welcome.

### Major features
- [x] Distributed training & testing with multiple GPUs and multiple machines.
- [x] High flexibility on various combinations of datasets, backbones, losses, etc.
- [x] GPU-based pseudo-label generation and k-reciprocal re-ranking with quite high speed.
- [x] Plug-and-play domain-specific BatchNorms for any backbones, sync BN is also supported.
- [x] A strong cluster baseline, providing high extensibility on designing new methods.
- [x] State-of-the-art methods and performances for both USL and UDA problems on object re-ID.

### Supported methods

Please refer to [MODEL_ZOO.md](docs/MODEL_ZOO.md) for trained models and download links.

| Method | Reference | USL | UDA |
| ------ | :---: | :-----: | :-----: |
| [UDA_TP](tools/UDA_TP) | [PR'20 (arXiv'18)](https://arxiv.org/abs/1807.11334) | ✓ | ✓ |
| SPGAN  | [CVPR'18](https://arxiv.org/abs/1711.07027) | n/a  |  ongoing |  
| SSG | [ICCV'19](https://arxiv.org/abs/1811.10144) | ongoing  | ongoing  |  
| [strong_baseline](tools/strong_baseline) | Sec. 3.1 in [ICLR'20](https://openreview.net/pdf?id=rJlnOhVYPS) | ✓ | ✓ |
| [MMT](tools/MMT/) | [ICLR'20](https://openreview.net/pdf?id=rJlnOhVYPS) | ✓  | ✓  |  
| [SpCL](tools/SpCL/) | [arXiv'20](https://arxiv.org/abs/2006.02713) | ✓ |  ✓  |  
| SDA  | [arXiv'20](https://arxiv.org/abs/2003.06650) | n/a  |  ongoing |  


## Updates

+ [2020-07-01] `OpenUnReID` v0.1.0 is released.

## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for installation and dataset preparation.

## Get Started

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) for the basic usage of `OpenUnReID`.

## License

`OpenUnReID` is released under the [Apache 2.0 license](LICENSE).

## Citation

If you use this toolbox or models in your research, please consider cite:
```
@inproceedings{ge2020mutual,
  title={Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification},
  author={Yixiao Ge and Dapeng Chen and Hongsheng Li},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=rJlnOhVYPS}
}

@misc{ge2020selfpaced,
    title={Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID},
    author={Yixiao Ge and Dapeng Chen and Feng Zhu and Rui Zhao and Hongsheng Li},
    year={2020},
    eprint={2006.02713},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
<!-- @misc{ge2020structured,
    title={Structured Domain Adaptation with Online Relation Regularization for Unsupervised Person Re-ID},
    author={Yixiao Ge and Feng Zhu and Rui Zhao and Hongsheng Li},
    year={2020},
    eprint={2003.06650},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
} -->


## Acknowledgement

Some parts of `openunreid` are learned from [torchreid](https://github.com/KaiyangZhou/deep-person-reid) and [fastreid](https://github.com/JDAI-CV/fast-reid). We would like to thank for their projects, which have boosted the research of supervised re-ID a lot. We hope that `OpenUnReID` could well benefit the research community of unsupervised re-ID by providing strong baselines and state-of-the-art methods.

## Contact

This project is developed by Yixiao Ge ([@yxgeee](https://github.com/yxgeee)), Tong Xiao ([@Cysu](https://github.com/Cysu)), Zhiwei Zhang ([@Dreamerzzw](https://github.com/Dreamerzzw)).
