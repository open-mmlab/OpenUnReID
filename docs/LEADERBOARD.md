## Leaderboard

A curated list of unsupervised and domain adaptive re-ID papers, as well as their results. To add some papers not included, you could create an issue or a pull request. **Note:** the following results are copied from their original papers.

### Contents

+ [Unsupervised learning (USL) on object re-ID](#unsupervised-learning-on-object-re-id)
  + [Market-1501](#market-1501)
  + [DukeMTMC-reID](#dukemtmc-reid)
  + [MSMT17](#msmt17)
+ [Unsupervised domain adaptation (UDA) on object re-ID](#unsupervised-domain-adaptation-on-object-re-id)
  + [Market-1501 -> DukeMTMC-reID](#market-1501---dukemtmc-reid)
  + [DukeMTMC-reID -> Market-1501](#dukemtmc-reid---market-1501)
  + [Market-1501 -> MSMT17](#market-1501---msmt17)
  + [DukeMTMC-reID -> MSMT17](#dukemtmc-reid---msmt17)


### Unsupervised learning on object re-ID

#### Market-1501

| Method | Venue | Code | mAP(%) | R@1(%) | R@5(%) | R@10(%) | Reference |
| ------ | :------: | :----: | :------: | :------: | :-------: | :------: | :------ |
| SpCL+ | arXiv'20 | [OpenUnReID](../tools/SpCL) | 76.0 | 89.5 | 96.2 | 97.5 | [Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID](https://arxiv.org/pdf/2006.02713.pdf) |
| MMT+ | ICLR'20 | [OpenUnReID](../tools/MMT) | 74.3 | 88.1 | 96.0 | 97.5 | [Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification](https://openreview.net/pdf?id=rJlnOhVYPS) |
| SpCL | arXiv'20 | [PyTorch](https://github.com/yxgeee/SpCL) | 72.6 | 87.7 | 95.2 | 96.9 | [Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID](https://arxiv.org/pdf/2006.02713.pdf) |
| HCT | CVPR'20 | [Empty](https://github.com/zengkaiwei/HCT) | 56.4 | 80.0 | 91.6 | 95.2 | [Hierarchical Clustering with Hard-batch Triplet Loss for Person Re-identification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zeng_Hierarchical_Clustering_With_Hard-Batch_Triplet_Loss_for_Person_Re-Identification_CVPR_2020_paper.pdf) |
| MMCL | CVPR'20 | [PyTorch](https://github.com/kennethwdk/MLCReID) | 45.5 | 80.3 | 89.4 | 92.3 | [Unsupervised Person Re-Identification via Multi-Label Classification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Unsupervised_Person_Re-Identification_via_Multi-Label_Classification_CVPR_2020_paper.pdf) |
| SSL | CVPR'20 | [PyTorch (Unofficial)](https://github.com/ryanaleksander/softened-similarity-learning) | 37.8 | 71.7 | 83.8 | 87.4 | [Unsupervised Person Re-identification via Softened Similarity Learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_Unsupervised_Person_Re-Identification_via_Softened_Similarity_Learning_CVPR_2020_paper.pdf) |
| BUC | AAAI'19 | [PyTorch](https://github.com/vana77/Bottom-up-Clustering-Person-Re-identification) | 38.3 | 66.2 | 79.6 | 84.5 | [A Bottom-up Clustering Approach to Unsupervised Person Re-identification](https://vana77.github.io/vana77.github.io/images/AAAI19.pdf) |

#### DukeMTMC-reID

| Method | Venue | Code | mAP(%) | R@1(%) | R@5(%) | R@10(%) | Reference |
| ------ | :------: | :----: | :------: | :------: | :-------: | :------: | :------ |
| SpCL+ | arXiv'20 | [OpenUnReID](../tools/SpCL) | 67.1 | 82.4 | 90.8 | 93.0 | [Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID](https://arxiv.org/pdf/2006.02713.pdf) |
| MMT+ | ICLR'20 | [OpenUnReID](../tools/MMT) | 60.3 | 75.6 | 86.0 | 89.2 | [Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification](https://openreview.net/pdf?id=rJlnOhVYPS) |
| SpCL | arXiv'20 | [PyTorch](https://github.com/yxgeee/SpCL) | 65.3 | 81.2 | 90.3 | 92.2 | [Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID](https://arxiv.org/pdf/2006.02713.pdf) |
| HCT | CVPR'20 | [Empty](https://github.com/zengkaiwei/HCT) | 50.7 | 69.6 | 83.4 | 87.4 | [Hierarchical Clustering with Hard-batch Triplet Loss for Person Re-identification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zeng_Hierarchical_Clustering_With_Hard-Batch_Triplet_Loss_for_Person_Re-Identification_CVPR_2020_paper.pdf) |
| MMCL | CVPR'20 | [PyTorch](https://github.com/kennethwdk/MLCReID) | 40.2 | 65.2 | 75.9 | 80.0 | [Unsupervised Person Re-Identification via Multi-Label Classification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Unsupervised_Person_Re-Identification_via_Multi-Label_Classification_CVPR_2020_paper.pdf) |
| SSL | CVPR'20 | [PyTorch (Unofficial)](https://github.com/ryanaleksander/softened-similarity-learning) | 28.6 | 52.5 | 63.5 | 68.9 | [Unsupervised Person Re-identification via Softened Similarity Learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_Unsupervised_Person_Re-Identification_via_Softened_Similarity_Learning_CVPR_2020_paper.pdf) |
| BUC | AAAI'19 | [PyTorch](https://github.com/vana77/Bottom-up-Clustering-Person-Re-identification) | 27.5 | 47.4 | 62.6 | 68.4 | [A Bottom-up Clustering Approach to Unsupervised Person Re-identification](https://vana77.github.io/vana77.github.io/images/AAAI19.pdf) |

#### MSMT17

| Method | Venue | Code | mAP(%) | R@1(%) | R@5(%) | R@10(%) | Reference |
| ------ | :------: | :----: | :------: | :------: | :-------: | :------: | :------ |
| MMCL | CVPR'20 | [PyTorch](https://github.com/kennethwdk/MLCReID) | 11.2 | 35.4 | 44.8 | 49.8 | [Unsupervised Person Re-Identification via Multi-Label Classification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Unsupervised_Person_Re-Identification_via_Multi-Label_Classification_CVPR_2020_paper.pdf) |

### Unsupervised domain adaptation on object re-ID

#### Market-1501 -> DukeMTMC-reID

| Method | Venue | Code | mAP(%) | R@1(%) | R@5(%) | R@10(%) | Reference |
| ------ | :------: | :----: | :------: | :------: | :-------: | :------: | :------ |
| SpCL+ | arXiv'20 | [OpenUnReID](../tools/SpCL) | 70.4 | 83.8 | 91.2 | 93.4 | [Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID](https://arxiv.org/pdf/2006.02713.pdf) |
| MMT+ | ICLR'20 | [OpenUnReID](../tools/MMT) | 67.7 | 80.3 | 89.9 | 92.9 | [Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification](https://openreview.net/pdf?id=rJlnOhVYPS) |
| SpCL | arXiv'20 | [PyTorch](https://github.com/yxgeee/SpCL) | 68.8 | 82.9 | 90.1 | 92.5 | [Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID](https://arxiv.org/pdf/2006.02713.pdf) |
| SDA | arXiv'20 | [PyTorch](https://github.com/yxgeee/SDA) | 61.4 | 76.5 | 86.6 | 89.7 | [Structured Domain Adaptation with Online Relation Regularization for Unsupervised Person Re-ID](https://arxiv.org/pdf/2003.06650.pdf) |
| MEB-Net | ECCV'20 | [PyTorch](https://github.com/YunpengZhai/MEB-Net) | 66.1 | 79.6 | 88.3 | 92.2 | [Multiple Expert Brainstorming for Domain Adaptive Person Re-identification](https://arxiv.org/pdf/2007.01546.pdf) |
| MMT | ICLR'20 | [PyTorch](https://github.com/yxgeee/MMT) | 65.1 | 78.0 | 88.8 | 92.5 | [Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification](https://openreview.net/pdf?id=rJlnOhVYPS) |
| AD-Cluster | CVPR'20 | [PyTorch](https://github.com/kennethwdk/MLCReID) | 54.1 | 72.6 | 82.5 | 85.5 | [AD-Cluster: Augmented Discriminative Clustering for Domain Adaptive Person Re-identification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhai_AD-Cluster_Augmented_Discriminative_Clustering_for_Domain_Adaptive_Person_Re-Identification_CVPR_2020_paper.pdf) |
| SNR | CVPR'20 | - | 58.1 | 76.3 | - | - | [Style Normalization and Restitution for Generalizable Person Re-identification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jin_Style_Normalization_and_Restitution_for_Generalizable_Person_Re-Identification_CVPR_2020_paper.pdf) |
| MMCL | CVPR'20 | [PyTorch](https://github.com/kennethwdk/MLCReID) | 51.4 | 72.4 | 82.9 | 85.0 | [Unsupervised Person Re-Identification via Multi-Label Classification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Unsupervised_Person_Re-Identification_via_Multi-Label_Classification_CVPR_2020_paper.pdf) |
| ECN++ | TPAMI'20 | - | 54.4 | 74.0 | 83.7 | 87.4 | [Learning to Adapt Invariance in Memory for Person Re-identification](https://ieeexplore.ieee.org/abstract/document/9018132) |
| UDA_TP | PR'20 | [PyTorch](https://github.com/LcDog/DomainAdaptiveReID) | 49.0 | 68.4 | 80.1 | 83.5 | [Unsupervised Domain Adaptive Re-Identification: Theory and Practice](https://arxiv.org/pdf/1807.11334.pdf) |
| SSG | ICCV'19 | [PyTorch](https://github.com/SHI-Labs/Self-Similarity-Grouping) | 53.4 | 73.0 | 80.6 | 83.2 | [Self-similarity Grouping: A Simple Unsupervised Cross Domain Adaptation Approach for Person Re-identification](https://openaccess.thecvf.com/content_ICCV_2019/papers/Fu_Self-Similarity_Grouping_A_Simple_Unsupervised_Cross_Domain_Adaptation_Approach_for_ICCV_2019_paper.pdf) |
| PCB-PAST | ICCV'19 | [PyTorch](https://github.com/zhangxinyu-xyz/PAST-ReID) | 54.3 | 72.4 | - | - | [Self-Training With Progressive Augmentation for Unsupervised Cross-Domain Person Re-Identification](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Self-Training_With_Progressive_Augmentation_for_Unsupervised_Cross-Domain_Person_Re-Identification_ICCV_2019_paper.pdf) |
| CR-GAN | ICCV'19 | - | 48.6 | 68.9 | 80.2 | 84.7 | [Instance-Guided Context Rendering for Cross-Domain Person Re-Identification](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Instance-Guided_Context_Rendering_for_Cross-Domain_Person_Re-Identification_ICCV_2019_paper.pdf) |
| PDA-Net | ICCV'19 | - | 45.1 | 63.2 | 77.0 | 82.5 | [Cross-Dataset Person Re-Identification via Unsupervised Pose Disentanglement and Adaptation](https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Cross-Dataset_Person_Re-Identification_via_Unsupervised_Pose_Disentanglement_and_Adaptation_ICCV_2019_paper.pdf) |
| UCDA | ICCV'19 | - | 31.0 | 47.7 | - | - | [A Novel Unsupervised Camera-aware Domain Adaptation Framework for Person Re-identification](https://openaccess.thecvf.com/content_ICCV_2019/papers/Qi_A_Novel_Unsupervised_Camera-Aware_Domain_Adaptation_Framework_for_Person_Re-Identification_ICCV_2019_paper.pdf) |
| ECN | CVPR'19 | [PyTorch](https://github.com/zhunzhong07/ECN) | 40.4 | 63.3 | 75.8 | 80.4 | [Invariance Matters: Exemplar Memory for Domain Adaptive Person Re-identification](https://arxiv.org/pdf/1904.01990.pdf) |
| HHL | ECCV'18 | [PyTorch](https://github.com/zhunzhong07/HHL) | 33.4 | 60.2 | 73.9 | 79.5 | [Generalizing A Person Retrieval Model Hetero- and Homogeneously](https://openaccess.thecvf.com/content_ECCV_2018/papers/Zhun_Zhong_Generalizing_A_Person_ECCV_2018_paper.pdf) |
| SPGAN | CVPR'18 | [PyTorch](https://github.com/Simon4Yan/eSPGAN) | 22.3 | 41.1 | 56.6 | 63.0 | [Image-Image Domain Adaptation with Preserved Self-Similarity and Domain-Dissimilarity for Person Re-identification](https://openaccess.thecvf.com/content_cvpr_2018/papers/Deng_Image-Image_Domain_Adaptation_CVPR_2018_paper.pdf) |
| TJ-AIDL | CVPR'18 | - | 23.0 | 44.3 | 59.6 | 65.0 | [Transferable Joint Attribute-Identity Deep Learning for Unsupervised Person Re-Identification](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Transferable_Joint_Attribute-Identity_CVPR_2018_paper.pdf) |
| PUL | TOMM'18 | [PyTorch](https://github.com/hehefan/Unsupervised-Person-Re-identification-Clustering-and-Fine-tuning) | 16.4 | 30.0 | 43.4 | 48.5 | [Unsupervised Person Re-identification: Clustering and Fine-tuning](https://hehefan.github.io/pdfs/unsupervised-person-identification.pdf) |

#### DukeMTMC-reID -> Market-1501

#### Market-1501 -> MSMT17

#### DukeMTMC-reID -> MSMT17
