## Model Zoo

For the leaderboard on public benchmarks, please refer to [LEADERBOARD.md](LEADERBOARD.md).

**Note:** all the models below are selected by the performance on validation sets.

### Unsupervised learning (USL) on object re-ID

- `Direct infer` models are directly tested on the re-ID datasets with ImageNet pre-trained weights.

#### Market-1501

<!-- | Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet | 2.2 | 6.7 | 14.9 | 20.1 | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet | 34.7 | 58.6 | 74.0 | 78.9 | ~2h | [[config]](https://drive.google.com/file/d/1qzb9aVND9ueXYkXxBYl-WDFZ7aY6AR7N/view?usp=sharing) [[model]](https://drive.google.com/file/d/1JPiB4TNPmsYw-qBwEQsg44T6sGy6m8F5/view?usp=sharing) [[log]](https://drive.google.com/file/d/1ImlMaZCpzriq9ScHDfKW6CLmzuCX8KkA/view?usp=sharing) |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 70.5 | 87.9 | 95.7 | 97.1 | ~2.5h | [[config]](https://drive.google.com/file/d/13Fwe6ser_JKPIXVmnJd3KfBhsivP0OMa/view?usp=sharing) [[model]](https://drive.google.com/file/d/1lRMCDfIyji58oodAMJkl6ucPs4Lx6iws/view?usp=sharing) [[log]](https://drive.google.com/file/d/1IlwrtkLj7nJd7AXszFKFfASADbxSOBQ4/view?usp=sharing) |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 74.3 | 88.1 | 96.0 | 97.5 | ~4.5h | [[config]](https://drive.google.com/file/d/16GNU2qQdnmX9qYaqoy9w_DxU9myXjBo4/view?usp=sharing) [[model]](https://drive.google.com/file/d/1y-cSb_6gyigbRNPcsIT1ixOpeg1A9WDg/view?usp=sharing) [[log]](https://drive.google.com/file/d/1lPNykPY6AgfMtsVrqcG-4IQO8wD--2mp/view?usp=sharing) |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet | 76.0 | 89.5 | 96.2 | 97.5 | ~2h | [[config]](https://drive.google.com/file/d/1D4IEJhlqPvd8OZocavg0UZrHqpyr60sR/view?usp=sharing) [[model]](https://drive.google.com/file/d/1zMKSKYwdNsg2qKJEHpwvjuxzKoH0e4uE/view?usp=sharing) [[log]](https://drive.google.com/file/d/1Xn1RjFFJPfmlCPI0MUdT-FyppIdiKIY-/view?usp=sharing) | -->

| Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet | 2.2 | 6.7 | 14.9 | 20.1 | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet | 34.7 | 58.6 | 74.0 | 78.9 | ~2h | [[config]](https://drive.google.com/file/d/1qzb9aVND9ueXYkXxBYl-WDFZ7aY6AR7N/view?usp=sharing) [[model]](https://drive.google.com/file/d/1JPiB4TNPmsYw-qBwEQsg44T6sGy6m8F5/view?usp=sharing) |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 70.5 | 87.9 | 95.7 | 97.1 | ~2.5h | [[config]](https://drive.google.com/file/d/13Fwe6ser_JKPIXVmnJd3KfBhsivP0OMa/view?usp=sharing) [[model]](https://drive.google.com/file/d/1lRMCDfIyji58oodAMJkl6ucPs4Lx6iws/view?usp=sharing) |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 74.3 | 88.1 | 96.0 | 97.5 | ~4.5h | [[config]](https://drive.google.com/file/d/16GNU2qQdnmX9qYaqoy9w_DxU9myXjBo4/view?usp=sharing) [[model]](https://drive.google.com/file/d/1y-cSb_6gyigbRNPcsIT1ixOpeg1A9WDg/view?usp=sharing) |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet | 76.0 | 89.5 | 96.2 | 97.5 | ~2h | [[config]](https://drive.google.com/file/d/1D4IEJhlqPvd8OZocavg0UZrHqpyr60sR/view?usp=sharing) [[model]](https://drive.google.com/file/d/1zMKSKYwdNsg2qKJEHpwvjuxzKoH0e4uE/view?usp=sharing) |

#### DukeMTMC-reID

<!-- | Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet | 2.3 | 7.5 | 14.7 | 18.1 | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet | 42.3 | 64.4 | 76.0 | 79.9 | ~2h | [[config]](https://drive.google.com/file/d/1GOrQBdYINXK-RQ8OANuVpBpYly8aYzOs/view?usp=sharing) [[model]](https://drive.google.com/file/d/1N8cALZkOzIEcKdSWkCbG83tQ-ADBwa5E/view?usp=sharing) [[log]](https://drive.google.com/file/d/1xktR52dIItFpYHtr0A4u8kTfkwFmk29v/view?usp=sharing) |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 54.7 | 72.9 | 83.5 | 87.2 | ~2.5h | [[config]](https://drive.google.com/file/d/1fiuKgedqg839vfZMCdzUEnWVfmXE26TT/view?usp=sharing) [[model]](https://drive.google.com/file/d/1BUoshDWxAtY-L5nNYo2zUnOF6PjiqpyN/view?usp=sharing) [[log]](https://drive.google.com/file/d/1ofH_LoRXQeUyTArzNFoX6IxFFgyz46Y9/view?usp=sharing) |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 60.3 | 75.6 | 86.0 | 89.2 | ~4.5h | [[config]](https://drive.google.com/file/d/1kXKdq-mZ-wiWrgsss5Ny_vmdTSnvLAhH/view?usp=sharing) [[model]](https://drive.google.com/file/d/11qtWjAgGtjCa_G3G1hWLWj0Mpko9N7D3/view?usp=sharing) [[log]](https://drive.google.com/file/d/1LGSSooEeNXOQWueRSJW0Ypw_bnpYQDch/view?usp=sharing) |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet | 67.1 | 82.4 | 90.8 | 93.0 | ~2h | [[config]](https://drive.google.com/file/d/1QXrH0apN0QqsgU0Bie8Vk7kXKDfYxFL_/view?usp=sharing) [[model]](https://drive.google.com/file/d/1B5nlhSj8AfTpzSW8bkx-LMj-Loa1ENul/view?usp=sharing) [[log]](https://drive.google.com/file/d/1ezMtn8ZtM80Gck_cCs6ZyJC9a6efJ5xF/view?usp=sharing) | -->

| Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet | 2.3 | 7.5 | 14.7 | 18.1 | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet | 42.3 | 64.4 | 76.0 | 79.9 | ~2h | [[config]](https://drive.google.com/file/d/1GOrQBdYINXK-RQ8OANuVpBpYly8aYzOs/view?usp=sharing) [[model]](https://drive.google.com/file/d/1N8cALZkOzIEcKdSWkCbG83tQ-ADBwa5E/view?usp=sharing) |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 54.7 | 72.9 | 83.5 | 87.2 | ~2.5h | [[config]](https://drive.google.com/file/d/1fiuKgedqg839vfZMCdzUEnWVfmXE26TT/view?usp=sharing) [[model]](https://drive.google.com/file/d/1BUoshDWxAtY-L5nNYo2zUnOF6PjiqpyN/view?usp=sharing) |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 60.3 | 75.6 | 86.0 | 89.2 | ~4.5h | [[config]](https://drive.google.com/file/d/1kXKdq-mZ-wiWrgsss5Ny_vmdTSnvLAhH/view?usp=sharing) [[model]](https://drive.google.com/file/d/11qtWjAgGtjCa_G3G1hWLWj0Mpko9N7D3/view?usp=sharing) |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet | 67.1 | 82.4 | 90.8 | 93.0 | ~2h | [[config]](https://drive.google.com/file/d/1QXrH0apN0QqsgU0Bie8Vk7kXKDfYxFL_/view?usp=sharing) [[model]](https://drive.google.com/file/d/1B5nlhSj8AfTpzSW8bkx-LMj-Loa1ENul/view?usp=sharing) |

#### MSMT17

<!-- | Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet | 0.7 | 3.5 | 7.3 | 9.7 | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() [[log]]() |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 15.5 | 37.8 | 50.4 | 56.3 | ~3h | [[config]]() [[model]]() [[log]]() |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 20.5 | 43.6 | 57.1 | 63.2 | ~5.5h | [[config]]() [[model]]() [[log]]() |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet | 19.6 | 42.8 | 55.4 | 61.1 | ~4.5h | [[config]]() [[model]]() [[log]]() | -->

| Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet | 0.7 | 3.5 | 7.3 | 9.7 | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 15.5 | 37.8 | 50.4 | 56.3 | ~3h | [[config]]() [[model]]() |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 20.5 | 43.6 | 57.1 | 63.2 | ~5.5h | [[config]]() [[model]]() |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet | 19.6 | 42.8 | 55.4 | 61.1 | ~4.5h | [[config]]() [[model]]() |

#### Veri

<!-- | Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet | 7.6 | 30.3 | 44.9 | 52.2 | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() [[log]]() |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 31.9 | 71.9 | 80.5 | 85.7 | ~3h | [[config]]() [[model]]() [[log]]() |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 35.8 | 76.8 | 83.7 | 87.4 | ~7h | [[config]]() [[model]]() [[log]]() |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet | 36.9 | 79.9 | 86.8 | 89.9 | ~5h | [[config]]() [[model]]() [[log]]() | -->

| Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet | 7.6 | 30.3 | 44.9 | 52.2 | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 31.9 | 71.9 | 80.5 | 85.7 | ~3h | [[config]]() [[model]]() |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 35.8 | 76.8 | 83.7 | 87.4 | ~7h | [[config]]() [[model]]() |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet | 36.9 | 79.9 | 86.8 | 89.9 | ~5h | [[config]]() [[model]]() |

#### VehicleID

<!-- | Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet |  |  |  |  | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() [[log]]() |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() [[log]]() |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() [[log]]() |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() [[log]]() | -->

| Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet |  |  |  |  | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() |

#### ... (TBD)

### Unsupervised domain adaptation (UDA) on object re-ID

- `Direct infer` models are trained on the source-domain datasets ([source_pretrain](../tools/source_pretrain)) and directly tested on the target-domain datasets.
- UDA methods (`MMT`, `SpCL`, etc.) starting from ImageNet means that they are trained end-to-end in only one stage without source-domain pre-training.

#### DukeMTMC-reID -> Market-1501

<!-- | Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | DukeMTMC-reID | 27.2 | 58.9 | 75.7 | 81.4 | ~1h | [[config]](https://drive.google.com/file/d/1_gnPfjwf9uTOJyg1VsBzbMNQ-SGuhohP/view?usp=sharing) [[model]](https://drive.google.com/file/d/1MH-eIuWICkkQ8Ka3stXbiTq889yUZjBV/view?usp=sharing) [[log]](https://drive.google.com/file/d/15NUJvltPs_oT_0pyTjaKaEqn4n5hiyJI/view?usp=sharing) |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | DukeMTMC-reID | 52.3 | 76.0 | 87.8 | 91.9 | ~2h | [[config]](https://drive.google.com/file/d/1NgbBQrM8jbnKJJHQ1WUZ1sPeXvH6luAd/view?usp=sharing) [[model]](https://drive.google.com/file/d/1ciAk7GxnShm8z25hVqarhaG_8fz_tiyX/view?usp=sharing) [[log]](https://drive.google.com/file/d/12-U3hmjhz3D3rtUJ-_vsTE5QkQ5LgxfU/view?usp=sharing) |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 75.6 | 90.9 | 96.6 | 97.8 | ~3h | [[config]](https://drive.google.com/file/d/1Oe5QQ-NEJy9YsQr7hsMr5CJlZ0XHJS5P/view?usp=sharing) [[model]](https://drive.google.com/file/d/18t9HOCnQzQlgkRkSs8uFaDFYioGRtcLO/view?usp=sharing) [[log]](https://drive.google.com/file/d/1kn77MKbCBDviauLDphCS-NnpfBj_gQXd/view?usp=sharing) |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 80.9 | 92.2 | 97.6 | 98.4 | ~6h | [[config]](https://drive.google.com/file/d/1iFiOLbrzVQcEtIlFvsDIcDf4FcT9Z60U/view?usp=sharing) [[model]](https://drive.google.com/file/d/1XGOrt1iTHQNuFPebBcNjPrkTEwBXXRr_/view?usp=sharing) [[log]](https://drive.google.com/file/d/1Hwpr3f0X_EMYzkMsiegF7dqX0WsYkMQw/view?usp=sharing) |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet | 78.2 | 90.5 | 96.6 | 97.8 | ~3h | [[config]](https://drive.google.com/file/d/1O8XxCJDzpI7VIRR7crh0kkOK8vebmIgj/view?usp=sharing) [[model]](https://drive.google.com/file/d/1LvrHptXgzWspN2jwYtom4L_jUKYHpU_z/view?usp=sharing) [[log]](https://drive.google.com/file/d/1oy45txWnreyWOp2Y-E5mwjIss7raBDAP/view?usp=sharing) | -->

| Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | DukeMTMC | 27.2 | 58.9 | 75.7 | 81.4 | ~1h | [[config]](https://drive.google.com/file/d/1_gnPfjwf9uTOJyg1VsBzbMNQ-SGuhohP/view?usp=sharing) [[model]](https://drive.google.com/file/d/1MH-eIuWICkkQ8Ka3stXbiTq889yUZjBV/view?usp=sharing) |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | DukeMTMC | 52.3 | 76.0 | 87.8 | 91.9 | ~2h | [[config]](https://drive.google.com/file/d/1NgbBQrM8jbnKJJHQ1WUZ1sPeXvH6luAd/view?usp=sharing) [[model]](https://drive.google.com/file/d/1ciAk7GxnShm8z25hVqarhaG_8fz_tiyX/view?usp=sharing) |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 75.6 | 90.9 | 96.6 | 97.8 | ~3h | [[config]](https://drive.google.com/file/d/1Oe5QQ-NEJy9YsQr7hsMr5CJlZ0XHJS5P/view?usp=sharing) [[model]](https://drive.google.com/file/d/18t9HOCnQzQlgkRkSs8uFaDFYioGRtcLO/view?usp=sharing) |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 80.9 | 92.2 | 97.6 | 98.4 | ~6h | [[config]](https://drive.google.com/file/d/1iFiOLbrzVQcEtIlFvsDIcDf4FcT9Z60U/view?usp=sharing) [[model]](https://drive.google.com/file/d/1XGOrt1iTHQNuFPebBcNjPrkTEwBXXRr_/view?usp=sharing) |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet | 78.2 | 90.5 | 96.6 | 97.8 | ~3h | [[config]](https://drive.google.com/file/d/1O8XxCJDzpI7VIRR7crh0kkOK8vebmIgj/view?usp=sharing) [[model]](https://drive.google.com/file/d/1LvrHptXgzWspN2jwYtom4L_jUKYHpU_z/view?usp=sharing) |

#### Market-1501 -> DukeMTMC-reID

<!-- | Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | Market-1501 | 28.1 | 49.3 | 64.3 | 69.7 | ~1h | [[config]](https://drive.google.com/file/d/1FOuW_Hwl2ASPx0iXeDNxZ1R9MwFBr3gx/view?usp=sharing) [[model]](https://drive.google.com/file/d/13dkhrjz-VIH3jCjIep185MLZxFSD_F7R/view?usp=sharing) [[log]](https://drive.google.com/file/d/1EDT4ymWGzExyxT0uRXIKBXefjyds79qp/view?usp=sharing) |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | Market-1501 | 45.7 | 65.5 | 78.0 | 81.7 | ~2h | [[config]](https://drive.google.com/file/d/1Dvd-D4lTYJ44SJK0gMpTJ-W8cTgMF0vD/view?usp=sharing) [[model]](https://drive.google.com/file/d/1805D3yqtY3QY8pM83BanLkMLBnBSBgIz/view?usp=sharing) [[log]](https://drive.google.com/file/d/1fl_APkPZXtTfYFLoENX9prd_vllIz3b7/view?usp=sharing) |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 60.4 | 75.9 | 86.2 | 89.8 | ~3h | [[config]](https://drive.google.com/file/d/1-y5o5j6_K037s1BKKlY5IHf-hJ37XEtK/view?usp=sharing) [[model]](https://drive.google.com/file/d/1IVTJkfdlubV_bfH_ipxIEsubraxGbQMI/view?usp=sharing) [[log]](https://drive.google.com/file/d/1dh50GH7HWi7KTEJ8J2mZg3AJ7j3adJEX/view?usp=sharing) |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 67.7 | 80.3 | 89.9 | 92.9 | ~6h | [[config]](https://drive.google.com/file/d/1KcRmKH-8VZudb6N-KHj12DhV3ECmdBuM/view?usp=sharing) [[model]](https://drive.google.com/file/d/1tgqTZDLIZQrPS56PF0Yguy6lfNdSAIa9/view?usp=sharing) [[log]](https://drive.google.com/file/d/16ZOpCglyvzctsnbbkWILCNVmZEGlv0LU/view?usp=sharing) |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet | 70.4 | 83.8 | 91.2 | 93.4 | ~3h | [[config]](https://drive.google.com/file/d/1ILiId7BF_49kv4dT1pcZE0HQEdeTPXjU/view?usp=sharing) [[model]](https://drive.google.com/file/d/17WQyMnS7PiDy3EpD2RJbk45LVxcRZNi2/view?usp=sharing) [[log]](https://drive.google.com/file/d/19x3a_ZX5XJIEWp3y-eK_2TdNxYwUktdj/view?usp=sharing) | -->

| Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | Market | 28.1 | 49.3 | 64.3 | 69.7 | ~1h | [[config]](https://drive.google.com/file/d/1FOuW_Hwl2ASPx0iXeDNxZ1R9MwFBr3gx/view?usp=sharing) [[model]](https://drive.google.com/file/d/13dkhrjz-VIH3jCjIep185MLZxFSD_F7R/view?usp=sharing) |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | Market | 45.7 | 65.5 | 78.0 | 81.7 | ~2h | [[config]](https://drive.google.com/file/d/1Dvd-D4lTYJ44SJK0gMpTJ-W8cTgMF0vD/view?usp=sharing) [[model]](https://drive.google.com/file/d/1805D3yqtY3QY8pM83BanLkMLBnBSBgIz/view?usp=sharing) |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 60.4 | 75.9 | 86.2 | 89.8 | ~3h | [[config]](https://drive.google.com/file/d/1-y5o5j6_K037s1BKKlY5IHf-hJ37XEtK/view?usp=sharing) [[model]](https://drive.google.com/file/d/1IVTJkfdlubV_bfH_ipxIEsubraxGbQMI/view?usp=sharing) |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 67.7 | 80.3 | 89.9 | 92.9 | ~6h | [[config]](https://drive.google.com/file/d/1KcRmKH-8VZudb6N-KHj12DhV3ECmdBuM/view?usp=sharing) [[model]](https://drive.google.com/file/d/1tgqTZDLIZQrPS56PF0Yguy6lfNdSAIa9/view?usp=sharing) |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet | 70.4 | 83.8 | 91.2 | 93.4 | ~3h | [[config]](https://drive.google.com/file/d/1ILiId7BF_49kv4dT1pcZE0HQEdeTPXjU/view?usp=sharing) [[model]](https://drive.google.com/file/d/17WQyMnS7PiDy3EpD2RJbk45LVxcRZNi2/view?usp=sharing) |

#### DukeMTMC-reID -> MSMT17

<!-- | Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet | 11.9 | 35.9 | 48.4 | 54.2 | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() [[log]]() |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 20.3 | 45.7 | 58.6 | 64.5 | ~3.5h | [[config]]() [[model]]() [[log]]() |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 25.7 | 51.1 | 63.9 | 69.2 | ~7h | [[config]]() [[model]]() [[log]]() |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet | 26.5 | 53.1 | 65.8 | 70.5 | ~6h | [[config]]() [[model]]() [[log]]() | -->

| Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet | 11.9 | 35.9 | 48.4 | 54.2 | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 20.3 | 45.7 | 58.6 | 64.5 | ~3.5h | [[config]]() [[model]]() |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 25.7 | 51.1 | 63.9 | 69.2 | ~7h | [[config]]() [[model]]() |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet | 26.5 | 53.1 | 65.8 | 70.5 | ~6h | [[config]]() [[model]]() |

#### MSMT17 -> DukeMTMC-reID

<!-- | Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet | 38.1 | 60.2 | 74.7 | 79 | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() [[log]]() |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 62.3 | 77.4 | 88.0 | 91.7 | ~2.5h | [[config]]() [[model]]() [[log]]() |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 66.6 | 79.1 | 88.9 | 91.7 | ~6h | [[config]]() [[model]]() [[log]]() |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet | 67.5 | 82.8 | 90.5 | 92.6 | ~5h | [[config]]() [[model]]() [[log]]() | -->

| Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet | 38.1 | 60.2 | 74.7 | 79 | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 62.3 | 77.4 | 88.0 | 91.7 | ~2.5h | [[config]]() [[model]]() |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 66.6 | 79.1 | 88.9 | 91.7 | ~6h | [[config]]() [[model]]() |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet | 67.5 | 82.8 | 90.5 | 92.6 | ~5h | [[config]]() [[model]]() |

#### Market-1501 -> MSMT17

<!-- | Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet | 9.6 | 29.4 | 41.5 | 46.8 | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() [[log]]() |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 22.8 | 49.3 | 61.7 | 67.2 | ~3.5h | [[config]]() [[model]]() [[log]]() |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 25.6 | 50.9 | 63.5 | 68.8 | ~7.5h | [[config]]() [[model]]() [[log]]() |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() [[log]]() | -->

| Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet | 9.6 | 29.4 | 41.5 | 46.8 | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 22.8 | 49.3 | 61.7 | 67.2 | ~3.5h | [[config]]() [[model]]() |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 25.6 | 50.9 | 63.5 | 68.8 | ~7.5h | [[config]]() [[model]]() |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() |

#### MSMT17 -> Market-1501

<!-- | Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet | 30.1 | 61.0 | 76.8 | 81.7 | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() [[log]]() |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 76.0 | 89.7 | 96.5 | 97.9 | ~3h | [[config]]() [[model]]() [[log]]() |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() [[log]]() |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() [[log]]() | -->

| Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet | 30.1 | 61.0 | 76.8 | 81.7 | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 76 | 89.7 | 96.5 | 97.9 | ~3h | [[config]]() [[model]]() |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 78.4 | 90.8 | 97.0 | 98.0 | ~6.5h | [[config]]() [[model]]() |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet | 77.5 | 89.7 | 96.1 | 97.6 | ~5h | [[config]]() [[model]]() |

#### PersonX -> DukeMTMC-reID

<!-- | Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet | 18.3 | 39.2 | 53.1 | 58.7 | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() [[log]]() |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 58.6 | 74.4 | 85.5 | 89.7 | ~2.5h | [[config]]() [[model]]() [[log]]() |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 64.1 | 78.2 | 87.3 | 90.8 | ~6h | [[config]]() [[model]]() [[log]]() |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() [[log]]() | -->

| Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet | 18.3 | 39.2 | 53.1 | 58.7 | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 58.6 | 74.4 | 85.5 | 89.7 | ~2.5h | [[config]]() [[model]]() |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 64.1 | 78.2 | 87.3 | 90.8 | ~6h | [[config]]() [[model]]() |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() |

#### PersonX -> Market-1501

<!-- | Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet | 20.3 | 46.4 | 63.5 | 71.1 | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() [[log]]() |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 74.2 | 89.4 | 96.1 | 97.5 | ~2.5h | [[config]]() [[model]]() [[log]]() |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 78.1 | 90.3 | 96.9 | 97.9 | ~6h | [[config]]() [[model]]() [[log]]() |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() [[log]]() | -->

| Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet | 20.3 | 46.4 | 63.5 | 71.1 | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 74.2 | 89.4 | 96.1 | 97.5 | ~2.5h | [[config]]() [[model]]() |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 78.1 | 90.3 | 96.9 | 97.9 | ~6h | [[config]]() [[model]]() |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() |

#### PersonX -> MSMT17

<!-- | Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet | 5.3 | 19.0 | 29.0 | 34.0 | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() [[log]]() |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 19.9 | 45.0 | 57.9 | 63.4 | ~3.5h | [[config]]() [[model]]() [[log]]() |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 24.2 | 49.4 | 62.6 | 68.0 | ~7.5h | [[config]]() [[model]]() [[log]]() |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() [[log]]() | -->

| Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet | 5.3 | 19.0 | 29.0 | 34.0 | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 19.9 | 45.0 | 57.9 | 63.4 | ~3.5h | [[config]]() [[model]]() |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 24.2 | 49.4 | 62.6 | 68.0 | ~7.5h | [[config]]() [[model]]() |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() |

#### VehicleID -> Veri

<!-- | Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet |  |  |  |  | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() [[log]]() |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() [[log]]() |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() [[log]]() |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() [[log]]() | -->

| Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet |  |  |  |  | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() |

#### Veri -> VehicleID

<!-- | Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet |  |  |  |  | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() [[log]]() |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() [[log]]() |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() [[log]]() |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() [[log]]() | -->

| Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet |  |  |  |  | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet |  |  |  |  | ~h | [[config]]() [[model]]() |

#### ... (TBD)
