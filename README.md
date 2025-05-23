[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ulip-2-towards-scalable-multimodal-pre/3d-point-cloud-classification-on-scanobjectnn)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-scanobjectnn?p=ulip-2-towards-scalable-multimodal-pre)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ulip-learning-unified-representation-of/3d-point-cloud-classification-on-modelnet40)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-modelnet40?p=ulip-learning-unified-representation-of)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ulip-learning-unified-representation-of/3d-point-cloud-classification-on-scanobjectnn)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-scanobjectnn?p=ulip-learning-unified-representation-of)

# ULIP-2: Towards Scalable Multimodal Pre-training For 3D Understanding (CVPR2024)

# ULIP: Learning a Unified Representation of Language, Images, and Point Clouds for 3D Understanding (CVPR2023)

[comment]: <> (---)

Official implementation of [ULIP-2: Towards Scalable Multimodal Pre-training For 3D Understanding](https://arxiv.org/abs/2305.08275)

Official implementation of [ULIP: Learning a Unified Representation of Language, Images, and Point Clouds for 3D Understanding](https://arxiv.org/abs/2212.05171)

[Project Website](https://tycho-xue.github.io/ULIP/)

# 📰 News

### 📢 [**05/23/2025**] Major Update: ULIP Resources Moved to Hugging Face!
Due to a new policy on Google Cloud Storage, **all ULIP-series datasets and models** have been reuploaded to Hugging Face:
👉 [**https://huggingface.co/datasets/SFXX/ulip**](https://huggingface.co/datasets/SFXX/ulip/tree/main)

If you're unable to download from the GCP bucket, please switch to Hugging Face for continued access and updates.

---

### 🗓️ [06/17/2024] [ULIP-2: Towards Scalable Multimodal Pre-training For 3D Understanding](https://arxiv.org/abs/2305.08275)
📦 The latest **CVPR 2024** version of the ensembled pre-trained model (10k xyzrgb points) is now available [here](https://storage.cloud.google.com/sfr-ulip-code-release-research/ULIP-2/models/ULIP-2-PointBERT-10k-colored-pc-pretrained.pt)

### 🗓️ [02/26/2024] ULIP-2 Accepted to CVPR 2024 🎉

### 🗓️ [06/09/2023] PointBERT ULIP-2 Model Released  
🔗 [Download here](https://storage.cloud.google.com/sfr-ulip-code-release-research/pretrained_models/ckpt_zero-sho_classification/pointbert_ULIP-2.pt)

### 🗓️ [06/09/2023] Smaller Version of ULIP - ShapeNet Triplets Released  
A 420GB subset is available at [this GCP link](https://console.cloud.google.com/storage/browser/sfr-ulip-code-release-research/shapenet-55), under the `only_rgb_depth_images` folder — the exact subset used by ULIP.  
Skip downloading the full rendered_images (~1TB) if not needed.

### 🗓️ [05/22/2023] ULIP - Objaverse and ShapeNet Triplets Uploaded  
📁 [GCP bucket here](https://console.cloud.google.com/storage/browser/sfr-ulip-code-release-research)

### 🗓️ [05/14/2023] ULIP-2 Released!

### 🗓️ [02/28/2023] ULIP Accepted to CVPR 2023 🔥🔥🔥


# Animation
![Pipeline Animation](assets/pipeline_8s_timing.gif)

[comment]: <> (---)

# What is ULIP
ULIP is a Model-agnostic Multimodal Pre-training Framework, which can leverage information from other modalities (Images, Language) to improve the ability to understand 3D data without introducing any extra latency.

[comment]: <> (---)

# Pipeline
![Overall Pipeline](assets/figure2_resize.gif)

[comment]: <> (---)

# Instructions
ULIP is a highly extensible multimodal pre-training framework, and it's model-architecture agnostic, meaning you can easily plug in any 3D backbone models and pre-train it using our framework to get a jump-start for various downstreaming tasks!
## [Install environments]
We pre-train ULIP on 8 Nvidia A100 GPUs, the code is tested with CUDA==11.0 and pytorch==1.10.1\
```conda create -n ulip python=3.7.15``` \
```conda activate ulip``` \
```conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge``` \
```pip install -r requirements.txt```\
\
[optional] \
If you want to pre-train PointNeXt, we embed a modified PointNeXt codebase inside the ./models/pointnext, please do the following to install it:
```
cd ./models/pointnext/PointNeXt \
bash update.sh \
bash install.sh \
```
## [Download datasets and initialize models, put them in the right paths.]
Download the used datasets and initialize models from [here](https://console.cloud.google.com/storage/browser/sfr-ulip-code-release-research). For now, you ONLY need to download "initialize_models", "modelnet40_normal_resampled", and "shapenet-55". You might need a gmail account to access it.\
After you download the datasets and initialize models, you can choose one of the following options: \
(1) Put it in or do a soft link to the data folder, by default the data folder should have the following structure:
```
./data |
-- ModelNet40.yaml |
-- ShapeNet-55.yaml |
-- dataset_3d.py |
-- dataset_catalog.json |
-- initialize_models |
-- labels.json |
-- modelnet40_normal_resampled |
-- shapenet-55 |
-- templates.json
```
(2) Change the paths accordingly (optional to do if you don't want to put/link downloaded files in the data folder):
```
# Change the "DATA_PATH", "PC_PATH", "IMAGE_PATH"
./data/ShapeNet-55.yaml
# Change the "DATA_PATH"
./data/ModelNet40.yaml
# Change the initialize_models address
./models/ULIP_models.py
Modify this line "pretrain_slip_model = torch.load('./data/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))"
```


## [Pre-train 3D backbones]
**Our framework is model architecture agonistic, currently four 3D backbones are supported:** \
**Pointnet2(ssg)**\
**PointBERT**\
**PointMLP**\
**PointNeXt**\
\
Please change the script to accommodate your system accordingly, this script is used to pre-train on 8 gpus by default. You can also modify the desired output folder in the script.
```
# the scripts are named by its correspoinding 3D backbone name.
bash ./scripts/(choose your pre-train script)
```

## [Test pre-trained models for zero-shot classification on ModelNet40]
You may also change the output path in the scripts as well.

```
bash ./scripts/(choose your test script) /path/to/your/checkpoint.pt
```
You may also change the output path in the scripts as well.

## [Pre-train & Test using different number of points]
Change the npoints argument in the scripts, by default its 8192. \
**Note: Currently we use FPS to subsample the 8192 points, which might slow down the training speed. If you'd like, you can choose to cache or save the pre-processed datasets with different number of points to speed up your pre-training.**

## [Pre-train your customized 3D backbones]
There are only two things you need to change to pre-train your own customized 3D backbones: \
(1) Define your own 3D backbone in ./models folder.\
We put a template "customized_backbone" here, you can refer to the comments to see the expected input and output shapes. You can also refer to how pointnet2 is defined here. \
(2) Use or modify this "ULIP_CUSTOMIZED" class in ./models/ULIP_models.py.\
Please refer to the comments in "ULIP_CUSTOMIZED" class, it should be straightforward to follow, and please be sure to change the "pc_feat_dims" accordingly (since we are agnostic to the point cloud output feature dimensions of your customized 3D backbones).


# Pre-trained models for zero-shot classification
ULIP-1 models, Zero-shot classification on ModelNet40, 8k points pre-train, 8k points test, best checkpoint:

| model                                                                                                                                                                   | top1 | top5 |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|------|
| [Pointnet2(ssg)](https://storage.cloud.google.com/sfr-ulip-code-release-research/pretrained_models/ckpt_zero-sho_classification/checkpoint_pointnet2_ssg.pt?authuser=0) | 57.7 | 78.9 |
| [PointMLP](https://storage.cloud.google.com/sfr-ulip-code-release-research/pretrained_models/ckpt_zero-sho_classification/checkpoint_pointmlp.pt?authuser=0)            | 60.0 | 79.4 |
| [PointBERT](https://storage.cloud.google.com/sfr-ulip-code-release-research/pretrained_models/ckpt_zero-sho_classification/checkpoint_pointbert.pt?authuser=0)          | 60.3 | 84.0 |
| [PointNeXt](https://storage.cloud.google.com/sfr-ulip-code-release-research/pretrained_models/ckpt_zero-sho_classification/checkpoint_pointnext.pt?authuser=0)          | 56.2 | 77.0 |

ULIP-2 models, pre-trained with 10k xyzrgb point clouds on the ensembled ULIP-Objaverse + ULIP-ShapeNet, check the [google drive]([sfr-ulip-code-release-research/ULIP-Objaverse_triplets](https://console.cloud.google.com/storage/browser/sfr-ulip-code-release-research/ULIP-Objaverse_triplets?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22)))).

Note that, ULIP-2 models are improved in the CVPR2024 version compared to the initial arxiv release, please refer to the new one for matching the cvpr2024 version numbers.

| model                                                                                                                                                                   | Objaverse-top1 | Objaverse-top5 | Modelnet40-top1 | Modelnet40-top5 |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|------| - | - |
|openshape-pointbert-10kxyzrgb-ensembled-objavserse-shapenet-abo-3d_future| 46.8 | 77.0 | 84.4 | 98.0 |
| [ULIP2-PointBERT-10kxyzrgb-ensembled-objaverse-shapenet](https://storage.cloud.google.com/sfr-ulip-code-release-research/ULIP-2/models/ULIP-2-PointBERT-10k-colored-pc-pretrained.pt) | 50.6 | 79.1 | 84.7 | 97.1 |

# License and term of use for the released pre-train datasets
The code is under https://github.com/salesforce/ULIP/blob/main/LICENSE.txt.

The released "ULIP - Objaverse Triplets" is under https://opendatacommons.org/licenses/by/1-0/, consistent with Objaverse's license.

The released "ULIP - ShapeNet Triplets" is under the terms of use from https://shapenet.org/terms, consistent with ShapeNet's terms of use.

# Citation
```bibtex
@inproceedings{xue2023ulip,
  title={Ulip: Learning a unified representation of language, images, and point clouds for 3d understanding},
  author={Xue, Le and Gao, Mingfei and Xing, Chen and Mart{\'\i}n-Mart{\'\i}n, Roberto and Wu, Jiajun and Xiong, Caiming and Xu, Ran and Niebles, Juan Carlos and Savarese, Silvio},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={1179--1189},
  year={2023}
}
@inproceedings{xue2024ulip,
  title={Ulip-2: Towards scalable multimodal pre-training for 3d understanding},
  author={Xue, Le and Yu, Ning and Zhang, Shu and Panagopoulou, Artemis and Li, Junnan and Mart{\'\i}n-Mart{\'\i}n, Roberto and Wu, Jiajun and Xiong, Caiming and Xu, Ran and Niebles, Juan Carlos and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={27091--27101},
  year={2024}}


# Contact
If you have any question about this project, please contact [lxue@salesforce.com](lxue@salesforce.com)
