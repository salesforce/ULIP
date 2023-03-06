[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ulip-learning-unified-representation-of/3d-point-cloud-classification-on-modelnet40)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-modelnet40?p=ulip-learning-unified-representation-of)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ulip-learning-unified-representation-of/3d-point-cloud-classification-on-scanobjectnn)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-scanobjectnn?p=ulip-learning-unified-representation-of)

# ULIP: Learning Unified Representation of Language, Image and Point Cloud for 3D Understanding

[comment]: <> (---)

Official implementation of ['ULIP: Learning Unified Representation of Language, Image and Point Cloud for 3D Understanding'](https://arxiv.org/abs/2212.05171)

[Project Website](https://tycho-xue.github.io/ULIP/)

# News
ULIP has been accepted by CVPR 2023! 🔥🔥🔥

# Animation
![Pipeline Animation](pipeline_8s_timing.gif)

[comment]: <> (---)

# Abstract
The recognition capabilities of current state-of-the-art 3D models are limited by datasets with a small number of annotated data and a pre-defined set of categories. In its 2D counterpart, recent advances have shown that similar problems can be significantly alleviated by employing knowledge from other modalities, such as language. Inspired by this, leveraging multimodal information for 3D modality could be promising to improve 3D understanding under the restricted data regime, but this line of research is not well studied. Therefore, we introduce ULIP to learn a unified representation of image, text, and 3D point cloud by pre-training with object triplets from the three modalities. To overcome the shortage of training triplets, ULIP leverages a pre-trained vision-language model that has already learned a common visual and textual space by training with massive image-text pairs. Then, ULIP learns a 3D representation space aligned with the common image-text space, using a small number of automatically synthesized triplets. ULIP is agnostic to 3D backbone networks and can easily be integrated into any 3D architecture. Experiments show that ULIP effectively improves the performance of multiple recent 3D backbones by simply pre-training them on ShapeNet55 using our framework, achieving state-of-the-art performance in both standard 3D classification and zero-shot 3D classification on ModelNet40 and ScanObjectNN. ULIP also improves the performance of PointMLP by around 3% in 3D classification on ScanObjectNN, and outperforms PointCLIP by 28.8% on top-1 accuracy for zero-shot 3D classification on ModelNet40. Our code and pre-trained models will be released.

[comment]: <> (---)

# Pipeline
![Overall Pipeline](figure2_resize.gif)

[comment]: <> (---)

# Instructions
ULIP is a highly extensible multimodal pre-training framework, and it's model-architecture agnostic, meaning you can easily plug in any 3D backbone models and pre-train it using our framework to get a jump-start for various downstreaming tasks!
## 【Install environments】
We pre-train ULIP on 8 Nvidia A100 GPUs, the code is tested with CUDA==11.0 and pytorch==1.10.1\
```conda create -n ulip python=3.7.15``` \
```conda activate ulip``` \
```conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge``` \
```pip install -r requirements.txt```

## 【Download datasets and initialize models, put them in the right paths.】
Download the used datasets and initialize models from [here](https://console.cloud.google.com/storage/browser/sfr-ulip-code-release-research). For now, you ONLY need to download "initialize_models", "modelnet40_normal_resampled", and "shapenet-55". You might need a gmail account to access it.\
After you download the datasets and initialize models, you can choose one of the following options: \
(1) Put it in or do a soft link to the data folder, by default the data folder should have the following structure: \
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


## 【Reproduce ULIP + Pointnet2(SSG)】
Please change the script to accommodate your system accordingly, this script is used to pre-train on 8 gpus by default. You can also modify the desired output folder in the script.
```
bash pretrain.sh
```
## 【Pre-train your customized 3D backbones】
There are only two things you need to change to pre-train your own customized 3D backbones: \
(1) Define your own 3D backbone in ./models folder.\
We put a template "customized_backbone" here, you can refer to the comments to see the expected input and output shapes. You can also refer to how pointnet2 is defined here. \
(2) Use or modify this "ULIP_CUSTOMIZED" class in ./models/ULIP_models.py.\
Please refer to the comments in "ULIP_CUSTOMIZED" class, it should be straightforward to follow, and please be sure to change the "pc_feat_dims" accordingly (since we are agnostic to the point cloud output feature dimensions of your customized 3D backbones).

# Code Release
We received many code-release requests, thus decided to opensource the pre-train part first to unblock people who want to use our framework. We plan to enrich the repo gradually.
 
# Citation

    @article{xue2022ulip,
      title={ULIP: Learning Unified Representation of Language, Image and Point Cloud for 3D Understanding},
      author={Xue, Le and Gao, Mingfei and Xing, Chen and Mart{\'\i}n-Mart{\'\i}n, Roberto and Wu, Jiajun and Xiong, Caiming and Xu, Ran and Niebles, Juan Carlos and Savarese, Silvio},
      journal={arXiv preprint arXiv:2212.05171},
      year={2022}
    }

# Contact
If you have any question about this project, please contact [lxue@salesforce.com](lxue@salesforce.com)
