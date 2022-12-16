[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ulip-learning-unified-representation-of/3d-point-cloud-classification-on-modelnet40)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-modelnet40?p=ulip-learning-unified-representation-of)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ulip-learning-unified-representation-of/3d-point-cloud-classification-on-scanobjectnn)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-scanobjectnn?p=ulip-learning-unified-representation-of)

# ULIP: Learning Unified Representation of Language, Image and Point Cloud for 3D Understanding

[comment]: <> (---)

Official implementation of ['ULIP: Learning Unified Representation of Language, Image and Point Cloud for 3D Understanding'](https://arxiv.org/abs/2212.05171)

[Project Website](https://tycho-xue.github.io/ULIP/)

# Animation
![Pipeline Animation](pipeline_8s_timing.gif)

[comment]: <> (---)

# Abstract
The recognition capabilities of current state-of-the-art 3D models are limited by datasets with a small number of annotated data and a pre-defined set of categories. In its 2D counterpart, recent advances have shown that similar problems can be significantly alleviated by employing knowledge from other modalities, such as language. Inspired by this, leveraging multimodal information for 3D modality could be promising to improve 3D understanding under the restricted data regime, but this line of research is not well studied. Therefore, we introduce ULIP to learn a unified representation of image, text, and 3D point cloud by pre-training with object triplets from the three modalities. To overcome the shortage of training triplets, ULIP leverages a pre-trained vision-language model that has already learned a common visual and textual space by training with massive image-text pairs. Then, ULIP learns a 3D representation space aligned with the common image-text space, using a small number of automatically synthesized triplets. ULIP is agnostic to 3D backbone networks and can easily be integrated into any 3D architecture. Experiments show that ULIP effectively improves the performance of multiple recent 3D backbones by simply pre-training them on ShapeNet55 using our framework, achieving state-of-the-art performance in both standard 3D classification and zero-shot 3D classification on ModelNet40 and ScanObjectNN. ULIP also improves the performance of PointMLP by around 3% in 3D classification on ScanObjectNN, and outperforms PointCLIP by 28.8% on top-1 accuracy for zero-shot 3D classification on ModelNet40. Our code and pre-trained models will be released.

[comment]: <> (---)

# Pipeline
![Overall Pipeline](figure2_resize.gif)

[comment]: <> (---)

# Code
Code will come soon, we are working hard on the code releasing now, thanks for the waiting.

# Citation

    @article{xue2022ulip,
      title={ULIP: Learning Unified Representation of Language, Image and Point Cloud for 3D Understanding},
      author={Xue, Le and Gao, Mingfei and Xing, Chen and Mart{\'\i}n-Mart{\'\i}n, Roberto and Wu, Jiajun and Xiong, Caiming and Xu, Ran and Niebles, Juan Carlos and Savarese, Silvio},
      journal={arXiv preprint arXiv:2212.05171},
      year={2022}
    }

# Contact
If you have any question about this project, please contact [lxue@salesforce.com](lxue@salesforce.com)
