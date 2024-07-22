# Semantic-based-Point-Cloud-Tasks
This is an open-source repository for semantic based point cloud tasks, and we aim to provide a comprehensive summary of various semantic based point cloud tasks.
## 1. Traditional Point Cloud Tasks with Semantic
### 1.1 Point Cloud Semantic Segmentation
#### 1.1.1 Top-down Segmentation Method
* [SOLOv2: Dynamic and Fast Instance Segmentation](https://neurips.cc/virtual/2020/public/poster_cd3afef9b8b89558cd56638c3631868a.html) \[2020 NIPS\] :octocat:[code](https://git.io/AdelaiDet)
* [SOLO: A Simple Framework for Instance Segmentation](https://ieeexplore.ieee.org/document/9536421) \[2022 TPAMI\] :octocat:[code](https://git.io/AdelaiDet)
* [NeuralBF: Neural Bilateral Filtering for Top-down Instance Segmentation on Point Clouds](https://arxiv.org/pdf/2207.09978v1) \[2023 WACV\] 

####  1.1.2 Bottom-up Semantic Segmentation Method
##### (1) Voxel-based Semantic Segmentation
* [Pointgrid: A deep network for 3d shape understanding](https://ieeexplore.ieee.org/document/8579057) \[2018 CVPR\] :octocat:[code](https://github.com/trucleduc/PointGrid)
* [Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution](https://www.semanticscholar.org/paper/Searching-Efficient-3D-Architectures-with-Sparse/769dbcb80cc1d2d17ae7c524644103d0f6595452) \[2020 ECCV\] :octocat:[code](https://github.com/mit-han-lab/spvnas)
* [VMNet: Voxel-Mesh Network for Geodesic-Aware 3D Semantic Segmentation](https://ieeexplore.ieee.org/document/9710530) \[2021 ICCV\] :octocat:[code](https://github.com/hzykent/VMNet)
* [GeoAuxNet: Towards Universal 3D Representation Learning for Multi-sensor Point Clouds](https://arxiv.org/pdf/2403.19220) \[2024 CVPR\] 
:octocat:[code](https://github.com/zhangshengjun2019/GeoAuxNet)
* [PanoOcc: Unified Occupancy Representation for Camera-based 3D Panoptic Segmentation](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_PanoOcc_Unified_Occupancy_Representation_for_Camera-based_3D_Panoptic_Segmentation_CVPR_2024_paper.pdf) \[2024 CVPR\] :octocat: [code](https://github.com/Robertwyq/PanoOcc)

##### (2) Projection-based Semantic Segmentation
* [Squeezeseg: Convolutional neural nets with recurrent crf for real-time road-object segmentation from 3d lidar point cloud](https://ieeexplore.ieee.org/document/8462926) \[2018 ICRA\] :octocat:[code](https://github.com/BichenWuUCB/SqueezeSeg)
* [Rangenet++: Fast and accurate lidar semantic segmentation](https://ieeexplore.ieee.org/document/8967762) \[2019 IROS\] :octocat:[code](https://github.com/PRBonn/lidar-bonnetal)
* [Squeezesegv2: Improved model structure and unsupervised domain adaptation for roadobject segmentation from a lidar point cloud](https://ieeexplore.ieee.org/abstract/document/8793495) \[2019 ICRA\] :octocat:[code](https://github.com/xuanyuzhou98/SqueezeSegV2)
* [Salsanext: Fast, uncertainty-aware semantic segmentation of lidar point clouds](https://link.springer.com/content/pdf/10.1007/978-3-030-64559-5_16) \[2020 ISVC\] :octocat:[code](https://github.com/TiagoCortinhal/SalsaNext)
* [Squeezesegv3: Spatially-adaptive convolution for efficient point-cloud segmentation](https://link.springer.com/chapter/10.1007/978-3-030-58604-1_1) \[2020 ECCV\] :octocat:[code](https://github.com/chenfengxu714/SqueezeSegV3)
* [Active learning based 3d semantic labeling from images and videos](https://ieeexplore.ieee.org/document/9430559) \[2022 TCSVT\] 
* [Efficient 3d scene semantic segmentation via active learning on rendered 2d images](https://ieeexplore.ieee.org/document/10158507) \[2023 TIP\] 
* [3d semantic segmentation of aerial photogrammetry models based on orthographic projection](https://ieeexplore.ieee.org/document/10119167) \[2023 TCSVT\]
* [Knowledge distillation from 3d to bird’s-eye-view for lidar semantic segmentation](https://ieeexplore.ieee.org/document/10220057) \[2023 ICME\] :octocat:[code](https://github.com/fengjiang5/Knowledge-Distillation-from-Cylinder3D-to-PolarNet)
* [Rangevit: Towards vision transformers for 3d semantic segmentation in autonomous driving](https://ieeexplore.ieee.org/document/10204428) \[2023 CVPR\] :octocat:[code](https://github.com/valeoai/rangevit)
* [Bird's-Eye-View Semantic Segmentation With Two-Stream Compact Depth Transformation and Feature Rectification](https://ieeexplore.ieee.org/abstract/document/10124335) \[2023 TIV\] 
* [Residual graph convolutional network for bird’seye-view semantic segmentation](https://ieeexplore.ieee.org/document/10483624) \[2024 WACV\] 

##### (3) Point-based Semantic Segmentation
* [Pointnet++: Deep hierarchical feature learning on point sets in a metric space](https://dl.acm.org/doi/abs/10.5555/3295222.3295263) \[2017 NIPS\] :octocat:[code](https://github.com/charlesq34/pointnet2)
* [Point transformer](https://ieeexplore.ieee.org/document/9710703) \[2021 ICCV\]
* [Backward attentive fusing network with local aggregation classifier for 3d point cloud semantic segmentation](https://ieeexplore.ieee.org/abstract/document/9410334) \[2021 TIP\] :octocat:[code](https://github.com/Xiangxu-0103/BAF-LAC)
* [Cga-net: Category guided aggregation for point cloud semantic segmentation](https://ieeexplore.ieee.org/document/9577467) \[2021 CVPR\] :octocat:[code](https://github.com/MCG-NJU/CGA-Net)
* [Point transformer v2: Grouped vector attention and partition-based pooling](https://papers.nips.cc/paper_files/paper/2022/hash/d78ece6613953f46501b958b7bb4582f-Abstract-Conference.html) \[2022 NIPS\] :octocat:[code](https://github.com/Gofinge/PointTransformerV2)
* [Dcnet: Large-scale point cloud semantic segmentation with discriminative and efficient feature aggregation](https://ieeexplore.ieee.org/document/10025770) \[2023 TCSVT\] :octocat:[code](https://github.com/fukunyin/DCNet)
* [Point transformer v3: Simpler, faster, stronger](https://arxiv.org/abs/2312.10035) \[2024 CVPR\] :octocat:[code](https://github.com/Pointcept/PointTransformerV3)
* [KPConvX: Modernizing Kernel Point Convolution with Kernel Attention](https://arxiv.org/pdf/2405.13194) \[2024 CVPR\] :octocat:[code](https://github.com/apple/ml-kpconvx)

##### (4) Unit-sets-based on Semantic Segmentation
* [Instance segmentation in 3d scenes using semantic superpoint tree networks](https://ieeexplore.ieee.org/document/9709996) \[2021 ICCV\] :octocat:[code](https://github.com/Gorilla-Lab-SCUT/SSTNet)
* [Sspc-net: Semi-supervised semantic 3d point cloud segmentation network](https://ojs.aaai.org/index.php/AAAI/article/view/16200) \[2021 AAAI\] :octocat:[code](https://github.com/MMCheng/SSPC-Net)
* [One thing one click: A self-training approach for weakly supervised 3d semantic segmentation](https://ieeexplore.ieee.org/abstract/document/9578763) \[2021 CVPR\] :octocat:[code](https://github.com/liuzhengzhe/One-Thing-One-Click)
* [Rpvnet: A deep and efficient range-point-voxel fusion network for lidar point cloud segmentation](https://ieeexplore.ieee.org/abstract/document/9709941) \[2021 ICCV\] :octocat:[code](https://github.com/GuoPingPan/RPVNet)
* [Superpoint-guided semisupervised semantic segmentation of 3d point clouds](https://ieeexplore.ieee.org/document/9811904) \[2022 ICRA\] 
* [Pointdc: Unsupervised semantic segmentation of 3d point clouds via cross-modal distillation and super-voxel clustering](https://ieeexplore.ieee.org/document/10377176) \[2023 ICCV\] :octocat:[code](https://github.com/SCUT-BIP-Lab/PointDC)
* [Point-to-voxel knowledge distillation for lidar semantic segmentation](https://ieeexplore.ieee.org/document/9879674) \[2022 CVPR\] :octocat:[code](https://github.com/cardwing/Codes-for-PVKD)
* [Pointdc: Unsupervised semantic segmentation of 3d point clouds via cross-modal distillation and super-voxel clustering](https://ieeexplore.ieee.org/document/10377176) \[2023 ICCV\] :octocat:[code](https://github.com/SCUT-BIP-Lab/PointDC)
* [Nested architecture search for point cloud semantic segmentation](https://ieeexplore.ieee.org/document/9919408) \[2023 TIP\] :octocat:[code](https://github.com/fanyang587/NestedNet)
* [Multi-to-single knowledge distillation for point cloud semantic segmentation](https://ieeexplore.ieee.org/document/10160496) \[2023 ICRA\] :octocat:[code](https://github.com/skyshoumeng/M2SKD)
* [Pointdistiller: Structured knowledge distillation towards efficient and compact 3d detection](https://ieeexplore.ieee.org/abstract/document/10205029) \[2023 CVPR\] :octocat:[code](https://github.com/RunpeiDong/PointDistiller)
* [Knowledge distillation from 3d to bird’s-eye-view for lidar semantic segmentation](https://ieeexplore.ieee.org/document/10220057) \[2023 ICME\] :octocat:[code](https://github.com/fengjiang5/Knowledge-Distillation-from-Cylinder3D-to-PolarNet)
  
#### 1.1.3 Refined Segmentation Goal
##### (1) Generalization of Segmentation
* [Complete & label: A domain adaptation approach to semantic segmentation of lidar point clouds](https://ieeexplore.ieee.org/document/9578920) \[2021 CVPR\]
* [Few-shot 3d point cloud semantic segmentation](https://ieeexplore.ieee.org/document/9577428) \[2021 CVPR\] :octocat:[code](https://github.com/Na-Z/attMPTI)
* [Perceptionaware multi-sensor fusion for 3d lidar semantic segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhuang_Perception-Aware_Multi-Sensor_Fusion_for_3D_LiDAR_Semantic_Segmentation_ICCV_2021_paper.pdf) \[2021 ICCV\] :octocat:[code](https://github.com/ICEORY/PMF)
* [Sparse-to-dense feature matching: Intra and inter domain cross-modal learning in domain adaptation for 3d semantic segmentation](https://ieeexplore.ieee.org/document/9710520) \[2021 ICCV\] :octocat:[code](https://github.com/leolyj/DsCML)
* [Seggroup: Seglevel supervision for 3d instance and semantic segmentation](https://ieeexplore.ieee.org/document/9833393) \[2022 TIP\] :octocat:[code](https://github.com/antao97/SegGroup)
* [Lif-seg: Lidar and camera image fusion for 3d lidar semantic segmentation](https://ieeexplore.ieee.org/document/10128757) \[2023 TMM\] 
* [Cross-modal learning for domain adaptation in 3d semantic segmentation](https://ieeexplore.ieee.org/document/9737217) \[2023 TPAMI\] :octocat:[code](https://github.com/valeoai/xmuda_journal)
* [Pointglr: Unsupervised structural representation learning of 3d point clouds](https://ieeexplore.ieee.org/document/9736689) \[2023 TPAMI\] :octocat:[code](https://github.com/raoyongming/PointGLR)
* [Prototype adaption and projection for few- and zero-shot 3d point cloud semantic segmentation](https://ieeexplore.ieee.org/document/10138737/) \[2023 TIP\] :octocat:[code](https://github.com/heshuting555/PAP-FZS3D)
* [A multi-phase camera-lidar fusion network for 3d semantic segmentation with weak supervision](https://ieeexplore.ieee.org/abstract/document/10035004) \[2023 TCSVT\] 
* [Geometry and uncertainty-aware 3d point cloud class-incremental semantic segmentation](https://ieeexplore.ieee.org/document/10204829) \[2023 CVPR\] :octocat:[code](https://github.com/leolyj/3DPC-CISS)
* [Novel class discovery for 3d point cloud semantic segmentation](https://ieeexplore.ieee.org/document/10203892) \[2023 CVPR\] :octocat:[code](https://github.com/LuigiRiz/NOPS)
* [Lasermix for semi-supervised lidar semantic segmentation](https://ieeexplore.ieee.org/document/10205234) \[2023 CVPR\] :octocat:[code](https://github.com/ldkong1205/LaserMix)
* [Less is more: Reducing task and model complexity for 3d point cloud semantic segmentation](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Less_Is_More_Reducing_Task_and_Model_Complexity_for_3D_CVPR_2023_paper.html) \[2023 CVPR\] :octocat:[code](https://github.com/l1997i/lim3d)
* [PLA: Language-Driven Open-Vocabulary 3D Scene Understanding](https://openaccess.thecvf.com/content/CVPR2023/html/Ding_PLA_Language-Driven_Open-Vocabulary_3D_Scene_Understanding_CVPR_2023_paper.html) \[2023 CVPR\] :octocat:[code](https://dingry.github.io/projects/PLA)
* [Growsp: Unsupervised semantic segmentation of 3d point clouds](https://ieeexplore.ieee.org/abstract/document/10203698) \[2023 CVPR\] :octocat:[code](https://github.com/vLAR-group/GrowSP)
* [MSeg3D: Multi-Modal 3D Semantic Segmentation for Autonomous Driving](https://ieeexplore.ieee.org/document/10203290) \[2023 CVPR\] :octocat:[code](https://github.com/jialeli1/lidarseg3d)
* [Label-guided knowledge distillation for continual semantic segmentation on 2d images and 3d point clouds](https://ieeexplore.ieee.org/document/10377766) \[2023 ICCV\] :octocat:[code](https://github.com/Ze-Yang/LGKD)
* [Zero-shot point cloud segmentation by semantic-visual aware synthesis](https://ieeexplore.ieee.org/document/10377426) \[2023 ICCV\] :octocat:[code](https://github.com/leolyj/3DPC-GZSL)
* [Walking Your LiDOG: A Journey Through Multiple Domains for LiDAR Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2023/html/Saltori_Walking_Your_LiDOG_A_Journey_Through_Multiple_Domains_for_LiDAR_ICCV_2023_paper.html) \[2023 ICCV\] :octocat:[code](https://github.com/saltoricristiano/lidog)
* [Hierarchical pointbased active learning for semi-supervised point cloud semantic segmentation](https://openaccess.thecvf.com/content/ICCV2023/html/Xu_Hierarchical_Point-based_Active_Learning_for_Semi-supervised_Point_Cloud_Semantic_Segmentation_ICCV_2023_paper.html) \[2023 ICCV\] :octocat:[code](https://github.com/SmiletoE/HPAL)
* [Cpcm: Contextual point cloud modeling for weaklysupervised point cloud semantic segmentation](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_CPCM_Contextual_Point_Cloud_Modeling_for_Weakly-supervised_Point_Cloud_Semantic_ICCV_2023_paper.html) \[2023 ICCV\] :octocat:[code](https://github.com/lizhaoliu-Lec/CPCM)
* [Towards open vocabulary learning: A survey](https://ieeexplore.ieee.org/document/10420487) \[2024 TPAMI\] 
* [RegionPLC: Regional Point-Language Contrastive Learning for Open-World 3D Scene Understanding](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_RegionPLC_Regional_Point-Language_Contrastive_Learning_for_Open-World_3D_Scene_Understanding_CVPR_2024_paper.html) \[2024 CVPR\] 
* [Dense supervision propagation for weakly supervised semantic segmentation on 3d point clouds](https://ieeexplore.ieee.org/document/10328634) \[2024 TCSVT\] 

##### (2) Real-time Segmentation
* [Lightningnet: Fast and accurate semantic segmentation for autonomous driving based on 3d lidar point cloud](https://ieeexplore.ieee.org/document/9102769) \[2020 ICME\] 
* [Lite-hdseg: Lidar semantic segmentation using lite harmonic dense convolutions](https://ieeexplore.ieee.org/document/9561171) \[2021 ICRA\] 
* [Stage-aware feature alignment network for real-time semantic segmentation of street scenes](https://ieeexplore.ieee.org/document/9583294) \[2022 TCSVT\] 
* [3d semantic segmentation of aerial photogrammetry models based on orthographic projection](https://ieeexplore.ieee.org/document/10119167) \[2023 TCSVT\]
* [Knowledge distillation from 3d to bird’s-eye-view for lidar semantic segmentation](https://ieeexplore.ieee.org/document/10220057) \[2023 ICME\] :octocat:[code](https://github.com/fengjiang5/Knowledge-Distillation-from-Cylinder3D-to-PolarNet)
* [Rangevit: Towards vision transformers for 3d semantic segmentation in autonomous driving](https://ieeexplore.ieee.org/document/10204428) \[2023 CVPR\] :octocat:[code](https://github.com/valeoai/rangevit)
* [Bird's-Eye-View Semantic Segmentation With Two-Stream Compact Depth Transformation and Feature Rectification](https://ieeexplore.ieee.org/abstract/document/10124335) \[2023 TIV\] 
* [Residual graph convolutional network for bird’seye-view semantic segmentation](https://ieeexplore.ieee.org/document/10483624) \[2024 WACV\] 

##### (3) Point-based Semantic Segmentation
* [Pointnet++: Deep hierarchical feature learning on point sets in a metric space](https://dl.acm.org/doi/abs/10.5555/3295222.3295263) \[2017 NIPS\] :octocat:[code](https://github.com/charlesq34/pointnet2)
* [Point transformer](https://ieeexplore.ieee.org/document/9710703) \[2021 ICCV\]
* [Backward attentive fusing network with local aggregation classifier for 3d point cloud semantic segmentation](https://ieeexplore.ieee.org/abstract/document/9410334) \[2021 TIP\] :octocat:[code](https://github.com/Xiangxu-0103/BAF-LAC)
* [Cga-net: Category guided aggregation for point cloud semantic segmentation](https://ieeexplore.ieee.org/document/9577467) \[2021 CVPR\] :octocat:[code](https://github.com/MCG-NJU/CGA-Net)
* [Point transformer v2: Grouped vector attention and partition-based pooling](https://papers.nips.cc/paper_files/paper/2022/hash/d78ece6613953f46501b958b7bb4582f-Abstract-Conference.html) \[2022 NIPS\] :octocat:[code](https://github.com/Gofinge/PointTransformerV2)
* [Dcnet: Large-scale point cloud semantic segmentation with discriminative and efficient feature aggregation](https://ieeexplore.ieee.org/document/10025770) \[2023 TCSVT\] :octocat:[code](https://github.com/fukunyin/DCNet)
* [Point transformer v3: Simpler, faster, stronger](https://arxiv.org/abs/2312.10035) \[2024 CVPR\] :octocat:[code](https://github.com/Pointcept/PointTransformerV3)

##### (4) Unit-sets-based on Semantic Segmentation
* [Instance segmentation in 3d scenes using semantic superpoint tree networks](https://ieeexplore.ieee.org/document/9709996) \[2021 ICCV\] :octocat:[code](https://github.com/Gorilla-Lab-SCUT/SSTNet)
* [Sspc-net: Semi-supervised semantic 3d point cloud segmentation network](https://ojs.aaai.org/index.php/AAAI/article/view/16200) \[2021 AAAI\] :octocat:[code](https://github.com/MMCheng/SSPC-Net)
* [One thing one click: A self-training approach for weakly supervised 3d semantic segmentation](https://ieeexplore.ieee.org/abstract/document/9578763) \[2021 CVPR\] :octocat:[code](https://github.com/liuzhengzhe/One-Thing-One-Click)
* [Rpvnet: A deep and efficient range-point-voxel fusion network for lidar point cloud segmentation](https://ieeexplore.ieee.org/abstract/document/9709941) \[2021 ICCV\] :octocat:[code](https://github.com/GuoPingPan/RPVNet)
* [Superpoint-guided semisupervised semantic segmentation of 3d point clouds](https://ieeexplore.ieee.org/document/9811904) \[2022 ICRA\] 
* [Pointdc: Unsupervised semantic segmentation of 3d point clouds via cross-modal distillation and super-voxel clustering](https://ieeexplore.ieee.org/document/10377176) \[2023 ICCV\] :octocat:[code](https://github.com/SCUT-BIP-Lab/PointDC)
* [Point-to-voxel knowledge distillation for lidar semantic segmentation](https://ieeexplore.ieee.org/document/9879674) \[2022 CVPR\] :octocat:[code](https://github.com/cardwing/Codes-for-PVKD)
* [Pointdc: Unsupervised semantic segmentation of 3d point clouds via cross-modal distillation and super-voxel clustering](https://ieeexplore.ieee.org/document/10377176) \[2023 ICCV\] :octocat:[code](https://github.com/SCUT-BIP-Lab/PointDC)
* [Nested architecture search for point cloud semantic segmentation](https://ieeexplore.ieee.org/document/9919408) \[2023 TIP\] :octocat:[code](https://github.com/fanyang587/NestedNet)
* [Multi-to-single knowledge distillation for point cloud semantic segmentation](https://ieeexplore.ieee.org/document/10160496) \[2023 ICRA\] :octocat:[code](https://github.com/skyshoumeng/M2SKD)
* [Pointdistiller: Structured knowledge distillation towards efficient and compact 3d detection](https://ieeexplore.ieee.org/abstract/document/10205029) \[2023 CVPR\] :octocat:[code](https://github.com/RunpeiDong/PointDistiller)
* [Knowledge distillation from 3d to bird’s-eye-view for lidar semantic segmentation](https://ieeexplore.ieee.org/document/10220057) \[2023 ICME\] :octocat:[code](https://github.com/fengjiang5/Knowledge-Distillation-from-Cylinder3D-to-PolarNet)

### 1.2 Point Cloud Compression
* [Inter-frame compression for dynamic point cloud geometry coding](https://ieeexplore.ieee.org/document/10380494) \[2024 TIP\] 
* []() \[\] :octocat:[code]()
### 1.3 Point Cloud Registration

* []() \[\] :octocat:[code]()
* []() \[\] :octocat:[code]()
### 1.4 Point Cloud Reconstruction
* []() \[\] :octocat:[code]()
* []() \[\] :octocat:[code]()
* []() \[\] :octocat:[code]()
* [Rfd-net: Point scene understanding by semantic instance reconstruction](https://ieeexplore.ieee.org/document/9578585) \[2021 CVPR\]
* [NeRF-LiDAR: Generating Realistic LiDAR Point Clouds with Neural Radiance Fields](https://arxiv.org/pdf/2304.14811) \[2024 AAAI\] :octocat:[code](https://github.com/fudan-zvg/NeRF-LiDAR)
  
## 2. New Point Cloud Tasks with Semantic

### 2.1 3D Scene Understanding
#### 2.1.1 Scene Graph Prediction
* []() \[\] :octocat:[code]()
* []() \[\] :octocat:[code]()
* []() \[\] :octocat:[code]()
#### 2.1.2 3D vision with language
* [Scan2cap: Context-aware dense captioning in rgb-d scans](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Scan2Cap_Context-Aware_Dense_Captioning_in_RGB-D_Scans_CVPR_2021_paper.pdf) \[2021 CVPR\] :octocat:[code](https://github.com/daveredrum/Scan2Cap)
* [Free-form description guided 3d visual graph network for object grounding in point cloud](https://ieeexplore.ieee.org/document/9710755) \[2021 ICCV\] :octocat:[code](https://github.com/PNXD/FFL-3DOG)
* [Spatiality-guided transformer for 3d dense captioning on point clouds](https://arxiv.org/abs/2204.10688) \[2022 arxiv\] :octocat:[code](https://spacap3d.github.io/)
* [X-trans2cap: Cross-modal knowledge transfer using transformer for 3d dense captioning](https://ieeexplore.ieee.org/document/9879338) \[2022 CVPR\]
* [3DJCG: A Unified Framework for Joint Dense Captioning and Visual Grounding on 3D Point Clouds](https://ieeexplore.ieee.org/document/9879358) \[2022 CVPR\]
* [A comprehensive survey of 3d dense captioning: Localizing and describing objects in 3d scenes](https://ieeexplore.ieee.org/document/10187165) \[2024 TCSVT\]
* [Pointnetvlad: Deep point cloud based retrieval for large-scale place recognition](https://ieeexplore.ieee.org/document/8578568) \[2018 CVPR\] :octocat:[code](https://github.com/mikacuy/pointnetvlad)
* [Text2Loc: 3D Point Cloud Localization from Natural Language](https://arxiv.org/pdf/2311.15977) \[2024 CVPR\] :octocat:[code](https://github.com/Yan-Xia/Text2Loc)

### 2.2 Point Cloud Semantic Scene Completion
* [See and think: Disentangling semantic scene completion](https://dl.acm.org/doi/10.5555/3326943.3326968) \[2018 NIPS\] :octocat:[code](https://github.com/ShiceLiu/SATNet)
* [3D Semantic Scene Completion from a Single Depth Image Using Adversarial Training](https://ieeexplore.ieee.org/abstract/document/8803174) \[2019 ICIP\] :octocat:[code](https://github.com/shurans/sscnet)
* [Cascaded context pyramid for full-resolution 3d semantic scene completion](https://ieeexplore.ieee.org/document/9008381) \[2019 ICCV\] 
* [3d sketch-aware semantic scene completion via semi-supervised structure prior](https://ieeexplore.ieee.org/document/9156418) \[2020 CVPR\] :octocat:[code](https://github.com/CV-IP/3D-SketchAware-SSC)
* [Attention-based multimodal fusion network for semantic scene completion](https://ojs.aaai.org/index.php/AAAI/article/view/6803) \[2020 AAAI\]
* [Semantic scene completion using local deep implicit functions on lidar data](https://ieeexplore.ieee.org/abstract/document/9477025) \[2022 TPAMI\]
* [Voxformer: Sparse voxel transformer for camerabased 3d semantic scene completion](https://ieeexplore.ieee.org/document/10203337) \[2023 CVPR\] :octocat:[code](https://github.com/NV1abs/VoxFormer)
* [Scpnet: Semantic scene completion on point cloud](https://ieeexplore.ieee.org/document/10203998) \[2023 CVPR\] :octocat:[code](https://github.com/SCPNet/Codes-for-SCPNet)
* [Occdepth: A depth-aware method for 3d semantic scene completion](https://arxiv.org/abs/2302.13540) \[2023 arxiv\] :octocat:[code](https://github.com/megvii-research/OccDepth)
* [Ddit: Semantic scene completion via deformable deep implicit templates](https://ieeexplore.ieee.org/document/10376787) \[2023 ICCV\] 
* [Cvsformer: Cross-view synthesis transformer for semantic scene completion](https://ieeexplore.ieee.org/abstract/document/10378387) \[2023 ICCV\] :octocat:[code](https://github.com/donghaotian123/CVSformer.)
* [Ndc-scene: Boost monocular 3d semantic scene completion in normalized device coordinates space](https://ieeexplore.ieee.org/document/10376597) \[2023 ICCV\] :octocat:[code](https://github.com/Jiawei-Yao0812/NDCScene)
* [Esc-net: Alleviating triple sparsity on 3d lidar point clouds for extreme sparse scene completion](https://ieeexplore.ieee.org/document/10409585) \[2024 TMM\]
* [Symphonize 3D Semantic Scene Completion with Contextual Instance Queries](https://arxiv.org/pdf/2306.15670) \[2024 CVPR\] :octocat:[code](https://github.com/hustvl/Symphonies)
* [Unleashing Network Potentials for Semantic Scene Completion](https://arxiv.org/pdf/2403.07560v1) \[2024 CVPR\] :octocat:[code](https://github.com/fereenwong/AMMNet)
* [SemCity: Semantic Scene Generation with Triplane Diffusion](https://arxiv.org/pdf/2403.07773v1) \[2024 CVPR\] :octocat:[code](https://github.com/zoomin-lee/SemCity)
  
### 2.3 Point Cloud Understanding
#### 2.3.1 Integration Tasks in Point Clouds
* [Jsis3d: Joint semantic-instance segmentation of 3d point clouds with multi-task pointwise networks and multi-value conditional random fields](https://ieeexplore.ieee.org/document/9412532) \[2020 ICPR\] :octocat:[code](https://github.com/pqhieu/jsis3d)
* [Associatively segmenting instances and semantics in point clouds](https://ieeexplore.ieee.org/document/8953321) \[2019 CVPR\] :octocat:[code](https://github.com/WXinlong/ASIS)
* [Semantic labeling and instance segmentation of 3d point clouds using patch context analysis and multiscale processing](https://ieeexplore.ieee.org/document/8590720) \[2020 TVCG\]
* [Jsnet++: Dynamic filters and pointwise correlation for 3d point cloud instance and semantic segmentation](https://ieeexplore.ieee.org/document/9932589) \[2023 TCSVT\] :octocat:[code](https://github.com/dlinzhao/JSNetPP)
* [Explore in-context learning for 3d point cloud understanding](https://papers.nips.cc/paper_files/paper/2023/file/8407d254b5baacf69ee977aa34f0e521-Paper-Conference.pdf) \[2023 NIPS\] :octocat:[code](https://github.com/fanglaosi/Point-In-Context)
* [Multi-Space Alignments Towards Universal LiDAR Segmentation](https://arxiv.org/pdf/2405.01538v1) \[2024 CVPR\] :octocat:[code](https://github.com/youquanl/M3Net)
* [X-3D: Explicit 3D Structure Modeling for Point Cloud Recognition](https://arxiv.org/pdf/2404.15010) \[2024 CVPR\] :octocat:[code](https://github.com/sunshuofeng/X-3D)
* [Geometrically-driven Aggregation for Zero-shot 3D Point Cloud Understanding](https://arxiv.org/pdf/2312.02244) \[2024 CVPR\] :octocat:[code](https://luigiriz.github.io/geoze-website/)
#### 2.3.2 Multi-modality 
* [Pointclip: Point cloud understanding by clip](https://ieeexplore.ieee.org/document/9878980) \[2022 CVPR\] :octocat:[code](https://github.com/ZrrSkywalker/PointCLIP)
* [Crosspoint: Self-supervised cross-modal contrastive learning for 3d point cloud understanding](https://ieeexplore.ieee.org/document/9878878) \[2022 CVPR\] :octocat:[code](https://github.com/MohamedAfham/CrossPoint)
* [Leaf: Learning frames for 4d point cloud sequence understanding](https://ieeexplore.ieee.org/document/10377208) \[2023 ICCV\]
* [Point-Bind & Point-LLM: Aligning Point Cloud with Multi-modality for 3D Understanding, Generation, and Instruction Following](https://arxiv.org/pdf/2309.00615) \[arxiv 2023\] :octocat:[code](https://github.com/ZiyuGuo99/Point-Bind_Point-LLM)
* [Echoes Beyond Points: Unleashing the Power of Raw Radar Data in Multi-modality Fusion](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a8f7f12b29d9b8c227785f6b529f63b7-Abstract-Conference.html) \[2023 NIPS\] :octocat:[code](https://github.com/tusen-ai/EchoFusion)
* [MM-Point: Multi-View Information-Enhanced Multi-Modal Self-Supervised 3D Point Cloud Understanding](https://arxiv.org/pdf/2402.10002) \[2024 AAAI\] :octocat:[code](https://github.com/HaydenYu/MM-Point)
  

