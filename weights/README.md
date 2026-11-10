# 模型权重

| 文件 | 用途 |
| --- | --- |
| depth_anything_v2_vitl.pth | DA2 Large，运行前按根目录README下载 |
| refinenet_patient1_fullres_best.pth | Patient1全分辨率实验的刚性位姿网络 |

RefineNet对应1920×1080输入、每路5通道和64×36特征token网格。配置位于configs/patient1_fullres.env。

RefineNet来自主实验result_snapshot/Patient1/02/checkpoints/best.pth；DA2权重由使用者提前从官方渠道下载。第三方代码的版权头保留，上游为[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)和[FoundationPose](https://github.com/NVlabs/FoundationPose)。
