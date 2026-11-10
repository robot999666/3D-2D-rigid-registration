# 共享模块

- `case_config.py`：病例、路径、图像参数、随机种子和权重选择。
- `pose_render.py`：网格投影、轮廓和肝脏区域渲染、DA2 输入图像生成。
- `da2_engine.py`：DA2 加载、深度推理与预览。
- `depth_augment.py`：深度归一化和训练增强。
- `intraop_masks.py`：人工掩码、质量检查和文件指纹。
- `evaluation.py`：批处理使用的 TRE/IC 评价。

各阶段共用图像尺寸、轮廓线宽和深度预处理。病例及目录可通过 `PATIENT_ID`、`FRAME_ID`、`CASE_ROOT`、`DATA_ROOT`、`RESULT_ROOT` 和 `WEIGHTS_ROOT` 设置；统一入口负责加载项目配置。
