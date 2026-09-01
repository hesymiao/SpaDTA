# spaDTA downstream 目录说明

这个目录目前混了三类东西：

1. `spaDTA` 当前主流程真正会用到的 downstream 代码
2. 可以单独直接运行的分析/画图脚本
3. 从旧目录搬过来、但默认路径仍然强依赖 `compare_method/ours` 或 `spatialmeta/Agent` 的脚本

下面先把每个脚本是干什么的理清楚。

## 当前主流程

- `workflow.py`
  - 当前 downstream 主实现。
  - 输入模型输出的 `h5ad`、GT、loss csv 等，生成统一的 downstream 结果目录。
  - 主要工作包括：
    - 读取模型结果
    - 合并病理标签
    - 重新组织 cluster / contribution 信息
    - 画 UMAP、spatial、violin、heatmap
    - 生成 summary json、README、annotated h5ad、tables

- `__init__.py`
  - 只是把 `workflow.py` 里的 `DownstreamConfig`、`run_downstream`、`run_downstream_for_sample`、`run_downstream_for_samples` 暴露出来。

- `run_downstream_for_run_existing_clusters.py`
  - 当前批量入口脚本。
  - 用 `workflow.py` 对一整批样本跑 downstream。
  - 默认是读取 `spaDTA/runs/model_runs/...`，输出到 `spaDTA/runs/downstream_runs/...`。

- `run_mousebrain_sample_downstream_existing_clusters.py`
  - 当前单样本入口脚本。
  - 用 `workflow.py` 对一个样本直接跑 downstream。
  - 现在默认样本是 `Y27_T`。

## 配置文件

- `defaults.py`
  - 只是存一些默认路径和常量。
  - 目前在 `downstream` 目录内部基本没有被实际用到。
  - 更像早期预留的配置文件。

## 生成任务 / 重建任务

- `run_sm_to_st_generation_experiment.py`
  - 训练并评估 `SM -> ST` 生成任务。
  - 不是调用 `workflow.py`，而是自己构建训练、切分、评估、导出结果。
  - 依赖 `spaDTA.model.model.DecAlignSpatialMetaLinear`。

- `run_st_to_sm_generation_experiment.py`
  - 训练并评估 `ST -> SM` 生成任务。
  - 结构和上一个脚本对应，只是方向反过来。

- `run_sm_to_st_generation_experiment_spatial_top_third.py`
  - `SM -> ST` 的 top 1/3 holdout 版本。
  - 同一样本内部按空间坐标切分，默认拿切片上方 1/3 做评估。
  - 推断时使用 train split 的平均 ST library size。

- `run_st_to_sm_generation_experiment_spatial_top_third.py`
  - `ST -> SM` 的 top 1/3 holdout 版本。
  - 输出仍然是 spot 级别 SM 生成结果。

- `evaluate_sm_to_st_superres.py`
  - 评估高分辨率 `SM -> ST` 超分辨生成结果。
  - 输入是 experiment summary、高分辨率 SM h5ad、目标 h5ad。
  - 会重建 joint adata，再把高分辨结果聚合回 spot 级别，与真实 ST 比较。

- `evaluate_sm_to_st_superres_fixed_train_mean_libsize.py`
  - `SM -> ST` 的 cell-level 生成评估版本。
  - 先生成 cell-level ST，再聚合回 spot 级别做指标。
  - 推断时使用 train split 的平均 ST library size。

- `plot_layer_gene_on_spatial_region.py`
  - 在指定空间区域内，把某个 layer 里的单个 gene 画回空间坐标。
  - 现在主要用于 top 1/3 holdout 区域可视化。

## 生物标志物 / 差异分析 / 富集分析

- `run_domain_multiomics_biomarker_analysis.py`
  - 对一个样本按 cluster 做 ST/SM 多组学 biomarker 分析。
  - 会跑 cluster-level marker ranking、dotplot、dendrogram、summary table。
  - 默认路径仍然指向 `compare_method/ours/runs/Y27_T/...`。

- `plot_domain_multiomics_biomarker_visuals.py`
  - 读取上一个脚本产出的 biomarker 结果，再生成热图和空间展示图。
  - 默认也是 `compare_method/ours/runs/Y27_T/...`。

- `plot_major_class_subtype_de_log2fc.py`
  - 针对单个 major class 内部 subtype，画 ST/SM differential log2FC。
  - 当前默认样本是 `Y27_T`。
  - 默认路径还是 `compare_method/ours/runs/...`。

- `plot_gt_major_class_subtype_de_log2fc.py`
  - 和上一个类似，但限制在 GT major class 的范围里做 subtype differential log2FC。
  - 默认路径也是旧 `compare_method/ours` 体系。

- `plot_subtype_fig3hi_like_enrichment.py`
  - 读取 subtype 相关表格，做 reference-style 的基因 GO 和代谢物 enrichment 可视化。
  - 默认样本 `Y27_T`，默认路径仍然在 `compare_method/ours/runs/...`。

- `find_plin2_like_metabolites.py`
  - 给定一个基因和目标 subtype，找空间模式和这个基因相似的代谢物。
  - 依赖 slice summary json 和 SM differential csv。

## 切片空间图 / 局部区域画图

- `plot_group_feature_expression_on_slice.py`
  - 在某个 marker-defined major group 或 subtype 区域里，画指定 ST gene / SM feature 的切片表达图。
  - 默认是 `Y27_T`。
  - 默认读取 `compare_method/ours/runs/Y27_T/...`。

- `plot_group_region_with_subtype_boundary.py`
  - 画一个 major group 的区域分布图，并用平滑边界描出一个 subtype。
  - 默认也是 `Y27_T` 和旧 `compare_method/ours` 路径。

- `plot_imm_region_on_slice.py`
  - 画单个样本的 major-group region 空间分布。
  - 默认路径也指向 `compare_method/ours/runs/Y27_T/...`。

- `refresh_y27_t_real_annotation_feature_panels.py`
  - 自动刷新 `Y27_T` 的 real-annotation feature panels。
  - 它本身不是核心分析逻辑，而是一个调度脚本。
  - 而且它默认调用的是 `compare_method/ours/downstream_code/plot_group_feature_expression_on_slice.py` 和 `plot_group_region_with_subtype_boundary.py`，不是本目录里的版本。
  - 说明这个脚本现在仍然耦合旧目录实现。

## 观测值 / 误差 / 局部可视化

- `plot_generated_vs_true_features_on_slice.py`
  - 把生成值和真实值画到整张 slice 上做对比。
  - 支持指定 true layer、pred layer 和 feature 列表。

- `plot_obs_value_on_full_slice.py`
  - 把一个 `obs` 列画回完整 slice。
  - 适合画 cluster、score、contribution 一类 spot-level 标量。

- `plot_obs_diff_on_full_slice.py`
  - 把两个 `obs` 列的差值画回完整 slice。

- `plot_spot_pcc_on_slice.py`
  - 从 eval h5ad 里算每个 spot 的 PCC，再画到原始切片坐标上。

## 结构/状态分析

- `run_mouse_thymus_single_slice_paga.py`
  - 对鼠胸腺单切片做 state graph / pseudotime / PAGA 分析。
  - 默认样本是 `m3_FMP`。
  - 默认读取 `compare_method/ours/runs/...`。

## 探针 / 鲁棒性分析

- `run_intervention_probe_downstream.py`
  - 用保存好的 embedding 做 intervention-style probe。
  - 主要衡量：
    - `homo_st` 能否预测 ST
    - `homo_sm` 能否预测 SM
    - `joint/fused` 对跨模态预测和互补性的作用
  - 依赖模型保存的 `X_emb_homo_st_decalign_linear`、`X_emb_homo_sm_decalign_linear`、`X_emb_homo_joint_decalign_linear`。

- `run_m3_fmp_noise_robustness.py`
  - 做 `m3_FMP` 的噪声鲁棒性实验。
  - 会构造带噪输入、调用旧训练脚本、重新扫 Leiden target count、对比 ARI/NMI。
  - 默认直接依赖：
    - `compare_method/ours/runs/...`
    - `spatialmeta/Agent/...`
  - 说明它当前不是 `spaDTA` 内部自洽脚本，而是和旧实验目录强绑定的分析脚本。

## 当前整理状态总结

- 真正属于当前 `spaDTA` downstream 主链路的，主要是：
  - `workflow.py`
  - `run_downstream_for_run_existing_clusters.py`
  - `run_mousebrain_sample_downstream_existing_clusters.py`

- 其余大多数脚本都属于“独立运行分析脚本”。

- 其中相当一部分虽然已经放进 `spaDTA/downstream`，但默认路径仍然直接写死在：
  - `compare_method/ours/runs/...`
  - `compare_method/ours/downstream_code/...`
  - `spatialmeta/Agent/...`

- 所以它们现在更像“旧实验脚本归档到这里”，还不能算彻底整理完成的 `spaDTA` 原生 downstream 模块。

## 下一步建议

如果继续整理，建议下一步按下面三类拆：

- `core`
  - 当前真正要保留和维护的 downstream 主流程

- `analysis`
  - 保留的独立分析脚本

- `legacy`
  - 明显耦合 `compare_method/ours` 或 `spatialmeta/Agent` 的旧实验脚本

这样后面判断哪些能删、哪些该改路径、哪些该并到主流程里，会清楚很多。
