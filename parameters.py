"""MRI 磁铁排布原理2：唯一可编辑配置是 WORKBENCH_CONFIG。

UI 与普通 CLI 使用同一 target / solver / manufacturing schema。
CONFIG 是给历史磁盘流程和测试保留的派生兼容视图，请勿独立修改。
旧式 JSON 的 fov_diameter_mm 明确迁移为平面域；dsv_diameter_mm 迁移为球域。
"""
from mri_magnet.workbench_config import default_config
from mri_magnet.config_schema import legacy_view
from mri_magnet.legacy_defaults import legacy_runtime_options

WORKBENCH_CONFIG = default_config()
WORKBENCH_CONFIG["mode"] = "de_novo"  # de_novo 不读取下面的源文件；seeded 才加载。
WORKBENCH_CONFIG["source_geometry"] = "examples/seeded/placement.json"
WORKBENCH_CONFIG["comsol_model"] = "../5.MRI_IORing_14Layers_Magnetization.mph"
# 拼装包络与单机热床分开审查；数组顺序是 X、Y、Z，单位 mm。
WORKBENCH_CONFIG["manufacturing"].update(
    printer_bed_mm=[480.0,480.0,255.0], machine_bed_mm=[255.0,255.0,255.0],
    bed_margin_per_side_mm=5.0, sectors_per_layer=4, allow_split_pocket_seams=True)
WORKBENCH_CONFIG["envelope"]["max_length_mm"] = 500.0  # 总高500，半高250。
WORKBENCH_CONFIG["search"]["max_radial_rings_per_layer"] = None  # 自动按包络/节距计算。
WORKBENCH_CONFIG["target"] = {
    "geometry": "sphere", "diameter_mm": 20.0, "z_mm": 0.0,
    "min_field_mT": 30.0, "max_ppm": 1000.0,
}
# 更改参数可在此写入，或在工作台表单/高级JSON中保存当前工程。
# WORKBENCH_CONFIG["solver"]["max_evaluations"] = 1200
# WORKBENCH_CONFIG["solver"]["max_workers"] = 2
# V6正式优化默认阵列自洽互耦；旧工程保留自己的模型，切换后必须重新求场。
# WORKBENCH_CONFIG["physics"].update(field_model="coupled", quadrature_order=2, cells_per_axis=1)
# WORKBENCH_CONFIG["physics"].update(max_unknowns=1200, max_matrix_mb=128.0, prepared_cache_size=4)
# WORKBENCH_CONFIG["search"]["optimize_azimuth_count"] = True
# WORKBENCH_CONFIG["target"]["geometry"] = "disk"  # 保留2cm、z=0平面FOV能力。
# 本轮完整工程参数独立保存在 工程案例_100mmDSV.json；不会覆盖旧工程。
# 按需工程分析位于右侧“谐波 / 公差”，不覆盖原优化或密集验收指标。
# 公差为均匀分布半幅：analysis.tolerance.remanence_half_range_fraction=0.015，
# analysis.tolerance.sector_translation_half_range_mm=0.15；不是高斯标准差。
# WORKBENCH_CONFIG["analysis"]["mutual_demag"]["quadrature_order"] = 2  # 8点目标体积平均
# WORKBENCH_CONFIG["analysis"]["tolerance"]["trials"] = 200
# WORKBENCH_CONFIG["de_novo"]["harmonic_order_weights"] = {"2": 1., "4": 1., "6": .25, "8": .1}

CONFIG = legacy_view(WORKBENCH_CONFIG, legacy_runtime_options())
