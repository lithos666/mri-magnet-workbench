# V6 新建模型的 FEM 对照

`mri_magnet.comsol_validation` 创建独立 Java / CLASS，实际执行几何、网格、稳态求解和指定点采样。它不读取或修改原始 MPH，也不改变 COMSOL 的全局安全设置。

## 调用

```python
job = prepare_fem_validation(design, sample, output_dir, comsol_root, settings)
report = run_fem_validation(job, predicted_fields, event_callback, cancel_check)
```

`sample` 使用同一组 `points_mm` 和非负体积/面积积分 `weights`。边界点可用于极值，但其权重为零，不改变平均场。接口会绑定设计 SHA256、采样指纹、Java 文件指纹和输出 CSV 指纹。球形 DSV 在全球边界点足够时另外比较 Bz 的 Z2/Z4/Z6/Z8；总不均匀度仍按 |B| 计算。

默认至少三个逐步加密网格，再在相同末级网格尺度下扩大空气域。只在网格和空气域变化都满足门槛时才设置 `fem_verified=true`。Python 对照同时满足平均场相对误差不超过 0.5%、PPM 差不超过 max(10 ppm, FEM PPM 的 10%) 时，才设置 `model_accuracy_passed=true`。

单点零磁标势只固定标量势的任意常数，不改变物理磁场。默认采用三次磁标势单元、CG/AMG 和 1e-10 相对求解容差；满足显式规模阈值的微型任务使用 PARDISO。默认空气半径为3000 mm，继续扩大检查外域收敛。独立子任务有超时和取消处理，异常或未收敛不会显示为已验证。

## 可选的精确周向扇区

`settings['symmetry_mode']='rotational_sector'` 适用于全部磁环列数相同、周向起始相位等价且磁块不穿过扇区侧面的理想阵列。每环建立一颗完整磁块，空气与目标区裁切为一个 2π/N 周期楔体，两侧使用严格的磁场镜面对称 B·n=0。全域观察点转入楔体，求出的 Bx/By 再旋转回原坐标；CSV 保留原始全域坐标。

显式选择 `rotational_sector` 时，遇到不同列数、不同相位或磁块穿越侧面会拒绝，不近似修补、不改变几何。默认 `auto` 自动检查上述条件，不满足时使用全域。报告同时记录完整阵列块数与实际 FEM 扇区块数。制造误差会破坏该理想对称性，公差扰动不能使用此路径。

扇区路径需先通过小阵列的全域 FEM 对照和独立解析矢量变换对照，再用于大案例。需要单独对照时可显式设置 `symmetry_mode='full'`。

## 状态含义

- `passed`：FEM 两种收敛及所提供 Python 模型的精度均通过。
- `fem_converged_model_accuracy_not_passed`：FEM 已收敛，但 Python 模型未达到精度目标。
- `fem_not_converged`：网格或空气域变化仍过大。
- `failed` / `cancelled`：保留日志，不能用作精度标定。

编译 CLASS 或完成一次网格求解都不能单独构成 FEM 验证。当前指标是给定冻结采样点上的结果，不能冒充整个 DSV 的严格全局极值证明。
