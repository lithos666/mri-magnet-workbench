# COMSOL 已存解读取与对标

这一部分只读取现有 MPH 及其中已经保存的解，不重新划分网格、不调用研究或求解器、不保存回原模型。它与“对优化方案重新运行 FEM”是两个不同的步骤。

## 已提供的接口

- `inspect_mph(path)`：使用 Python 标准库读取 MPH 内的 ZIP/XML；报告参数、物理接口、三组磁化、变量、空气球、网格、数据集、已有数值表和解的快照；前后核对原文件 SHA256。
- `prepare_existing_solution_export(mph_path, points_mm, output_dir, comsol_root, ...)`：生成只读取现有解的 Java 文件、点坐标表、任务说明和 MPH 审查报告。
- `run_existing_solution_export(job, timeout=600)`：调用本机 `comsolcompile.exe`、`comsolbatch.exe`。Java 使用 `ModelUtil.loadCopy` 读取副本，`Interp.getData()` 插值读取 `Bx/By/Bz`，批处理指定 `-nosave`，不保存模型。Python 捕获输出并写新的 CSV。
- `read_field_csv(path, coordinate_unit=None, field_unit=None)`、`write_field_csv(path, points_mm, fields_T)`：采用明确单位的 CSV。
- `compare_field_tables(predicted, reference, coordinate_tolerance_mm=1e-6)`：按坐标一一匹配，报告带符号的分量误差和向量误差。不会自动改变剩磁、反转场方向或缩放磁场来拟合。

读取函数返回 `{"points_mm": [[x,y,z],...], "fields_T": [[Bx,By,Bz],...]}`。CSV 的标准表头是：

```text
x_mm,y_mm,z_mm,Bx_T,By_T,Bz_T
```

也支持 `x (mm),y (mm),z (mm),mfnc.Bx (mT),...`。缺乏单位的 `x,y,z,Bx,By,Bz` 表头必须显式指定坐标单位和磁场单位。坐标支持 mm/cm/m，磁场支持 T/mT/uT。缺点、重复点、无法匹配的点和非有限数值均会报错。

## 使用自己的 FEM 模型

仓库不附带大型 MPH、已有求解结果或个人运行日志。请传入自己保存过解的 MPH，先用 `inspect_mph` 确认物理接口、解与数据集标签，再生成采样任务。不要把一条截线上的极值当作整个平面 FOV 或球形 DSV 的极值。

## 解释对标结果

快速模型和有限元模型有各自的假设。首先应保持相同几何、尺寸、磁化、剩磁、采样坐标和评价域。基于固定均匀磁化的解析模型与当前有限回复磁导率 FEM 可能存在物理模型差异，不能仅用一个缩放系数掩盖。

现有解成功读取，代表“Python → Java编译 → COMSOL批处理 → 已存解采样 → CSV回读”链路可用；不代表优化候选已经重新求解，也不证明网格与外域已经收敛。文件系统访问被 COMSOL 安全设置拒绝时，脚本会报错并保留日志，不会以 Python 预测冒充 FEM 数值。

## 参考文档

- [COMSOL 6.2 Interp](https://doc.comsol.com/6.2/doc/com.comsol.help.comsol/comsol_api_results.50.074.html)
- [COMSOL 6.2 Windows批处理与-nosave](https://doc.comsol.com/6.2/doc/com.comsol.help.comsol/comsol_ref_running.36.31.html)
- [COMSOL 6.2 ModelUtil.loadCopy](https://doc.comsol.com/6.2/doc/com.comsol.help.comsol/api/com/comsol/model/util/ModelUtil.html)
