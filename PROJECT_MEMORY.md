# 🧠 农作物识别项目记忆与开发全景归档 (Project Memory & Architecture Archive)

本文档归档了本项目的整体架构、三大创新算法细节、端侧离线 ONNX 量化导出链路、Android 客户端项目结构以及关键踩坑解决方案。

---

## 🌾 一、 项目定位与指标 (Overview & Benchmarks)

本系统是一个基于 **CLIP + LoRA + 三大创新模块** 的农作物生育期智能识别系统，支持 **玉米、小麦、棉花** 三大作物共 15 个生育阶段的诊断。

* **验证集准确率 (Val Acc)**：**93.53%**（对比基线 CLIP + LoRA 84.73% 提升了 **+8.8%**）。
* **作物与阶段清单 (15类)**：
  - **🌽 玉米 (Corn)**: `corn_seedling` (出苗期), `corn_jointing` (拔节期), `corn_tasseling` (抽穗期), `corn_filling` (灌浆期), `corn_maturity` (成熟期)
  - **🌾 小麦 (Wheat)**: `wheat_seedling` (出苗期), `wheat_tillering` (分蘖期), `wheat_jointing` (拔节期), `wheat_heading` (抽穗期), `wheat_maturity` (成熟期)
  - **🌸 棉花 (Cotton)**: `cotton_seedling` (苗期), `cotton_squaring` (蕾期), `cotton_flowering` (开花期), `cotton_boll_setting` (结铃期), `cotton_boll_opening` (吐絮期)

---

## 🚀 二、 三大核心创新模块 (Three Innovation Modules)

1. **置信度引导路由 (Confidence-aware Routing)** (`models/confidence_router.py`)
   - 克服传统硬路由只选 top-1 的弊端。当作物概率模糊（低于阈值 0.7）时，软路由多分支加权融合，防止相似阶段（如玉米拔节 vs 小麦拔节）误判。
2. **生育期关系建模 (Phenology-aware Relation)** (`models/phenology_graph.py`)
   - 利用高斯邻接矩阵和图卷积网络 (GCN)，建模生长阶段的时间连续性。
3. **Adaptive LoRA Rank** (`models/adaptive_lora.py`)
   - 按作物视觉复杂度分配 LoRA 秩：玉米 `rank=4`，小麦 `rank=8`，棉花 `rank=16`。

---

## 📱 三、 端侧离线 Android 部署与 ONNX 导出链路

为了实现偏远农田完全脱网离线运行，建立了完整的 **LoRA 融合 + 路由向量化 + ONNX INT8 量化** 导出体系：

### 1. 核心转换技术点
* **LoRA 在线融合 (Weight Fusion)**：
  根据推理逻辑 $W_{fused} = W_{orig} + \frac{1}{3} \sum_{c=0}^{2} \frac{\alpha_c}{r_c} (B_c \cdot A_c)^T$，将 LoRA 专家参数直接融合回原生的 `nn.Linear` 中，消除移动端额外的 LoRA 分支计算与显存占用。
* **路由向量化 (Tensor Masking)**：
  使用 `torch.cat` 与 `mask` 掩码重构 Python 条件分支，确保 `torch.onnx.export` 顺利导出计算图。

### 2. 导出与量化产物
* **FP32 原生 ONNX**：`saved_models/onnx/crop_model_fp32.onnx` (1166.91 MB)
* **INT8 动态量化 ONNX**：`saved_models/onnx/crop_model_int8.onnx` (**293.26 MB**, 压缩率 **-74.9%**)
* **残差校验指标**：FP32 MSE = $3.66 \times 10^{-14}$；INT8 Top-1 分类与 PyTorch 100% 保持一致。

### 3. 工具脚本说明
* [scripts/export_onnx.py](file:///c:/Users/Vicitior/Desktop/%E6%96%B0%E5%BB%BA%E6%96%87%E4%BB%B6%E5%A4%B9%20%284%29/crop_recognition/scripts/export_onnx.py)：融合 LoRA 并导出端到端 ONNX 模型。
* [scripts/quantize_onnx.py](file:///c:/Users/Vicitior/Desktop/%E6%96%B0%E5%BB%BA%E6%96%87%E4%BB%B6%E5%A4%B9%20%284%29/crop_recognition/scripts/quantize_onnx.py)：使用 ONNX Runtime 内存模型载入规避 Windows 文件锁 Bug，完成 INT8 动态量化。
* [scripts/verify_onnx.py](file:///c:/Users/Vicitior/Desktop/%E6%96%B0%E5%BB%BA%E6%96%87%E4%BB%B6%E5%A4%B9%20%284%29/crop_recognition/scripts/verify_onnx.py)：比对 PyTorch vs ONNX 模型的软分布 MSE 误差与 Top-1 匹配度。

---

## 🛠️ 四、 Android 客户端完整工程架构 (`android_app/`)

在项目根目录下生成了可以直接导入 Android Studio 的工程：

* **核心模块文件**：
  - `ImageUtils.kt`: 图像调整为 `336 x 336` 并按 ImageNet 标准 RGB 均值与标准差归一化为 NCHW `FloatBuffer`，支持 HARDWARE 位图安全转换。
  - `CropRecognitionEngine.kt`: 集成 `onnxruntime-android:1.17.0`，在协程后台线程中执行离线推理，小缓冲区拷贝防 OOM 闪退。
  - `AgronomicKnowledge.kt`: 15 类阶段中英文双语农艺养护（水肥病虫害）数据库。
  - `MainActivity.kt` & `activity_main.xml`: 动态权限申请（修复拍照闪退）、右上角 `🌐 English / 🌐 中文` 自由无缝切换、Material3 翡翠绿高颜值卡片与 Top-3 匹配渲染。
* **已解决的 Gradle 踩坑事项**：
  - **Windows 中文路径拦截**：因父路径包含 `新建文件夹 (4)`，Android Gradle 插件会拦截编译。已在 [android_app/gradle.properties](file:///c:/Users/Vicitior/Desktop/%E6%96%B0%E5%BB%BA%E6%96%87%E4%BB%B6%E5%A4%B9%20%284%29/crop_recognition/android_app/gradle.properties) 中配置 `android.overridePathCheck=true` 成功规避。
  - **模型资源部署**：293MB 的 INT8 模型已自动存入 `android_app/app/src/main/assets/crop_model_int8.onnx`。

---

## 📖 五、 快速维护与使用指南

1. **Web 端使用**：
   ```bash
   python app.py
   ```
2. **重新导出与量化 ONNX**：
   ```bash
   python scripts/export_onnx.py
   python scripts/quantize_onnx.py
   python scripts/verify_onnx.py
   ```
3. **Android 端导入**：
   用 Android Studio 打开 `android_app` 目录，点击 `Sync Project with Gradle Files` 后直接点击 `Run 'app'` 即可。
