# 基于 OpenCV 的技术图纸矢量化与 SVG 导出

**[English](README.md)**

一个 OpenCV 作品集项目，用于技术图纸清洗、几何检测和 SVG 矢量化。

## 处理流程

```
PNG/JPEG/PDF 页面图像 → 预处理 → 几何检测 → 图元归一化 → SVG/JSON 导出 → 叠加图与报告
```

## 快速开始

```shell
# 安装依赖
uv sync --extra dev

# 生成合成测试夹具
uv run tdv-make-fixtures -o data/fixtures/synthetic

# 矢量化单张图像
uv run tdv-vectorize data/fixtures/synthetic/composite.png -o data/results/runs/my-run

# 批量处理目录
uv run tdv-vectorize data/fixtures/synthetic -o data/results/runs/batch

# 与 ground truth 对比评估
uv run tdv-evaluate data/fixtures/synthetic -o data/results/runs/eval

# 所有命令无需外部 API 密钥
```

## 输出内容

| 资产 | 格式 | 说明 |
|------|------|------|
| SVG | `.svg` | 带颜色分层的矢量渲染（直线、圆、弧、多段线） |
| 图元 | `.json` | 机器可读的结构化几何数据 |
| 叠加图 | `.png` | 检测结果叠加在清洗后的图像上 |
| 中间阶段 | `.png` | 每个预处理步骤的图像，便于对比 |
| 报告 | `.json`/`.md` | 各夹具的精确率/召回率/F1 指标 |

## 配置说明

详见 `configs/default.yaml`。所有参数通过 `pydantic` 类型化，支持运行时覆盖：

```shell
uv run tdv-vectorize input.png -c my_config.yaml -o results/
```

## 功能特性

- **预处理：** 灰度化、去噪（fastNlMeans/双边滤波）、CLAHE 对比度增强、自适应/OTSU 阈值、Hough/minAreaRect 旋转校正、基于轮廓的透视矫正
- **几何检测：** 概率 Hough 直线检测、Hough 圆检测、基于轮廓的弧检测、多边形近似
- **归一化：** 共线线段合并、端点吸附、基于长度的噪声过滤
- **导出：** 带 `<g>` 分层的 SVG、JSON 图元、DXF（通过 ezdxf）
- **PDF 输入：** 通过 pypdfium2（无需系统 poppler）
- **确定性：** 相同输入 + 配置 → 字节级一致的 JSON 和 SVG

## 已知限制

- Hough 参数需要针对每张图像调优以获得最佳检测效果；默认值在干净线条图纸上表现最好
- 旋转校正/透视矫正可能会对干净的合成图像产生错位（专为真实扫描/拍摄图纸设计）
- 弧检测为启发式方法（轮廓最小外接矩形拟合）；精度有波动
- DXF 导出仅支持基本图元（直线、圆、弧、多段线）——不支持块、属性或高级功能
- 不包含深度学习、大模型或 OCR 集成（设计如此）
- 像素 IoU 指标假设二值化清洗后的图像；噪声背景会影响精度

## 无需外部 API 密钥

所有处理均在本地完成。无需云服务、无需 API 密钥、无需网络连接。

## 许可证

MIT
