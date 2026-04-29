<img width="1648" height="918" alt="imgs_binning_logo" src="https://github.com/user-attachments/assets/6038bea4-df60-40c2-8958-f6a71951c8a7" />

# BINNING!BINNING!BINNING!: jupyter_woe_binner(JWB)
# — Interactive WOE Binning Tool for Jupyter 

<p align="center">
  <b>— Binning it! —</b><br>
  <sub>Interactive Weight of Evidence Binning Tool for Credit Risk Modeling</sub>
</p>

---

## 📖 项目介绍

在信用风控建模中，**WOE（Weight of Evidence）分箱** 是特征工程的核心步骤。通过将连续变量离散化为若干区间（箱），并计算每个区间的 WOE 值和 IV（Information Value），可以：

- **提升模型稳定性**：减少极端值和噪声对模型的干扰
- **捕捉非线性关系**：将连续变量与目标变量之间的非线性关系转化为线性可解释的 WOE 编码
- **增强可解释性**：每个箱的坏账率、WOE 值直观可读，便于业务决策
- **满足监管要求**：分箱后的评分卡模型更符合金融监管对模型可解释性的要求

然而，传统的分箱方式往往依赖代码反复试错，效率低下。**Binning it!** 将分箱过程变为 **Jupyter Notebook 中的可视化交互操作**，让风控建模人员可以：

- 🖱️ **点击选择** 箱子，一键合并或分裂
- 📊 **实时预览** Distribution、Bad Rate、WOE 三图联动
- 🚀 **合并提示** 支持表格中提示合并分箱，智能且方便处理合并
- 🧠 **智能分裂** 基于最优 IV 的决策树算法自动寻找最佳分裂点
- 📋 **表格联动** 柱状图，线图，双向柱状图，表单，变动一目了然
- 🔄 **变量切换** 支持批量变量分箱，Next/Last 快速导航

---

## 🚀 安装步骤

### 环境要求

- Python >= 3.7
- Jupyter Notebook 或 JupyterLab

### 安装

```bash
# 克隆项目
git clone https://github.com/EddIeZhao/jupyter_woe_binner.git
cd jupyter_woe_binner

# 安装依赖
pip install -e .
```

### 依赖项

| 包 | 版本要求 |
|---|---------|
| ipywidgets | >= 7.6 |
| plotly | >= 4.0 |
| pandas | >= 1.0 |
| numpy | — |
| ipyevents | >= 0.9 |

> **JupyterLab 用户** 需额外安装扩展：
> ```bash
> jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyterlab-plotly
> ```

---

## 📝 使用示例

### 单变量分箱

```python
import pandas as pd
import numpy as np
from jupyter_woe_binner import BinningWidget

# 准备数据
np.random.seed(42)
n = 2000
amount = np.random.gamma(2, 50000, n)
prob = 0.3 - 0.2 * (amount - amount.min()) / (amount.max() - amount.min())
target = (np.random.rand(n) < prob).astype(int)
df = pd.DataFrame({'adj_finance_amt': amount, 'target': target})

# 启动交互分箱工具
widget = BinningWidget(df, var_name='adj_finance_amt',
                       target_name='target',
                       event_flag=1,
                       non_event_flag=0,
                       max_bins=6)
widget.display()
```

**操作方式：**

| 操作 | 方式 |
|------|------|
| 选择箱子 | 在 Select 多选框中按住 Ctrl 点击，或点击图表柱子 |
| 合并两个相邻箱 | 选中两个箱 → 点击 **⬌ Merge** 或按 `Ctrl+Shift+W` |
| 分裂一个箱 | 选中一个箱 → 点击 **⬍ Split** 或按 `Ctrl+Shift+Q` |
| 确认分箱 | 点击 **✓ Confirm** |
| 重置 | 点击 **↺ Reset** |

**获取分箱结果：**

```python
widget.bins
# [-inf, 50000.0, 100000.0, 150000.0, inf]
```

### 多变量批量分箱

```python
from jupyter_woe_binner import BinningWidgetList

widget_list = BinningWidgetList(df,
    var_name=['adj_finance_amt', 'credit_6m_usage', 'annual_income', 'age'],
    target_name='target',
    event_flag=1,
    non_event_flag=0,
    max_bins=6)
widget_list.display()
```

**多变量操作：**

| 操作 | 方式 |
|------|------|
| 上一个变量 | 点击 **◀ Last** |
| 下一个变量 | 点击 **Next ▶** |
| 确认当前变量分箱 | 点击 **✓ Confirm**（导航栏显示 ✓ 标记） |

**获取所有变量分箱结果：**

```python
widget_list.bins
# {
#   'adj_finance_amt': [-inf, 50000.0, 100000.0, inf],
#   'credit_6m_usage': [-inf, 0.3, 0.7, inf],
#   'annual_income': [-inf, 20000.0, 50000.0, inf],
#   'age': [-inf, 30, 50, inf]
# }
```

---

## 🔧 技术说明

### 交互技术栈

| 技术 | 用途 |
|------|------|
| **ipywidgets** | Jupyter 内嵌交互控件（按钮、多选框、输出区域） |
| **plotly FigureWidget** | 可交互图表，支持点击事件绑定和实时数据更新 |
| **ipyevents** | DOM 事件监听，实现键盘快捷键（Ctrl+Shift+W/Q） |
| **pandas Styler** | 表格内嵌柱状图（%Total、WOE 双向柱状图） |

### 核心交互机制

1. **图表点击选中**：通过 `FigureWidget.data[i].on_click()` 绑定点击事件，点击柱子切换选中状态，`selectedpoints` 属性高亮显示选中箱

2. **智能分裂算法**：遍历箱内所有唯一值作为候选分裂点，计算每个分裂点的总 IV 值，选择使 IV 最大化的分裂点。若所有分裂点的 IV 均不大于当前 IV，则提示不可分裂

3. **图表重建**：合并/分裂后箱数变化，完全重建 `FigureWidget` 并替换 UI 中的旧图表，确保子图标题、坐标轴等正确更新

4. **键盘快捷键**：通过 `ipyevents.Event` 监听 `keydown` 事件，绑定到整个 UI 容器（而非独立 VBox），确保事件能被正确捕获

5. **多变量导航**：`BinningWidgetList` 为每个变量维护独立的 `BinningWidget` 实例，通过替换 `_content.children` 实现变量切换，各变量的分箱状态互不干扰

### 图表设计

- **Distribution 图**：水平堆叠柱状图，Good（蓝）/Bad（红），柱内显示占比百分比
- **Bad Rate 图**：水平折线图，标记点 + 数值标签
- **WOE 图**：水平柱状图，正值蓝色向右、负值红色向左，0 轴居中虚线标注；相邻箱 WOE 差异 < 0.1 时显示虚线 + "merge?" 提示
- **配色**：论文级低色度配色（Steel Blue / Terracotta / Dark Navy），专业且不刺眼

---

## 📄 License

MIT License

Copyright (c) 2026 EddIe Zhao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 📬 联系方式

**EddIe Zhao**

- GitHub: [https://github.com/EddIeZhao](https://github.com/EddIeZhao)
# BINNING!BINNING!BINNING!: jupyter_woe_binner(JWB)
# — Interactive WOE Binning Tool for Jupyter 

<p align="center">
  <b>— Binning it! —</b><br>
  <sub>Interactive Weight of Evidence Binning Tool for Credit Risk Modeling</sub>
</p>

---

## 📖 项目介绍

在信用风控建模中，**WOE（Weight of Evidence）分箱** 是特征工程的核心步骤。通过将连续变量离散化为若干区间（箱），并计算每个区间的 WOE 值和 IV（Information Value），可以：

- **提升模型稳定性**：减少极端值和噪声对模型的干扰
- **捕捉非线性关系**：将连续变量与目标变量之间的非线性关系转化为线性可解释的 WOE 编码
- **增强可解释性**：每个箱的坏账率、WOE 值直观可读，便于业务决策
- **满足监管要求**：分箱后的评分卡模型更符合金融监管对模型可解释性的要求

然而，传统的分箱方式往往依赖代码反复试错，效率低下。**Binning it!** 将分箱过程变为 **Jupyter Notebook 中的可视化交互操作**，让风控建模人员可以：

- 🖱️ **点击选择** 箱子，一键合并或分裂
- 📊 **实时预览** Distribution、Bad Rate、WOE 三图联动
- 🚀 **合并提示** 支持表格中提示合并分箱，智能且方便处理合并
- 🧠 **智能分裂** 基于最优 IV 的决策树算法自动寻找最佳分裂点
- 📋 **表格联动** 柱状图，线图，双向柱状图，表单，变动一目了然
- 🔄 **变量切换** 支持批量变量分箱，Next/Last 快速导航

---

## 🚀 安装步骤

### 环境要求

- Python >= 3.7
- Jupyter Notebook 或 JupyterLab

### 安装

```bash
# 克隆项目
git clone https://github.com/EddIeZhao/jupyter_woe_binner.git
cd jupyter_woe_binner

# 安装依赖
pip install -e .
```

### 依赖项

| 包 | 版本要求 |
|---|---------|
| ipywidgets | >= 7.6 |
| plotly | >= 4.0 |
| pandas | >= 1.0 |
| numpy | — |
| ipyevents | >= 0.9 |

> **JupyterLab 用户** 需额外安装扩展：
> ```bash
> jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyterlab-plotly
> ```

---

## 📝 使用示例

### 单变量分箱

```python
import pandas as pd
import numpy as np
from jupyter_woe_binner import BinningWidget

# 准备数据
np.random.seed(42)
n = 2000
amount = np.random.gamma(2, 50000, n)
prob = 0.3 - 0.2 * (amount - amount.min()) / (amount.max() - amount.min())
target = (np.random.rand(n) < prob).astype(int)
df = pd.DataFrame({'adj_finance_amt': amount, 'target': target})

# 启动交互分箱工具
widget = BinningWidget(df, var_name='adj_finance_amt',
                       target_name='target',
                       event_flag=1,
                       non_event_flag=0,
                       max_bins=6)
widget.display()
```

**操作方式：**

| 操作 | 方式 |
|------|------|
| 选择箱子 | 在 Select 多选框中按住 Ctrl 点击，或点击图表柱子 |
| 合并两个相邻箱 | 选中两个箱 → 点击 **⬌ Merge** 或按 `Ctrl+Shift+W` |
| 分裂一个箱 | 选中一个箱 → 点击 **⬍ Split** 或按 `Ctrl+Shift+Q` |
| 确认分箱 | 点击 **✓ Confirm** |
| 重置 | 点击 **↺ Reset** |

**获取分箱结果：**

```python
widget.bins
# [-inf, 50000.0, 100000.0, 150000.0, inf]
```

### 多变量批量分箱

```python
from jupyter_woe_binner import BinningWidgetList

widget_list = BinningWidgetList(df,
    var_name=['adj_finance_amt', 'credit_6m_usage', 'annual_income', 'age'],
    target_name='target',
    event_flag=1,
    non_event_flag=0,
    max_bins=6)
widget_list.display()
```

**多变量操作：**

| 操作 | 方式 |
|------|------|
| 上一个变量 | 点击 **◀ Last** |
| 下一个变量 | 点击 **Next ▶** |
| 确认当前变量分箱 | 点击 **✓ Confirm**（导航栏显示 ✓ 标记） |

**获取所有变量分箱结果：**

```python
widget_list.bins
# {
#   'adj_finance_amt': [-inf, 50000.0, 100000.0, inf],
#   'credit_6m_usage': [-inf, 0.3, 0.7, inf],
#   'annual_income': [-inf, 20000.0, 50000.0, inf],
#   'age': [-inf, 30, 50, inf]
# }
```

---

## 🔧 技术说明

### 交互技术栈

| 技术 | 用途 |
|------|------|
| **ipywidgets** | Jupyter 内嵌交互控件（按钮、多选框、输出区域） |
| **plotly FigureWidget** | 可交互图表，支持点击事件绑定和实时数据更新 |
| **ipyevents** | DOM 事件监听，实现键盘快捷键（Ctrl+Shift+W/Q） |
| **pandas Styler** | 表格内嵌柱状图（%Total、WOE 双向柱状图） |

### 核心交互机制

1. **图表点击选中**：通过 `FigureWidget.data[i].on_click()` 绑定点击事件，点击柱子切换选中状态，`selectedpoints` 属性高亮显示选中箱

2. **智能分裂算法**：遍历箱内所有唯一值作为候选分裂点，计算每个分裂点的总 IV 值，选择使 IV 最大化的分裂点。若所有分裂点的 IV 均不大于当前 IV，则提示不可分裂

3. **图表重建**：合并/分裂后箱数变化，完全重建 `FigureWidget` 并替换 UI 中的旧图表，确保子图标题、坐标轴等正确更新

4. **键盘快捷键**：通过 `ipyevents.Event` 监听 `keydown` 事件，绑定到整个 UI 容器（而非独立 VBox），确保事件能被正确捕获

5. **多变量导航**：`BinningWidgetList` 为每个变量维护独立的 `BinningWidget` 实例，通过替换 `_content.children` 实现变量切换，各变量的分箱状态互不干扰

### 图表设计

- **Distribution 图**：水平堆叠柱状图，Good（蓝）/Bad（红），柱内显示占比百分比
- **Bad Rate 图**：水平折线图，标记点 + 数值标签
- **WOE 图**：水平柱状图，正值蓝色向右、负值红色向左，0 轴居中虚线标注；相邻箱 WOE 差异 < 0.1 时显示虚线 + "merge?" 提示
- **配色**：论文级低色度配色（Steel Blue / Terracotta / Dark Navy），专业且不刺眼

---

## 📄 License

MIT License

Copyright (c) 2026 EddIe Zhao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 📬 联系方式

**EddIe Zhao**

- GitHub: [https://github.com/EddIeZhao](https://github.com/EddIeZhao)
