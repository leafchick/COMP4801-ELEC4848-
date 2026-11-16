# 讨论注意力的早期识别 / Early Recognition of Attention Episodes

用 Social Signal 的 firm-day 注意力指数切出「讨论爆发」事件，并在只看到事件前 20% / 40% / 60% 时，预测它最终收敛成**单峰、双峰还是平台**。

当前进度：数据可用性验证 + 分段 + 形状提示 + 出图，已跑通。模型代码已就位，等人工标注填完即可训练。

---

## 目录

```
.
├── data/                     # 不进 git
│   ├── raw/                  # social_signal_index.dta, crsp_daily.parquet
│   └── interim/              # episodes.csv, episodes_to_label.csv
├── notebooks/
│   ├── 01_data_and_episodes.ipynb
│   └── figures/              # episodes_preview.png, shape_check.png
├── src/
│   ├── load_data.py          # 三个 loader，全部带 as_of
│   ├── episodes.py           # 分段 + 形状提示 + 出图
│   └── models.py             # 逻辑回归 + LightGBM
└── README.md
```

## 环境

Python 3.12，依赖见 `requirements.txt`。

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

跑一遍：

```bash
python -m src.load_data      # 数据自检：行数、股票数、覆盖密度
python -m src.episodes       # 分段 + 写 CSV + 出预览图
```

或打开 `notebooks/01_data_and_episodes.ipynb`。

---

## 数据

### 注意力（主信号）

`data/raw/social_signal_index.dta` —— Cookson, Lu, Mullins & Niessner (2024), *The Social Signal*, JFE。

**不要直接下整个 1.42 GB 的 replication package。** 那个包里除了这个指数，其余是作者明确标注的 **stand-in 替身数据**（StockTwits / Twitter / Seeking Alpha / CRSP / Compustat 都是专有的，不能公开）。真正要用的只有 `Social signal index/` 目录下的三个文件，合计约 11.5 MB：

| 文件 | 用途 |
|---|---|
| `social_signal_index.dta` | 821,534 条 firm-day 指数，本文主数据 |
| `socal_signal_index.do` | 指数构造过程（Stata） |
| `README.docx` | 作者对指数的说明 |

获取方式：从 [Mendeley 页面](https://data.mendeley.com/datasets/xffyybvw4j/1) 下载 ZIP 后只解压该目录；或用 HTTP Range 只取这三个文件（1.42 GB → 8 MB）：

<details>
<summary>只取 index 文件的脚本（点开）</summary>

```python
import urllib.request, struct, zlib, os

URL = "https://data.mendeley.com/public-api/zip/xffyybvw4j/download/1"
SIZE = 1529725468
H = {"User-Agent": "Mozilla/5.0",
     "Referer": "https://data.mendeley.com/datasets/xffyybvw4j/1"}

def get(a, b):
    h = dict(H); h["Range"] = f"bytes={a}-{b}"
    return urllib.request.urlopen(urllib.request.Request(URL, headers=h), timeout=180).read()

t = get(SIZE - 65536, SIZE - 1)          # end-of-central-directory
i = t.rfind(b"PK\x05\x06")
cd_size, cd_off = struct.unpack("<II", t[i + 12 : i + 20])
cd = get(cd_off, cd_off + cd_size - 1)   # central directory

p = 0
while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
    f = struct.unpack("<IHHHHHHIIIHHHHHII", cd[p:p + 46])
    nlen, elen, clen = f[10], f[11], f[12]
    name = cd[p + 46 : p + 46 + nlen].decode()
    if "Social signal index" in name:
        lh = get(f[16], f[16] + 29)
        n2, e2 = struct.unpack("<HH", lh[26:30])
        start = f[16] + 30 + n2 + e2
        raw = get(start, start + f[8] - 1)
        data = zlib.decompress(raw, -15) if f[4] == 8 else raw
        open(os.path.join("data/raw", os.path.basename(name)), "wb").write(data)
        print("saved", name, len(data))
    p += 46 + nlen + elen + clen
```
</details>

### 价格（标签）

CRSP 日度，从 WRDS 导出，至少含 `permno, date, ret`，存成 `data/raw/crsp_daily.parquet`。

`load_data.py` 里已经写好 `load_crsp`、`add_forward_returns`（R20）、`add_forward_drawdown`（MDD20）。没有文件时会直接报错，不会用假数据糊过去。

---

## 方法

### 分段规则（写死，不调优）

```python
HI = 3.0          # 开始：attention >= 过去 60 个交易日均值的 3 倍
LO = 1.5          # 结束：回落到基线的 1.5 倍以下
QUIET_DAYS = 5    # ... 并连续保持 5 个交易日
WINDOW = 60       # 基线窗口
MIN_OBS = 10      # 窗口内至少 10 个观测日
MIN_DAYS = 3      # 事件至少 3 个观测日
MAX_DAYS = 60     # 超过 60 天视为水平位移，不是爆发
```

`HI` / `LO` 是在 **exp(z)** 上比的，不是原始指数。原因：发布的是标准化主成分，中位数 −0.21，可以取负值；「3 倍均值」在负数上没有定义（3 × 负均值还是负数，几乎每天都触发）。`exp(z)` 单调且恒正，比值 3 等价于指数比基线高 `ln 3 ≈ 1.10`。

**基线在事件开始时冻结。** 如果基线每天重算，爆发会把自己抬高：一次 5 倍冲击持续三周后，滚动均值被拉到 2–3 倍，比值自然掉到 1.5 以下，平台型在五周左右被误判结束。冻结后，「回落到爆发前的正常水平」才是结束条件。

### 四类事件

| 状态 | 含义 | 是否标注 |
|---|---|---|
| `complete` | 上升段和下降段都看到了 | ✅ |
| `censored` | 峰值落在第一天或最后一天 → 上升或下降段没被观测到 | ❌ |
| `long` | 超过 60 个交易日，是水平位移不是爆发 | ❌ |
| `open` | 到 as_of 还没结束 | ❌ |

`censored` 占了 54%，不是一个可以忽略的边角：指数只在高讨论日存在，很多爆发的第一天就已经是峰值。硬给这种曲线贴「单峰」标签，模型学到的会是「衰减 = 单峰」。

### 形状提示

`shape_auto` 只是给人工标注加速用的初稿：两个显著峰（prominence ≥ 0.40）且中间谷底低于 0.6 → 双峰；顶部平坦日占比 ≥ 0.4 → 平台；否则单峰。**最终以 `shape` 列为准。**

---

## 当前结果（`as_of = 2021-12-31`）

| 指标 | 数值 |
|---|---|
| firm-day 观测 | 821,534 |
| 股票数 | 1,500 |
| 交易日 | 2,518（2012-01-03 → 2021-12-31） |
| 面板密度 | 21.8% |
| 切出事件 | 1,751 |
| 有事件的股票 | 624 / 1,500 |
| 可标注事件 | **745**（368 只股票） |
| 可标注事件中位长度 | 8 个交易日 |
| 自动形状提示 | 单峰 484 / 双峰 258 / 平台 3 |

三个必须写进论文的事实：

1. **54% 的事件被截断**（峰值在首/末日），能被完整观测形状的只有 745 个。
2. **平台型只有 41 个**，三类严重不平衡；训练时必须报 macro-F1 和每类召回率，不能只报 accuracy。
3. **只有 41% 的股票（624/1500）在 2012–2021 间出现过至少一次爆发**；能标注的只有 368 只。这就是「受众范围」的真实答案——不是「大部分股票」，而是「有过讨论爆发的这部分」。

---

## 标注任务与初标

最终任务定为**二分类：单峰 vs 非单峰**。

三类（single / double / plateau）在 745 条里平台只占 1–2%，训不出来：带类别权重时逻辑回归为了抓 1 条平台会误报 162 条（precision 0.01），不带权重时 LightGBM 一条平台都不预测（recall 0），而 accuracy 0.71 完全掩盖了这个问题。合并成 `multi` 之后正负比约 3:1，三个时间块都有两类，macro-F1 从 0.43 升到 0.61。

745 条已做过两遍三分类初标，保留作为**参考**（不是最终标签）：

| | single | double | plateau |
|---|---|---|---|
| 规则（`shape_auto`） | 484 | 258 | 3 |
| 第二遍（看图） | 563 | 173 | 9 |

折算成二分类：规则 484 / 261，第二遍 563 / 182，**一致率 87.2%，Cohen's κ = 0.699**，95 条不一致。

最终标签由人工在 `notebooks/label_all.html` 里重标（745 张图内嵌，双击打开）。规则和第二遍只作为参考标签显示，可以一键隐藏。

这一轮对比还抓到一个真实的 bug：原来的平台判据是「≥40% 的天数在高位」，在 4–5 天的短事件里，只要前两天高就会被判成平台。已改成「连续 ≥3 天在高位，且占比 ≥25%」，平台从 41 条降到 3 条。



## `as_of` 约定

所有对外函数第一个参数之后就是 `as_of`，内部只保留 `date <= as_of` 的行。两个例外是标签构造器（`add_forward_returns`、`add_forward_drawdown`），它们**只能**用于造标签，不能进特征。

t 日的注意力指数视为 t 日收盘后可得，因此标签从 t+1 开始算。

---

## 已知限制

- 只有注意力指数，没有原始帖子数/新闻条数；无法区分「公司自产内容」和「独立报道」。要做来源归因，需要 FNSPID 的 publisher 字段（见下一步）。
- 指数是份额型主成分，不是绝对讨论量；跨股票的绝对热度不可比。
- 形状标签是「规则 + 第二遍看图」两遍三分类初标，折算二分类后 κ = 0.699；最终标签仍需人工重标，口径由人定。
- 平台型只占 1–2%，已合并进 `multi`；如果论文要单独讨论平台，只能作描述性统计。
- CRSP 标签还没接上，收益相关结论一律未验证。

## 下一步

1. 打开 `notebooks/label_all.html` 标 745 条（单峰 / 非单峰）→ 导出 `episodes_labeled.csv` 放进 `data/interim/`。
2. `python -m src.models` —— 20/40/60% 前缀 → 二分类，逻辑回归 + LightGBM，按 2018 年切分。
3. 接 CRSP：R20 / MDD20，做「首次可靠判断时价格已经走了多少」的窗口分析。
4. 财报日虚拟变量：用 8-K/`rdq` 把事件型披露和非事件型宣传分开，检验 C2。
