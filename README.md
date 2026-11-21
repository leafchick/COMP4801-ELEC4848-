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
│   ├── models.py             # 形状分类：逻辑回归 + LightGBM
│   └── returns.py            # CRSP 面板 + 收益/回撤模型 A–C
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

## 协作

### 仓库里有什么、没有什么

**有**：代码（`src/`）、报告（`M2_report.md`）、标注页（`notebooks/label_all.html`、`kappa_sample.html`）、WRDS 查询（`notebooks/wrds_extract.sql`）、图（`notebooks/figures/`）。

**没有**：`data/` 全部内容。原因有两个——体积，以及 CRSP 的许可只覆盖 HKU 订阅用户，不能随仓库分发。

所以新人 clone 下来之后，`data/` 是空的，必须自己准备。下面逐项说明。

### 数据获取

| 文件 | 怎么拿 | 能不能给别人 |
|---|---|---|
| `data/raw/social_signal_index.dta` | 见下方「注意力」一节，Mendeley 公开下载 | ✅ CC BY 4.0，可以共享 |
| `data/raw/crsp_*.parquet` | 自己的 HKU WRDS 账号，跑 `notebooks/wrds_extract.sql` | ❌ 许可限制，不能共享 |
| `data/raw/fnspid_*.csv` | HuggingFace `Zihan1004/FNSPID`，需要重新流式解析 | ✅ CC BY-NC 4.0 |

**如果新成员没有 WRDS 账号**：他可以跑通 `src/load_data.py`、`src/episodes.py`、`src/models.py`（只需要 Social Signal），但 `src/returns.py` 和 `src/news.py` 跑不了。分工时要注意这一点。

### 四个可复现的入口

```bash
python -m src.load_data     # 数据自检
python -m src.episodes      # 分段 → data/interim/episodes.csv
python -m src.models        # 形状分类（需要 episodes_labeled.csv）
python -m src.returns       # CRSP 面板 + Model A–C + 行动窗口
python -m src.news          # Model C vs D
```

### 分支约定

- `main` 永远保持可运行，不直接在上面改
- 每人开自己的分支：`git checkout -b feat/xxx`，完成后提 PR
- 提交前跑一遍 `python -m src.load_data`，确认没把数据路径写死
- **不要提交 `data/` 里的任何东西**（`.gitignore` 已经挡了，但别用 `git add -f`）

### 会被 git 忽略的

`data/`、`.venv/`、`wheels/`、`.matplotlib/`、`.idea/`、`*.pdf`、`*.pptx`、`browser-screenshots/`。

课程文档（proposal、guideline、deck）留在本地；如果想让它们进版本库，删掉 `.gitignore` 最后两行即可。

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

最终标签：95 条分歧已由人工逐条裁定（52 单峰 / 43 非单峰），其余 650 条沿用「规则与第二遍一致」的结果。最终分布 **单峰 528 / 非单峰 217**（29.1%）。非单峰事件的中位长度 16 天，单峰 6 天——标签有经济含义，不是噪声。

## 初步结果（`as_of = 2021-12-31`，训练 2012–2017 / 测试 2020–2021）

多数类基线：accuracy 0.686，macro-F1 0.407。

| 前缀 | 模型 | accuracy | macro-F1 | AUC | 非单峰召回 |
|---|---|---|---|---|---|
| 20% | logistic | 0.610 | 0.590 | 0.642 | 0.615 |
| 20% | LightGBM | 0.669 | 0.603 | 0.636 | 0.418 |
| 40% | logistic | 0.617 | 0.606 | 0.704 | 0.714 |
| 40% | LightGBM | **0.700** | **0.676** | 0.735 | 0.681 |
| 60% | logistic | 0.631 | 0.622 | 0.752 | 0.758 |
| 60% | LightGBM | 0.669 | 0.652 | **0.762** | 0.714 |

两个必须如实报告的结论：

1. **AUC 随观测增加单调上升**（0.64 → 0.70 → 0.75），这是「看得越多判断越准」的直接证据，也是项目的核心曲线。
2. **accuracy 打不过多数类基线**（0.686），只有 LightGBM 在 40% 处以 0.700 略微超过。因为基线只要永远猜「单峰」就能到 68.6%，而模型要在 29% 的少数类上做取舍。论文里必须同时报 macro-F1 和 AUC，只报 accuracy 会得出相反结论。


这一轮对比还抓到一个真实的 bug：原来的平台判据是「≥40% 的天数在高位」，在 4–5 天的短事件里，只要前两天高就会被判成平台。已改成「连续 ≥3 天在高位，且占比 ≥25%」，平台从 41 条降到 3 条。



## 收益模型（Model A–C）

CRSP 面板：1,322,791 行，624 只股票，2011-12-01 至 2022-12-30。**216 条退市收益已并入**日收益序列。720 / 745 个事件能算出完整的 20 天前瞻窗口。

结果（测试集 162 个事件，2018–2019 为间隔，AUC）：

| 结果变量 | A 价格历史 | B + 注意力 | C + 形状特征 |
|---|---|---|---|
| P(R20 > 0) | 0.580 | 0.591 | **0.619** |
| P(R20 > 10%) | 0.625 | 0.630 | **0.653** |
| P(MDD20 < −20%) | **0.920** | 0.915 | 0.902 |

**这里有一个必须警惕的陷阱**：回撤的 0.92 几乎全部来自波动率，不是注意力。

| 单变量 | MDD20 的 AUC |
|---|---|
| `vol60`（过去 60 日波动率） | **0.907** |
| `lsize`（市值） | 0.890 |
| `turnover` | 0.876 |
| `peak_z`（注意力峰值） | 0.606 |

`corr(mdd20, vol60) = −0.605`；按波动率五分组，MDD20 中位数从 −5.1% 单调走到 −43.5%。高波动股票本来就回撤深，所以「预测回撤」接近同义反复。论文里若直接报 0.92，评委会立刻问这一点。

可行的改法：把回撤按自身波动率标准化（`mdd20 / vol60`），或把 `vol60` 作为强制控制变量后，只看注意力的**增量**。

另一方面，注意力 + 形状对**收益方向**的增量是实的：0.580 → 0.619，`P(R20>10%)` 0.625 → 0.653。幅度不大，但方向一致。

LightGBM 在全部结果上都不如逻辑回归——训练集只有 404 个事件，树模型过拟合。这个要如实写。

## 行动窗口（actionable window）

形状模型在测试集上给出置信度 `p_true`，以 0.6 为「可以下判断」的门槛（覆盖率约 50%）。下表的收益单位是百分比，`r_before` 从事件起点算到判断日，`r_after` 从判断日往后 20 个交易日。

| 检查点 | 组 | n | 判断前涨跌 | 判断后 20 日 | 后段为负的比例 |
|---|---|---|---|---|---|
| 20% | 已判断 | 138 | +0.7% | **−8.5%** | 63.8% |
| 40% | 已判断 | 127 | +0.1% | **−9.8%** | 67.7% |
| 60% | 已判断 | 139 | +2.8% | **−9.1%** | 71.9% |
| 60% | 未判断 | 144 | +6.2% | −5.6% | 61.8% |

三个结论：

1. **到能判断的时候，价格几乎还没动**（60% 检查点中位 +2.8%）。所以「太晚」不是问题。
2. **但之后没有剩余上涨可以吃，反而是下跌**：判断后 20 天中位 −9.1%，且 72% 的事件为负。在 2020–2021 整体上涨的样本里，这个反向更值得注意。
3. **置信度越高，后续越差**：已判断组 −9.1%，未判断组 −5.6%。模型识别出的「确定性」对应的是**衰减信号**，不是买入信号。

这与 Barber, Huang, Odean & Schwarz (2022) 的注意力驱动买入→价格压力→反转一致。它意味着项目的正确叙事不是「早买」，而是「识别到讨论爆发进入后半程时，风险已经不对称」。

局限必须写清楚：只有 162 个测试事件、窗口重叠、未计交易成本、只做多头框架。

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
