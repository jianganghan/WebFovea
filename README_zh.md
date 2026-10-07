# WebFovea

[English](README.md) | 中文

**模型想对了，点击却点歪了。** WebFovea 是一个基于视觉的网页 Agent：看截图，在真实网站上点击、输入、滚动，最后交出一个可核验的答案。它在 **[WebRetriever Challenge 2026](https://mininglamp-ai.github.io/WebRetriever_Challenge/) 中获得第 2 名**，最终得分 **57.0 / 100**。

[![tests](https://github.com/jianganghan/WebFovea/actions/workflows/tests.yml/badge.svg)](https://github.com/jianganghan/WebFovea/actions/workflows/tests.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2610.03036-b31b1b.svg)](https://arxiv.org/abs/2610.03036)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<p align="center"><img src="docs/demo.gif" width="620" alt="WebFovea 在 Tableau 仪表板上答题：选中 Bolivia，再悬停表格单元格，直到 tooltip 显示 2024 年 Group Grievance 得分 6.7"></p>
<p align="center"><sub>一次 25 步真实网站任务中的关键步骤（即技术报告第 4 节的案例）。</sub></p>

## 核心观点

强的多模态大模型是必要条件，但不是充分条件。模型的决策要经过 *harness*（模型与网页之间的那层代码）才能落到浏览器上。每一步都是一次往返，要过四关：

| 环节 | 真实网站上出过的问题（来自我们的运行记录） | WebFovea 的做法 |
|---|---|---|
| **解析 Parsing** | 模型自造的 `<\|eot_id\|>` 之类聊天模板结束符被敲进搜索框（4.9% 的任务受影响）；模型一口气编出多步轨迹和假答案 | 只取第一个动作；剥掉特殊 token 和动作后的杂尾 |
| **执行 Execution** | 每次点击都落在目标坐标的 3/4 处；原生下拉框点不动；按键落到错误的焦点上；iframe 里的点击被吞掉 | 坐标空间对齐；`select`；`fill`；结尾换行即回车；iframe 内 DOM 点击兜底 |
| **反馈 Feedback** | iframe 里其实点成功了，却被报告成「页面无变化」，模型因此放弃了正确操作；输入失败在截图上看不出来 | DOM 与子 frame 指纹；告诉模型点中的是哪个元素；`fill` 回读；tooltip 文本哈希 |
| **观察 Observation** | 数值只在悬停时出现；目标在长页面很靠下的位置；数字被自定义字体遮住 | 截图尺寸上限；`hover`；`find_text`；`read_text`；规则化触发条件 |
| **护栏 Guardrails** | 绕过页面直接走 URL 或数据接口；无休止地重复同一动作；worker 崩溃或 API 失败导致丢题 | 封闭动作集；硬约束；执行层强制预算；滚动熔断；重试与共享队列调度 |

四次官方提交用的都是同一个模型（`claude-opus-4-6`），分数从 31.0 涨到 57.0，靠的是 harness 的改动（在真实网站运行间波动的范围内）：

| 提交（2026） | 隐藏集得分 | 主要新增 |
|---|---:|---|
| 8 月 29 日 | 31.0 | 坐标对齐、`select`、`hover`、`reload`；封闭动作集、硬约束 |
| 8 月 31 日 | 41.0 | `find_text`、`back`；共享队列调度 |
| 9 月 1 日 | 46.0 | 特殊 token 剥离；`fill`、换行回车；`read_text`；滚动熔断修复 |
| 9 月 2 日 | **57.0** | iframe 兜底；`find_text` / `read_text` 搜索子 frame；80 步；模型调用重试 |

完整设计、每个组件的证据（包括负面结果）和失败分析见[技术报告](https://arxiv.org/abs/2610.03036)。

## 快速上手

需要：Python 3.10+、Playwright 的 Chromium、Anthropic API key。在 macOS 上开发；CI 在 Linux 上跑测试。

```bash
git clone https://github.com/jianganghan/WebFovea.git && cd WebFovea
pip install -r requirements.txt    # 或：conda env create -f environment.yml
playwright install chromium

# key 只从环境变量读，不写进 config.json
export VLM_API_KEY=sk-ant-...
bash scripts/run_local.sh data/example_tasks.json runs/demo
```

每道题在 `runs/demo/` 下有一个目录：`result.json`（答案 `agent_answer`、动作、思考、访问过的 URL、状态）、`trajectory/*.png`（每一步模型看到的截图）、`capture.json`（本题的网络请求）。

并发：`bash scripts/run_local.sh tasks.json runs/x 4`（每路一个浏览器）。关掉浏览器：`bash scripts/run_local.sh --kill`。

**成本**：每一步都会重发最近 5 张截图，输入远多于输出，平均每步约 10.6k 输入 token、127 输出 token。用 `claude-opus-4-6` 跑 100 题约 $170–230。建议先跑几题试试。

### 在公开 Protocol III 题上评测

100 道公开 Protocol III 题在 Hugging Face 的 [WebRetriever 数据集](https://huggingface.co/datasets/Mininglamp-2718/WebRetriever)里（需申请访问，CC BY 4.0），自带参考答案：

```bash
bash scripts/run_local.sh path/to/protocol3.json runs/p3 4
# 加 --fuzzy：规则判不了的答案交给 LLM 判等
python eval/score.py --runs runs/p3 --gold path/to/protocol3.json
```

`eval/score.py` 用规则档位（精确、日期、数值）加可选的 LLM 判等来逼近官方口径。官方是 LLM 判等加人工复核，而且部分公开题的参考答案已经随网站变化而过期，所以**本地分数不能和排行榜直接比较**。WebFovea 就是在这批公开题上开发的；官方的 57.0 出自另一套隐藏题。

## 配置

`config.json` 里是模型端点和全部开关，与提交版本一致（只去掉了 API key）。每个开关都能用环境变量 `WEBFOVEA_<KEY>` 临时覆盖（如 `WEBFOVEA_MAX_STEPS=40`），优先级：环境变量 > `config.json["agent"]` > 代码默认值。每个参数为什么取这个值，写在 `src/agent/runtime_config.py` 里。`api_base` / `api_model` 可以换成任何 OpenAI 兼容端点，但 prompt、坐标处理和 token 剥离都是针对 `claude-opus-4-6` 调的。

## 目录结构

```
src/agent/
├── main.py             # 任务调度：worker、单题预算、结果落盘
├── task_dispatch.py    # 共享任务队列，含崩溃补偿
├── agent.py            # 模型调用、回复解析、坐标映射
├── prompts.py          # 动作空间与硬约束
├── web_controller.py   # 执行层：各动作、反馈、iframe 兜底
└── runtime_config.py   # 全部可调参数及取值依据
scripts/
├── run_local.sh        # 启动 Chromium 并运行 agent
├── run.sh              # 对已在运行的浏览器运行 agent
└── launcher.py         # 在任何机器上保持相同运行条件
eval/score.py           # Protocol III 本地打分器
tests/                  # 270+ 个离线测试，零 token
```

`src/agent/` 就是比赛中拿到 57.0 分的那个 agent。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试不需要浏览器、网络或模型，专门覆盖真实运行中**不报错、只是悄悄失效**的问题：prompt 里写了某个动作，解析层或执行层却没接上；监听器跨题泄漏；worker 崩溃悄悄丢题；配置写错导致行为改变。详见 [tests/README.md](tests/README.md)。

## 局限

- **只用了一个模型。** 所有结果都基于 `claude-opus-4-6`。四环节的视角与模型无关，但部分修复（坐标对齐、token 剥离）针对的是这个模型及其 API 的特有问题。
- **真实网站。** 网站会改版、拦截自动化，或在某些网络下无法访问；同一版本重复运行，结果也会有波动。
- **为评测而建。** Agent 做的是信息检索类问答。硬约束要求它留在给定网站内、不能按 URL 跳转；它不是为需要登录或会产生副作用的任务设计的。

## 后续方向

来自技术报告的后续工作，大致按「收益 / 投入」排序：

- **不同步骤交给不同模型。** 每一步需要的能力不同：定位控件、读密集表格、判断何时收尾。每一步可以交给大模型、小模型、规则或经典视觉算法，四个环节和护栏保持不变。
- **用经典视觉方法加强反馈与观察。** 用 OCR 交叉核对从截图读出的数值；用感知哈希检测 canvas 和图片上的变化（DOM 指纹看不到这类变化）；用模板匹配识别反复出现的控件。
- **独立的答案核验器。** 在 `finished` 之前，另起一次上下文干净的模型调用，只给它任务、最后一张截图和候选答案，专门检查「读错行、读错表」这类错误。
- **改用原生 API：prompt caching 加 tool-use 动作。** 不受比赛 API 限制时，预计可省 55–70% 的成本，并且不再需要解析自由文本，大部分解析失败也随之消失。
- **混合观察。** 在截图旁附上可交互元素及其坐标的列表。需要 A/B 验证，因为混合方式并不一定更好。
- **下载解析与任务类型技能。** 直接读取 PDF 和表格类数据源；按任务类别给出简短的检查清单，并积累各站点的操作要点。
- **合成任务与微调。** 在同一批网站上生成带程序化标准答案的任务，把教师轨迹蒸馏到更小的开源模型，以降低成本，并把 prompt 里的规则内化到模型中。

欢迎就其中任何一项提 issue 或 PR。

## 引用

见 [README.md](README.md#citation)。

## 致谢

WebFovea 的起点是明略科技（Mininglamp Technology）[WebRetriever](https://github.com/Mininglamp-AI/WebRetriever) 基准的参考实现（MIT 协议），其动作格式沿用 [UI-TARS](https://github.com/bytedance/UI-TARS)。感谢 WebRetriever Challenge 组委会提供基准与评测设施。

## 许可证

[MIT](LICENSE)。
