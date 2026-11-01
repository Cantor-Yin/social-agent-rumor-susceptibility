# 大模型驱动的社交智能体谣言易感性与干预策略研究

本仓库对应论文《大模型驱动的社交智能体谣言易感性与干预策略研究》。

论文 PDF：<https://www.cjist.com.cn/previewFile?id=102122500&type=pdf&lang=zh>


## 目录

| 路径 | 内容 |
|---|---|
| `code/` | 智能体与三种实验条件的模拟代码 |
| `data/` | 正文使用的 8 条事实冲突谣言 |
| `results/` | 各实验条件下的模拟记录 |
| `analysis/` | 汇总表、绘图数据和笔记本 |

## 运行

您需要在 `code/agent.py` 里把对应模型的 `port` 和 `VLLM_API_KEY` 改成您所用LLM服务的端口和 `--api-key`。

您可以在仓库根目录运行以下命令，以运行主方针代码

```bash
python code/main.py --condition no_intervention
python code/main.py --condition prompt_baseline
python code/main.py --condition self_prompt
```

## 引用

如果本工作对您有所帮助，请您引用本工作：

```
殷勇杰, 袁靖炜, 宫庆媛, 等. 大模型驱动的社交智能体谣言易感性与干预策略研究[J]. 智能科学与技术学报, 2025, 07(04): 517-532. DOI: 10.11959/j.issn.2096-6652.202545.

YIN Yongjie, YUAN Jingwei, GONG Qingyuan, et al. Rumor susceptibility and intervention strategies of large language model-driven social agents[J]. Chinese Journal of Intelligent Science and Technology, 2025, 07(04): 517-532. DOI: 10.11959/j.issn.2096-6652.202545.
```
