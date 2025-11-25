# Towards Trustworthy Knowledge Graph Reasoning: An Uncertainty Aware Perspective

> 题目：迈向可信的知识图谱推理：一个不确定性感知的视角
>
> 会议：AAAI 2025

## 研究问题

如何为 KG-LLM 的问答系统引入严格的不确定性量化机制，以提升其在高风险场景下的可信度和可靠性。

## 主要内容

### 理论基础

#### Conformal Prediction

给定一个用户设定的错误率容忍度 $α$，共形预测可以产生一个预测集$C(X_{test})$，并保证这个集合包含真实答案 $y$ 的概率至少是 $1-\alpha$。这个概率被称为**覆盖率**。

1. **定义非共形分数**： $s=S(x,y)$，这是一个衡量预测“不好”程度的函数。分数越高，表示预测 $y$ 与输入 $x$ 越不匹配，即不确定性越高。在本文中，这个函数通常是文本相似度的倒数或负值。分数越高，表示 $x_i$ 和 $y_i$ 的一致性越差。

2. **在校准集上计算分位数**：定义校准集是$\mathcal{D}^{cal}=\{(x_{i},y_{i})\}_{i=1}^{n}$，然后对校准集上的每一个样本$ (x_i,y_i)$，计算其非共形分数$s_{i}=S(x_{i},y_{i})$，得到的所有分数集合记为$S^{cal}$，然后根据用户定义的错误率$\alpha$，找到对应的分位数$q_{\alpha}^{S,\mathcal{D}_{cal}}$：
   $$
   q_{\alpha}^{S,\mathcal{D}_{cal}}=Quant(\{S(x,y)|(x,y)\in\mathcal{D}_{cal}\} , \frac{\lceil(n+1)(1-\alpha)\rceil}{n})
   $$
   $\frac{\lceil(n+1)(1-\alpha)\rceil}{n}$是一个概率，目的是理论上保证最终预测集的覆盖率至少是$1-\alpha$。

3. **构建预测集**：对于一个待测样本 $X_{test}$ 和所有可能的候选答案 $y$，我们计算它们的不确定性分数 $S(X_{test}, y)$。只有当候选答案 $y$ 的不确定性分数 $S(X_{test}, y)$ **小于或等于**这个阈值 $q_{\alpha}$ 时，该候选 $y$ 才会被纳入最终的预测集 $C(X_{test})$。
   $$
   C(X_{test})=\{y|y\in\mathcal{Y},S(X_{test},y)\le q_{\alpha}^{\mathcal{S},\mathcal{D}_{cal}}\}
   $$

#### Learn Then Test



### 方法

## 实验