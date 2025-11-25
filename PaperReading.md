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

在UAG这样的多步骤系统中，每一步（如检索、评估）都有自己的错误率 $(α1,α2,...)$。多步骤组合后，误差会积累，导致整体不再满足用户的风险要求。LTT提供了一种数据驱动的方法，来为每个组件寻找一组**有效的错误率配置** $λ=(α1,α2,α3)$，使得整个系统的**总体错误率**不超过 $α$。
$$
\mathbb{P}(sup_{\lambda\in\Lambda_{valid}}\mathbb{E}[L_{\lambda}|\mathcal{D}_{cal}]\le\alpha)\ge1-\delta
$$
在至少 $1-\delta$ 的置信度（概率）下，我们保证所选出的有效配置集合 $\Lambda_{valid}$ 中的所有配置 $\lambda$，它们各自产生的期望损失（即错误率 $\mathbb{E}[L_{\lambda}]$）的最大值，都不会超过用户最初设定的目标错误率 $\alpha$。

### 方法

UAG框架主要包含三个主要组件：**UQ-aware Candidate Retriever**（不确定性感知候选检索器）、**UQ-aware Candidate Evaluator**（不确定性感知候选评估器）和 **Global Error Rate Controller**（全局错误率控制器）

> 核心思想：用 conformal prediction 控制每一步的概率风险，再用 LTT 框架解决多步骤误差传播问题。

最终目标是构建一个预测集合 $C(X_{test})$，满足：
$$
\mathbb{P}(\mathbb{P}(\hat{e}\in\mathcal{Y}_{test},\forall\hat{e}\in C_{\lambda}(X_{test})|\mathcal{D}_{cal})\ge1-\alpha)\ge1-\delta
$$

#### 不确定性感知候选检索器

之前的大多数方法在KG上进行多跳图遍历和路径搜索时，为了找到潜在答案，通常采用 Top-K 候选选择这种基于启发式的方法。**缺乏理论基础**。

这个组件负责在KG上进行遍历，寻找可能的答案实体，它通过两个共形预测步骤来实现。

- **检索候选路径**：在遍历知识图谱的时候，决定下一步该走那条边。路径扩展规则为：
  $$
  \{s|s\in\mathcal{N}(v),S_{1}(Q||(||_{i=0}^{j-1}r_{i}),r_{j})<q_{\alpha_{1}}^{S_{1},\mathcal{D}_{cal}}\}
  $$
  其中 $Q$ 是问题，||表示拼接操作，$S_1$ 是文本相似度评分函数，$q_{\alpha_{1}}$ 是基于错误率 $\alpha_1$ 计算出的分位数阈值。$v$ 是当前节点, $s$ 是节点 $v$ 的一个邻居节点。

  **解释**：对于当前节点 $v$，考察它的每一个邻居 $s$。将问题 $Q$ 与路径上的所有关系拼接起来，与候选关系 $r_j$ 计算非共形分数。只要得分低于合格阈值 $q$，就会选择 $s$ 为下一跳扩展节点。

- **检索候选邻居**：判断当前访问的节点本身是否应该被加入到候选答案集合中。
  $$
  \{s|s\in\mathcal{N}(v),S_{1}(Q,||_{i=0}^{j}r_{i})<q_{\alpha_{2}}^{S_{1},\mathcal{D}_{cal}}\}
  $$
  对于当前节点 $v$，计算问题 $Q$ 与从起点到 $v$ 的完整路径的拼接文本的相似度。如果相似度分数高于阈值 $q$，就把这个节点加入候选集。

#### 不确定性感知候选评估器

现有框架在评估阶段，通常依赖LLM对检索到的信息进行直接推理，这样做会产生不可靠的输出缺乏理论保证。另一方面检索到的候选集可能过大，可能会包含很多不相关实体。

给定检索到的候选集 $\mathcal{C}$ 和推理路径 $\mathcal{P}$，以及 LLM 生成函数 $\Phi$。最终答案集定义为与 LLM 生成内容相似度满足阈值的候选：
$$
\{a\in\mathcal{C}|S_{1}(a,\Phi(\mathcal{P}))<q_{\alpha_{3}}^{S_{1},\mathcal{D}_{cal}}\}
$$


#### 全局错误率控制器





## 实验







# Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with  Large Language Models

> 标题：图约束推理：基于大语言模型的知识图上的可信推理
>
> 会议：ICML 2025

## 研究问题



## 主要内容

## 实验