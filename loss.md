# 论文标题：Can Knowledge-Graph-based Retrieval Augmented Generation Really Retrieve What You Need?

## Loss 设计目标

1. **奖励分解：**将最终奖励有效地分解到轨迹上的每一个中间状态上。
2. **策略对齐：**让模型生成路径的概率与奖励一致。、
3. **有限探索：**路径不是在大型 KG 上找，而是在局部领域内进行受限的探索。

## Loss整体结构

$$
\mathcal{L}_{DBLE}(s_{t}) = \sum_{i=0}^{k} \left[\log F(s_{t}) - \log F(s_{t+1,i}^{\prime}) + r_{\theta}(s_{t},a_{t,i}^{\prime}) - \log \sum_{j=0}^{k}e^{r_{\theta}(s_{t},a_{t,j}^{\prime})}\right]^{2}
$$

详细平衡损失+局部探索损失。$$log F(·)$$ 由 **Flow Head** 预测；$$ r_{\theta}(\cdot) $$由 **Policy Head** 预测（随后用于 Softmax 计算 $\log P(\cdot)$）

# 论文标题：Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning

## Loss 设计目标

RoG 的损失函数旨在解决以下关键问题：

1. **幻觉问题**：LLM 在推理过程中容易产生不准确的推理步骤

2. **过程监督缺失**：需要让 LLM 学会基于检索到的推理路径进行忠实推理
3. **LLM 缺乏 KG 知识**：LLM 无法直接生成基于 KG 的忠实关系路径

因此，RoG 设计了**双任务优化框架**，通过联合训练**规划模块**和**检索推理模块**来解决这些问题。

## Loss 整体结构

$$
\mathcal{L} = -\underbrace{\log P_{\theta}(a \mid q, Z_{K}^*, G)}_{\text{检索推理优化}} - \underbrace{\frac{1}{|Z^*|} \sum_{z \in Z^*} \log P_{\theta}(z \mid q)}_{\text{规划优化}}
$$

**解释：**ROG损失函数的推导始于一个概率模型设定，然后通过引入变分推断来优化其证据下界（ELBO），最终分解为两个可以实际优化的损失项：**规划损失**和**推理损失**。

1. 建立概率模型与最终目标，先基于问题生成一个关系路径，再基于这个计划和知识图谱生成最终答案。

$$
P_{\theta}(a|q,\mathcal{G})=\sum_{z\in\mathcal{Z}}P_{\theta}(a|q,z,\mathcal{G})P_{\theta}(z|q)
$$

2. 直接优化上式困难，因此论文采用了最大化证据下界（ELBO）来优化这个目标函数。ELBO将目标分解为两个部分（具体分解过程见RoG笔记）：

$$
log~P(a|q,\mathcal{G})\ge\mathbb{E}_{z\sim Q(z)}[log~P_{\theta}(a|q,z,\mathcal{G})]-D_{KL}(Q(z)||P_{\theta}(z|q))
$$

最大化这个ELBO目标就是**最大化**期望$\mathbb{E}_{z\sim Q(z)}[log~P_{\theta}(a|q,z,\mathcal{G})]$，**最小化**$D_{KL}(Q(z)||P_{\theta}(z|q))$

$$Q(z)$$是一个后验分布来近似未知的后验分布$$P(z|a,q,\mathcal{G})$$,在实际的规划优化中，由于计算所有有效路径很困难，该分布被进一步近似为**仅使用连接问题实体 $e_q$ 和答案实体 $e_a$ 之间的最短路径 $\mathcal{Z}^{*}$**:
$$
Q(z)\simeq Q(z|a,q,\mathcal{G})=\begin{cases}\frac{1}{|\mathcal{Z}|},\exists w_{z}(e_{q},e_{a})\in\mathcal{G},\\ 0,else,\end{cases}
$$

3. 将定义好的$$Q(z)$$带入ELBO的两项：

   1. 规划损失：$\mathcal{L}_{plan} = D_{KL}(Q(z|a, q, G)∥P_{\theta}(z|q)) $=$ \mathbb{E}_{z\sim Q(z|a,q,G)}[log Q(z|a, q, G) − log P_{\theta}(z|q)]=$$−\mathbb{E}_{z\sim Q(z|a,q,G)}log P_{\theta}(z|q) + CONST$

      $\mathcal{L}_{plan} \simeq-\frac{1}{|\mathcal{Z}^{*}|}\sum_{z\in\mathcal{Z}^{*}}log~P_{\theta}(z|q)$

   2. 推理损失:$\mathcal{L}_{reason} = \mathbb{E}_{z\sim Q(z|a,q,\mathcal{G})}[log~P_{\theta}(a|q,z,\mathcal{G})] \simeq log~P_{\theta}(a|q,\mathcal{Z}_{K}^{*},\mathcal{G})$

4. 优化目标：$\mathcal{L} = \log P_{\theta}(a|q, \mathcal{Z}_{K}^{*}, \mathcal{G}) + \frac{1}{|\mathcal{Z}^{*}|} \sum_{z \in \mathcal{Z}^{*}} \log P_{\theta}(z|q)$

# 论文标题：Knowledge Graph-Enhanced Large Language Models via Path Selection

## Loss设计目标

训练一个编码器$$M$$，使其能够衡量问题 q 与路径 p 之间的语义相关性； 识别出那些能帮助 LLM 生成正确答案的路径，即使这些路径与问题在表面上语义不直接相关，也能被正确识别。

## Loss整体结构

该损失函数是基于余弦相似度的成对排序损失
$$
\mathcal{L}=\sum_{q}max(cos(h_{q},h_{q}^{-})-cos(h_{q},h_{q}^{+})+\eta,0)
$$
其中，$$h_q=M(q)$$:输入问题q的编码向量；$h_{q}^{+}=M(p_{q}^{+})$:正样本知识路径$$p_{q}^{+}$$编码向量；$h_{q}^{-}=M(p_{q}^{-})$:负样本知识路径$$p_{q}^{-}$$编码向量；

正样本：当某个路径 p 加入 prompt 后，LLM 从错误输出变为正确输出；

负样本：加入路径 p 后，LLM 的输出仍然错误；

# 论文标题：
