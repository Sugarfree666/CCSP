# 毕业设计资料

[nuolade/disease-kb: 常见疾病相关信息构建knowledge graph](https://github.com/nuolade/disease-kb)

[honeyandme/RAGQnASystem: 本项目设计了一个基于 RAG 与大模型技术的医疗问答系统，利用 DiseaseKG 数据集与 Neo4j 构 建知识图谱，结合 BERT 的命名实体识别和 34b 大模型的意图识别，通过精确的知识检索和问答生成， 提升系统在医疗咨询中的性能，解决大模型在医疗领域应用的可靠性问题。](https://github.com/honeyandme/RAGQnASystem?tab=readme-ov-file)





# 这种方式，数据集会保存到"/本地路径"中
huggingface-cli download 数据集名称 --repo-type dataset --token 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   --local-dir 本地路径

huggingface_hub download rmanluo/RoG-webqsp --repo-type dataset --token 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 --local-dir D:\PyCode\study_project\study_pytorch\study_demo1\webqsp_dataset



# ideas？

## GraphFlow与Beyond the answer

> 对推理路径进行PSE离线打分，作为Reward。

​	可以对数据集进行PSE离线打分，作为Reward。在多跳问答或图检索中，模型不仅要找到可以支持回答的正确路径，这个路径还应该尽可能的合理。

1. GraphFlow 的目标是学习一个生成路径的策略，使采样出的路径分布与其质量成比例。但是reward只使用二值或者粗略奖励信号无法区分“好的程度”
2. PSE 能告诉我们一条路径“推理得好不好”，而 GraphFlow 的 F(s) 则需要一个高质量的信号告诉它“该学哪些路径”。这两者**刚好形成互补**。



> 论文：Knowledge Graph-extended Retrieval Augmented Generation for Question Answering
>
> KG的引入可能提高多跳但降低简单检索的性能？



## 从静态问答到反事实推理

​	现有RAG系统擅长回答**静态事实类问题**：比如X是什么？Y在哪一年发生的？在真实场景中还有一些问题是：

> 如果某件事情发生or不发生，会导致什么？

​	这类问题常常不存在知识库中，因此传统RAG只能检索到相关信息，无法模拟具体推演过程。能不能让RAG不只是取相关资料，还要动态模拟一个被扰动的知识库。



## 解决知识冲突

> 问题： 用户问“A公司的营收是多少？”
>
> 知识库：财报显示营收100亿，新闻报道说80亿，分析师说是90亿。

普通 RAG 会混淆或随机选一个。不是给出一个数字，而是生成一段解释：“关于A公司营收存在争议，财报口径为100亿，但如果扣除XX业务，新闻报道认为是80亿。”

论文：

## 否定性问题判定

知识库中没有提到的东西，LLM非常容易产生幻觉去填补空白。

> 解决的目标问题是：区分“由于缺失数据导致的不知道”和“逻辑上推导出的不存在”

论文：

1. **Towards Trustworthy Knowledge Graph Reasoning: AnUncertainty Aware Perspective**





## 复杂约束的问题（Complex Constraint Satisfaction）



> 我想找一种高血压药物，要求：1. 不含‘磺胺’成分（因为我过敏）；2. 不会与我正在服用的‘阿司匹林’产生拮抗作用；3. 属于医保乙类。请推荐并解释。”
>
> 

论文：

1. KG-Agent: An Efficient Autonomous Agent Framework for Complex Reasoning over Knowledge Graph
2. **Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models**
3. Generating SPARQL from Natural Language Using Chain-of-Thoughts Prompting





**写一个更完整的流程；数据集用多答案数据集改造**



在KG增强LLM领域，通过提出不同的检索方法来增强RAG性能已经很难超越SOTA方法的性能。所以我想能不能在处理的问题上做文章，比如说现有研究都是为了解决多跳问题，多实体问题，其实还有一种问题是复杂约束的问题，如果可以在这种问题上做文章，提高对这种问题的RAG性能也是一个不错的想法。现在我的想法是利用GoT的思想，把思维建模为图结构来解决复杂约束问题。我发你的文件中是一些初步思路，现在需要细化步骤，也就是具体每一步应该怎么做。











这是一个极具学术价值且逻辑非常自洽的切入点。将 **KG 的结构特征（Topology/Structure）** 直接映射到 **GoT 的推理图构建（Reasoning Graph Construction）** 上，能够完美体现“KG 增强 LLM”的核心奥义——不仅仅是把 KG 当作数据库，而是把 KG 的**图论属性**作为 LLM 推理的**导航仪**。

基于 GoT 论文的核心机制，我为你设计了一个名为 **SA-GoT (Structure-Aware Graph of Thoughts)** 的框架。



### 核心理念：同构推理 (Isomorphic Reasoning)



GoT 论文指出，GoT 的优势在于能模拟任意的图结构 1111。

SA-GoT 的核心假设：最优的思维图结构，应该与知识图谱中相关子图的数据拓扑结构保持“同构”或“形态一致”。

- **KG 数据稠密（Hub节点）** $\rightarrow$ **GoT 思维发散（宽度优先，High $k$）**。
- **KG 数据稀疏（长尾节点）** $\rightarrow$ **GoT 思维深挖（深度优先，Chain）**。
- **KG 社区隔离（Community）** $\rightarrow$ **GoT 桥接思维（Bridge Thought）**。

------



### 具体实现细节：三个关键映射机制





#### 1. 基于“节点度”的自适应生成 (Degree-Adaptive Generation)



这一步决定了 GoT 的 `Generate` 操作如何根据 KG 结构调整“宽度”。

- **问题场景**：

  - 用户输入约束 $C_1$: “国产品牌”。在 KG 中，“China”或“Domestic_Brand”连接了成千上万个实体。这是一个 **Hub Node（超级节点）**。
  - 用户输入约束 $C_2$: “RTX 4090”。在 KG 中，这只连接了少数高端机型。这是一个 **Tail Node（长尾节点）**。

- **GoT 构建策略**：

  - **探测 (Probe)**：在生成思维前，Controller 先对约束对应的 KG 实体进行轻量级探测，获取其 **度（Degree）**。

  - **动态分支因子 ($k$)**：

    - **如果 Degree > 阈值 (Hub)**：强制 GoT 执行 **爆炸式生成 (Explosive Generation)**。设置 $k=High$ (如 5-10)。LLM 必须将这个大概念拆解为子类（如：联想系、华为系、小米系...）。

      - *原理*：GoT 论文提到分解任务能减小输入规模 2。对于 Hub 节点，必须拆解才能避免后续检索超时或精度丢失。

    - **如果 Degree < 阈值 (Tail)**：设置 $k=1$。LLM 直接生成单一的验证思维，不做拆解。

- **结果**：推理图在“国产品牌”处非常宽（Width大），在“RTX 4090”处非常窄。**推理图的形状完美拟合了数据分布。**



#### 2. 基于“社区结构”的聚合路径规划 (Community-Guided Aggregation)



这一步决定了 GoT 的 `Aggregate` 操作如何选择合并顺序。

- **问题场景**：
  - 你有三个思维节点：$V_{brand}$ (联想), $V_{usage}$ (商务), $V_{game}$ (游戏)。
  - 在 KG 中，“联想”和“商务”可能属于同一个 **社区（Community，如 ThinkPad 圈子）**，连接非常紧密；而“商务”和“游戏”在 KG 中拓扑距离很远（通常互斥）。
- **GoT 构建策略**：
  - **结构亲和度 (Structural Affinity)**：Controller 计算待聚合节点在 KG 中的 **拓扑距离** 或 **Jaccard 系数**。
  - **优先聚合 (Priority Aggregation)**：
    - 优先聚合 **同社区（Intra-community）** 的思维节点。因为它们的交集往往非空且 Volume 较大，能保留更多信息。
    - *Action*: 先聚合 $V_{brand} + V_{usage}$ $\rightarrow$ 得到 $V_{ThinkPad}$。
  - **推迟聚合 (Delayed Aggregation)**：
    - 推迟聚合 **跨社区（Inter-community）** 的节点（如商务+游戏），直到各自的子图已经足够具体。
- **结果**：GoT 不再盲目地两两聚合，而是沿着 KG 的“纹理”进行合并，效率最高。



#### 3. 基于“连通性”的桥接精炼 (Connectivity-Based Refinement)



这一步利用 GoT 的 `Refine` 解决冲突，但引入了图论中的路径寻找。

- **问题场景**：
  - $V_{A}$ (轻薄) 和 $V_{B}$ (高性能) 聚合后结果为空（KG 中这两个概念没有直接交集实体）。
- **GoT 构建策略**：
  - **多跳寻径 (Multi-hop Pathfinding)**：Controller 不仅仅让 LLM 瞎猜“放宽条件”，而是去 KG 中寻找连接 $V_{A}$ 和 $V_{B}$ 的 **最短路径** 或 **桥接节点**。
  - **KG 发现**：发现 $V_{A}$ 和 $V_{B}$ 虽然没有直接交集，但它们都连接到一个中间节点 $V_{bridge}$: "全能本 (All-rounder)" 或 "Creator Laptop (创作本)"。
  - **生成桥接思维 (Bridge Thought Generation)**：
    - GoT 自动插入一个新的思维节点 $V_{new}$ ("查找全能本/创作本")。
    - 修改图结构：$V_{A} \rightarrow V_{new} \leftarrow V_{B}$。
- **结果**：GoT 的图结构动态演化，自动“生长”出了一个桥梁节点来连接断裂的逻辑。

------



### 具体案例演示：SA-GoT 解决“国产+游戏+轻薄+<8000”



让我们看看引入结构感知后，GoT 图是如何构建的：

Code snippet

```
graph TD
    UserQuery[Query] --> Probe{KG Structure Probe}
    
    %% 1. 基于节点度的自适应生成
    Probe --"Brand=Hub (Degree High)"--> Gen_Brand[Generate k=3: <br>1. Lenovo Series<br>2. Xiaomi/Redmi<br>3. Honor/Huawei]
    Probe --"Gaming=Fuzzy"--> Gen_Game[Generate k=1: <br>Gaming Constraint]
    Probe --"Light=Constraint"--> Gen_Light[Generate k=1: <br>Weight < 1.5kg]
    
    %% 2. 基于社区的聚合
    subgraph "Aggregation Phase"
        %% KG显示 Xiaomi与Gaming连接更紧密 (社区A)
        Gen_Game & Gen_Brand --"High Connectivity"--> Agg_Xiaomi_Game[Agg: Redmi G Pro]
        
        %% KG显示 Lenovo与Gaming连接紧密 (社区B)
        Gen_Game & Gen_Brand --"High Connectivity"--> Agg_Lenovo_Game[Agg: Legion/GeekPro]
        
        %% KG显示 Huawei与Gaming连接稀疏 (社区隔离)
        Gen_Game & Gen_Brand --"Low Connectivity"--> Prune_Huawei[Prune/Discard]
    end
    
    %% 3. 冲突与桥接精炼
    Agg_Lenovo_Game --> Check1{Check Weight < 1.5kg}
    Check1 --"Fail (All > 2kg)"--> Refine_Lenovo
    
    subgraph "Structural Refinement"
        Refine_Lenovo --"KG Pathfinding"--> Bridge[Found Bridge Node in KG: <br> 'Slim' Series / 'Air' Series]
        Bridge --> New_Constraint[Refine: Look for 'Legion Slim' or 'YOGA Pro']
    end
    
    New_Constraint --> FinalResult
```



### 为什么这具有高创新性 (High Novelty)？



1. **超越了单纯的 RAG**：你不是在检索，你是在**对齐 (Aligning)**。你对齐了“人类思维的推理过程（GoT）”和“世界知识的存储结构（KG）”。

2. **理论支撑扎实**：

   - GoT 论文提供了“任意图变换”的操作基础 3333。

     

     

   - KG 图论提供了“度、社区、连通性”的计算基础。

   - 你的工作是这两者的**数学级联 (Mathematical Cascade)**。

3. **可解释性极强**：

   - 为什么 GoT 在这步分了3个叉？因为 KG 里这是一个 Hub 节点。
   - 为什么 GoT 先合并这两个条件？因为 KG 里它们属于同一个社区。
   - 这种解释性是目前“黑盒 LLM”极其缺乏的。









