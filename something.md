# 毕业设计资料

[nuolade/disease-kb: 常见疾病相关信息构建knowledge graph](https://github.com/nuolade/disease-kb)

[honeyandme/RAGQnASystem: 本项目设计了一个基于 RAG 与大模型技术的医疗问答系统，利用 DiseaseKG 数据集与 Neo4j 构 建知识图谱，结合 BERT 的命名实体识别和 34b 大模型的意图识别，通过精确的知识检索和问答生成， 提升系统在医疗咨询中的性能，解决大模型在医疗领域应用的可靠性问题。](https://github.com/honeyandme/RAGQnASystem?tab=readme-ov-file)



# ideas？

## GraphFlow与Beyond the answer

> 对推理路径进行PSE离线打分，作为Reward。

​	可以对数据集进行PSE离线打分，作为Reward。在多跳问答或图检索中，模型不仅要找到可以支持回答的正确路径，这个路径还应该尽可能的合理。

1. GraphFlow 的目标是学习一个生成路径的策略，使采样出的路径分布与其质量成比例。但是reward 只使用二值或者粗略奖励信号无法区分“好的程度”
2. PSE 能告诉我们一条路径“推理得好不好”，而 GraphFlow 的 F(s) 则需要一个高质量的信号告诉它“该学哪些路径”。这两者**刚好形成互补**。



**检索时用普通图转化成超图来增强LLM？**

















