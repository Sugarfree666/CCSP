# 复杂约束性问题

> 1. 我想购买一台苹果品牌的笔记本电脑，要求内存至少为 16GB，且重量低于1.5公斤，价格不高于8000元。
> 2. 我想找一种高血压药物，要求：1. 不含‘磺胺’成分（因为我过敏）；2. 不会与我正在服用的‘阿司匹林’产生拮抗作用；3. 属于医保乙类。
> 3. 汤姆·汉克斯演过哪些战争题材且上映时间在1995年之后的高评分电影？

## 思路

根据 GoT 框架解决复杂约束问题（G,T,E,R）。

T：思维变换：聚合思维，精炼思维，生成思维。

将这个问题的推理结构建模为一个有向图。节点代表thought，边代表thought之间的依赖关系。

具体问题：

输入问题 $Q$：我想购买一台苹果品牌笔记本电脑，内存至少为 16GB，必须搭载M2芯片，价格不高于8000元。

1. 初始化阶段：将查询 $Q$ 初始化为根节点 $V_{root}$ 。

2. 通过prompt将根节点分解为4个子约束任务，使用**生成思维**生成四个思考结点

   1. $V_1$ 查找苹果品牌笔记本电脑（Anchor）
   2. $V_2$ 查找内存至少为16GB的笔记本电脑（Filter）
   3. $V_3$ 查找搭载M2芯片的笔记本电脑（Anchor）
   4. $V_4$ 查找价格不高于8000元的笔记本电脑（Filter）

   > - 为了提高子思维的精确度，在生成思维之前，要告诉LLM中KG有哪些属性，避免它生成一些无关的思维。
   > - 在生成思维前，参考KG返回的约束实体的度的信息：
   >   - 如果Degree特别大，作为Filter节点。其他的可以直接作为Anchor思维节点
   >   - Filter节点在查询时不做查询操作，Anchor节点作查询操作

   将每个子查询使用KG查询（KG查询方法），得到查询集合 $A_i$；

   > 

   对查询集合评分&排名：如果Score太低，进行重新思考或修正思考，也就是**精炼思维**。

3. 使用**聚合思维**将思考$V_{1},V_{3}$和$V_{2},V_{4}$分别聚合。得到的聚合结果利用**生成思维** 得到$V_{12}$和 $V_{34}$ 作为下一思考结点。

   1. $V_{13}$：查找既是苹果品牌，又是M2芯片的笔记本电脑。
   2. $V_{24}$：查找内存至少为16GB，但价格不高于8000元的电脑。

   > 对于思考节点中的实体，根据社区检测算法找到在KG中联系紧密的实体进行聚合，不再盲目的聚合。
   >
   > 在这个例子中，使 $V_1$和 $V_3$ 聚合，$V_2$ 和 $V_4$ 聚合；如果是两个Anchor节点进行聚合，则要把思维聚合后在两个查询的结果集合中，找到满足思维的答案 $A$。如果是两个Filter思考节点聚合，则进行多规则思考节点的生成。

   - 如果查询集合 $A$ 为空，回溯思考，使用**精炼思维**基于原思考节点生成一个新的思考节点。
     - 比如说 $A$ 为空，$V_{13}$ 节点进行回溯，对 $V_3$ 点进行适当的放宽条件。
     - 然后用这个节点与 $V_1$ 节点重新聚合。

4. 使用**聚合思维**对$V_{13}$和 $V_{24}$ 进行聚合，利用**生成思维**得到 $V_{final}$思考结点，包含着满足各个约束的笔记本电脑信息。前面 $A$ 集合也就是 $A_{final}$，$V_{final}$根据这个 $A_{final}$ 得到最后的**答案集合**。

   将得到的答案集合和问题 $Q$ 输入给LLM生成答案。




# 数据集构造

原始问题 (1-to-N) $\rightarrow$ 获取答案集属性 $\rightarrow$ 构造过滤条件 $\rightarrow$ 生成新问题 $\rightarrow$ 验证

原始种子数据集：WikiWebQuestions 

- 先在 WikiWebQuestions 数据集中找到答案大于等于5的QA。
- 然后查询答案实体的属性，为构造过滤条件作准备。











魔塔社区api：[概览 · 魔搭社区](https://modelscope.cn/my/overview)

硅基流动：[SiliconCloud](https://cloud.siliconflow.cn/me/account/ak)

[【抽奖】抽一个月的节点 - 福利羊毛 - LINUX DO](https://linux.do/t/topic/1287921/35)







# prompt

You are a search-query optimization expert. Your task: take an **Original Question** plus a set of **Constraints** and rewrite them as a single, natural, fluent English complex question that a real person might ask.

Rules:

1. **Single output** — produce exactly one question in English (one sentence) and **nothing else**.
2. **Preserve logic exactly.** Translate symbolic operators into clear English:
   -  >→ “more than”, >= → “at least”
   - < → “less than”, <= → “at most”
   - = → “exactly” (or “that are” when natural)
   - AND → “and”; OR → “or”; NOT → “but not” / “excluding”
      Maintain the order and grouping implied by the constraints.
3. **Map property names to natural phrases:**
   - starring → “starring [Actor]” or “featuring [Actor]”
   - released in after YYYY→ “released after YYYY” (or “released after [date]” if full date)
   - duration / P2047→ “runtime” or “runtime of more/less than X minutes”
   - genre → “which [genre] films” (e.g., “which drama films”)
   - directed by → “directed by [Director]”
   - box office → “box office” or “box office gross” and convert large numbers to readable form (“$711.9 million”).
4. **Normalize numbers and currencies** to readable English (e.g., \$711909412.25 → “\$711.9 million” or “about $711.9 million” depending on exactness).
5. **Dates and relative terms**: Use absolute wording (“released after 2009”, not “after 2009.0”).
6. **Fix typos and grammar** from the original question while keeping intent unchanged.
7. **Tone & phrasing**: Sound like a natural user query: start with “Which”, “List”, or “Can you list” as appropriate; prefer concise phrasing.
8. **Do not add** additional filters, assumptions, or explanations beyond the provided constraints.
9. If constraints imply a class (e.g., genre), prefer “Which [genre] films…”; if constraints specify a person (starring/director), put that close to the subject for readability.
10. The answer to the original question has only one correct option after being constrained and restricted.

Examples:

- Input Original: what movies does taylor lautner play in?
   Constraints: starring is 'Taylor Lautner' AND released in after 2009 AND duration is more than 94 min AND box office is less than \$711.9M AND directed by is 'Garry Marshall'
   Output:Which movie starring Taylor Lautner and directed by Garry Marshall was released after 2009, has a runtime longer than 94 minutes, and a box office gross of less than $711.9 million?
- Input Original: what are the 5 biggest cities in the usa?
   Constraints: founded in before 1792 AND population is more than 149.7K AND elevation is more than 18 m AND area is less than 1.2K sq km
   Output: Which US city founded before 1792 has a population of more than 149.7 thousand, an elevation above 18 meters, and an area of less than 1.2 thousand square kilometers?









```
[{
  "original_question": "what did shawnee smith play in?",
  "source_id": "WebQTrn-115",
  "original_answer_count": 32,
  "constraint_description": "is a is 'film' AND released in after 1995 AND duration is less than 109.50 min AND genre is 'horror film' AND directed by is 'Kevin Greutert' AND starring is 'Chester Bennington'",
  "constraint_logic": "(P31 is 'film') AND (P577 > 1995.0) AND (P2047 < 109.5) AND (P136 is 'horror film') AND (P57 is 'Kevin Greutert') AND (P161 is 'Chester Bennington')",
  "new_ground_truth": [
    "Q676284"
  ],
  "new_answer_count": 1
},
  {
    "original_question": "what movie did angelina jolie direct?",
    "source_id": "WebQTrn-124",
    "original_answer_count": 6,
    "constraint_description": "released in before 2019 AND duration is more than 85.75 min AND genre is 'drama film' AND budget is less than $65.0M",
    "constraint_logic": "(P577 < 2019.0) AND (P2047 > 85.75) AND (P136 is 'drama film') AND (P2130 is '65000000')",
    "new_ground_truth": [
      "Q15146380"
    ],
    "new_answer_count": 1
  },
  {
    "original_question": "who was elected president of the philippines?",
    "source_id": "WebQTrn-157",
    "original_answer_count": 45,
    "constraint_description": "is a is 'human' AND gender is 'male' AND born in before 1907 AND died in before 1980 AND occupation is 'politician' AND educated at is 'Harvard Law School'",
    "constraint_logic": "(P31 is 'human') AND (P21 is 'male') AND (P569 < 1907.75) AND (P570 < 1980.0) AND (P106 is 'politician') AND (P69 is 'Harvard Law School')",
    "new_ground_truth": [
      "Q656969"
    ],
    "new_answer_count": 1
  },
    {
    "original_question": "what time zones are there in the us?",
    "source_id": "WebQTrn-193",
    "original_answer_count": 16,
    "constraint_description": "country is 'United States' AND is a is 'time zone'",
    "constraint_logic": "(P17 is 'United States') AND (P31 is 'time zone')",
    "new_ground_truth": [
      "Q3446496"
    ],
    "new_answer_count": 1
  },
    {
    "original_question": "what are the 5 biggest cities in the usa?",
    "source_id": "WebQTrn-194",
    "original_answer_count": 5,
    "constraint_description": "founded in after 1672 AND elevation is less than 130 m AND population is more than 149.7K AND continent is 'North America'",
    "constraint_logic": "(P571 > 1672.5) AND (P2044 < 130.0) AND (P1082 > 149749.5) AND (P30 is 'North America')",
    "new_ground_truth": [
      "Q65"
    ],
    "new_answer_count": 1
  },
  {
    "original_question": "what countries did queen victoria rule?",
    "source_id": "WebQTrn-197",
    "original_answer_count": 5,
    "constraint_description": "founded in before 1924 AND continent is 'Europe' AND population is more than 11.0M",
    "constraint_logic": "(P571 < 1924.5) AND (P30 is 'Europe') AND (P1082 > 11023911.5)",
    "new_ground_truth": [
      "Q174193"
    ],
    "new_answer_count": 1
  },  {
    "original_question": "who won golden boot?",
    "source_id": "WebQTrn-206",
    "original_answer_count": 5,
    "constraint_description": "league is 'Premier League' AND height is less than 1.86 m AND born in before 1987 AND citizenship is 'United Kingdom'",
    "constraint_logic": "(P118 is 'Premier League') AND (P2048 < 1.865) AND (P569 < 1987.0) AND (P27 is 'United Kingdom')",
    "new_ground_truth": [
      "Q234576"
    ],
    "new_answer_count": 1
  },
   {
    "original_question": "where is the thames river located?",
    "source_id": "WebQTrn-208",
    "original_answer_count": 9,
    "constraint_description": "is a is 'ceremonial county of England' AND population is more than 707.2K AND area is less than 1.7K sq km",
    "constraint_logic": "(P31 is 'ceremonial county of England') AND (P1082 > 707237.5) AND (P2046 is '1662.5177')",
    "new_ground_truth": [
      "Q23276"
    ],
    "new_answer_count": 1
  }
]
```
