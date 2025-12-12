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




## 数据集构造

> 原始问题 (1-to-N) $\rightarrow$ 获取答案实体属性 $\rightarrow$ 构造过滤条件 $\rightarrow$ 生成新问题 $\rightarrow$ 验证
>

原始种子数据集：WikiWebQuestions，下载地址：(https://github.com/stanford-oval/wikidata-emnlp23)

格式如下：

```json
[  {
    "id": "WebQTrn-0",
    "utterance": "what is the name of justin bieber brother?",
    "sparql": "PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> SELECT DISTINCT ?x WHERE { wd:Q34086 wdt:P3373 ?x. ?x wdt:P21 wd:Q6581097. }"
  }
]
```

- 下载 WikiWebQuestions 数据集后，在其中的 train 数据集找到答案大于等于 5 的QA。

```json
  {
    "original_id": "WebQTrn-60",
    "question": "what countries do people speak portuguese?",
    "original_sparql": "PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> SELECT DISTINCT ?x WHERE { ?x wdt:P37 wd:Q5146; wdt:P31/wdt:P279* wd:Q6256. }",
    "answers": [
      "Q574",
      "Q45",
      "Q155",
      "Q916",
      "Q983",
      "Q1007",
      "Q1011",
      "Q1029",
      "Q1039",
      "Q19356759",
      "Q27304761"
    ],
    "answer_count": 11
  }
```

- 然后查询答案实体的属性，提取每条数据里的answer列表，批量向Wikidata查询这些实体的属性，存到answer字段后面。
  - 定义一些高质量属性，也就是程序去Wikidata中要查的属性。
  - 加载 seed_1_to_n_questions.json，循环处理每一个问题。
  - 拿到该问题的所有 QID，对每一个QID 扔进查询函数进行查询。
  - 把查询得到的属性保存到"answers_attributes"字段。
  - 最后保存为data_with_attributes.json文件。

```json
"answers_attributes": {
      "Q4880234": {
        "P31": [
          "written work"
        ],
        "P136": [
          "children's novel"
        ],
        "P407": [
          "English"
        ]
    }
}
```


- 接下来就是利用data_with_attributes.json构造过滤条件。**逆向工程**，在答案列表中确定一个答案，然后尝试通过添加约束条件，将答案限制在**一个**。

```json
[  {
    "original_question": "what books did beverly cleary right?",
    "source_id": "WebQTrn-18",
    "original_answer_count": 28,
    "constraint_description": "is a is 'literary work' AND released in after 1956 AND award received is 'Newbery Medal'",
    "constraint_logic": "(P31 is 'literary work') AND (P577 > 1956.75) AND (P166 is 'Newbery Medal')",
    "new_ground_truth": [
      "Q5246906"
    ],
    "new_answer_count": 1
  }]
```


- 构造prompt，让LLM根据生成的约束根据原问题进行重写。
- 验证。







## prompt

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





> 魔塔社区api：[概览 · 魔搭社区](https://modelscope.cn/my/overview)
>
> 硅基流动：[SiliconCloud](https://cloud.siliconflow.cn/me/account/ak)
>
> (https://github.com/chatanywhere/GPT_API_free)

```python
# free-gpt-key
sk-wZPm2CCFydnh7Nuh9vuaMBLYiJxBxP0MsIMwp6rGZ87JVzkF
https://api.chatanywhere.tech
# 硅基流动
sk-iedkedhtzkamboikwwoamudadmxmuwvrxwovbedjzvcycqda
https://api.siliconflow.cn/v1/
```



```python
INTERESTING_PROPS = [
    # --- 基础分类 ---
    "P31",    # Instance of (是...的实例: 人, 电影, 城市)
    "P279",   # Subclass of (是...的子类: 属于生物, 属于交通工具)

    # --- 人物相关 (People) ---
    "P569",   # Date of birth (出生日期) -> 适合: 年代约束 (e.g., born after 1990)
    "P570",   # Date of death (死亡日期)
    "P27",    # Country of citizenship (国籍) -> 适合: 地点约束
    "P106",   # Occupation (职业) -> 适合: 角色过滤 (e.g., acts as Director)
    "P166",   # Award received (获得奖项) -> 适合: 高级约束 (e.g., won Nobel Prize)
    "P69",    # Educated at (毕业院校) -> 适合: 校友约束
    "P21",    # Sex or gender (性别) -> 适合: 人口统计学约束 (慎用，但在数据集中常见)

    # --- 创意作品 (Creative Works: 电影, 书籍, 音乐) ---
    "P577",   # Publication date (发布/上映时间) -> 适合: 时间约束
    "P136",   # Genre (流派) -> 适合: 类型约束 (e.g., Horror movies)
    "P57",    # Director (导演) -> 适合: 关系约束
    "P161",   # Cast member (演员阵容) -> 适合: 合作关系约束
    "P175",   # Performer (表演者/歌手)
    "P2047",  # Duration (时长) -> 适合: 数值约束 (e.g., > 120 minutes)
    "P2142",  # Box office (票房) -> 适合: 商业数值约束 (e.g., > 1 Billion USD)
    "P2130",  # Cost/Budget (成本/预算)
    "P407",   # Language of work (语言)

    # --- 地理与行政 (Geography & Places) ---
    "P17",    # Country (所属国家)
    "P1082",  # Population (人口) -> 适合: 规模约束 (e.g., > 5 million people)
    "P2046",  # Area (面积) -> 适合: 大小约束
    "P2044",  # Elevation above sea level (海拔) -> 适合: 地形约束
    "P30",    # Continent (所属洲)
    "P1376",  # Capital of (是...的首都) -> 适合: 政治地位约束

    # --- 组织与体育 (Organizations & Sports) ---
    "P571",   # Inception (成立时间)
    "P159",   # Headquarters location (总部所在地)
    "P1128",  # Employees (员工数量) -> 适合: 企业规模约束
    "P118",   # League (所属联盟: NBA, 英超等)
    "P112",   # Founder (创始人)

    # --- 科学与物理 (Science & Objects) ---
    "P2048",  # Height (高度 - 建筑/人)
    "P2067",  # Mass (质量/重量)
    "P186",   # Material used (由...材料制成)
    "P61",    # Discoverer or inventor (发现者/发明者)
]
```





