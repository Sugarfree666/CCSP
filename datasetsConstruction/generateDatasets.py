import json
import time
from openai import OpenAI  # 确保安装了 openai 库: pip install openai

# 配置你的 API (这里以 DeepSeek 或 OpenAI 为例，通用格式)
client = OpenAI(
    api_key="sk-olxsNswN3LrJjFKOduxcUsn3LGiGUXPTz7X0v7uu5udabPrz",

    base_url="https://api.chatanywhere.tech"
)


def generate_natural_question(original_q, constraint_desc):
    # ---------------------------------------------------------
    # 1. System Prompt: 定义规则、逻辑转换方式和Few-Shot示例
    # ---------------------------------------------------------
    system_instruction ="""# Role
You are a Search Query Optimization Expert. Your goal is to synthesize natural, human-like complex questions from a "Base Question" and a set of "Logical Constraints".

# Task
Rewrite the provided **Original Question** and **Constraint Description** into a **single, fluent English sentence**.

# Input Data Explanation
- **Original Question**: A simple query (may contain typos or plural forms like "what are the 5 cities").
- **Constraint Description**: A structured text string containing logic (e.g., "AND", "OR"), comparison operators, and raw property values.

# Critical Rules

1.  **Enforce Singularity**: 
    - The constraints restrict the result to **exactly one correct answer**. 
    - Even if the Original Question asks for plural items (e.g., "what movies", "top 5 cities"), you MUST change the phrasing to singular (e.g., "Which movie...", "Which city...").

2.  **Natural Language Translation**:
    - **Symbol mapping**:
      - `>` → "more than", "over", "longer than"
      - `<` → "less than", "under", "shorter than"
      - `>=` → "at least"
      - `<=` → "at most"
    - **Logic mapping**:
      - `released in after YYYY` → "released after YYYY" (Remove the redundant "in")
      - `AND` → Combine naturally with commas or "and".
    - **Property mapping**:
      - `duration` / `P2047` → "runtime of..." or "...long"
      - `box office` → "box office gross of..."
      - `starring` / `directed by` → Use natural prepositions (e.g., "starring X", "directed by Y").

3.  **Number & Unit Normalization**:
    - Convert raw numbers to readable formats: 
      - `$711909412.25` → "$711.9 million"
      - `149.7K` → "149.7 thousand" or "over 149,000"
    - Ensure dates are grammatically correct (e.g., "released after 2009").

4.  **Tone & Format**:
    - Start with "Which", "What", or "Can you list" (singular preferred).
    - Fix any typos in the original question (e.g., "right" -> "write").
    - **Output ONLY the rewritten question string. No explanations.**

# Examples

**User Input:**
Original: "what movies does taylor lautner play in?"
Constraints: "starring is 'Taylor Lautner' AND released in after 2009 AND duration is more than 94 min AND box office is less than $711.9M AND directed by is 'Garry Marshall'"

**Assistant Output:**
Which movie starring Taylor Lautner and directed by Garry Marshall was released after 2009, has a runtime longer than 94 minutes, and a box office gross of less than $711.9 million?

**User Input:**
Original: "what are the 5 biggest cities in the usa?"
Constraints: "founded in before 1792 AND population is more than 149.7K AND elevation is more than 18 m AND area is less than 1.2K sq km"

**Assistant Output:**
Which US city founded before 1792 has a population of more than 149.7 thousand, an elevation above 18 meters, and an area of less than 1.2 thousand square kilometers?

# Current Task
Original: "{original_question}"
Constraints: "{constraint_description}"""

    # ---------------------------------------------------------
    # 2. User Prompt: 填入当前具体的变量
    # ---------------------------------------------------------
    user_input = f"""
Please rewrite this specific query:

Original Question: "{original_q}"
Constraints: "{constraint_desc}"

Output:
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_input}
            ],
            temperature=0.3,
        )
        content = response.choices[0].message.content.strip()

        # 简单的后处理：有时候模型会忍不住加引号
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]

        return content
    except Exception as e:
        print(f"Error calling API: {e}")
        return None


# 1. 读取原始数据
input_file = '../datasets/complex_constraint_dataset.json'
output_file = '../datasets/complex_constraint_dataset_rewrite_queries.json'

with open(input_file, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# 2. 循环处理
total = len(dataset)
print(f"开始处理 {total} 条数据...")

for index, item in enumerate(dataset):
    original_q = item.get('original_question', '')
    constraint = item.get('constraint_description', '')

    # 简单的去重/清洗逻辑：去除 "released in after" 这种生成的中间语法的冗余词
    clean_constraint = constraint.replace("released in after", "released after")

    print(f"[{index + 1}/{total}] 生成中: {original_q} + {clean_constraint}")

    new_question = generate_natural_question(original_q, clean_constraint)

    if new_question:
        # 将生成的新问题写入 item
        item['complex_question'] = new_question
        print(f"   -> Result: {new_question}")
    else:
        item['complex_question'] = "GENERATION_FAILED"

    # 避免速率限制，适当休眠（可选）
    # time.sleep(0.5)

# 3. 保存结果
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"\n处理完成！结果已保存至 {output_file}")