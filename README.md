How to Run the Code
====

Ensure the data lake directory (./data_lake) contains the CSV tables (generated or provided separately).
Set the OpenAI API key as an environment variable:

* On Unix/Linux/Mac: export OPENAI_API_KEY=your_api_key_here
* On Windows: set OPENAI_API_KEY=your_api_key_here
* In Kaggle: Use import os; os.environ['OPENAI_API_KEY'] = 'your_key' in a cell before running.


Optionally, create a query.txt file with queries and judgments (format: query\tjudgment per line) for custom inputs.
Run the script: python main.py

* The script executes Task1 (schema summarization), Task2 (table search), and Task3 (evaluation) sequentially.
* Outputs are printed to the console, and summaries are saved to ./table_summaries.csv.

Required Packages or Dependencies
====
* pandas
* numpy
* openai
* sentence-transformers

Install them via pip: pip install pandas numpy openai sentence-transformers

Example Queries and Outputs
====

Example Query 1: "Which table contains information about employee salaries?"

* Matched Table: emp_sal_h
* LLM Evaluation: Rating: 5. Explanation: The table "emp_sal_h" is highly relevant to the query about employee salaries. The column "sal" directly corresponds to salary information, and the presence of employee IDs ("e_id") and effective dates ("eff_dt") further supports the context of salary data. There is no missing information for the query, and all aspects of the table are relevant.
* Reflection: The LLM's judgment aligns with my own, as both assessments recognize the table "emp_sal_h" as highly relevant and comprehensive for querying employee salary history.

Example Query 2: "Find the table with payment options and fees."

* Matched Table: pay_methods
* LLM Evaluation: Rating: 5. Explanation: The table "pay_methods" is highly relevant to the query as it provides information on payment options ("pay_desc") and associated processing fees ("proc_fee_pct"), which directly aligns with the user's request for payment options and fees. The presence of an "active_flg" column also indicates which payment methods are currently active, adding further useful context. There is no missing information or irrelevant aspects in the table concerning the query.
* Reflection: The LLM's judgment aligns with my own, as both assessments recognize the table "pay_methods" as highly relevant to the query by providing comprehensive information on payment options and associated fees.

Key Insights and Challenges Encountered
====

Task1
----
1. Task1的入手消耗了比较多的时间：阅览了几个经典的kaggle表格数据（Boston，Titanic）之后决定还是先用数据合成的方法来测试一下框架。
2. 根据任务描述，以及kaggle的列名情况，推测这个表格模式挖掘的任务可能需要考虑一些列名略缩的设计；甚至一些非结构化数据？（datalake设想，未实现）
3. 使用Prompt生成了一些原始数据，提示词见dataprompt.txt。
4. 设计三个prompt之后；原本想使用multiagent debate来讨论表格，只不过一来目前数据没有那么复杂（看上去总结已经足够好），二来三个主流模型API分别购买有点expensive，于是加入TO DO LIST。

Task2
----
1. **基于嵌入匹配的低准确性**：
    - 疑似所使用模型参数规模较小，导致次优排名或遗漏匹配（例如，薪资查询遗漏 emp_sal_h）。
    - 原因：小型模型规模限制了对技术总结的语义捕捉；分数分布扁平化。
    - 解决方案：在混合策略中引入自适应扩展（从 top-3 到 top-5，然后全 LLM 回退）以包括更多候选。降低阈值（例如 0.1）并添加调试打印以检查分数。作为备选，回退到 OpenAI 嵌入以提高质量。
2. **混合模式中 LLM 来自分数的偏见**：
    - 当提示中包含分数时，LLM 过度依赖它们，向高分但语义弱的候选偏向。
    - 原因：提示设计无意中强调了量化方面。
    - 解决方案：在最终混合版本中从 LLM 提示中移除分数，指示 LLM “优先语义匹配”并“忽略外部偏见”。这将嵌入纯粹视为过滤器，最大化定性判断。
3. **Token 效率和成本**：
    - 纯 LLM 提示使用所有总结消耗高。
    - 原因：目前task1给出的总结是没有节省token的设计的。
    - 解决方案：混合过滤将输入减少到 top-K 总结（例如 3-5），将 token 减少 70-80%。自适应回退确保鲁棒性，而无需总是默认全输入。

Task3
----
1. 发现task3的任务和task1中添加的GPT-5作为'Critic'的任务有些重复了。
2. 于是尝试喂给task3的评价者原始数据和query，来尝试丰富评价维度。



PlanB
----
尝试使用LangGraph实现更灵活的框架
