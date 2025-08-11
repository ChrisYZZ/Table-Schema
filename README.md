LLM-Powered Table Discovery and Evaluation
This project implements a simple assistant for schema summarization, table search, and result evaluation using Large Language Models (LLMs), based on a collection of CSV tables.
How to Run the Code

Ensure the data lake directory (./data_lake) contains the CSV tables (generated or provided separately).
Set the OpenAI API key as an environment variable:

On Unix/Linux/Mac: export OPENAI_API_KEY=your_api_key_here
On Windows: set OPENAI_API_KEY=your_api_key_here
In Kaggle: Use import os; os.environ['OPENAI_API_KEY'] = 'your_key' in a cell before running.


Optionally, create a query.txt file with queries and judgments (format: query\tjudgment per line) for custom inputs.
Run the script: python table_assistant.py

The script executes Task1 (schema summarization), Task2 (table search), and Task3 (evaluation) sequentially.
Outputs are printed to the console, and summaries are saved to ./table_summaries.csv.



Required Packages or Dependencies

pandas
numpy
openai
sentence-transformers

Install them via pip: pip install pandas numpy openai sentence-transformers
Example Queries and Outputs
Example Query 1: "Which table contains information about employee salaries?"

Matched Table: emp_sal_h
LLM Evaluation: Rating: 5. Explanation: The table "emp_sal_h" is highly relevant to the query about employee salaries. The column "sal" directly corresponds to salary information, and the presence of employee IDs ("e_id") and effective dates ("eff_dt") further supports the context of salary data. There is no missing information for the query, and all aspects of the table are relevant.
Reflection: The LLM's judgment aligns with my own, as both assessments recognize the table "emp_sal_h" as highly relevant and comprehensive for querying employee salary history.

Example Query 2: "Find the table with payment options and fees."

Matched Table: pay_methods
LLM Evaluation: Rating: 5. Explanation: The table "pay_methods" is highly relevant to the query as it provides information on payment options ("pay_desc") and associated processing fees ("proc_fee_pct"), which directly aligns with the user's request for payment options and fees. The presence of an "active_flg" column also indicates which payment methods are currently active, adding further useful context. There is no missing information or irrelevant aspects in the table concerning the query.
Reflection: The LLM's judgment aligns with my own, as both assessments recognize the table "pay_methods" as highly relevant to the query by providing comprehensive information on payment options and associated fees.

Key Insights and Challenges Encountered
Key insights: The hybrid approach in Task2 (embedding similarity + LLM judgment) provides robust table matching, outperforming pure LLM or embedding methods in handling vague queries. Integrating multiple prompt styles in Task1 enhances schema summaries by combining analytical, relational, and dictionary perspectives. Reading custom queries from a file improves flexibility.
Challenges: Parsing LLM responses for consistent table names required careful string handling. Managing API costs and response variability from GPT-4o was addressed through temperature control and retries. Data quality issues in tables (e.g., abbreviations, nulls) occasionally affected summary accuracy, suggesting potential for column name normalization. In environments like Kaggle, handling parallelism warnings from tokenizers and environment variables needed explicit settings.
