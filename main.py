import pandas as pd
import numpy as np
import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

# 常量定义：API key 从环境变量获取（为公开代码隐藏）
API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
DATA_LAKE_DIR = "./data_lake"
SUMMARY_OUTPUT_PATH = "./table_summaries.csv"
CLIENT = OpenAI(api_key=API_KEY)

# 全局变量：存储Task2的匹配结果（列表，每个元素是{'query': str, 'matched_table': str}）
MATCH_RESULTS = []

def build_prompt_a(table_name, sample_data):
    """构建Prompt A：数据分析师风格"""
    return f"""
You are a senior data analyst.
Below is a CSV table sample from an unknown enterprise system. Your task is to:
1. Identify what this table is about.
2. Describe each column's purpose in plain English.
3. Mention who might use this table and for what kind of task.
Table Name: {table_name}
Sample Data:
{sample_data}
Please summarize the schema clearly and concisely.
"""

def build_prompt_b(table_name, sample_data):
    """构建Prompt B：模式识别专家风格"""
    return f"""
You are a schema recognition expert.
Based on the data sample, identify logical relationships between fields and explain their semantic meanings.
Table Name: {table_name}
Sample Data:
{sample_data}
"""

def build_prompt_c(table_name, sample_data):
    """构建Prompt C：数据字典生成器风格"""
    return f"""
Convert this table into a human-readable data dictionary.
Table Name: {table_name}
Sample Data:
{sample_data}
"""

def call_llm(prompt):
    """调用LLM生成响应"""
    return CLIENT.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are an expert in tabular data interpretation."},
                  {"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=600
    ).choices[0].message.content

def integrate_summaries(summary_a, summary_b, summary_c):
    """整合三种总结为最终总结（简单拼接作为示例，实际可优化）"""
    return f"A: {summary_a}\nB: {summary_b}\nC: {summary_c}"

def run_task1():
    """执行Task1：生成表格总结"""
    summaries = []
    first_summary_details = None  # 用于存储第一个表格的细节以供打印
    for idx, fname in enumerate(os.listdir(DATA_LAKE_DIR)):
        if fname.endswith(".csv"):
            table_name = fname.replace(".csv", "")
            df = pd.read_csv(os.path.join(DATA_LAKE_DIR, fname))
            sample_data = df.head(3).to_dict(orient='records')
            
            prompt_a = build_prompt_a(table_name, sample_data)
            summary_a = call_llm(prompt_a)
            
            prompt_b = build_prompt_b(table_name, sample_data)
            summary_b = call_llm(prompt_b)
            
            prompt_c = build_prompt_c(table_name, sample_data)
            summary_c = call_llm(prompt_c)
            
            final_summary = integrate_summaries(summary_a, summary_b, summary_c)
            summaries.append({"table_name": table_name, "final_summary": final_summary})
            
            # 保存第一个表格的细节用于打印
            if idx == 0:
                first_summary_details = {
                    "summary_a": summary_a,
                    "summary_b": summary_b,
                    "summary_c": summary_c,
                    "final_summary": final_summary
                }
    
    pd.DataFrame(summaries).to_csv(SUMMARY_OUTPUT_PATH, index=False)
    print("Task1 completed: Summaries saved.")
    
    # 打印第一个表格的部分总结
    if first_summary_details:
        print("\nFirst table summary preview:")
        print("Prompt A output:\n", first_summary_details["summary_a"], "\n")
        print("Prompt B output:\n", first_summary_details["summary_b"], "\n")
        print("Prompt C output:\n", first_summary_details["summary_c"], "\n")
        print("Final integrated summary:\n", first_summary_details["final_summary"])

def load_table_summaries():
    """加载表格总结"""
    df = pd.read_csv(SUMMARY_OUTPUT_PATH)
    return dict(zip(df['table_name'], df['final_summary']))

def load_raw_tables():
    """加载原始表格"""
    raw_tables = {}
    for file in os.listdir(DATA_LAKE_DIR):
        if file.endswith('.csv'):
            table_name = file.replace('.csv', '')
            raw_tables[table_name] = pd.read_csv(os.path.join(DATA_LAKE_DIR, file))
    return raw_tables

def llm_only_match(query, table_summaries):
    """Task2策略1：纯LLM匹配"""
    summaries_text = "\n".join([f"Table: {table}\nSummary: {summary}" for table, summary in table_summaries.items()])
    prompt = f"Given these table summaries:\n{summaries_text}\n\nUser Query: {query}\n\nSelect the most relevant table and explain why."
    return call_llm(prompt)

def embedding_match(query, table_summaries):
    """Task2策略2：嵌入相似度匹配"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    summaries = list(table_summaries.values())
    summary_embs = model.encode(summaries, convert_to_tensor=True)
    query_emb = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_emb, summary_embs)[0].cpu().numpy()
    top_idx = np.argmax(cos_scores)
    top_table = list(table_summaries.keys())[top_idx]
    return f"Top table: {top_table} (Score: {cos_scores[top_idx]:.4f})"

def hybrid_match(query, table_summaries, top_n=1, initial_k=3):
    """Task2策略3：混合匹配（嵌入过滤 + LLM）"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    summaries = list(table_summaries.values())
    summary_embs = model.encode(summaries, convert_to_tensor=True)
    query_emb = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_emb, summary_embs)[0].cpu().numpy()
    
    indices = np.argsort(cos_scores)[::-1][:initial_k]
    candidates_text = "\n".join([f"Table: {list(table_summaries.keys())[i]}\nSummary: {summaries[i]}" for i in indices])
    
    prompt = f"Given these candidate table summaries:\n{candidates_text}\n\nUser Query: {query}\n\nSelect the {top_n} most relevant table(s) and explain why."
    return call_llm(prompt)

def evaluate_retrieved_table(query, table_name, raw_tables):
    """Task3：使用LLM评估检索表格相关性"""
    if table_name not in raw_tables:
        return "Table not found."
    df = raw_tables[table_name]
    columns = ', '.join(df.columns.tolist())
    sample_data = df.head(3).to_string(index=False)
    prompt = f"""
User query: {query}
Retrieved table name: {table_name}
Columns: {columns}
Sample data:
{sample_data}
Rate relevance (1-5) and explain. Identify missing/irrelevant info.
"""
    return call_llm(prompt)

def generate_reflection(my_judgment, llm_evaluation):
    """Task3：生成反思"""
    prompt = f"""
My judgment: {my_judgment}
LLM evaluation: {llm_evaluation}
Reflect on agreement in 1-2 sentences.
"""
    return call_llm(prompt)

def load_queries_and_judgments(file_path="query.txt"):
    """从query.txt读取查询和人工标签"""
    queries = []
    judgments = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        queries.append(parts[0])
                        judgments.append(parts[1])
    except FileNotFoundError:
        print(f"{file_path} not found, using default queries and judgments.")
    
    # 如果读取失败或为空，使用默认;确保demo可以完成
    if not queries:
        queries = [
            "Which table contains information about employee salaries?",
            "Find the table with payment options and fees.",
            "Something about products and stock."
        ]
        judgments = [
            "5/5 for emp_sal_h: Direct salary data in 'sal' column.",
            "5/5 for pay_methods: Payment options and fees in columns.",
            "5/5 for product_catalog: Products and stock in columns."
        ]
    
    return queries, judgments

def run_task2():
    """执行Task2：测试查询，并将结果封装到全局MATCH_RESULTS"""
    global MATCH_RESULTS
    MATCH_RESULTS = []  # 清空以防重复运行
    table_summaries = load_table_summaries()
    queries, _ = load_queries_and_judgments()  # 只用queries
    
    for query in queries:
        print(f"\nQuery: {query}")
        
        # Task2策略示例（使用混合作为默认）
        match_response = hybrid_match(query, table_summaries)
        print("Hybrid Match Response:", match_response)
        
        # 解析匹配表格
        table_name = match_response.split("Table: ")[1].split(" ")[0].strip() if "Table: " in match_response else None
        if table_name:
            MATCH_RESULTS.append({'query': query, 'matched_table': table_name})

def run_task3():
    """执行Task3：使用全局MATCH_RESULTS作为输入进行评估"""
    global MATCH_RESULTS
    raw_tables = load_raw_tables()
    _, my_judgments = load_queries_and_judgments()  # 只用judgments
    
    for i, result in enumerate(MATCH_RESULTS):
        query = result['query']
        table_name = result['matched_table']
        print(f"\nQuery: {query} (Matched Table: {table_name})")
        
        # Task3评估
        evaluation = evaluate_retrieved_table(query, table_name, raw_tables)
        print("LLM Evaluation:", evaluation)
        
        # 反思
        reflection = generate_reflection(my_judgments[i], evaluation)
        print("Reflection:", reflection)

if __name__ == "__main__":
    # Task1
    run_task1()
    
    # Task2
    run_task2()
    
    # Task3
    run_task3()
