import os
import pandas as pd
from typing import Dict, List, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import numpy as np
import re  # Added for better score parsing

# Load environment variables
load_dotenv()

# ========== State Definition ==========
class ProjectState(TypedDict):
    """Project state for Table Discovery and Evaluation"""
    data_lake_dir: str
    summary_output_path: str
    raw_tables: Optional[Dict[str, pd.DataFrame]]
    table_summaries: Optional[Dict[str, str]]
    queries: List[str]
    my_judgments: List[str]
    match_results: List[Dict[str, str]]
    evaluations: Optional[List[str]]
    reflections: Optional[List[str]]
    error: Optional[str]
    llm: Optional[ChatOpenAI]
    embedding_model: Optional[SentenceTransformer]
    strategy: str  # Added: 'hybrid', 'llm_only', or 'embedding' for Task2 fallback
    retry_count: int  # Added: Track retry attempts for fallback

# ========== Model Initialization ==========
def initialize_models(state: ProjectState) -> ProjectState:
    """Initialize models"""
    print("[Graph Node: initialize_models] Starting - Role: Load LLM and embedding models for subsequent tasks.")
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            state['llm'] = ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=api_key)
            print("✓ OpenAI model initialized successfully")
        except Exception as e:
            print(f"✗ OpenAI model initialization failed: {str(e)}")
            state['llm'] = None
    else:
        print("⚠️ OPENAI_API_KEY not found, will use offline mode")
        state['llm'] = None
    
    try:
        state['embedding_model'] = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Embedding model initialized successfully")
    except Exception as e:
        print(f"✗ Embedding model initialization failed: {str(e)}")
        state['embedding_model'] = None
    
    # Initialize strategy and retry
    state['strategy'] = 'hybrid'
    state['retry_count'] = 0
    
    print("[Graph Node: initialize_models] Completed.")
    return state

# ========== Load Raw Tables ==========
def load_raw_tables(state: ProjectState) -> ProjectState:
    """Load all CSV tables from data_lake_dir"""
    print("[Graph Node: load_raw_tables] Starting - Role: Read CSV files into DataFrames for raw data access.")
    try:
        raw_tables = {}
        files = [f for f in os.listdir(state['data_lake_dir']) if f.endswith('.csv')]
        for i, file in enumerate(files, 1):
            print(f"Loading file {i}/{len(files)}: {file}")
            table_name = file.replace('.csv', '')
            raw_tables[table_name] = pd.read_csv(os.path.join(state['data_lake_dir'], file))
        state['raw_tables'] = raw_tables
        print(f"✓ Loaded {len(raw_tables)} tables")
    except Exception as e:
        state['error'] = str(e)
        print(f"✗ Failed to load tables: {str(e)}")
    
    print("[Graph Node: load_raw_tables] Completed.")
    return state

# ========== Generate Table Summaries (Task1) ==========
def generate_summaries(state: ProjectState) -> ProjectState:
    """Generate schema summaries for each table using three prompt styles and integrate"""
    print("[Graph Node: generate_summaries] Starting - Role: Create natural language summaries for each table schema (Task1).")
    if state.get('error') or not state.get('llm') or not state.get('raw_tables'):
        print("Skipping due to missing LLM or raw tables.")
        print("[Graph Node: generate_summaries] Completed (skipped).")
        return state
    
    try:
        summaries = []
        tables = list(state['raw_tables'].items())
        for i, (table_name, df) in enumerate(tables, 1):
            print(f"Processing table {i}/{len(tables)}: {table_name}")
            sample_data = df.head(3).to_dict(orient='records')
            
            # Prompt A: Data Analyst Style
            print(f"  Generating Prompt A for {table_name}...")
            prompt_a = f"""
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
            summary_a = state['llm'].invoke(prompt_a).content
            
            # Prompt B: Schema Recognition Expert Style
            print(f"  Generating Prompt B for {table_name}...")
            prompt_b = f"""
You are a schema recognition expert.
Based on the data sample, identify logical relationships between fields and explain their semantic meanings.
Table Name: {table_name}
Sample Data:
{sample_data}
"""
            summary_b = state['llm'].invoke(prompt_b).content
            
            # Prompt C: Data Dictionary Generator Style
            print(f"  Generating Prompt C for {table_name}...")
            prompt_c = f"""
Convert this table into a human-readable data dictionary.
Table Name: {table_name}
Sample Data:
{sample_data}
"""
            summary_c = state['llm'].invoke(prompt_c).content
            
            # Integrate summaries
            final_summary = f"A: {summary_a}\nB: {summary_b}\nC: {summary_c}"
            summaries.append({"table_name": table_name, "final_summary": final_summary})
            print(f"  Integrated summary for {table_name} completed.")
        
        pd.DataFrame(summaries).to_csv(state['summary_output_path'], index=False)
        state['table_summaries'] = {s['table_name']: s['final_summary'] for s in summaries}
        print("✓ Task1: Summaries generated and saved")
        
        # Preview first summary
        if summaries:
            first = summaries[0]
            print("\nFirst table summary preview:")
            print(f"Table: {first['table_name']}")
            print("Final integrated summary:\n", first['final_summary'])
    except Exception as e:
        state['error'] = str(e)
        print(f"✗ Task1 failed: {str(e)}")
    
    print("[Graph Node: generate_summaries] Completed.")
    return state

# ========== Perform Table Search (Task2) ==========
def perform_search(state: ProjectState) -> ProjectState:
    """Perform natural language table search using selected strategy"""
    print(f"[Graph Node: perform_search] Starting - Role: Match queries to tables using {state['strategy']} strategy (Task2).")
    if state.get('error') or not state.get('table_summaries') or not state.get('embedding_model') or not state.get('llm'):
        print("Skipping due to missing dependencies.")
        print("[Graph Node: perform_search] Completed (skipped).")
        return state
    
    try:
        state['match_results'] = []
        for j, query in enumerate(state['queries'], 1):
            print(f"Processing query {j}/{len(state['queries'])}: {query}")
            
            if state['strategy'] == 'hybrid':
                # Hybrid match
                summaries = list(state['table_summaries'].values())
                summary_embs = state['embedding_model'].encode(summaries, convert_to_tensor=True)
                query_emb = state['embedding_model'].encode(query, convert_to_tensor=True)
                cos_scores = util.cos_sim(query_emb, summary_embs)[0].cpu().numpy()
                
                indices = np.argsort(cos_scores)[::-1][:3]  # initial_k=3
                candidates_text = "\n".join([f"Table: {list(state['table_summaries'].keys())[i]}\nSummary: {summaries[i]}" for i in indices])
                
                prompt = f"Given these candidate table summaries:\n{candidates_text}\n\nUser Query: {query}\n\nSelect the most relevant table and explain why."
                match_response = state['llm'].invoke(prompt).content
            elif state['strategy'] == 'llm_only':
                # LLM-only fallback
                summaries_text = "\n".join([f"Table: {table}\nSummary: {summary}" for table, summary in state['table_summaries'].items()])
                prompt = f"Given these table summaries:\n{summaries_text}\n\nUser Query: {query}\n\nSelect the most relevant table and explain why."
                match_response = state['llm'].invoke(prompt).content
            elif state['strategy'] == 'embedding':
                # Embedding-only fallback (Unified format for parsing)
                summaries = list(state['table_summaries'].values())
                summary_embs = state['embedding_model'].encode(summaries, convert_to_tensor=True)
                query_emb = state['embedding_model'].encode(query, convert_to_tensor=True)
                cos_scores = util.cos_sim(query_emb, summary_embs)[0].cpu().numpy()
                top_idx = np.argmax(cos_scores)
                top_table = list(state['table_summaries'].keys())[top_idx]
                match_response = f"Table: {top_table} - Score: {cos_scores[top_idx]:.4f}"  # Unified "Table: " prefix
            else:
                raise ValueError("Invalid strategy")
            
            table_name = match_response.split("Table: ")[1].split(" - ")[0].strip() if "Table: " in match_response else None  # Adjusted parsing for score
            if table_name:
                state['match_results'].append({'query': query, 'matched_table': table_name})
                print(f"  Matched: {table_name}")
    except Exception as e:
        state['error'] = str(e)
        print(f"✗ Task2 failed: {str(e)}")
    
    print("[Graph Node: perform_search] Completed.")
    return state

# ========== Evaluate Results (Task3) ==========
def evaluate_results(state: ProjectState) -> ProjectState:
    """Evaluate retrieved tables using raw data"""
    print("[Graph Node: evaluate_results] Starting - Role: Assess search results with raw data and reflect (Task3).")
    if state.get('error') or not state.get('match_results') or not state.get('raw_tables') or not state.get('llm'):
        print("Skipping due to missing dependencies.")
        print("[Graph Node: evaluate_results] Completed (skipped).")
        return state
    
    try:
        state['evaluations'] = []
        state['reflections'] = []
        for k, result in enumerate(state['match_results'], 1):
            query = result['query']
            table_name = result['matched_table']
            print(f"Evaluating result {k}/{len(state['match_results'])} for query: {query}")
            df = state['raw_tables'].get(table_name)
            if not df:
                continue
            
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
            evaluation = state['llm'].invoke(prompt).content
            state['evaluations'].append(evaluation)
            print(f"  Evaluation: {evaluation}")
            
            # Reflection
            my_judgment = state['my_judgments'][k-1]
            refl_prompt = f"""
My judgment: {my_judgment}
LLM evaluation: {evaluation}
Reflect on agreement in 1-2 sentences.
"""
            reflection = state['llm'].invoke(refl_prompt).content
            state['reflections'].append(reflection)
            print(f"  Reflection: {reflection}")
    except Exception as e:
        state['error'] = str(e)
        print(f"✗ Task3 failed: {str(e)}")
    
    print("[Graph Node: evaluate_results] Completed.")
    return state

# ========== Condition for Fallback ==========
def check_evaluation(state: ProjectState) -> str:
    """Check if evaluation is satisfactory; if not, fallback to another strategy"""
    # Parse evaluations for scores (use regex to find first number after "Rating:")
    if state.get('evaluations'):
        scores = []
        for eval_str in state['evaluations']:
            match = re.search(r'Rating:\s*(\d+)', eval_str)
            score = int(match.group(1)) if match else 3  # Default if parse fails
            scores.append(score)
        avg_score = sum(scores) / len(scores)
        if avg_score < 4 and state['retry_count'] < 2:
            state['retry_count'] += 1
            old_strategy = state['strategy']
            if state['strategy'] == 'hybrid':
                state['strategy'] = 'llm_only'
            elif state['strategy'] == 'llm_only':
                state['strategy'] = 'embedding'
            print(f"Average score {avg_score:.1f} < 4, retrying with strategy: {state['strategy']} (from {old_strategy})")
            return "perform_search"  # Fallback to search with new strategy
        # If all retries fail, note potential future fallback to Task1
        # Future: if state['retry_count'] >= 2: return "generate_summaries"  # Regenerate summaries if needed
    return END  # Proceed to end if satisfactory

# ========== Load Queries and Judgments ==========
def load_queries_and_judgments(file_path="query.txt") -> tuple[List[str], List[str]]:
    """Load queries and judgments from file"""
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

# ========== Create Workflow ==========
def create_workflow() -> StateGraph:
    """Create LangGraph workflow for the mini-project"""
    workflow = StateGraph(ProjectState)
    
    # Add nodes
    workflow.add_node("initialize_models", initialize_models)
    workflow.add_node("load_raw_tables", load_raw_tables)
    workflow.add_node("generate_summaries", generate_summaries)
    workflow.add_node("perform_search", perform_search)
    workflow.add_node("evaluate_results", evaluate_results)
    
    # Set flow with conditional edges
    workflow.set_entry_point("initialize_models")
    workflow.add_edge("initialize_models", "load_raw_tables")
    workflow.add_edge("load_raw_tables", "generate_summaries")
    workflow.add_edge("generate_summaries", "perform_search")
    workflow.add_edge("perform_search", "evaluate_results")
    workflow.add_conditional_edges(
        "evaluate_results",
        check_evaluation,
        {"perform_search": "perform_search", END: END}
    )
    
    return workflow.compile()

# ========== Main Function ==========
def run_project(data_lake_dir: str = "./data_lake", summary_output_path: str = "./table_summaries.csv"):
    """Run the mini-project workflow"""
    queries, my_judgments = load_queries_and_judgments()
    
    initial_state = ProjectState(
        data_lake_dir=data_lake_dir,
        summary_output_path=summary_output_path,
        raw_tables=None,
        table_summaries=None,
        queries=queries,
        my_judgments=my_judgments,
        match_results=[],
        evaluations=None,
        reflections=None,
        error=None,
        llm=None,
        embedding_model=None,
        strategy='hybrid',  # Initial strategy
        retry_count=0  # Initial retry count
    )
    
    app = create_workflow()
    try:
        result = app.invoke(initial_state)
        if result.get("error"):
            print(f"✗ Project failed: {result['error']}")
            return False
        else:
            print("✓ Project completed!")
            return True
    except Exception as e:
        print(f"✗ Workflow execution failed: {str(e)}")
        return False

if __name__ == "__main__":
    run_project()
