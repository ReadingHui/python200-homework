import pandas as pd
import json
import os
from dotenv import load_dotenv
from scipy.stats import pearsonr

# smolagents imports
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent

# Set-up
if load_dotenv():
    print('Successfully loaded environment variables from .env')
else:
    print('Warning: could not load environment variables from .env')


# Pre-task: Load the Data
DATA_PATH = "assignments_01/outputs/merged_happiness.csv"
df = None
API_KEY = os.environ["OPENAI_API_KEY"]

# Task 1: Define Your Tools
# Tool 1: load_happiness_data
@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory. Returns the metadata dictionary.
    
    Returns:
        A DICTIONARY containing dataset metadata (NOT a DataFrame).
        Example:
            {
                "shape": (156, 9),
                "columns": ["Country", "Score", "GDP"]
            }
    """
    global df
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, index_col=0)
    else:
        dfs = []
        for year in range(2015, 2025):
            year_df = pd.read_csv(f'./happiness_project/world_happiness_{year}.csv', sep=';')
            if year == 2024:
                year_df = year_df.rename({'Ladder score': 'Happiness score'}, axis=1)
            for f in ['Happiness score', 
                        'GDP per capita',
                        'Social support',
                        'Freedom to make life choices',
                        'Generosity',
                        'Perceptions of corruption']:
                year_df[f] = year_df[f].str.replace(',', '.').astype(float)
            year_df['Year'] = year
            dfs.append(year_df)
        df = pd.concat(dfs)    
    return {
        "shape": df.shape,
        "columns": df.columns,
    }

# Tool 2: summarize_column
@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset using df.describe().
    
    Args:
        column: The name of the column to describe.

    Returns:
        A dict of the description, or an error dict.
    """
    global df
    if df is None:
        return {"error": "No data is loaded."}
    if not column:
        return {"error": "Column is missing."}        
    if column not in df.columns:
        return {"error": "Column not found."}
    return df[column].describe().to_dict()

# Tool 3: compute_correlation
@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.
    
    Args:
        col1: Column name for first column. 
        col2: Column name for second column.
    
    Returns:
        Returns a dict containing the name of both columns, correlation coefficient and p-value, or the error dict.
    """
    global df
    if df is None:
        return {"error": "No data is loaded."}
    if not col1:
        return {"error": "col1 not provided."}
    if not col2:
        return {"error": "col2 not provided."}
    if col1 not in df.columns:
        return {
            "error": f"column '{col1}' is not in {df.columns.tolist()}"
        }
    if col2 not in df.columns:
        return {
            "error": f"column '{col2}' is not in {df.columns.tolist()}"
        }
    res = pearsonr(df[col1], df[col2])
    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(res.statistic, 4),
        "p_value": round(res.pvalue, 4)
    }

# Tool 4: get_top_n_countries
@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.
    
    Args:
        column: Column name of the column to be ranked by
        year: year to be filtered
        n: value of N in "top N countries"
    
    Returns:
        The list of the top n rows as a list of dicts (each dict has "country" and the requested column value) or the error dict.
    """
    global df
    if df is None:
        return {"error": "No data is loaded."}
    if not column:
        return {"error": "column not provided."}
    if column not in df.columns:
        return {
            "error": f"column '{column}' is not in {df.columns.tolist()}"
        }
    if not isinstance(year, int):
        return {"error": "year has to be an int."}
    if year < 2015 or year > 2024:
        return {"error": "year out of range, year has to be between 2015 to 2024."}
    if n < 1:
        return {"error": f"n = {n} is out of range, it has to be a positive integer."}

    top_df = df[df['Year'] == year].copy()
    top_list = top_df.sort_values(by=column, ascending=False).head(n)[["Country", column]].rename({"Country": "country"}, axis=1).to_dict(orient='records')
    return top_list

# Task 2: Build the Agent
model = OpenAIServerModel(api_key=API_KEY, model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)
if __name__ == "__main__":
    # Task 3: Run Guided Queries
    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to assignments_07/outputs/happiness_by_region.png.",
    ]

    print("=== Task 3: Run Guided Queries ===")
    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)
        print()

    # Task 4: Your Own Questions
    # My query 1
    my_query_1 = "What is the median of GDP per capita?"   # replace with your question
    response_1 = agent.run(my_query_1, reset=False)
    print(response_1)
    # Comment: Did this trigger tool use, code generation, or both?
    # This triggered the tool use of summarize_column

    # My query 2
    my_query_2 = "Plot GDP per capita over the years as a line chart, with each line represents the n-th rank of country in that year up to n=5. Label each point by the country. You don't have access to the full DataFrame, use other tools to get the highest 5 GDP per capita each year if necessary. Save the plot to assignments_07/outputs/gdp_per_capita_by_region.png"   # replace with your question
    response_2 = agent.run(my_query_2, reset=False)
    print(response_2)
    # Comment: Did this trigger tool use, code generation, or both?
    # This triggered both tool use (get_top_n_countries) and code generation, as the model need to get the top n country with value, then write a code to plot the required graph.





# --- Reflection ---
#
# 1. In Query 3, how did the agent communicate whether the correlation was statistically
#    significant? Did it use the p-value correctly? What threshold did it apply?
# A: The agent communicated the correlation through executing the tool of compute_correlation 
#    on the columns 'GDP per capita' and 'Happiness score'. However, it did not show its reasoning
#    on why it is statiscally significant, it just stated the result.
#
# 2. Did any of the agent's responses surprise you — either by being more capable than
#    you expected, or less? Describe one specific example.
# A: The agent surprised me that it was unable to load the df correctly, trying to convert the output
#    of the load_happiness_data() directly to the desired df, when it was actually just the metadata dict.
#    This shows the model did not fully commit to the docstring provided by the function, the pre-trained
#    data is still heavily influencing its decision.
#
# 3. What one additional tool would make this agent meaningfully more useful?
#    Describe what it would do and what kind of question it would help the agent answer.
#    (You do not need to implement it.)
# A: A load_dataframe() function, where it just return the loaded DataFrame. The model now has no access to the loaded df,
#    as the structure of the code updates the df in its memory without returning in any of the tool. Hence, the model actually
#    has no info on the whole DataFrame, which causes it unable to plot the correct graph in the last provided prompt. Giving
#    the tool to load the df into the agent's sandbox will significantly increase its ability to work with the dataset.