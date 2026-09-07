# --- ML vs. LLM in Pipelines ---
# Q1
# In a comment block, explain the difference between what the ML classifier produces and what the LLM produces in this week's pipeline. 
# Why does each tool do what it does? 
# What would go wrong if you tried to swap them — using the LLM to make the binary good/skip prediction and the ML model to write the recommendation?

# ML classifier produces the prediction of the weather, whether it is suitable for running or not. As for LLM, it produces natural language explanation
# into why it is classified that way. ML classifier is a logistic regression, which computes the probability of whether it is suitable for running or not,
# while LLM is generating natural language, which is best at providing natural explanation on why the classification worked that way.
# If I try to swap them, they will be doing the job a lot worse, LLM prediction is not deterministic, and ML classifier cannot generate natural language at all.

# Q2
# For each task below, write one sentence in a comment block stating whether you would use a trained ML model, an LLM, or deterministic code, and why:
# * Converting a date string like "2023-07-04" to day-of-week
# Deterministic code, as we can expect the same logic to be used everytime, with reproducible result.
# * Classifying a job posting as "entry-level", "mid-level", or "senior" based on freeform text
# LLM, as the posting and analysis are in natural language, which requires reading comprehension and judgment that rule-based code handles poorly.
# * Predicting customer churn given 15 numeric features and a labeled training dataset
# Trained ML model, as we want deterministic outcome, with a feature-target relation yet to be discovered.
# * Normalizing inconsistent city names ("NYC", "New York City", "New York, NY") to a canonical form
# Deterministic code, as we want the city names to be consistent after the normalization.
# * Summing a column of revenue figures
# Deterministic code, as LLM is bad at math and ML model is not required for basic summation.

# Q3
# In a comment block, answer: what is incremental processing, and why is it important for this pipeline? 
# What would happen — in terms of cost and data correctness — if the transform script re-processed all 365 records every time it ran?
# Incremental processing means only updating the record that has not been processed yet, while skipping the ones that are done. It is important because it reduces costs and prevent redoing completed work.
# If it re-processed all 365 records every time it ran, it will cost a lot more than needed. Data will also be overwritten, where data correctness cannot be guaranteed.

# --- Prompt Design ---
# Q1
# Alternative system prompt: 
# "You are writing a two-sentence running recommendation for a daily weather summary app. "
# "You will receive weather conditions for a single day and a machine learning prediction "
# "about whether the day is good for running. "
# "Write exactly two sentence — first sentance states the prediction and the second sentence explains the reasoning.""
# "Be direct, practical, and specific to the conditions. "
# "Do not use bullet points, headers, or phrases like 'Based on the data'."

# To change the validation logic, I would change the if len(sentences) > 2 to (!= 3 or sentences[-1] != "")
# This makes sure there are 2 sentences. 

# Q2
from time import sleep

def call_with_retry(client, messages, max_retries=3):
    for tries in range(max_retries):
        try:
            response = client.chat.completions.create(
                model='gpt-4.1-mini',
                messages=messages,
                tool_choice='auto',  # model chooses whether to use a tool
            )
            return response
        except Exception as e:
            print(f"Error Message: {e}, try #{tries + 1}, retrying in 2 seconds...")
            sleep(2)
    return None

# This will be used in the pipeline as the enriching step, where the API data is sent to GPT using this helper function, so it will retry up to 3 times instead of directly returning None upon
# the first failure, adding robustness to the pipeline.
