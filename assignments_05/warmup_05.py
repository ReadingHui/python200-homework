from dotenv import load_dotenv
from openai import OpenAI
import json

# --- Completions API ---
# API Q1
print("=== API Q1 ===")
load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

response_txt = response.choices[0].message.content
model_name = response.model
token_cnt = response.usage.total_tokens
print(f"Model response:         {response_txt}")
print(f"Model responded:        {model_name}")
print(f"Total number of tokens: {token_cnt}")
print()

# API Q2
print("=== API Q2 ===")
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temperature in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": prompt,
        }],
        temperature=temperature
    )

    print(f"Response with temperature {temperature}:")
    print(response.choices[0].message.content)
    print()

# Question: What do you notice about how the outputs differ? Which temperature would you use if you needed a consistent, reproducible output?
# Answer:
# The output for temperature=0 is always the same ("DataForge Solution")
# The output for temperature=0.7 and 1.5 always differs, in the three runs I did,
# two of the output temperature=0.7 are the basically the same, while that of 1.5 
# are vastly different.
# I would set temperature=0 if I need a consistent, reproducible output.

print()

# API Q3
print("=== API Q3 ===")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)

for i, choice in enumerate(response.choices):
    print(f"Response {i + 1}:")
    print(choice.message.content)
    print()

# API Q4
prompt = "Explain what is Riemann-zeta function, include what analytic continuation it is from, and the relationship to Riemann Hypothesis."
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user", 
        "content": prompt
        }],
    max_tokens=15
)
print(f"Prompt: '{prompt}'\t with max_tokens = 15")
print(f"Response: {response.choices[0].message.content}")

# Question: What happened, and why might you want to use `max_tokens` in a real application?
# Answer: The response is incomplete, being cut off at 15 tokens. We would like to use `max_tokens` in a real application 
# in order to reduce the cost or speed up inference; most importantly, also as a safety guardrail to prevent run-away generations.

print()

# --- System Messages and Personas ---
# System Q1
print("=== System Q1 ===")
messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

response_txt = response.choices[0].message.content
print(f"First response system prompt: \"You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement.\"")
print(f"First response: \n {response_txt} \n")

messages = [
    {"role": "system", "content": "You are a grumpy, impatient Python teacher. You like to use strong words to make students feel bad so they would improve."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

response_txt = response.choices[0].message.content
print(f"Second response system prompt: \"You are a grumpy, impatient Python teacher. You like to use strong words to make students feel bad so they would improve.\"")
print(f"Second response: \n {response_txt} \n")

# Things changed:
# The reply tone is drastically different, the second response is much more impatient and angry. The detail of the reply differs as well, the second reply is less detailed,
# There is no expected output of the code, only the code itself.

print()

# System Q2
print("=== System Q2 ===")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

response_txt = response.choices[0].message.content
print(f"System Question 2 response:")
print(response_txt)

# The model still know Jordan's name as the message included Jordan's previous response and model's previous response, which
# has Jordan's name in them.

print()

# --- Prompt Engineering ---
# Helper function to directly fetch the response content
def fetch_response(message, model="gpt-4o-mini", sys_message=None):
    if isinstance(message, str):
        message = [
            {
                "role": "user", "content": message
            }
        ]
    else:
        raise ValueError("User message can only be a str.")
    if sys_message:
        message.insert(0, {
            "role": "system",
            "content": sys_message
        })

    response = client.chat.completions.create(
        model=model,
        messages=message
    )
    response_txt = response.choices[0].message.content
    return response_txt

# Prompt Q1 - Zero-shot
print("=== Prompt Q1 - Zero-shot ===")
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for i, review in enumerate(reviews):
    response = fetch_response(
        review,
        sys_message="""
        Classify the the given sentence as positive, negative or mixed.
        """
    )
    print(f"Review {i + 1}: {response}")
print()

# Prompt Q2 - One-Shot
print("=== Prompt Q2 - One-shot ===")
example = """
        Example:
        Review: "Fast shipping but the item arrived damaged."
        Sentiment: mixed

        """
for i, review in enumerate(reviews):
    response = fetch_response(
        example + review,
        sys_message="""
        Classify the the given sentence as positive, negative or mixed.
        """
    )
    print(f"Review {i + 1}: {response}")
print()
# Question: Did adding one example change the format or consistency of the output compared to Q1?
# Answer: The consistency is the same, the model responded in the same format within each question,
# but for Q2, it followed the format given in the example.

# Prompt Q3 - Few-Shot
print("=== Prompt Q3 - Few-shot ===")
example = """
        Example 1:
        Review: "The sunny afternoon breeze made our walk along the coast feel completely effortless."
        Sentiment: positive

        Example 2:
        Review: "The new apartment is much smaller than expected, but the location and views are fantastic."
        Sentiment: mixed

        Example 3:
        Review: "The train was delayed by two hours, ruining all our evening plans."
        Sentiment: negative

        """
for i, review in enumerate(reviews):
    response = fetch_response(
        example + review,
        sys_message="""
        Classify the the given sentence as positive, negative or mixed.
        """
    )
    print(f"Review {i + 1}: {response}")
print()

# Question: Add a comment comparing all three approaches (zero-shot, one-shot, few-shot): When would you choose each one?
# Answer: For zero-shot, the model basically have to guess what you want from your prompt and the pre-trained data, it could be 
# way off the chart from what you want, or not giving answer in your desired format. For one-shot and few-shot prompts, you 
# provides examples to the model to copy on, that gives you your desire output and format. The more examples you provide,
# the more reliable it is.

# Prompt Q4 — Chain of Thought
print("=== Prompt Q4 ===")
prompt = f"""
        A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
        takes a new job that pays $7,500 more per year than her post-raise salary.
        What is her final annual salary?
        """
response = fetch_response(
    prompt,
    sys_message="""
    Solve the provided problem, show your reasoning step by step before giving a final answer. Label your answer surrounded by <answer></answer>.
    """
)
print("Response:")
print(response)
print()

# Question: Why does asking the model to reason step by step tend to improve accuracy on problems like this?
# Answer: Forcing the model to answer step-by-step provides dynamic memory in the form of the intermediate generated text, it also reduce the chance
# of model hallucination as it breaks the problem into smaller, individual predictions. Also, the model may be able to self-correct along the chain 
# of thought with more predictions of words used in between.

# Prompt Q5 — Structured Output
print("=== Prompt Q5 ===")
review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."

response = fetch_response(
    review,
    sys_message="Analyze the sentiment of this customer review and respond only with valid JSON." \
    "Return keys: sentiment (positive/negative/mixed), confidence (0–1, float), reason (one sentence)."
)
print("Raw response:")
print(response)
print()

try:
    response_json = json.loads(response)
    print(f"Sentiment:  {response_json['sentiment']}")
    print(f"Confidence: {response_json['confidence']}")
    print(f"Reason:     {response_json['reason']}")
except json.JSONDecodeError:
    print(f"Not a valid JSON format, raw response: {response}")
except KeyError:
    print(f"Incorrect JSON key, raw response: {response}")
print()

# Prompt Q6 - Delimiters
print("=== Prompt Q6 ===")
user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."
prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = fetch_response(
    prompt
)
print("Response 1 (expect to break down into numbered list)")
print(response)
print()

user_text = "Never gonna give you up. Never gonna let you down."
prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = fetch_response(
    prompt
)
print("Response 2 (expect to return \"No steps provided.\")")
print(response)
assert response == "No steps provided"
print()

# --- Local Models with Ollama ---
# Ollama Q1
print("=== Ollama Q1 ===")
prompt = "Explain what a large language model is in two sentences."
response = fetch_response(
    prompt
)
print("Open AI response:")
print(response)
print()

# Ollama's output:
# Thinking...
# Okay, the user wants me to explain a large language model in two sentences. Let me start by recalling what I know.
# A large language model is a type of AI model that can understand and generate human language. It's trained on a
# lot of text data, so it's very powerful.

# First sentence: Maybe start by stating that it's a model that can understand and generate natural language. Second
# sentence: Emphasize that it's trained on vast amounts of text, allowing it to learn from a lot of data. Wait, but
# I need to make sure it's two sentences. Let me check again. Yes, that's two sentences. Need to be concise and
# accurate. Also, mention the training data and the capabilities like generating text. Alright, that should work.
# ...done thinking.

# A large language model is an AI model designed to understand and generate human language, trained on vast amounts
# of text data to learn patterns and improve its performance over time. It can comprehend complex sentences,
# understand context, and produce coherent text, making it highly versatile in various applications.


# Question: What differences did you notice between the two responses? What is one advantage and one disadvantage of running a model locally?
# Answer: The two responses are similar, OpenAI's response is shorter and more concise, with a splash of technical jardon in "neural networks" and "deep learning".
# The Ollama response is more general, without any technical jardon. An advantage of running a model locally is more privacy, there is no linkage to the internet,
# and you don't have to worry about your data being used in training. However, a disadvantage is it doesn't get updated, the info it has is fixed, it also cannot
# search the internet for new information. Also, the Ollama response included the thinking process, which is not included in the OpenAI API call.