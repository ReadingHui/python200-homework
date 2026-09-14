from dotenv import load_dotenv
from openai import OpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.readers.file import PyMuPDFReader # To read brightleaf_pdfs
import json

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

response_txt = response.choices[0].message.content
print(response_txt)

# --- RAG Concepts ---
# Concepts Q1
# Scenario A:
# As the team is updating their library with a huge 
# amount of PDFs every quarter, fine-tuning requires
# quaterly re-train and prompt engineering requires a
# huge context window, so RAG will be the best approach.

# Scenario B:
# As they are focused on a very specific brand voice, they
# are unlikely to change the style, fine-tuning will force 
# the model to focus on the particular style with little to
# none training data on the web.

# Scenario C:
 # Since the document is only two pages, the entire text easily fits 
 # directly into the LLM's context window, making prompt engineering 
 # (in-context learning) the fastest and most cost-effective approach. 
 # Building a RAG pipeline or fine-tuning a model would add unnecessary 
 # architecture and cost for a one-off task with no need for scalability 
 # or dynamic retrieval across larger document sets.

# Concept Q2
# A confident, polished tone exploits human trust heuristics, making authoritative 
# outputs feel inherently reliable and discouraging users from double-checking facts. 
# Conversely, an output containing "I am not sure" signals low confidence, prompting 
# the user to independently verify the information before taking action.
# In software development, an LLM might confidently suggest installing a non-existent 
# software package or library (a package hallucination). Because the model presents the 
# command with total certainty, a developer runs it without verification—allowing attackers 
# who registered that fake package name on PyPI/npm to execute malicious code on the 
# developer's system.

# Concept Q3
# Original order:
# steps = [
#     "Generate a response from the LLM",      
#     "Extract text from source documents",
#     "Receive the user's query",
#     "Retrieve the most relevant chunks",
#     "Convert text chunks into embeddings",
#     "Inject retrieved chunks into the prompt",
#     "Split text into chunks",
#     "Embed the user's query",
# ]

# Correct order:
# steps = [
#     "Extract text from source documents",         # This step load the text of the document into the Python script
#     "Split text into chunks",                     # This step splits the document text into chunks so that it fits in the LLM context window
#     "Convert text chunks into embeddings",        # This step embeds the text chunks into sementic vectors in the embedding space
#     "Receive the user's query",                   # This step takes user's input
#     "Embed the user's query",                     # This step embeds the user input into sementic vectors in the embedding space
#     "Retrieve the most relevant chunks",          # This step compares the user input and document embeddings by cosine similarity to find the most relevant chunks
#     "Inject retrieved chunks into the prompt",    # This step fetchs the original chunk text and inject it to the prompt
#     "Generate a response from the LLM",           # This step pass the prompt through the LLM and generate the response
# ]

# --- Keyword RAG ---
import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    # Needed to add "your" to the stopwords list to correctly filter Q1
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "your", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]
# Keyword Q1
query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

print("=== Keyword Q1 ===")
simple_keyword_retrieval(query, documents, verbose=True)
print()

# Keyword Q2
query = "Do you have anything without caffeine?"

print("=== Keyword Q2 ===")
simple_keyword_retrieval(query, documents, verbose=True)
print()

# In this question, none of the document was chosen.
# The keyword RAG got it logically (with the code logic) as there is no explicit "caffine" mentioned in any document,
# but naturally it should've chosen `menu.txt`, as it contains all the info on what drinks are served.
# Vector embedded semantic search would've done the job much better.

# Keyword Q3
# With the code logic, I would predict none should be chosen, because none of the query word appeared in the documents.
query = "How do I sign up for rewards?"
print("=== Keyword Q3 ===")
simple_keyword_retrieval(query, documents, verbose=True)
print()
# The prediction is correct, although we do not want it to be that way. Without sementic match, it is really a hit or miss
# with whether the keywords matches exactly.

# --- Semantic RAG Concepts ---
# Semantic Q1
# 1) Q: What is a vector embedding? (1-2 sentences)
#    A: Vector embedding is a method to convert strings into a n-tuple of numbers, which represents the direction of the string
#       in that sementic vector space. Similar sementic string will have similar directions, which means their cosine similarity 
#       will be high.
# 2) Q: Two text chunks have cosine similarity scores of 0.85 and 0.30 with a given query. Which chunk is more relevant, and 
#       what does that number tell you about the relationship between the texts?
#    A: The chunk with cosine similarity score of 0.85 will be more relevant, as the higher the score, the smaller their angle 
#       difference, which means they points to similar direction, translating to similar sementic meaning.
# 3) Q: Why can semantic search find a relevant chunk even when none of the exact words from the query appear in the chunk?
#    A: Semantic search finds a relevant chunk by computing the cosine similarity score between the query and the embedded chunks.
#       As the embedded vector direction encodes the semantic meaning of a chunk, by comparing the direction using cosine similarity
#       score, we can retrieve the closest meaning chunk to the query even if none of the exact words appeared.

# Semantic Q2
# | Feature                    | Keyword RAG                       | Semantic RAG                                           |
# |----------------------------|-----------------------------------|--------------------------------------------------------|
# | What is compared?          | Exact word overlap                | High-dimensional embedding vectors (conceptual meaning)|
# | What is retrieved?         | Full document                     | Specific document chunks / text passages               |
# | Can it handle synonyms?    | No                                | Yes                                                    |
# | Storage format             | Plain text dictionary             | Vectorized database                                    |
# | Relevance score            | Number of overlapping keywords    | Cosine similarity score                                |

# --- LlamaIndex ---
brightleaf_path = "assignments_06/resources/brightleaf_pdfs"


# LlamaIndex Q1
# Load documents directly from PDFs in the folder
docs = SimpleDirectoryReader(
    brightleaf_path,
    file_extractor={ # Used PyMuPDFReader to correctly read the PDF, default pypdf parsed the PDF as gibberish.
        ".pdf": PyMuPDFReader()
    }
    ).load_data()

# Build a vector index automatically (handles chunking + embeddings)
index = VectorStoreIndex.from_documents(docs)
questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]
query_engine = index.as_query_engine(similarity_top_k=3)
print("=== Llama Q1 ===")
for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)

# For Q1 "What employee benefits does BrightLeaf offer?":
# - The response is highly relevant, it listed out most of the benefits program from the "employee_benefits.pdf".
# - The response was confident, there were no hesistation phrases at all, just a plain list of the programs.
# - For 2nd and 3rd top matching chunk, they were surprising from a different document at all, first one from "mission_statmemt.pdf",
#   while the second comes from "partnerships.pdf", where both are completely irrelevant to the question, yet still being at a high 
#   Similarity score. This is kind of strange.

# For Q2 "What are BrightLeaf's security policies?":
# - The response is highly relevant to the question, it is basically a copy and paste of the important info from `security_policy/pdf`.
# - The response was confident, no hesistation phrase at all.
# - Again, the second and third most relevant document extracted had a high cosine similarity score, but nothing wrong other than that.

# LlamaIndex Q2
print("=== Llama Q2 ===")
print("similarity_top_k=1")

query_engine = index.as_query_engine(similarity_top_k=1)
q = questions[0]
print(f"\nQ: {q}")
response = query_engine.query(q)
print("A:", response)

for node_with_score in response.source_nodes:
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print("-" * 30)

print("=== Llama Q2 ===")
print("similarity_top_k=5")

query_engine = index.as_query_engine(similarity_top_k=5)
q = questions[0]
print(f"\nQ: {q}")
response = query_engine.query(q)
print("A:", response)

for node_with_score in response.source_nodes:
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print("-" * 30)

# The response in similarity_top_k=5 is a bit more verbose than that of similarilty_top_k=1
# The content are basically identical, with similarity_top_k=5 included a little bit more detal.
# In that case, including more retrieved context is not always better, especially when each chunk already
# included everything the model needed to answer the full prompt.

# LlamaIndex Q3
print("=== Llama Q3 ===")
query_engine = index.as_query_engine(similarity_top_k=3)
question = [
    "How does the mission of the company align with how they treat their employees and finding their partnership?"
]
for q in question:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    print("-" * 30)
    
    for node_with_score in response.source_nodes:
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Retreived Chunks: {node_with_score.node.get_content()}")
        print("-" * 30)

# I expected the model would state the mission of the company first, then pick the relevant part of the `employee_benefits.pdf` and `partnership.pdf` to
# elaborate on each mission statements. However, the model just pick out some points from each document without stating the mission statements, let alone
# using examples on each one of them, instead it just answered vaguely, albeit choosing relevant info.

# To improve the system, we can implement custom default prompt by using the `PromptTemplate` function to enforce a more in-depth prompt.
# Also, we can introduce sub-query system to break down a complex prompt by the SubQuestionQueryEngine module.

# LlamaIndex Question 4
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

# Index and query from Q1
query_engine = index.as_query_engine(similarity_top_k=1)

# Create Judge LLM
llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

# Define evaluator
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

# BrightLeaf query
# Get response to query
q = "What employee benefits does BrightLeaf offer?"
response = query_engine.query(q)
print(f"BrightLeaf query: {q}")
print()

# Evaluate faithfulness and relevancy
faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))
print()

relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print("Relevancy Result: " + str(relevancy_result.score))
print("-" * 30)

# BrightLeaf query (Out of Context)
# Get response to query
q = "Which LLM does BrightLeaf use?"
response = query_engine.query(q)
print(f"Out of context query: {q}")
print()

# Evaluate faithfulness and relevancy
faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))
print()

relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print("Relevancy Result: " + str(relevancy_result.score))

# Q: What does a faithfulness score of 1.0 mean? What would a score of 0.0 indicate?
# A: A faithfulness score of 1.0 means the LLM retrieved info from the documents only, did not hallucinated any facts. A score of 0.0 means
# the LLM hallucinated facts in the response.

# Q: What does a relevancy score measure, and how is it different from faithfulness?
# A: The relevancy score measures whether the response from the LLM is relevant to the question, instead of whether it made up the answer or not.

# Q: Did the scores change between your two queries? If so, why do you think that happened?
# A: The Relevancy score changed from 1.0 to 0.0 across the first to the second prompt. I think that is becuase the second prompt asked for a question
#    with answer that could not be found in the documents. By staying true to the info in the database, the model searched for the most relevant chunk
#    and answered with the info in the chunk (obviously not relevant), hence getting a relevancy score of 0.0 but still 1.0 on faithfulness.

# Q: What is the "LLM-as-a-judge" approach, and why is it used for RAG evaluation instead of a simple accuracy metric?
# A: The "LLM-as-a-judge" approach uses another LLM as a judge to determine whether a response from the target LLM is truthful (without hallucination)
#    and relevant (is it answering the question). This approach is used instead of a simple accuracy metric because the responses from LLM are natural
#    languages instead of numeric values, so we cannot use ordinary accuracy score. The best way to evaluate natural language would be another LLM as
#    they are trained on natural language.