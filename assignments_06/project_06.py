from dotenv import load_dotenv
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
import os
import logging

# Silence HTTP request logs from httpx / OpenAI API calls
logging.getLogger("httpx").setLevel(logging.WARNING)

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# Step 1: Setup

print("=== Step 1: Load the Documents ===")
docs_dir = Path("assignments_06/resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

# Step 2: Load the Documents
docs = SimpleDirectoryReader(
    docs_dir,
    # file_extractor={ # Used PyMuPDFReader to correctly read the PDF, default pypdf parsed the PDF as gibberish.
    #     ".pdf": PyMuPDFReader()
    # }
    ).load_data()
print("=== Step 2: Load the Documents ===")
print(f"Loaded {len(docs)} documents.")
print(f"File names: {[doc.metadata["file_name"] for doc in docs]}")

# Step 3: Build the Index and Query Engine
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k=3)
print("Index built successfully. Ready to answer questions.")

# Step 4: Query the Assistant
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]
for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    top_source_node = response.source_nodes[0]
    print("=== Top response result ===")
    print(f"Document name: {top_source_node.metadata["file_name"]}")
    print(f"Similarity Score: {top_source_node.score:.4f}")
    print(f"Text Snippet: {top_source_node.node.get_content()[:200]}...")
    print("-" * 30)
print()
# Q: Did the assistant sound confident and accurate? Did any of the answers surprise you?
# A: The assistant sounded confident and accurate. All the answers are correct, but the response to the question
#    "Do you offer any dairy-free milk options?" was surprising in the document it drawn from was actually the wrong document.
#    The correct document that contains that information is supposed to be "menu.txt", bu the model's highest similarity score
#    came from "seasonal_specials.txt".

# Step 5: Find a Failure
print("Step 5: Find a Failure")
hard_question = "How are the menus align with the company's belief?"
print(f"\nQ: {hard_question}")
response = query_engine.query(hard_question)
print("A:", response)
print("-" * 30)

source_nodes = response.source_nodes
for i, s in enumerate(source_nodes):
    print(f"=== Document {i + 1} ===")
    print(f"Document name: {s.metadata["file_name"]}")
    print(f"Similarity Score: {s.score:.4f}")
    print(f"Text Snippet: {s.node.get_content()[:200]}...")
    print("-" * 30)

# Q: What you asked and why you expected it to be hard
# A: The question was "How are the menus align with the company's belief?", it is hard because the question was vague, it didn't specify which menu (normal, seasonal, wholesale and catering).
#    Also, the question required a comprehension from multiple documents, including `our_story.txt`, `menu.txt`, `seasonal_specials.txt` and `wholesale_catering.txt`, before it can compare the 
#    menus and how they are related to the founders' beliefs.
# Q: What went wrong — wrong retrieval, missing information, the model guessed anyway?
# A: The retrieval was incorrect, the prompt asked for how the menus align with the beliefs, the model is expected to analyze the menus, and find the relevance between the choice of drinks to 
#    their beliefs (e.g. good coffee and strong communities go together, fair trade to farmers). Instead, the model is distracted to the loyalty program, which has no relation to the menus at
#    all, while not elaborting on how the menus align with the belief, no exmaple given what so ever.
# Q: When the retrieval failed, did the model's tone change — did it become less certain, or did it still sound confident even when it was wrong? What does this suggest about trusting AI-generated responses?
# A: The tone of the model didn't change at all, it was still as certain as before even when it was wrong. This means we can never be trusting AI-generated responses 100% without verification.
# Q: What you would change about the system to improve it?
# A: I would change the system so that if the model is confident that it requires multiple documents to answer the prompt, it will retrieve every document and summarize the needed passage, 
#    before re-reading the summary and answer the question. If we want even stronger behaviour, we can introduce the chain-of-thought (CoT) process, which demands the model to think in steps,
#    gathering all the information separately and get to the answer step by step.

# Step 6: Reflection
# Q: The lesson built semantic RAG manually — chunking, embedding, and indexing took many lines of code. How many lines did the equivalent LlamaIndex implementation take in your project? What does that 
#    tell you about the value of using a framework?
# A: The LlamaIndex basically takes 1 line to do all three: `VectorStoreIndex.from_documents(docs)`. This drastically reduces the size of the code and make it much easier to manage and maintain. Using a
#    framework improves the scalability, reduces the maintenance cost and simplify codes. This also speed up the development cycle a lot, and allow a quicker production of an MVP.
# Q: You have now built a system that answers questions from real documents. Describe a different use case — not a coffee shop — where this approach would add genuine value to a business or organization.
# A: This approach will also be an immense help in sales-focused business. These business usually have a lot of client files that describes their past purchase, point of contact, preferences and history.
#    An RAG system can help sales representatives to retrieve all the relevant data in a much easier and user-friendly form, say they can just ask "What product is more popular among clients?", "What is
#    the most satisfied aspect of our product?", then they can proceed with using the information to better assist the clients, or even build new client relationships.
# Q: What is one failure mode that RAG cannot fully prevent, even when retrieval is working correctly?
# A: Hallucination. With all the pre-training on known data, the model usually adapted a professional and confident tone when generating the responses. This stays even when they don't actually know the 
#    correct answer (both from the document and pre-trained data). At its root, an LLM is just a word chain generator, it tries to predict the most probable word token that appears next. Hence, it will
#    certainly hallucinate and give a confident tone on an incorrect answer.