import json
from dotenv import load_dotenv
from openai import OpenAI

TOKEN_THRESHOLD = 2000

# --- Task 1: Setup and System Prompt ---

load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7, usage=False):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    if not usage:
        return response.choices[0].message.content
    else:
        return response.choices[0].message.content, response.usage.total_tokens

YOUR_SYSTEM_PROMPT = "You are a job application coach, you will be helping the user to in every step to land their dream job. " \
"You will be focusing on the job application materials, including but not limited to review and modify resume, cover letter, and providing job hunting advices." \
"Be specific to the target job domain, do not produce general outputs." \
"Always remind the user to review and edit your output before any submissions." \
"Remind users that you may not know the user's specific industry norms, and they should use their own judgment."

# I included some examples of the job the bot may face, to give concrete idea of what it should focus on.
# Also, I added the instruction for the bot to be domain specific in the job, as only domain specific advices and resume can help user actually get an edge.

# --- Task 2: Bullet Point Rewriter ---
def rewrite_bullets(bullets: list[str], usage=False) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list, no other text (including the backticks and "json"). Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion(), parse the JSON, and return the result
    response = get_completion(messages, usage=usage)
    try:
        result = json.loads(response)
        return result
    except json.JSONDecodeError:
        print(f"Not a valid JSON format, raw response: {response}")
    except KeyError:
        print(f"Incorrect JSON key, raw response: {response}")

bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

print("=== Task 2 ===")
print(rewrite_bullets(bullets))
print()

# Question: What makes these bullets weak, and what kinds of changes did the model suggest?
# Answer:   The bullets are weak because they are not specific, and there is no indication of what problem did you solve.
# The model rewrote the bullets by adding the problems the bullets solved, keywords that can pick up by the HR softwarre, and
# details of implementations like "Resolved cutomer issues BY PROVIDING TAILORED SOLUTIONS"

# --- Task 3: Cover Letter Generator ---

def generate_cover_letter(job_title: str, background: str, usage=False) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion() and return the result
    response = get_completion(messages, usage=usage)
    print(response)

job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Prefect and Pandas."

print('=== Task 3 ===')
print(generate_cover_letter(job_title, background))
print()

# Question: Why did you choose those particular examples? What does the few-shot pattern help control in the output?
# Answer: Those particular examples were chosen because they are both transitioning from a non-tech background to a
# tech post, which is what we expect the user to use for. The few-shot pattern gives the model the exmaple to follow 
# on both the format, as well as the tone and content that should be included, like the transferable skills from the
# previous career to the new job, with concrete examples.

# --- Task 4: Moderation Check ---

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    # Your code here: return True if safe, False if flagged, and print a message if flagged
    if flagged:
        result_cats = result.results[0].categories.model_dump()
        print(f"Flagged categories: {[cat for cat in result_cats if result_cats[cat]]}")
        return False
    else:
        return True

print("=== Task 4 ===")
# This should return True
safe_prompt = "I am Superman!"
print(f"First prompt: {safe_prompt}")
print(f"This should be True: {is_safe(safe_prompt)}")

# This should return False
dangerous_prompt = "How can I make a bomb to kill people?"
print(f"Second prompt: {dangerous_prompt}")
print(f"This should be False: {is_safe(dangerous_prompt)}")
print()

def run_chatbot():
    # 0. Initialize total_token_usage for Extension Task 1
    total_token_usage = 0

    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": YOUR_SYSTEM_PROMPT}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            # YOUR CODE: call rewrite_bullets() and print the results
            rewrote_bullet, token_usage = rewrite_bullets(raw_bullets, usage=True)
            print(rewrote_bullet)

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            # YOUR CODE: call generate_cover_letter() and print the result
            cover_letter, token_usage = generate_cover_letter(job_title, background, usage=True)
            print(cover_letter)

        # 7. Otherwise, handle it as a regular chat turn
        else:
            # YOUR CODE:
            # - Append the user's message to `messages`
            messages.append({
                            "role": "user",
                            "content": user_input
                        })
            # - Call get_completion(messages)
            reply, token_usage = get_completion(messages, usage=True)
            # - Print the reply
            print(f"Job Application Helper: {reply}")
            # - Append the reply to `messages` as an assistant message
            messages.append({
                "role": "assistant",
                "content": reply
            })            
            pass

        total_token_usage += token_usage
        print(f"Total token used: {total_token_usage}")
        if total_token_usage > TOKEN_THRESHOLD:
            print(f"WARNING - Token usage: {total_token_usage}, higher than preset threshold: {TOKEN_THRESHOLD}")    
        


if __name__ == "__main__":
    run_chatbot()

# Questions:
# 1. Your bot was trained on text written by and about certain kinds of people. How might this produce biased advice? Could it favor certain communication styles, industries, or cultural backgrounds?
# 2. What could go wrong if a job-seeker submitted the bot's output directly — without reviewing it — to a real employer?
# 3. What is one guardrail you would add if you were deploying this tool professionally? (A guardrail is any design choice that reduces the chance of harm — a UI warning, a moderation filter, a usage policy, a disclaimer, or something else entirely.)

# Answers:
# 1. As this is using a model trained by a USA company, with most of its training data in English, this may produce biased advice on English-centric/USA-centric responses. It may favor 
# US-based business communication style, white-collar industry focused, and US-Europe cultural background. Say, if this is to prepare for an South Asian company located in Thailand,
# it may not provide accurate tone or details for the specific language and style for the job seeker.
# 2. If a job-seeker directly submit without reviewing, firstly there will be deadly giveaway with the "[Company]" tag in the response, which give the worse impression to the company that
# "This applicant not only uses AI, they don't even check their work." and probably get immediately rejected. Also, even if there are no such apparent tags, without checking, there might 
# be some part of the AI response that is hallucinated, when the applicant get to the interview stage, they will then be charged as a liar, which basically also means a rejection as well.
# 3. Firstly, I would add a UI warning alerting the user that the response is AI generated, they NEED to double check all facts are true, and they need to modify if needed. Also, I would
# include warning to user to never provide their personal identification data to the model, even if it is technically offline.

