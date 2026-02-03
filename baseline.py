import os
import json
import numpy as np
from openai import OpenAI
import tqdm

# -----------------------------
# Configuration
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CHAT_MODEL = "gpt-4o" 
CHAT_MODEL = "gpt-5.2" 
EMBED_MODEL = "text-embedding-3-large"

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """You are a helpful assistant. Please answer questions accurately and concisely. 
Answer in the same language as the question. Answers should be one sentence long.
Base answers ONLY on data from Harel insurance website: https://www.harel-group.co.il/insurance.
If the question is not related to insurance or if the answer cannot be found on the Harel website, respond with "I don't know".
You must cite sources for every factual statement.
If no citation is available, say you cannot verify the claim.
"""

# -----------------------------
# Utility functions
# -----------------------------
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_embedding(text):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding


def get_chat_response(prompt):

    response = client.responses.create(
        model=CHAT_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        tools=[
            {"type": "web_search"}
        ],
    )
    
    res = ""
    try:
        for item in response.output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        res += content.text + "\n"
                        # for ann in content.annotations:
                        #     print("Citation:", ann.url)
    except Exception as e:
        print(f"Error retrieving citations: {e}")

    return res


# -----------------------------
# Predefined prompts & references
# -----------------------------
with open('ex2.json', 'r', encoding='utf-8') as f:
    test_samples = json.load(f)

test_cases = []

for insurance_type, samples in test_samples.items():
    for sample in samples:
        test_cases.append({'prompt': sample['שאלה'], 'reference': sample['תשובה'], 'type': insurance_type})


# -----------------------------
# Main evaluation loop
# -----------------------------
results = []

for case in tqdm.tqdm(test_cases):
    try:
        prompt = case["prompt"]
        reference = case["reference"]

        generated = get_chat_response(prompt)

        gen_emb = get_embedding(generated)
        ref_emb = get_embedding(reference)

        similarity = cosine_similarity(gen_emb, ref_emb)

        results.append({
            "type": case["type"],
            "prompt": prompt,
            "generated_response": generated,
            "reference_response": reference,
            "similarity_score": similarity
        })

    except Exception as e:
        print(f"Error processing case with prompt '{case['prompt']}': {e}")

# -----------------------------
# Output results
# -----------------------------
for r in results:
    print("=" * 60)
    print(f"Prompt:\n{r['prompt']}\n")
    print(f"Generated:\n{r['generated_response']}\n")
    print(f"Reference:\n{r['reference_response']}\n")
    print(f"Similarity Score: {r['similarity_score']:.4f}")

# Save results to a JSON file
with open(f'evaluation_results_{CHAT_MODEL}.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)