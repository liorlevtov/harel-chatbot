import os
import json
import time
import numpy as np
from openai import OpenAI
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


NEBIUS_API_KEY = 'v1.CmMKHHN0YXRpY2tleS1lMDBldHBiMzYyY3JuMngxcXYSIXNlcnZpY2VhY2NvdW50LWUwMGtieTJqN2p6ajljYXJuczILCKeFo8wGEL26q1s6DAiliLuXBxDA0NfVA0ACWgNlMDA.AAAAAAAAAAGNSitzi_mVnjLQCBIM0OeiIYDXqXQJwYLBqfTkFWqTVMAo_oZW5fhZCxCmfkh7rz9-U72xMILMxWQ7a8fAxkYG'

MODEL_CHOICES = {
    'ds': "deepseek-ai/DeepSeek-R1-0528-fast",
    'qwen': "Qwen/Qwen3-32B-fast", 
    'gemma2tiny': "google/gemma-2-2b-it",
    'gemma3': 'google/gemma-3-27b-it-fast',
}

MODELS = {}
MODELS['classification'] = MODEL_CHOICES['gemma3']  # model for classifying question category

os.environ['NEBIUS_API_KEY'] = NEBIUS_API_KEY


convert_category = {
    'דירה': 'Apartment Insurance',
    'עסקים': 'Business Insurance',
    'רכב': 'Car Insurance',
    'שיניים': 'Dental Insurance',
    'בריאות': 'Health Insurance',
    'חיים': 'Life Insurance',
    'משכנתא': 'Mortgage Insurance',
    'נסיעות': 'Travel Insurance',
    'אחר': 'General Insurance',
}



client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=NEBIUS_API_KEY,
)


def prep():
    # prepare a separate vector db for each insurance category
    pass


def classify_question(question):
    """ Classify question into insurance category using a simple keyword-based approach or a small classification model """

    SYS_PROMPT = """
        You are a helpful assistant that classifies insurance-related questions into the following categories:
        1. Apartment Insurance
        2. Business Insurance
        3. Car Insurance
        4. Dental Insurance
        5. Health Insurance
        6. Life Insurance
        7. Mortgage Insurance
        8. Travel Insurance
        If the question does not belong to any of these categories, classify it as "Other".
        Answer with the category name only.
    """
    # start_time = time.time()
    completion = client.chat.completions.create(
        model = MODELS['classification'],
        messages=[
            {
                "role": "system",
                "content": SYS_PROMPT
            },
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0
    )
    category = completion.choices[0].message.content.strip()
    if len(category.split(' ')) > 2:
        category = ' '.join(category.split()[-2:])  # take only the last two words if multiple are returned (e.g. due to thinking)
    if category not in convert_category.values():
        category = "Other"
    # print(f"Classified category: {category}")
    # print(f"Classification time: {time.time() - start_time:.2f} seconds")
    return category
    

def flow(question):
    # 0. ensure vector dbs exist and are up to date
    # 1. classify question into insurance category
    category = classify_question(question)
    # 2. retrieve relevant documents from the corresponding vector db
    # 3. generate answer based on retrieved documents
    # 4. check if answer is satisfactory, if not, refine answer or retrieve more documents
    # 5. return final answer and time taken
    return category


def get_test_questions(test_q_file='ex2.json'):
    with open(test_q_file, 'r', encoding='utf-8') as f:
        test_samples = json.load(f)

    test_cases = []

    for insurance_type, samples in test_samples.items():
        for sample in samples:
            test_cases.append({'prompt': sample['שאלה'], 'reference': sample['תשובה'], 'type': insurance_type})

    return test_cases


def evaluate_category(results):
    y_true = [r['true_category'].replace('Insurance', '').strip() for r in results]
    y_pred = [r['predicted_category'].replace('Insurance', '').strip() for r in results]

    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap='Blues')
    plt.title(f'Detected Category CM | {MODELS["classification"]}')
    plt.savefig(f'props/category_confusion_matrix_{MODELS["classification"]}.png'.replace('/', '_'))


def run():
    questions = get_test_questions()
    results = []

    for q in questions:
        # print("Question:", q['prompt'])
        answer = flow(q['prompt'])
        results.append({
            'prompt': q['prompt'],
            'predicted_category': answer,
            'true_category': convert_category[q['type']]
        })
    evaluate_category(results)


if __name__ == "__main__":    
    run()