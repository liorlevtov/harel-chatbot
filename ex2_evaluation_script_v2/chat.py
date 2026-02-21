import ast
import json
import time
import uuid
import argparse
import urllib.request
import urllib.error

def query_chatbot(question, base_url, endpoint, model):
    """
    Sends a single question to the chatbot API (OpenAI Completions API compatible)
    and returns the assistant's reply and latency.
    """
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    headers = {"Content-Type": "application/json"}

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}]
    }

    data = json.dumps(payload).encode("utf-8")

    start_time = time.time()
    try:
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"Error querying chatbot: {e}")
        return "Error", [], time.time() - start_time

    latency = time.time() - start_time

    # Extract the answer and sources (OpenAI-like completions format)
    choice = response_data.get("choices", [{}])[0]
    answer = choice.get("text", "").strip()
    sources = choice.get("sources", [])
    
    return answer, sources, latency


def main():
    parser = argparse.ArgumentParser(description="Query a local chatbot via REST API.")
    parser.add_argument("--input", required=True, help="Path to the questions TXT file")
    parser.add_argument("--host", default="127.0.0.1", help="Chatbot host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Chatbot port (default: 8000)")
    parser.add_argument("--endpoint", default="/v1/chat/completions", help="Chatbot API endpoint (default: /v1/chat/completions)")
    parser.add_argument("--model", default="best-chatbot-ever", help="Chatbot model (default: best-chatbot-ever)")

    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    with open(args.input, "r", encoding="utf-8") as f:
        data = f.read()
        questions = ast.literal_eval(data)

    # Generate submission ID
    submission_id = str(uuid.uuid4()).split("-")[0]
    output_filename = f"{submission_id}_conversations.json"

    conversations = []

    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Querying chatbot...")
        answer, sources, bot_latency = query_chatbot(q, base_url, args.endpoint, args.model)

        conversation = {
            "turns": [
                {
                    "role": "user",
                    "content": q,
                    "latency": 0.0  # No latency for first user turn
                },
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "latency": bot_latency
                }
            ],
            "metadata": {}
        }
        conversations.append(conversation)
        print(f"[{i}/{len(questions)}] Done ({bot_latency:.2f}s)")

    # Save output
    with open(output_filename, "w", encoding="utf-8") as out:
        json.dump(conversations, out, ensure_ascii=False, indent=2)

    print(f"Saved {len(conversations)} conversations to {output_filename}")


if __name__ == "__main__":
    main()
