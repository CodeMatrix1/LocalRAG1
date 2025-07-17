from query_rag import response
from langchain_community.llms.ollama import Ollama
import argparse

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("expected_response", type=str, help="expected_response.")
    args = parser.parse_args()
    query_text=args.query_text
    expected_response=args.expected_response
    query_and_validate(query_text,expected_response)


def query_and_validate(question: str, expected_response: str):
    response_text = response(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = Ollama(model="mistral")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print(f"Correct ,Response: {evaluation_results_str_cleaned}")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print(f"Incorrect, Response: {evaluation_results_str_cleaned}")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )

if __name__ == "__main__":
    main()