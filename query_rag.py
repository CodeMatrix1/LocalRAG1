
from langchain.vectorstores.chroma import Chroma
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.prompts import ChatPromptTemplate
from Embeddings import embeddings_function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def load_model():
    model_path = "models/phi-2"
    
    print("🔁 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if torch.cuda.is_available():
        print("🚀 CUDA available: Loading model on GPU (FP16)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        print("⚠️ CUDA not available: Loading model on CPU (may be slow)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32
        )

    return tokenizer, model


def response(query, tokenizer, model):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings_function())
    retrieved = db.similarity_search_with_score(query, k=3)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in retrieved])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text,
        question=query
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=300)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    sources = [doc.metadata.get("id", None) for doc, _ in retrieved]
    print("\n✅ Response:")
    print(response_text)
    print("\n📚 Sources:", sources)
    return response_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()

    tokenizer, model = load_model()
    response(args.query_text, tokenizer, model)

if __name__ == "__main__":
    main()



#"online"
# from langchain.vectorstores.chroma import Chroma
# import argparse
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from langchain.prompts import ChatPromptTemplate
# from Embeddings import embeddings_function
# from langchain_community.llms.ollama import Ollama

# CHROMA_PATH = "chroma"

# model = Ollama(model="mistral")

# PROMPT_TEMPLATE = """
# Answer the question based only on the following context:

# {context}

# ---

# Answer the question based on the above context: {question}
# """

# def response(query):
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings_function())
#     retrieved = db.similarity_search_with_score(query, k=3)

#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in retrieved])
#     prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
#         context=context_text,
#         question=query
#     )

#     response_text = model.invoke(prompt)

#     sources = [doc.metadata.get("id", None) for doc, _ in retrieved]
#     print("\n✅ Response:")
#     print(response_text)
#     print("\n📚 Sources:", sources)
#     return response_text

# # -----------------------
# # CLI Entrypoint
# # -----------------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("query_text", type=str, help="The query text.")
#     args = parser.parse_args()
#     response(args.query_text)

# if __name__ == "__main__":
#     main()
