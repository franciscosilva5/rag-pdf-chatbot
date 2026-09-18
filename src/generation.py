from openai import OpenAI


MODEL_NAME = "gpt-5.6-luna"


def build_context(results):
    parts = []

    for result in results:
        parts.append(
            f"[Page {result['page']}]\n{result['text']}"
        )

    return "\n\n".join(parts)


def generate_answer(
    query,
    results,
    api_key,
):
    client = OpenAI(api_key=api_key)

    context = build_context(results)

    prompt = f"""
You are a retrieval-augmented assistant.

Answer the user's question using ONLY the document context below.

Rules:
- Do not use outside knowledge.
- If the answer is not supported by the context, say exactly:
  "I don't have enough information in the document to answer that."
- Be concise and factual.
- Cite supporting pages using the format [Page X].
- Do not invent citations.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}
"""

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
    )

    return response.output_text
