import openai
openai.api_key = "YOUR_API_KEY"

def generate_answer(query, chunks):
    context = "\n".join(chunks)

    prompt = f"""Answer using context:
{context}

Question: {query}"""

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response['choices'][0]['message']['content']
