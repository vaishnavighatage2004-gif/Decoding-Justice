import os
from groq import Groq

GROQ_API_KEY = os.environ.get(
    "GROQ_API_KEY",
    ""
)

groq_client = Groq(
    api_key=GROQ_API_KEY
)


def ask_groq(
    prompt,
    model_name="llama-3.1-8b-instant",
    max_tokens=500,
    temperature=0.2
):

    prompt = prompt[:4000]

    response = groq_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature
    )

    return response.choices[0].message.content