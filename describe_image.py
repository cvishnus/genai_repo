"""Describe an image with a local Ollama model (via litellm)."""
import base64
import sys

import litellm

MODEL = sys.argv[2] if len(sys.argv) > 2 else "gemma:2b"
PATH = sys.argv[1] if len(sys.argv) > 1 else "image/pic.png"

with open(PATH, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

response = litellm.completion(
    model=f"ollama_chat/{MODEL}",
    api_base="http://localhost:11434",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
        ],
    }],
)
print(response.choices[0].message.content)
