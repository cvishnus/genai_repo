"""Call Claude models through the Delivery Hero LiteLLM gateway using the litellm SDK.

Prerequisite: the local reverse proxy must be running
(`brew services start dp-devinfra` or `dp-devinfra litellm proxy`, v1.24.0+).
"""
import argparse
import json
import os
import sys
import urllib.request

import litellm
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://localhost:36253")
API_KEY = os.environ.get("LITELLM_API_KEY", "cloudflare")


def _model_name(model: str) -> str:
    # "litellm_proxy/" tells the SDK to talk to a LiteLLM proxy at api_base.
    return model if model.startswith("litellm_proxy/") else f"litellm_proxy/{model}"


def _messages(prompt: str, system: str | None) -> list[dict]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def list_models() -> list[str]:
    req = urllib.request.Request(
        f"{BASE_URL}/v1/models", headers={"Authorization": f"Bearer {API_KEY}"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return sorted(m["id"] for m in json.load(resp)["data"])


def chat(prompt: str, model: str, system: str | None = None, max_tokens: int = 1024) -> str:
    response = litellm.completion(
        model=_model_name(model),
        messages=_messages(prompt, system),
        api_base=BASE_URL,
        api_key=API_KEY,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def stream_chat(prompt: str, model: str, system: str | None = None, max_tokens: int = 1024):
    stream = litellm.completion(
        model=_model_name(model),
        messages=_messages(prompt, system),
        api_base=BASE_URL,
        api_key=API_KEY,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def main() -> None:
    parser = argparse.ArgumentParser(description="Call Claude via the LiteLLM gateway")
    parser.add_argument("prompt", nargs="?")
    parser.add_argument("--model", default=os.environ.get("LITELLM_MODEL"))
    parser.add_argument("--system")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    try:
        if args.list_models:
            print("\n".join(list_models()))
            return
        if not args.prompt:
            parser.error("prompt is required unless --list-models is used")
        if not args.model:
            parser.error("set --model or LITELLM_MODEL (use --list-models to see options)")

        if args.stream:
            for token in stream_chat(args.prompt, args.model, args.system, args.max_tokens):
                print(token, end="", flush=True)
            print()
        else:
            print(chat(args.prompt, args.model, args.system, args.max_tokens))
    except (litellm.exceptions.APIConnectionError, urllib.error.URLError):
        sys.exit(
            f"Cannot reach {BASE_URL}. Start the proxy: `brew services start dp-devinfra` "
            "or `dp-devinfra litellm proxy`."
        )


if __name__ == "__main__":
    main()
