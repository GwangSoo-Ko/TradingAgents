"""Live smoke for Vertex Claude thinking/effort/max_tokens wiring.

Verifies that a real Vertex Model Garden Claude endpoint accepts the exact
request shape ``VertexAnthropicClient`` builds — ``max_tokens`` plus
``model_kwargs = {"thinking": {...}, "output_config": {"effort": ...}}`` — which
``ChatAnthropicVertex._default_params`` spreads into ``messages.create(**params)``.

To keep the surface tight (and avoid the heavy optional ``[vertex]`` extra), this
calls the raw ``AnthropicVertex`` SDK's ``messages.create`` directly: it is the
same underlying call ``ChatAnthropicVertex`` makes, so a green run confirms the
endpoint accepts our parameters end-to-end. Auth is ADC (no vendor API key).

Usage:
    GOOGLE_CLOUD_PROJECT=<your-gcp-project> python scripts/smoke_vertex_thinking.py
    python scripts/smoke_vertex_thinking.py --project <your-gcp-project> --location global

기본 프로젝트는 없다: GOOGLE_CLOUD_PROJECT 나 --project 가 꼭 있어야 한다. main.py 는
이제 Vertex 에서 돌지 않는다; 그 스모크는 scripts/smoke_anthropic_thinking.py(직접 API)다.

Requires: ``gcloud auth application-default login`` (or GOOGLE_APPLICATION_CREDENTIALS)
and the ``anthropic[vertex]`` extra (for google-auth). Tiny cost: 2 calls,
max_tokens=64. Exit code 0 = all accepted, 1 = a call was rejected, 2/3 = setup.
"""

from __future__ import annotations

import argparse
import os
import sys

# 계층형 Claude 블록의 (model, effort) 쌍 (main.py 가 직접 API 로 옮기기 전의 것;
# vertex_anthropic 공급자는 여전히 같은 노브를 받는다).
CASES = [
    ("claude-opus-5-5", "max"),
    ("claude-sonnet-5-5", "high"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
    args = parser.parse_args()

    if not args.project:
        print("ERROR: set GOOGLE_CLOUD_PROJECT or pass --project", file=sys.stderr)
        return 2

    try:
        import google.auth  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: google-auth missing ({exc!r}). Install: pip install -e \".[vertex]\"",
              file=sys.stderr)
        return 3

    import anthropic
    from anthropic import AnthropicVertex

    print(f"project={args.project} location={args.location} anthropic={anthropic.__version__}\n")
    client = AnthropicVertex(project_id=args.project, region=args.location)

    rc = 0
    for model, effort in CASES:
        label = f"{model} effort={effort} thinking=adaptive"
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=64,
                thinking={"type": "adaptive"},
                output_config={"effort": effort},
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            print(f"[ok]   {label}")
            print(f"       stop_reason={resp.stop_reason} out_tokens={resp.usage.output_tokens} text={text!r}")
        except TypeError as exc:
            rc = 1
            print(f"[FAIL] {label}  TypeError (SDK rejected a kwarg): {exc}")
        except Exception as exc:  # noqa: BLE001
            rc = 1
            status = getattr(exc, "status_code", None)
            suffix = f" [{status}]" if status else ""
            print(f"[FAIL] {label}  {type(exc).__name__}{suffix}: {str(exc)[:300]}")

    print("\nRESULT:", "all accepted" if rc == 0 else "at least one call rejected — see above")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
