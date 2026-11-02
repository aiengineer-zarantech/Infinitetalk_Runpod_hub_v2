import requests
import os
import sys

# Secrets only from the environment, never in this file: export RUNPOD_API_KEY=... before running
API_KEY = os.environ.get("RUNPOD_API_KEY", "").strip() or sys.exit("Set RUNPOD_API_KEY in the environment")

query = """
query {
  __type(name: "Mutation") {
    fields {
      name
    }
  }
}
"""

res = requests.post("https://api.runpod.io/graphql", headers={"Authorization": f"Bearer {API_KEY}"}, json={"query": query})
fields = res.json().get("data", {}).get("__type", {}).get("fields", [])
endpoint_muts = [f["name"] for f in fields if "endpoint" in f["name"].lower()]
print("Endpoint mutations:", endpoint_muts)
