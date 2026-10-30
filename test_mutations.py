import requests
import os
import sys

API_KEY = os.environ["RUNPOD_API_KEY"]

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
