import os
from openai import OpenAI
import numpy as np

model = "nvidia/nemotron-3.5-embed-1b" #use model which supports embeddings
API_KEY = os.environ.get("NVIDIA_API_KEY")

if API_KEY is None:
    raise ValueError("NVIDIA_API_KEY environment variable is not set.")

client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=API_KEY)


def get_embeddings(text, input_type="query"):
    response = client.embeddings.create(model=model, input=text,extra_body={"input_type": input_type, "truncate": "NONE"})
    return response.data[0].embedding


SENTENCE = [
    "I deposited cash at the bank",
    "The bank approved my home loan",
    "We sat by the steep river bank",
    "The river bank was covered in mud",
    "Jenkins pipeline failed to build",
    "The Ansible playbook ran successfully",
    "Terraform plan shows infrastructure changes",
]

embeddings = {s: get_embeddings(s) for s in SENTENCE}
# print(f"Embeddings for individual words: {embeddings}")
dims = len(list(embeddings.values())[0])
print(f"Each sentence = vector of {dims} dimensions\n")


# ── 2. COSINE SIMILARITY ───────────────────────────────────────────────

def cosine_similarity(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

pairs = [
    ("I deposited cash at the bank",
     "The bank approved my home loan"),        # same meaning: financial bank

    ("We sat by the steep river bank",
     "The river bank was covered in mud"),      # same meaning: river bank

    ("I deposited cash at the bank",
     "We sat by the steep river bank"),         # same word "bank", different meaning

    ("Jenkins pipeline failed to build",
     "The Ansible playbook ran successfully"),  # both DevOps, different tools

    ("I deposited cash at the bank",
     "Terraform plan shows infrastructure changes"),  # totally unrelated
]

print("── Cosine Similarity ─────────────────────────────────────────────")
print(f"{'Score':>6}  Pair")
print("─" * 70)
for a, b in pairs:
    score = cosine_similarity(embeddings[a], embeddings[b])
    print(f"{score:.3f}   '{a[:35]}...'")
    print(f"         '{b[:35]}...'")
    print()


# ── 3. SEMANTIC SEARCH (mini RAG preview) ─────────────────────────────

def find_most_similar(query: str, corpus: dict) -> list[tuple]:
    q_emb = get_embeddings(query)
    scores = [(s, cosine_similarity(q_emb, emb)) for s, emb in corpus.items()]
    return sorted(scores, key=lambda x: x[1], reverse=True)

print("── Semantic Search ───────────────────────────────────────────────")
queries = [
    "money and finance",
    "DevOps automation tools",
    "nature and outdoors",
]
for query in queries:
    print(f"\nQuery: '{query}'")
    results = find_most_similar(query, embeddings)
    for sentence, score in results[:2]:
        print(f"  {score:.3f}  {sentence}")


# ── 4. VECTOR ARITHMETIC ──────────────────────────────────────────────

print("\n── Vector Arithmetic ─────────────────────────────────────────────")
words = {w: get_embeddings(w) for w in
         ["king", "man", "woman", "queen", "prince",
          "jenkins", "ansible", "terraform", "docker"]}

def vector_arithmetic(pos: list, neg: list, candidates: list) -> tuple:
    result = np.zeros(len(list(words.values())[0]))
    for w in pos:   result += np.array(words[w])
    for w in neg:   result -= np.array(words[w])
    scores = {c: cosine_similarity(result.tolist(), words[c]) for c in candidates}
    best = max(scores, key=scores.get)
    return best, scores

# Classic: king - man + woman = ?
best, scores = vector_arithmetic(
    pos=["king", "woman"], neg=["man"],
    candidates=["queen", "prince", "woman", "king"]
)
print(f"\nking - man + woman = '{best}'")
for w, s in sorted(scores.items(), key=lambda x: x[1], reverse=True):
    print(f"  {s:.3f}  {w}")

# DevOps: jenkins - pipeline_tool + config_management = ?
best2, scores2 = vector_arithmetic(
    pos=["ansible", "jenkins"], neg=["docker"],
    candidates=["terraform", "ansible", "jenkins", "docker"]
)
print(f"\nansible + jenkins - docker = '{best2}'")
for w, s in sorted(scores2.items(), key=lambda x: x[1], reverse=True):
    print(f"  {s:.3f}  {w}")

print("\n✓ Day 2 complete.")