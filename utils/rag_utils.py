import chromadb
import torch
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModel
)

client = chromadb.PersistentClient(
    path="./legal_db"
)

collection = client.get_or_create_collection(
    name="legal_docs"
)

tokenizer = AutoTokenizer.from_pretrained(
    "law-ai/InLegalBERT"
)

model = AutoModel.from_pretrained(
    "law-ai/InLegalBERT"
)


def embed_text(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    vec = torch.nn.functional.normalize(
        outputs.last_hidden_state[:, 0, :],
        p=2,
        dim=1
    )

    return vec.cpu().numpy().flatten().tolist()


def add_doc_to_chroma(text):

    emb = embed_text(text)

    collection.add(
        documents=[text],
        embeddings=[emb],
        ids=[str(np.random.randint(1000, 9999))]
    )


def chroma_search(query, k=3):

    q_emb = embed_text(query)

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=k
    )

    return res