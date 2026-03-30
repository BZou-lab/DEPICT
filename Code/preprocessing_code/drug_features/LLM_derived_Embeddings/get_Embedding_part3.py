import openai
import pandas as pd

openai.api_key = '' # please fill in your API key

compounds_info = pd.read_csv("./RawData/compounds_with_yaml_part3.csv")

def compute_embeddings_batch(text_list):
    """
    text_list : list[str]  (≤ ~2000 short YAML docs keeps you under 8k tokens)
    Returns    : list[list[float]]  (one 512-dim vector per text)
    """
    resp = openai.embeddings.create(
        model="text-embedding-3-large",
        input=text_list,
        dimensions=512,
        encoding_format="float",
    )
    # API returns results in the same order as input
    return [d.embedding for d in resp.data]

batch_size = 15
embeds = []
texts   = compounds_info["augmented_yaml"].tolist()

for i in range(0, len(texts), batch_size):
    embeds.extend(compute_embeddings_batch(texts[i:i+batch_size]))

compounds_info["embedding"] = embeds

embedding_df = pd.DataFrame(compounds_info['embedding'].to_list(), index=compounds_info['pert_iname'])

# Optionally, you can rename the columns to something like "dim_0", "dim_1", ..., "dim_511"
embedding_df.columns = [f"dim_{i}" for i in range(embedding_df.shape[1])]

embedding_df.to_csv("./RawData/gptEmbed_Jul9_part3.csv")
