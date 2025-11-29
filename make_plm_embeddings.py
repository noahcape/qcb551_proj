from main import parse_SCOPe_file
from transformers import EsmTokenizer, EsmModel, AutoTokenizer, AutoModel, BertTokenizer, BertModel
import torch
import pandas as pd

# NOTE: I renamed the csv's afterwards to remove t_ (# layers) and UR50D (training dataset)
# we typically refer to the esm2 instances just by _M (# of params)
model_name = (
    #'esm2_t6_8M_UR50D'
    #'esm2_t12_35M_UR50D'
    #'esm2_t30_150M_UR50D'
    'prot_bert'
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using {device}")

def plm_embed(model_name):
    if 'esm2' in model_name:
        tokenizer = EsmTokenizer.from_pretrained('facebook/' + model_name)
        model = EsmModel.from_pretrained('facebook/' + model_name, add_pooling_layer=False).to(device).eval()
        add_spaces = False
    elif model_name == 'prot_bert':
        tokenizer = BertTokenizer.from_pretrained('Rostlab/' + model_name, do_lower_case=False)
        model = BertModel.from_pretrained('Rostlab/' + model_name, add_pooling_layer=False).to(device).eval()
        add_spaces = True

    def embed_fn(p):
        if add_spaces:
            p = " ".join(p)

        tok = tokenizer(p, return_tensors='pt')
        for k, v in tok.items(): tok[k] = v.to(device)

        with torch.no_grad():
            h = model(**tok).last_hidden_state
        return h.squeeze()[1:-1].mean(axis=0).cpu().numpy().tolist()

    return embed_fn

if __name__ == "__main__":
    # structural class
    #parse_SCOPe_file(plm_embed(model_name), f'{model_name}_embeddings.csv')

    # subcellular localization
    df = pd.read_csv("functional_annotations.csv")
    df = df.rename(columns={'location': 'type'})

    embed_fn = plm_embed(model_name)
    df["embedding"] = df["Sequence"].apply(lambda seq: embed_fn(seq))
    
    df.to_parquet(f"{model_name}_functional.parquet")

