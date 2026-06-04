import torch, esm
import numpy as np

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

sequence = "MAKWGEGDPRWIVEERADATNVNNWHWTERDASNWSTDKLKTLFLAVQVQNEEGKCEVTEVSKLDGEASINNRKGKLIFFYEWSVKLNWTGTSKSGVQYKGHVEIPNLSDENSVDEVEISVSLAKDEPDTNLVALMKEEGVKLLREAMGIYISTLKTEFTQGMILPTMNGESVDPVGQPALKTEERKAKPAPSKTQARPVGVKIPTCKITLKETFLTSPEELYRVFTTQELVQAFTHAPATLEADRGGKFHMVDGNVSGEFTDLVPEKHIVMKWRFKSWPEGHFATITLTFIDKNGETELCMEGRGIPAPEEERTRQGWQRYYFEGIKQTFGYGARLF"

with torch.no_grad():
    _, _, tokens = batch_converter([("protein", sequence)])
    results = model(tokens, repr_layers=[33])
    emb = results["representations"][33][0, 1:-1, :].cpu().numpy()

np.save("test_embeddings/demo_protein.npy", emb)
print(f"Saved: {emb.shape}")