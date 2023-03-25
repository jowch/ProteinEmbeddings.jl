module ProteinEmbeddings

import Base: show

using PyCall

export
# models
Model,
ESM1,
ESM1B_T33_650M_UR50S,

ESM2,
ESM2_T33_650M_UR50D,
ESM2_T36_3B_UR50D,
ESM2_T48_15B_UR50D,

# embedder
ProteinEmbedder, embed, show


include("model.jl")

function __init__()
    py"""
    import torch
    import esm
    import numpy as np

    from typing import List, Optional

    
    class Embedder:
        def __init__(self, model: str):
            if model == "esm1b_t33_650M_UR50S":
                self.model, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
                self.batch_converter = self.alphabet.get_batch_converter()
                self.embedding_dims = 1280
                self.embedding_layer = 33
            elif model == "esm2_t33_650M_UR50D":
                self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
                self.batch_converter = self.alphabet.get_batch_converter()
                self.embedding_dims = 1280
                self.embedding_layer = 33
            elif model == "esm2_t36_3B_UR50D":
                self.model, self.alphabet = esm.pretrained.esm2_t36_3B_UR50D()
                self.batch_converter = self.alphabet.get_batch_converter()
                self.embedding_dims = 2560
                self.embedding_layer = 36
            elif model == "esm2_t48_15B_UR50D":
                self.model, self.alphabet = esm.pretrained.esm2_t48_15B_UR50D()
                self.batch_converter = self.alphabet.get_batch_converter()
                self.embedding_dims = 5120
                self.embedding_layer = 48

            self.device = "cpu"
            self.model.eval()

        def embed(self, seqs: List[str]):
            batch_labels, batch_strs, batch_tokens = self.batch_converter(seqs)
            batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

            # Extract per-residue representations (on CPU)
            with torch.no_grad():
                results = self.model(
                    batch_tokens,
                    repr_layers=[self.embedding_layer],
                    return_contacts=False
                )

            token_representations = results["representations"][self.embedding_layer].numpy()

            # Generate per-sequence representations via averaging
            # NOTE: token 0 is always a beginning-of-sequence token, so the
            # first residue is token 1.
            representations = np.zeros((self.embedding_dims, len(seqs)))

            for i, tokens_len in enumerate(batch_lens):
                representations[:, i] = token_representations[i, 1 : tokens_len - 1].mean(0)

            return representations
    """
end

end # ProteinEmbeddings