module ProteinEmbeddings

import Base: show
import Statistics: mean

using PyCall
using BioSequences

export
# models
Model,
ESM1,
ESM1B_T33_650M_UR50S,

ESM2,
ESM2_T33_650M_UR50D,

ProteinEmbedder, embed, show


include("model.jl")

function __init__()
    py"""
    import torch
    import esm
    import numpy as np

    from typing import List

    
    class Embedder:
        def __init__(self, model: str):
            if model == "esm1b_t33_650M_UR50S":
                self.model, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
                self.batch_converter = self.alphabet.get_batch_converter()
                self.embedding_dims = 1280
            elif model == "esm2_t33_650M_UR50D":
                self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
                self.batch_converter = self.alphabet.get_batch_converter()
                self.embedding_dims = 1280

            # self.use_gpu = torch.cuda.is_available()
            # self.device = "cuda" if self.use_gpu else "cpu"
            self.device = "cpu"
            self.model.eval()

            # if self.use_gpu:
            #     self.model = self.model.cuda()

        def embed(self, seqs: List[str]):
            batch_labels, batch_strs, batch_tokens = self.batch_converter(seqs)
            batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

            # Extract per-residue representations (on CPU)
            with torch.no_grad():
                # if self.use_gpu:
                #     batch_tokens = batch_tokens.to(device = self.device, non_blocking=True)
                results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)

            token_representations = results["representations"][33].numpy()

            # Generate per-sequence representations via averaging
            # NOTE: token 0 is always a beginning-of-sequence token, so the
            # first residue is token 1.
            representations = []
            for i, tokens_len in enumerate(batch_lens):
                representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

            return np.array(representations)
    """
end

end # ProteinEmbeddings