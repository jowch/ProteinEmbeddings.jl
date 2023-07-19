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
ProteinEmbedder, embed, show, modelname, modeldims, modeldepth


include("model.jl")

function __init__()
    py"""
    import torch
    import esm
    import numpy as np

    from typing import List, Optional


    TRUNCATION_SEQ_LENGTH = 1022
    
    class Embedder:
        def __init__(self, model: str):
            if model == "esm1b_t33_650M_UR50S":
                self.model, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
                self.batch_converter = self.alphabet.get_batch_converter()
                self.embedding_dims = 1280
                self.default_embedding_layer = 33
            elif model == "esm2_t33_650M_UR50D":
                self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
                self.batch_converter = self.alphabet.get_batch_converter()
                self.embedding_dims = 1280
                self.default_embedding_layer = 33
            elif model == "esm2_t36_3B_UR50D":
                self.model, self.alphabet = esm.pretrained.esm2_t36_3B_UR50D()
                self.batch_converter = self.alphabet.get_batch_converter()
                self.embedding_dims = 2560
                self.default_embedding_layer = 36
            elif model == "esm2_t48_15B_UR50D":
                self.model, self.alphabet = esm.pretrained.esm2_t48_15B_UR50D()
                self.batch_converter = self.alphabet.get_batch_converter()
                self.embedding_dims = 5120
                self.default_embedding_layer = 48

            self.embedding_types = ["mean", "per_tok"]
            self.device = "cpu"
            self.model.eval()

        def _embed(
                self,
                seqs: List[str], repr_layers: List[int], include: List[str] = ["mean"],
                toks_per_batch = 4096, extra_toks_per_seq = 1, truncation_seq_length = 1022
            ):
            # Adapted from esm/scripts/extract.py
            assert all(-(self.model.num_layers + 1) <= i <= self.model.num_layers for i in repr_layers)
            repr_layers = [(i + self.model.num_layers + 1) % (self.model.num_layers + 1) for i in repr_layers]

            dataset = esm.FastaBatchedDataset(range(len(seqs)), seqs)
            batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=extra_toks_per_seq)
            data_loader = torch.utils.data.DataLoader(
                dataset, collate_fn=self.batch_converter, batch_sampler=batches
            )

            results = [None] * len(seqs)

            with torch.no_grad():
                for batch_idx, (labels, strs, toks) in enumerate(data_loader):
                    # if torch.cuda.is_available() and not args.nogpu:
                    #     toks = toks.to(device="cuda", non_blocking=True)
        
                    out = self.model(toks, repr_layers=repr_layers, return_contacts=False)
        
                    logits = out["logits"].to(device="cpu")
                    representations = {
                        layer: t.to(device="cpu") for layer, t in out["representations"].items()
                    }
        
                    for i, label in enumerate(labels):
                        truncate_len = min(truncation_seq_length, len(strs[i]))
                        # Call clone on tensors to ensure tensors are not views into a larger representation
                        # See https://github.com/pytorch/pytorch/issues/1995
                        result = {}

                        if "per_tok" in include:
                            result["representations"] = {
                                layer: t[i, 1 : truncate_len + 1].clone().numpy()
                                for layer, t in representations.items()
                            }
                        if "mean" in include:
                            # Generate per-sequence representations via averaging
                            # NOTE: token 0 is always a beginning-of-sequence token, so the
                            # first residue is token 1.
                            result["mean_representations"] = {
                                layer: t[i, 1 : truncate_len + 1].mean(0).clone().numpy()
                                for layer, t in representations.items()
                            }

                        results[int(label)] = result

            return results

        def embed(self, seqs: List[str], layers: List[int], include: List[str] = ["mean"]):
            results = self._embed(seqs, repr_layers=layers, include=include)
            representations = np.zeros((len(layers), self.embedding_dims, len(seqs)))

            for i, layer in enumerate(layers):
                for j, result in enumerate(results):
                    representations[i, :, j] = result["mean_representations"][layer]

            return representations
    """
end

end # ProteinEmbeddings