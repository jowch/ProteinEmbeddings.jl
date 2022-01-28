"""
    ProteinEmbedder()

A struct that can be used as a function to compute peptide sequence embeddings.
This struct contains `PyObject` references to the ESM-1b PyTorch model from
facebookresearch/esm (Rives, et al.). It can be called like a function.

# Examples
```julia-repl
julia> embed = ProteinEmbedder()
ProteinEmbedder(model = ESM-1b)

julia> embed(aa"LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES")
1-element Vector{Matrix{Float32}}:
 [-1.239253f-5 0.06010988 … 0.11572349 0.01225175]

julia> embed("K A <mask> I S Q")
1-element Vector{Matrix{Float32}}:
 [-0.051009744 -0.3597334 … 0.33589444 -0.34698522]
```
"""
struct ProteinEmbedder
    name::String
    embed::PyObject
    use_gpu::Bool
end

function ProteinEmbedder()
    py"""
    import torch
    import esm
    import numpy as np

    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()

    use_gpu = torch.cuda.is_available()

    if use_gpu:
        model = model.cuda()

    batch_size = 25

    def get_batches(data):
        for i in range(0, len(data), batch_size):
            yield data[i:min(i + batch_size, len(data))]

    def _embed_batched(data, return_contacts=False):
        model.eval()
        _, _, tokens = batch_converter(data)

        if use_gpu:
            tokens = tokens.to(device = "cuda", non_blocking=True)

        with torch.no_grad():
            results = model(tokens, repr_layers=[33], return_contacts=return_contacts)

        token_representations = results["representations"][33].cpu()
        embeddings = np.zeros((len(data), token_representations.shape[2]))

        for i, (_, sequence) in enumerate(data):
            embeddings[i, :] = token_representations[i, 1:len(sequence) + 1].mean(0).numpy()
            contacts.append(token_representations)

        if return_contacts:
            return embeddings, results["contacts"].cpu()
        else:
            return embeddings

    def embed(data):
        model.eval()

        if len(data) > batch_size:
            return np.vstack([_embed_batched(batch) for batch in get_batches(data)])
        else:
            return _embed_batched(data)
    """

    ProteinEmbedder("ESM-1b", py"embed", py"use_gpu")
end

function show(io::IO, embedder::ProteinEmbedder)
    print(io, "ProteinEmbedder(model = $(embedder.name), gpu = $(embedder.use_gpu))")
end

function (embedder::ProteinEmbedder)(sequences::Vector{String}, seqs_per_batch = 25)
    data = map(sequences) do sequence
        if length(sequence) > 1024
            @warn "Truncating sequence: $sequence"
            ("", sequence[1:1022])
        else
            ("", sequence)
        end
    end

    embedder.embed(data)
end

function (embedder::ProteinEmbedder)(sequences::Vector{LongSequence{AminoAcidAlphabet}})
    embedder(string.(sequences))
end

function (embedder::ProteinEmbedder)(sequence::LongSequence{AminoAcidAlphabet})
    embedder(string(sequence))
end

function (embedder::ProteinEmbedder)(sequence::String)
    embedder([sequence])
end
