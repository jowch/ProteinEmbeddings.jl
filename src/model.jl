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

    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()

    use_gpu = torch.cuda.is_available()

    if use_gpu:
        model = model.cuda()

    def embed(data):
        _, _, tokens = batch_converter(data)

        if use_gpu:
            tokens = tokens.to(device = "cuda", non_blocking = True)

        model.eval()

        with torch.no_grad():
            results = model(tokens, repr_layers=[33], return_contacts=False)

        return results["representations"][33].cpu().numpy()
    """

    ProteinEmbedder("ESM-1b", py"embed", py"use_gpu")
end

function show(io::IO, embedder::ProteinEmbedder)
    print(io, "ProteinEmbedder(model = $(embedder.name), gpu = $(embedder.use_gpu))")
end

function _embed_sequences(embedder::ProteinEmbedder, data::AbstractArray{Tuple{String, String}, 1})
    token_representations = embedder.embed(data)
    embeddings = zeros(length(data), size(token_representations, 3))

    for (i, (_, sequence)) in enumerate(data)
        embeddings[i, :] = mean(token_representations[i, 2 : length(sequence) + 1, :]; dims = 1)
    end

    embeddings
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

    if length(sequences) > seqs_per_batch
        batches = [view(data, i:min(i+seqs_per_batch-1, length(sequences)))
                   for i = 1:seqs_per_batch:length(sequences)]

        representations = map(enumerate(batches)) do (i, batch)
            _embed_sequences(embedder, batch)
        end

        return vcat(representations...)
    else
        return _embed_sequences(embedder, data)
    end
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
