import Base: show
import Statistics: mean

using PyCall
using BioSequences

"""
    ProteinEmbedder()

A struct that can be used as a function to compute peptide sequence embeddings.
This struct contains `PyObject` references to the ESM-1b PyTorch model from
facebookresearch/esm (Rives, et al.). It can be called like a function.

# Examples
```julia-repl
julia> embed = ProteinEmbedder()
ProteinEmbedder(model = ESM-1b)

julia> embed(aa"")
```
"""
struct ProteinEmbedder
    name::String
    model::PyObject
    batch_converter::PyObject
end

function ProteinEmbedder()
    esm = pyimport("esm")

    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()

    ProteinEmbedder("ESM-1b", model, batch_converter)
end

function show(io::IO, embedder::ProteinEmbedder)
    print(io, "ProteinEmbedder(model = $(embedder.name))")
end

function (embedder::ProteinEmbedder)(data::Vector{Tuple{String, String}})
    _, _, batch_tokens = embedder.batch_converter(data)

    # TODO: replace current implementation with this one. Julia variables aren't
    # assigned in @pywith blocks at the moment.

    # torch = pyimport("torch")

    # @pywith torch.no_grad() begin
    #     results = embedder.model(batch_tokens, repr_layers=[33], return_contacts=true)
    # end

    py"""
    import torch

    with torch.no_grad():
        results = $(embedder.model)($batch_tokens, repr_layers=[33], return_contacts=True)
    """

    token_representations = py"results"["representations"][33].numpy()
    sequence_representations = map(enumerate(data)) do (i, (_, seq))
        mean(token_representations[i, 2 : length(seq) + 1, :]; dims = 1)
    end

    sequence_representations
end

function (embedder::ProteinEmbedder)(data::Vector{Tuple{String, LongAA}})
    embedder(map(data) do (name, sequence)
        (name, string(sequence))
    end)
end

function (embedder::ProteinEmbedder)(sequence::LongAA, name::String = "")
    embedder(string(sequence), name)
end

function (embedder::ProteinEmbedder)(sequence::String, name::String = "")
    embedder([(name, sequence)])
end
