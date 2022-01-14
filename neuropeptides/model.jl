import Base: show
import Statistics: mean

using PyCall
using BioSequences


struct ProteinEmbedder
    name::String
    model::PyObject
    batch_converter::PyObject
end

function ProteinEmbedder()
    esm = pyimport("esm")

    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()

    ProteinEmbedder("esm1b_t33_650M_UR50S", model, batch_converter)
end

function show(io::IO, embedder::ProteinEmbedder)
    print(io, "ProteinEmbedder(model = $(embedder.name))")
end

function (embedder::ProteinEmbedder)(data::Vector{Tuple{String, LongAA}})
    data = map(data) do (name, seq)
        (name, String(seq))
    end

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

function (embedder::ProteinEmbedder)(sequence::LongAA, name::String = "")
    embedder([(name, sequence)])
end

@pywith pybuiltin("open")("Manifest.toml","r") as f begin
    manifest = f.readlines()
end
