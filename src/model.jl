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
    model::PyObject
    batch_converter::PyObject
    use_gpu::Bool
end

function ProteinEmbedder()
    torch = pyimport("torch")
    esm = pyimport("esm")

    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()

    use_gpu = torch.cuda.is_available()

    if use_gpu
        model = model.cuda()
    end

    ProteinEmbedder("ESM-1b", model, batch_converter, use_gpu)
end

function show(io::IO, embedder::ProteinEmbedder)
    print(io, "ProteinEmbedder(model = $(embedder.name), gpu = $(embedder.use_gpu))")
end

function (embedder::ProteinEmbedder)(sequences::Vector{String})
    data = map(sequences) do sequence
        ("", sequence)
    end

    embedder.model.eval()

    _, _, tokens = embedder.batch_converter(data)

    if embedder.use_gpu
        tokens = tokens.to(device = "cuda", non_blocking = true)
    end

    # TODO: replace current implementation with this one. Julia variables aren't
    # assigned in @pywith blocks at the moment.

    # @pywith torch.no_grad() begin
    #     results = embedder.model(batch_tokens, repr_layers=[33], return_contacts=true)
    # end

    py"""
    import pytorch

    with $torch.no_grad():
        results = $(embedder.model)($tokens, repr_layers=[33], return_contacts=True)
    """

    token_representations = py"results"["representations"][33].cpu().numpy()

    representations = zeros(length(sequences), size(token_representations, 3))

    for (i, sequence) in enumerate(sequences)
        representations[i, :] = mean(token_representations[i, 2 : length(sequence) + 1, :]; dims = 1)
    end

    representations
end

function (embedder::ProteinEmbedder)(sequences::Vector{LongAA})
    embedder(string.(sequences))
end

function (embedder::ProteinEmbedder)(sequence::LongAA)
    embedder(string(sequence))
end

function (embedder::ProteinEmbedder)(sequence::String)
    embedder([sequence])
end
