AASequence = LongSequence{AminoAcidAlphabet}

abstract type Model end
abstract type ESM1B_T33_650M_UR50S <: Model end
const ESM1 = ESM1B_T33_650M_UR50S

abstract type ESM2_T33_650M_UR50D <: Model end
const ESM2 = ESM2_T33_650M_UR50D

"""
    modelname(M <: Model)

Returns the name of the model represented by the type `M` in python.
"""
@inline modelname(::Type{ESM1B_T33_650M_UR50S}) = "esm1b_t33_650M_UR50S" 
@inline modelname(::Type{ESM2_T33_650M_UR50D}) = "esm2_t33_650M_UR50D"

"""
    modeldims(M <: Model)

Returns the embedding dimension of the model represented by the type `M` in python.
"""
@inline modeldims(::Type{ESM1B_T33_650M_UR50S}) = 1280
@inline modeldims(::Type{ESM2_T33_650M_UR50D}) = 1280

"""
    ProteinEmbedder{<:Model}()
    ProteinEmbedder{}(<:Model)

A struct that can be used as a function to compute peptide sequence embeddings.
This struct contains `PyObject` references to the ESM-1b PyTorch model from
facebookresearch/esm (Rives, et al.). It can be called like a function.

# Examples
```julia-repl
julia> embedder = ProteinEmbedder{ESM2}()
ProteinEmbedder{ESM2_T33_650M_UR50D}

julia> another_embedder = ProteinEmbedder(ESM2)
ProteinEmbedder{ESM2_T33_650M_UR50D}

julia> typeof(embedder) == typeof(another_embedder)
true

julia> embedder("LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES")
1-element Vector{Matrix{Float32}}:
 [-1.239253f-5 0.06010988 … 0.11572349 0.01225175]

julia> embed(aa"LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES")
1-element Vector{Matrix{Float32}}:
 [-1.239253f-5 0.06010988 … 0.11572349 0.01225175]

julia> embed("K A <mask> I S Q")
1-element Vector{Matrix{Float32}}:
 [-0.051009744 -0.3597334 … 0.33589444 -0.34698522]
```
"""
struct ProteinEmbedder{M <: Model}
    model::PyObject
end

function ProteinEmbedder{M}() where {M<:Model}
    ProteinEmbedder{M}(py"Embedder($(modelname(M)))")
end

@inline ProteinEmbedder(::Type{M}) where {M <: Model} = ProteinEmbedder{M}()

function show(io::IO, ::ProteinEmbedder{M}) where {M}
    print(io, "ProteinEmbedder{$(M)}")
end

@inline (model::ProteinEmbedder)(xs...) = _embed(model, xs...)

"""
    embed(model::ProteinEmbedder, x)

Computes an embedding using the `model`. Input `x` can be any string-like type
or array of string-like types. If `x` is a single sequence, the embedding is
returned as a vector. If `x` is an array of `n` sequences, the embedding is returned
as an `n` x d matrix of embeddings.
"""
@inline embed(model::ProteinEmbedder, xs...) = _embed(model, xs...)

function _embed(embedder::ProteinEmbedder, sequence::AbstractVector{Tuple{String, String}})
    embedder.model.embed(sequence)
end

@inline _embed(embedder::ProteinEmbedder, xs...) = _embed(embedder, collect(xs))
@inline _embed(embedder::ProteinEmbedder, sequence) =
    _embed(embedder, [_format(sequence)])[1, :]

@inline _embed(embedder::ProteinEmbedder, sequences::AbstractVector) =
    _embed(embedder, _format.(sequences))

@inline _format(sequence) = _format(string(sequence))
@inline _format(sequence::String) = (string(), sequence)
