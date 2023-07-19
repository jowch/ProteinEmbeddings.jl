abstract type Model end
abstract type ESM1B_T33_650M_UR50S <: Model end
const ESM1 = ESM1B_T33_650M_UR50S

abstract type ESM2_T33_650M_UR50D <: Model end
abstract type ESM2_T36_3B_UR50D <: Model end
abstract type ESM2_T48_15B_UR50D <: Model end
const ESM2 = ESM2_T33_650M_UR50D


"""
    modelname(M <: Model)

Returns the name of the model represented by the type `M`.
"""
@inline modelname(::Type{ESM1B_T33_650M_UR50S}) = "esm1b_t33_650M_UR50S" 
@inline modelname(::Type{ESM2_T33_650M_UR50D}) = "esm2_t33_650M_UR50D"
@inline modelname(::Type{ESM2_T36_3B_UR50D}) = "esm2_t36_3B_UR50D"
@inline modelname(::Type{ESM2_T48_15B_UR50D}) = "esm2_t48_15B_UR50D"

"""
    modeldims(M <: Model)

Returns the embedding dimension of the model represented by the type `M`.
"""
@inline modeldims(::Type{ESM1B_T33_650M_UR50S}) = 1280
@inline modeldims(::Type{ESM2_T33_650M_UR50D}) = 1280
@inline modeldims(::Type{ESM2_T36_3B_UR50D}) = 2560
@inline modeldims(::Type{ESM2_T48_15B_UR50D}) = 5120

"""
    modeldepth(M <: Model)

Returns the number of layers in the model represented by the type `M`.
"""
@inline modeldepth(::Type{ESM1B_T33_650M_UR50S}) = 33
@inline modeldepth(::Type{ESM2_T33_650M_UR50D}) = 33
@inline modeldepth(::Type{ESM2_T36_3B_UR50D}) = 36
@inline modeldepth(::Type{ESM2_T48_15B_UR50D}) = 48


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

@inline (model::ProteinEmbedder)(xs...; kwargs...) = _embed(model, xs...; kwargs...)

# TODO: batching should be done on the python side
# TODO: add support for an arbitrary list of embedding layers

"""
    embed(model::ProteinEmbedder, x; batch_size)

Computes an embedding using the `model`. Input `x` can be any string-like type
or array of string-like types. If `x` is a single sequence, the embedding is
returned as a vector. If `x` is an array of `n` sequences, the embedding is returned
as an d x `n` matrix of embeddings. The `batch_size` keyword argument can be used
to control the number of sequences to process at a time. This is useful for large
arrays of sequences, where the memory usage can be controlled by setting a smaller
batch size. The default batch size is 500. The `batch_size` keyword argument is
ignored if `x` is a single sequence.
"""
@inline embed(model::ProteinEmbedder, x; kwargs...) = _embed(model, x; kwargs...)

function _embed(embedder::ProteinEmbedder{M}, sequences::Vector{String}; layers = [modeldepth(M)]) where {M <: Model}
    @assert all(0 .<= layers .<= modeldepth(M)) "Requires 1 <= layer <= $(modeldepth(M))"
    embedder.model.embed(sequences, layers)
end

@inline _embed(embedder::ProteinEmbedder, sequence; kwargs...) = _embed(embedder, [_format(sequence)]; kwargs...)
@inline _embed(embedder::ProteinEmbedder, sequences::AbstractVector; kwargs...) =
    _embed(embedder, vec(_format.(sequences)); kwargs...)

@inline _format(sequence::AbstractString) = string(sequence)
