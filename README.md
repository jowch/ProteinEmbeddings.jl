# Protein Embeddings with FAIR-ESM

This package provides a Julia interface for
[FAIR-ESM](https://github.com/facebookresearch/esm), first published in [Rives
et al., _PNAS_, 2021](https://www.pnas.org/doi/full/10.1073/pnas.2016239118) as
ESM-1 and [Lin et al., _Science_,
2022](https://www.science.org/doi/full/10.1126/science.ade2574) as ESM-2. The
sequence representations produced by these models (as backbones) can be used as
starting points for downstream analyses.

## Installation

At the moment, this package is not yet registered in [Julia's General
registry](https://github.com/JuliaRegistries/General), though this is planned
for the future. For now, the way to use this package is through a clone of the
repository. Alternatively, you may use a [Local
Registry](https://github.com/GunnarFarneback/LocalRegistry.jl). If you or your
org already maintains a LocalRegistry, you may register this package to that
repository.

First, clone this repository. Then in Julia's Pkg mode, first run `instantiate`
followed by `build` to set the package up. This will also install a copy of
miniconda via [Conda.jl](https://github.com/JuliaPy/Conda.jl) along with some
Python dependencies including `pytorch` and `fair-esm`.

## Usage

### Generating Embeddings

The following is an example of how one can generate embeddings:

```julia
using ProteinEmbeddings

target = "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES"

model = ProteinEmbedder(ESM2_T36_3B_UR50D)
# or, equivalently,
# model = ProteinEmbedder{ESM2_T36_3B_UR50D}()

embedding = embed(model, target) # => 2560 dim Vector
```

Here, we instantiate an `ESM2_T36_3B_UR50D` model (see
[FAIR-ESM](https://github.com/facebookresearch/esm) for more information about
the available models). Then, we use the `embed` method to compute the model
representation for our `target` amino acid sequence.

The model to be used is specified by a type of the same name written with all capital letters (_e.g._, `ESM2_T36_3B_UR50D`).

In addition to single sequences, we can also produce embeddings for a `Vector` of sequences.

```julia
targets = [
    "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
    "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
    "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
    "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
]

embeddings = embed(model, targets)  # => 2560 x 4 Matrix
```

### Available Models

The list of available models can be found at the top of the
[`src/model.jl`](src/model.jl) file.

```julia
abstract type ESM1B_T33_650M_UR50S <: Model end
const ESM1 = ESM1B_T33_650M_UR50S

abstract type ESM2_T33_650M_UR50D <: Model end
abstract type ESM2_T36_3B_UR50D <: Model end
abstract type ESM2_T48_15B_UR50D <: Model end
const ESM2 = ESM2_T33_650M_UR50D
```

One can get the "name" of the model as appears in the
[FAIR-ESM](https://github.com/facebookresearch/esm) repository using the
`modelname` method. The size of each model's representations can be found using
the `modeldims` method.

```julia
modelname(ESM2_T36_3B_UR50D)
# => "esm2_t36_3B_UR50D"

modeldims(ESM2_T36_3B_UR50D)
# => 2048
```
