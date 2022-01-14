module ProteinEmbeddings

import Base: show
import Statistics: mean

using PyCall
using BioSequences


export ProteinEmbedder, show

include("model.jl")

end