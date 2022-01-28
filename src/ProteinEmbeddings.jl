module ProteinEmbeddings

import Base: show
import Statistics: mean

using PyCall
using BioSequences


export ProteinEmbedder, show, embed, contact

include("model.jl")

end