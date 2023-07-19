module BioSequencesExt

using ProteinEmbeddings
using BioSequences

@inline ProteinEmbeddings._format(sequence::LongAA) = string(sequence)

end