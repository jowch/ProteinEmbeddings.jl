module BioSequencesExt

using ProteinEmbeddings
using BioSequences

@inline ProteinEmbeddings._format(sequence::LongAA) = ProteinEmbeddings._format(string(sequence))

end