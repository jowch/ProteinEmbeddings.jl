using FileIO, JLD2

@testset "Embedder" begin
    embedder = ProteinEmbedder{ESM2}()
    another_embedder = ProteinEmbedder(ESM2)

    @test typeof(embedder) == ProteinEmbedder{ESM2_T33_650M_UR50D}
    @test typeof(embedder) == typeof(another_embedder)
end

@testset "Embed" begin
    using BioSequences

    embedder = ProteinEmbedder{ESM2}()

    LL37 = "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES"
    aaLL37 = LongAA(LL37)

    y = load("ll37.jld2", "LL37")

    @test all(isapprox.(y, embed(embedder, LL37); atol = 1f-4))
    @test all(isapprox.(y, embed(embedder, aaLL37); atol = 1f-4))

    @test all(isapprox.(y, embedder(LL37); atol = 1f-4))
    @test all(isapprox.(y, embedder(aaLL37); atol = 1f-4))

    Y = [y y]

    @test all(isapprox.(Y, embed(embedder, [LL37, aaLL37]); atol = 1f-4))
    @test all(isapprox.(Y, embedder([LL37, aaLL37]); atol = 1f-4))

    @test all(isapprox.(Y, embed(embedder, LL37, aaLL37); atol = 1f-4))
    @test all(isapprox.(Y, embedder(LL37, aaLL37); atol = 1f-4))

    # check if batching returns the correct result
    @test all(isapprox.(Y, embed(embedder, [LL37, LL37]; batch_size = 1); atol = 1f-4))

    L5 = [LL37, LL37, LL37, LL37, LL37]
    Y5 = [y y y y y]
    @test all(isapprox.(Y5, embed(embedder, L5; batch_size = 2); atol = 1f-4))
end
