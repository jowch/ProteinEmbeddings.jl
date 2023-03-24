using FileIO, JLD2

@testset "Embedder" begin
    embedder = ProteinEmbedder{ESM2}()
    another_embedder = ProteinEmbedder(ESM2)

    @test typeof(embedder) == ProteinEmbedder{ESM2_T33_650M_UR50D}
    @test typeof(embedder) == typeof(another_embedder)

end

@testset "Embed" begin
    embedder = ProteinEmbedder{ESM2}()

    LL37 = "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES"
    aaLL37 = LongAA(LL37)

    y = load("ll37.jld2", "LL37")

    @test y == embed(embedder, LL37)
    @test y == embed(embedder, aaLL37)

    @test y == embedder(LL37)
    @test y == embedder(aaLL37)

    Y = permutedims([y y], (2, 1))

    @test Y == embed(embedder, [LL37, aaLL37])
    @test Y == embedder([LL37, aaLL37])

    @test Y == embed(embedder, LL37, aaLL37)
    @test Y == embedder(LL37, aaLL37)
end
