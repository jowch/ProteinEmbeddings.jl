using FileIO, JLD2

@testset "Embedder" begin
    embedder = ProteinEmbedder{ESM2}()
    another_embedder = ProteinEmbedder(ESM2)

    @test typeof(embedder) == ProteinEmbedder{ESM2_T33_650M_UR50D}
    @test typeof(embedder) == typeof(another_embedder)
end

@testset "Embed" begin
    using BioSequences

    model_type = ESM2
    embedder = ProteinEmbedder{model_type}()

    LL37 = "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES"
    aaLL37 = LongAA(LL37)

    y = load("ll37.jld2", "LL37")
    y_33 = y[33:33, :, :]  # 1 x 1280 x 1

    # default layer is 33
    @test all(isapprox.(y_33, embed(embedder, LL37); atol = 1f-4))
    @test all(isapprox.(y_33, embed(embedder, aaLL37); atol = 1f-4))

    @test all(isapprox.(y_33, embedder(LL37); atol = 1f-4))
    @test all(isapprox.(y_33, embedder(aaLL37); atol = 1f-4))

    Y = reshape([y_33;;; y_33], 1, size(y_33, 2), 2)

    @test all(isapprox.(Y, embed(embedder, [LL37, aaLL37]); atol = 1f-4))
    @test all(isapprox.(Y, embedder([LL37, aaLL37]); atol = 1f-4))

    # check all layers
    @test all(isapprox.(y, embed(embedder, LL37; layers = 1:modeldepth(model_type)); atol = 1f-4))
end
