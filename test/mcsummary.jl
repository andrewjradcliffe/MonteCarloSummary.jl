@testset "NaN behavior" begin
    A = [NaN 1 2; 3 4 5; 6 7 8]
    # `quantile` should throw an ArgumentError, but it gets mangled
    @test_throws CompositeException mcsummary(A, dim=1)
    @test_throws CompositeException mcsummary(A, dim=2)
end

@testset "±Inf behavior" begin
    A = [Inf 2; 3 4]
    B = mcsummary(A, dim=1)
    @test B[1, 1] == Inf
    # Inf - Inf generates NaN
    @test isnan(B[1, 2])
    @test isnan(B[1, 3])
    @test all(isinf, B[1, 4:end])

    A = [-Inf 2; 3 4]
    B = mcsummary(A, dim=1)
    @test B[1, 1] == -Inf
    # Inf - Inf generates NaN
    @test isnan(B[1, 2])
    @test isnan(B[1, 3])
    @test all(==(-Inf), B[1, 4:end])

    # Each row or column with an Inf will suffer same problem.
    A = [Inf 2 3; 4 5 6; 7 8 Inf; 9 10 11]
    B = mcsummary(A, dim=1)
    @test all(isnan, B[1, 2:3])
    @test all(isnan, B[3, 3:3])
    # However, other rows are still ok
    @test all(!isnan, B[2, :])

    B = mcsummary(A, dim=2)
    @test all(isnan, B[1, 2:3])
    @test all(isnan, B[3, 2:3])

    @test all(!isnan, B[2, :])
    @test all(!isnan, B[4, :])
end


@testset "views" begin
    for T in (Float32, Float64, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64)
        A = T.(collect(reshape(1:120, 2, 3, 4, 5)))
        a = view(A, :, :, 1, 1)

        for dim = (1, 2)
            for multithreaded = (true, false)
                @test mcsummary(a, dim=dim, multithreaded=multithreaded) ==
                    mcsummary(collect(a), dim=dim, multithreaded=multithreaded)
            end
        end

        b = view(A, :, 1, :, 1)
        for dim = (1, 2)
            for multithreaded = (true, false)
                @test mcsummary(b, dim=dim, multithreaded=multithreaded) ==
                    mcsummary(collect(b), dim=dim, multithreaded=multithreaded)
            end
        end

        c = view(A, :, 2, 3, :)
        for dim = (1, 2)
            for multithreaded = (true, false)
                @test mcsummary(c, dim=dim, multithreaded=multithreaded) ==
                    mcsummary(collect(c), dim=dim, multithreaded=multithreaded)
            end
        end
    end
end

@testset "probabilities" begin
    A = [1.0 2.0; 3.0 4.0]
    p = (0.0, 1.0)
    B = mcsummary(A, p, dim=1)
    for T in (Int, Rational)
        @test mcsummary(A, T.(p), dim=1) == B
    end
end
