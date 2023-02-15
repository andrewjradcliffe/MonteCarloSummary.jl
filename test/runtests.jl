using MonteCarloSummary
using Test

const tests = [
    "mcsummary.jl",
]

for t in tests
    @testset "Test $t" begin
        include(t)
    end
end
