module TimePtycho
import FFTW
import PhysicalConstants: CODATA2018
import Unitful: ustrip
import LinearAlgebra: ldiv!, mul!
import Random: randperm, rand

const c = ustrip(CODATA2018.c_0)

fwhm_to_σ(fwhm; power=2) = fwhm / (2 * (2 * log(2))^(1 / power))

gauss(x, σ; x0=0, power=2) = exp(-1/2 * ((x-x0)/σ)^power)
gauss(x; x0=0, power=2, fwhm) = gauss(x, fwhm_to_σ(fwhm; power=power), x0 = x0, power=power)

function mock_trace_SHG(τfwhm, λ0, τrange, ωmin, ωmax; Nτ=101, dispersion=Float64[])
    trange = 8τfwhm
    ω0 = 2π*c/λ0
    f0 = ω0/2π

    fmax = ωmax/2π
    fmaxrel = fmax-f0 # maximum frequency shift
    δt = 1/2fmaxrel # Nyquist limit
    samples = 2^(ceil(Int, log2(trange/δt)))
    trange_even = δt*samples
    δω = 2π/trange_even

    Nω = collect(range(0, length=samples))
    t = @. (Nω-samples/2)*δt # time grid, centre on 0
    v = @. (Nω-samples/2)*δω # freq grid relative to ω0
    # v = FFTW.fftshift(v)
    ω = v .+ ω0

    φ = zeros(length(ω))
    for (n, φi) in enumerate(dispersion)
        φ .+= FFTW.fftshift(v).^(n+1)./factorial(n+1) * φi
    end

    function Et(τ)
        amp = sqrt.(gauss.(t; fwhm=τfwhm))
        Ef = FFTW.fft(amp)
        Ef .*= exp.(1im .* (φ .+ FFTW.fftshift(v).*τ))
        FFTW.ifft(Ef)
    end
    Et0 = Et(0)

    trace = zeros((samples, Nτ))
    τ = collect(range(-τrange/2, τrange/2, Nτ))
    for (ii, τi) in enumerate(τ)
        Esig = Et0 .* Et(τi)
        trace[:, ii] .= abs2.(FFTW.fft(Esig))
    end

    trace = FFTW.fftshift(trace, 1)
    return τ, v, trace, Et0
end

mutable struct Ptychographer{gT, iT, ftT}
    geometry::gT
    interaction::iT
    FT::ftT
    τ::Vector{Float64}
    ω::Vector{Float64}
    measured::Matrix{Float64}
    measA::Matrix{Float64}
    reconstructed::Matrix{Float64}
    testpulse::Vector{ComplexF64}
    gatepulse::Vector{ComplexF64}
    gateshift::Vector{ComplexF64}
    iters::Int
    errors::Vector{Float64}
    buffer::Vector{ComplexF64}
    ψ::Vector{ComplexF64}
    ψp::Vector{ComplexF64}
    φ::Vector{Float64}
    diff::Vector{ComplexF64}
end

function Ptychographer(interaction, geometry, τ, ω, trace)
    testpulse = rand(ComplexF64, length(ω))
    FT = FFTW.plan_fft(testpulse, 1, flags=FFTW.PATIENT)
    inv(FT) # plan inverse FFT
    s = sign.(trace)
    measA = @. s * sqrt(abs(trace))
    rec = zeros(Float64, size(trace))
    iters = 0
    errors = Float64[]
    gatepulse = rand(ComplexF64, length(ω))
    gateshift = zeros(ComplexF64, length(ω))
    buffer = copy(gateshift)
    ψ = copy(gateshift)
    ψp = copy(gateshift)
    diff = copy(gateshift)
    φ = zeros(Float64, length(ω))

    Ptychographer(geometry, interaction, FT, τ, FFTW.fftshift(ω),
                  trace, measA, rec, testpulse, gatepulse, gateshift,
                  iters, errors, buffer, ψ, ψp, φ, diff)

end

function doiter!(pt::Ptychographer; random_order=true, α=0.8)
    for ii in randperm(size(pt.measured, 2))
        pt.gatepulse .= pt.testpulse
        pt.gateshift .= pt.gatepulse
        τshift!(pt, pt.gateshift, pt.τ[ii])
        signal_field!(pt.ψ, pt.interaction, pt.testpulse, pt.gateshift)

        mul!(pt.buffer, pt.FT, pt.ψ)
        pt.φ .= angle.(pt.buffer)
        pt.buffer .= pt.measA[:, ii] .* exp.(1im .* pt.φ)
        ldiv!(pt.ψp, pt.FT, pt.buffer)

        @. pt.diff = pt.ψp - pt.ψ

        m = maximum(pt.gateshift) do gi
            abs2(gi)
        end

        for jj in eachindex(pt.testpulse)
            pt.testpulse[jj] += pt.diff[jj] * α * conj(pt.gateshift[jj])/m
        end
    end
    pt.iters += 1
    pt.gatepulse .= pt.testpulse
    update_recon!(pt)
end

function update_recon!(pt::Ptychographer)
    for ii in axes(pt.measured, 2)
        τshift!(pt.gateshift, pt, pt.gatepulse, pt.τ[ii])
        signal_field!(pt.ψ, pt.interaction, pt.testpulse, pt.gateshift)
        mul!(pt.buffer, pt.FT, pt.ψ)
        for jj in axes(pt.measured, 1)
            pt.reconstructed[jj, ii] = abs2(pt.buffer[jj])
        end
    end
    ηnom = sum(eachindex(pt.measured)) do ii
        pt.measured[ii] * pt.reconstructed[ii]
    end
    ηden = sum(pt.reconstructed) do ri
        ri^2
    end
    η = ηnom/ηden
    e = sum(eachindex(pt.reconstructed)) do ii
        (pt.measured[ii] - η*pt.reconstructed[ii])^2
    end
    push!(pt.errors, sqrt(1/length(pt.reconstructed) * e))
end

function τshift!(Et, τ, ω, FT, buffer)
    mul!(buffer, FT, Et)
    @. buffer *= exp(1im * ω * τ)
    ldiv!(Et, FT, buffer)
end

function τshift!(dest, Et, τ, ω, FT, buffer)
    mul!(buffer, FT, Et)
    @. buffer *= exp(1im * ω * τ)
    ldiv!(dest, FT, buffer)
end

function τshift!(pt::Ptychographer, Et, τ)
    τshift!(Et, τ, pt.ω, pt.FT, pt.buffer)
end

function τshift!(dest, pt::Ptychographer, Et, τ)
    τshift!(dest, Et, τ, pt.ω, pt.FT, pt.buffer)
end

abstract type AbstractGeometry end
abstract type AbstractInteraction end

struct FROG <: AbstractGeometry end
struct XFROG <: AbstractGeometry end

struct SHG <: AbstractInteraction end

function signal_field!(dest, int::SHG, test, gate)
    dest .= test .* gate
end


end
