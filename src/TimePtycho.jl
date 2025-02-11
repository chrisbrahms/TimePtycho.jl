module TimePtycho
import FFTW
import PhysicalConstants: CODATA2018
import Unitful: ustrip
import Statistics: mean
import LinearAlgebra: ldiv!, mul!
import Random: randperm, rand
import Dierckx: Spline1D
import DSP: unwrap
import ImageSegmentation: seeded_region_growing, labels_map

const c = ustrip(CODATA2018.c_0)

fwhm_to_σ(fwhm; power=2) = fwhm / (2 * (2 * log(2))^(1 / power))

gauss(x, σ; x0=0, power=2) = exp(-1/2 * ((x-x0)/σ)^power)
gauss(x; x0=0, power=2, fwhm) = gauss(x, fwhm_to_σ(fwhm; power=power), x0 = x0, power=power)

function mock_trace_SHG(τfwhm, λ0, τrange, ωmin, ωmax; Nτ=101, dispersion=Float64[])
    trange = 16τfwhm
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
    return τ, v, t, trace, Et0
end

function mock_trace_SFG(τfwhm_test, λtest, τfwhm_gate, λgate, τrange, ωmax; Nτ=101, dispersion=Float64[])
    trange = 8*sqrt(τfwhm_test^2 + τfwhm_gate^2)
    ω0 = 2π*c/λtest + 2π*c/λgate
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

    function gate(τ)
        amp = sqrt.(gauss.(t; fwhm=τfwhm_gate))
        Ef = FFTW.fft(amp)
        Ef .*= exp.(1im .* (FFTW.fftshift(v).*τ))
        FFTW.ifft(Ef)
    end

    amptest = sqrt.(gauss.(t; fwhm=τfwhm_test))
    Eftest = FFTW.fft(amptest)
    Eftest .*= exp.(1im .* φ)
    test = FFTW.ifft(Eftest)

    trace = zeros((samples, Nτ))
    τ = collect(range(-τrange/2, τrange/2, Nτ))
    for (ii, τi) in enumerate(τ)
        Esig = test .* gate(τi)
        trace[:, ii] .= abs2.(FFTW.fft(Esig))
    end

    trace = FFTW.fftshift(trace, 1)
    return τ, v, t, trace, test, gate(0.0)
end

function regrid(z, λ, trace; ω0=:moment, z0=:peak, trange=nothing, τ0shift=0,
                             zunit=:m, λmin=-Inf, λmax=Inf)
    if zunit == :m
        τ = 2z/c
    elseif zunit == :fs
        τ = z*1e-15
    end

    idcs = (λ .>= λmin) .&& (λ .<= λmax)

    λ = λ[idcs]
    trace = trace[idcs, :]
    
    ωraw = 2π*c./λ
    ω0raw = getω0raw(ωraw, trace, ω0)
    if isnothing(trange)
        trange = 2*(maximum(τ) - minimum(τ))
    end
    ωmax = maximum(abs.(ωraw .- ω0raw))

    δt = π/ωmax
    samples = 2^(ceil(Int, log2(trange/δt)))
    trange_even = δt*samples
    δω = 2π/trange_even

    Nω = collect(range(0, length=samples))
    t = @. (Nω-samples/2)*δt # time grid, centre on 0
    ω = @. (Nω-samples/2)*δω # freq grid relative to ω0

    δω = ω[2] - ω[1]
    Δω = length(ω)*δω
    @assert δt ≈ 2π/Δω
    @assert length(t) == length(ω)

    trace = reverse(trace ./ ωraw.^2; dims=1)

    out = zeros(Float64, (samples, length(z)))
    for ii in axes(out, 2)
        spl = Spline1D(reverse(ωraw) .- ω0raw, trace[:, ii]; k=3, bc="zero")
        out[:, ii] .= spl.(ω)
    end

    ωmarg = dropdims(sum(out; dims=1); dims=1)
    if z0 == :peak
        τ0 = τ[argmax(ωmarg)]
    elseif z0 == :moment
        τ0 = sum(ωmarg .* τ)/sum(ωmarg)
    end
    τ .- τ0 .- τ0shift, ω, ω0raw, t, out/maximum(abs.(out))
end

getω0raw(ωraw, trace, ω0::Number) = ω0

function getω0raw(ωraw, trace, ω0::Symbol)
    if ω0 == :moment
        zmarg = dropdims(sum(trace; dims=2); dims=2)
        return sum(zmarg .* ωraw)/sum(zmarg)
    else
        error("Invalid ω0 mode $ω0")
    end
end

function sub_dark!(λtrace, trace, λdark, dark)
    idcs = minimum(λtrace) .<= λdark .<= maximum(λtrace)
    dark = avg(dark)
    trace .-= dark[idcs]
end

function marg_correct!(trace, ωtrace, ω0, λfund, fund; threshold=0)
    ωfund = reverse(2π*c./λfund)
    fundω = reverse(fund) ./ ωfund.^2

    Iω_fund = Spline1D(ωfund .- ω0/2, fundω; k=3, bc="zero").(ωtrace)
    Iω_fund_t = FFTW.ifft(FFTW.ifftshift(Iω_fund))
    IωSHG = real(FFTW.fftshift(FFTW.fft(Iω_fund_t .^ 2)))
    IωSHG ./= maximum(IωSHG)

    marg = dropdims(sum(trace; dims=2); dims=2)
    marg ./= maximum(marg)

    m = maximum(marg)

    for ii in eachindex(marg)
        if marg[ii] > threshold*m
            trace[ii, :] .*= IωSHG[ii] / marg[ii]
        else
            trace[ii, :] .= 0
        end
    end
end

function marg_correct_SD!(trace, ωtrace, λfund, fund)
    ωfund = reverse(2π*c./λfund)
    fundω = reverse(fund) ./ ωfund.^2
    ω0_fund = sum(fundω .* ωfund)/sum(fundω)

    Iω_fund = Spline1D(ωfund .- ω0_fund, fundω; k=3, bc="zero").(ωtrace)
    Iω_fund ./= maximum(Iω_fund)

    marg = dropdims(sum(trace; dims=2); dims=2)
    marg ./= maximum(marg)

    for ii in eachindex(marg)
        if marg[ii] > 0
            trace[ii, :] .*= Iω_fund[ii] / marg[ii]
        else
            trace[ii, :] .= 0
        end
    end
end

function threshold!(trace, rtol; remove_islands=true)
    val = maximum(trace)*rtol
    if remove_islands
        trace_bin = trace .>= val
        maxidx = argmax(trace)
        seeds = [
            (maxidx, 1),
            (CartesianIndex(size(trace, 1), 1), 2),
            (CartesianIndex(size(trace, 1), size(trace, 2)), 3),
            (CartesianIndex(1, size(trace, 2)), 4),
            (CartesianIndex(1, 1), 5)
        ]
        seg = seeded_region_growing(trace_bin, seeds)
        trace[labels_map(seg) .≠ 1] .= 0
    else
        trace[trace .< val] .= 0
    end
end

avg(dark::AbstractVector) = dark
avg(dark::AbstractMatrix) = dropdims(mean(dark; dims=2); dims=2)

mutable struct Ptychographer{gT, iT, ftT}
    geometry::gT
    interaction::iT
    FT::ftT
    τ::Vector{Float64}
    ω::Vector{Float64}
    t::Vector{Float64}
    measured::Matrix{Float64}
    measA::Matrix{Float64}
    support::BitVector
    reconstructed::Matrix{Float64}
    testpulse::Vector{ComplexF64}
    gatepulse::Vector{ComplexF64}
    gateshift::Vector{ComplexF64}
    iters::Int
    errors::Vector{Float64}
    buffer::Vector{ComplexF64}
    ψ::Vector{ComplexF64}
    ψf::Vector{ComplexF64}
    ψp::Vector{ComplexF64}
    ψfp::Vector{ComplexF64}
    diff::Vector{ComplexF64}
    best_test::Vector{ComplexF64}
    best_gate::Vector{ComplexF64}
    best_error::Float64
    best_iter::Int64
    test_saved::Vector{ComplexF64}
    gate_saved::Vector{ComplexF64}
    rec_candidate::Matrix{Float64}
    err_candidate::Float64
end

function Ptychographer(interaction, geometry, τ, ω, trace, support=nothing)
    δω = ω[2] - ω[1]
    N = length(ω)
    Δω = N*δω
    δt = 2π/Δω
    idx = collect(1:N)
    t = @. (idx - N/2)*δt
    marg = dropdims(sum(trace; dims=1); dims=1)
    testpulse = complex(Spline1D(τ, marg; k=3, bc="zero").(t))
    testpulse ./= sqrt(maximum(abs2.(testpulse)))
    FT = FFTW.plan_fft(testpulse, 1, flags=FFTW.PATIENT)
    inv(FT) # plan inverse FFT
    s = sign.(trace)
    measA = @. s * sqrt(abs(trace))
    rec = zeros(Float64, size(trace))
    iters = 0
    errors = Float64[Inf]
    gatepulse = copy(testpulse)
    gateshift = zeros(ComplexF64, length(ω))
    buffer = copy(gateshift)
    ψ = copy(gateshift)
    ψf = copy(gateshift)
    ψp = copy(gateshift)
    ψfp = copy(gateshift)
    diff = copy(gateshift)
    if isnothing(support)
        support = trues(length(ω))
    end

    Ptychographer(geometry, interaction, FT, τ, FFTW.fftshift(ω), t,
                  trace, FFTW.fftshift(measA, 1), FFTW.fftshift(support), rec, testpulse,
                  gatepulse, gateshift,
                  iters, errors, buffer, ψ, ψf, ψp, ψfp, diff, copy(testpulse), copy(testpulse),
                  Inf, 0, copy(testpulse), copy(testpulse), copy(rec), Inf)

end

function doiter!(pt::Ptychographer;
                 α=(0.6, 1.0), soft_thr=false, γ=1e-3,
                 force_improvement=false, update_gate=true)
    pt.gate_saved .= pt.gatepulse
    pt.test_saved .= pt.testpulse
    for ii in randperm(size(pt.measured, 2))
        pt.gateshift .= pt.gatepulse
        τshift!(pt, pt.gateshift, pt.τ[ii])
        signal_field!(pt.ψ, pt.interaction, pt.testpulse, pt.gateshift)

        mul!(pt.ψf, pt.FT, pt.ψ)
        for jj in eachindex(pt.ψfp)
            if pt.support[jj] || ~soft_thr
                a = abs(pt.ψf[jj])
                if a > 0
                    pt.ψfp[jj] = pt.ψf[jj]/a * pt.measA[jj, ii]
                # else
                    # pt.ψfp[jj] = pt.measA[jj, ii] * exp(1im*2π*rand())
                end
            else
                pt.ψfp[jj] = fγ(real(pt.ψf[jj]), γ) + 1im*fγ(imag(pt.ψf[jj]), γ)
            end
        end
        ldiv!(pt.ψp, pt.FT, pt.ψfp)

        @. pt.diff = pt.ψp - pt.ψ

        m = maximum(abs2, pt.gateshift)

        α_ = getα(α)

        if isa(pt.geometry, XFROG)
            pt.buffer .= pt.testpulse # save old testpulse for gate update
        end

        for jj in eachindex(pt.testpulse)
            pt.testpulse[jj] += pt.diff[jj] * α_ * conj(pt.gateshift[jj])/m
        end
        
        if isa(pt.geometry, XFROG)
            if update_gate
                # buffer now contains the old test pulse before the update
                m = maximum(abs2, pt.buffer)
                for jj in eachindex(pt.gateshift)
                    pt.gateshift[jj] += pt.diff[jj] * α_ * conj(pt.buffer[jj])/m
                end
                τshift!(pt.gatepulse, pt, pt.gateshift, -pt.τ[ii])
                if isa(pt.interaction, XPM)
                    # pt.gatepulse .= exp.(1im.*angle.(pt.gatepulse))
                    pt.gatepulse ./= abs.(pt.gatepulse)
                elseif isa(pt.interaction, TG)
                    pt.gatepulse .= abs.(pt.gatepulse)
                end
            end
        elseif isa(pt.interaction, SHG)
            pt.gatepulse .= pt.testpulse
        elseif isa(pt.interaction, TG)
            pt.gatepulse .= abs2.(pt.testpulse)
        end
    end
    err = update_recon!(pt)
    if force_improvement && err > pt.errors[end]
        pt.testpulse .= pt.test_saved
        pt.gatepulse .= pt.gate_saved
    else
        pt.reconstructed .= pt.rec_candidate
        push!(pt.errors, err)
    end
    if err < pt.best_error
        pt.best_error = err
        pt.best_iter = pt.iters
        pt.best_test .= pt.testpulse
        pt.best_gate .= pt.gatepulse
    end
    pt.iters += 1
    end

getα(α::Tuple) = α[1] + (α[2]-α[1])*rand()
getα(α::Number) = α

function update_recon!(pt::Ptychographer)
    for ii in axes(pt.measured, 2)
        τshift!(pt.gateshift, pt, pt.gatepulse, pt.τ[ii])
        signal_field!(pt.ψ, pt.interaction, pt.testpulse, pt.gateshift)
        mul!(pt.buffer, pt.FT, pt.ψ)
        FFTW.fftshift!(pt.ψf, pt.buffer, 1)
        for jj in axes(pt.measured, 1)
            pt.rec_candidate[jj, ii] = abs2(pt.ψf[jj])
        end
    end
    ηnom = sum(eachindex(pt.measured)) do ii
        pt.measured[ii] * pt.rec_candidate[ii]
    end
    ηden = sum(pt.rec_candidate) do ri
        ri^2
    end
    η = ηnom/ηden
    e = sum(eachindex(pt.rec_candidate)) do ii
        (pt.measured[ii] - η*pt.rec_candidate[ii])^2
    end
    return sqrt(1/length(pt.rec_candidate) * e)
end

function make_recon!(dest, pt::Ptychographer, testpulse, gatepulse)
    for ii in axes(pt.measured, 2)
        τshift!(pt.gateshift, pt, gatepulse, pt.τ[ii])
        signal_field!(pt.ψ, pt.interaction, testpulse, pt.gateshift)
        mul!(pt.buffer, pt.FT, pt.ψ)
        FFTW.fftshift!(pt.ψf, pt.buffer, 1)
        for jj in axes(pt.measured, 1)
            dest[jj, ii] = abs2(pt.ψf[jj])
        end
    end
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

function init_from_spec!(pt::Ptychographer, λ, Iλ, ω0; which=:test, phase=nothing)
    ωraw = reverse(2π*c./λ)
    Iωraw = reverse(Iλ) ./ ωraw.^2

    Iω = Spline1D(ωraw .- ω0, Iωraw; k=3, bc="zero").(pt.ω)
    Iω[Iω .< 0] .= 0

    Eω = complex(sqrt.(Iω))

    if ~isnothing(phase)
        Eω .*= exp.(1im*phase)
    end

    # the extra fftshift places the pulse at the centre
    # of the time window
    Et = FFTW.fftshift(FFTW.ifft(Eω))
    Et ./= sqrt(maximum(abs2.(Et)))

    if which == :test
        pt.testpulse .= Et
    elseif which == :gate
        pt.gatepulse .= Et
    else
        error("`which` must be :test or :gate")
    end
end


fγ(x, γ) = x >= γ ? x - γ*sign(x) : zero(x)

abstract type AbstractGeometry end
abstract type AbstractInteraction end

struct FROG <: AbstractGeometry end
struct XFROG <: AbstractGeometry end

struct SHG <: AbstractInteraction end
struct XPM <: AbstractInteraction end
struct TG <: AbstractInteraction end

function signal_field!(dest, int, test, gate)
    dest .= test .* gate
end


end
