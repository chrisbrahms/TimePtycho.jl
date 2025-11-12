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

function mock_trace_TG(τfwhm, λ0, τrange, ωmax; Nτ=101, dispersion=Float64[], blur=0)
    trange = 8*τfwhm + τrange
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
        Esig = Et0 .* abs2.(Et(τi))
        trace[:, ii] .= abs2.(FFTW.fft(Esig))
    end

    if blur > 0
        kernel = FFTW.fftshift(gauss.(τ; fwhm=blur)) # gaussian centred at the beginning of the time window
        kernelFT = FFTW.fft(kernel)'
        traceFT = FFTW.fft(trace, 2) # transform along delay axis
        trace .= abs.(FFTW.ifft(traceFT.*kernelFT, 2)) # FFT convolution
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