import TimePtycho
import FFTW
import PyPlot: plt, pygui
import Random: randn
import Statistics: mean
pygui(true)

λ0 = 800e-9
ω0 = 4π*TimePtycho.c/λ0

τ, ω, t, trace, target = TimePtycho.mock_trace_SHG(
    10e-15, λ0, 250e-15, 0, 4e15;
    dispersion=[10e-30, 800e-45, 5000e-60],
    Nτ=101)

SNR = 5000

trace .+= randn(size(trace))/SNR*maximum(trace)
# trace[trace .< 1e-*maximum(trace)] .= 0
marg = dropdims(mean(trace; dims=2); dims=2)
support = marg .> maximum(marg)/SNR
# support = (ω .> -0.5e15) .&& (ω .< 0.5e15)
##
pt = TimePtycho.Ptychographer(TimePtycho.SHG(), TimePtycho.FROG(), τ, ω, trace, support)

for _ in 1:100
    TimePtycho.doiter!(pt; soft_thr=true)
end

##
meas = copy(pt.measured)
meas ./= maximum(meas)
meas[meas .< 1e-4] .= 0

shift = t[argmax(abs2.(pt.testpulse))] - t[argmax(abs2.(target))]
TimePtycho.τshift!(pt, pt.testpulse, shift)
plt.figure()
plt.subplot(2, 2, 1)
plt.pcolormesh(τ*1e15, ω*1e-15, 10log10.(meas/maximum(meas)))
plt.clim(-40, 0)
plt.subplot(2, 2, 2)
plt.pcolormesh(τ*1e15, ω*1e-15, 10log10.(pt.reconstructed/maximum(pt.reconstructed)))
plt.clim(-40, 0)
plt.subplot(2, 2, 3)
plt.semilogy(pt.errors)
plt.subplot(2, 2, 4)
plt.plot(1e15*t, abs2.(pt.testpulse))
# plt.plot(abs2.(pt.gatepulse))
plt.plot(1e15*t, abs2.(target))

