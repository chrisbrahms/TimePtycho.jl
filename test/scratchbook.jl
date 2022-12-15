import TimePtycho
import FFTW
import PyPlot: plt, pygui
import Random: randn
pygui(true)

τ, ω, trace, target = TimePtycho.mock_trace_SHG(
    10e-15, 800e-9, 100e-15, 0, 4e15;
    dispersion=[10e-30, 800e-45],
    Nτ=501)

# trace .+= randn(size(trace))
# trace[trace .< 1e-*maximum(trace)] .= 0
##
pt = TimePtycho.Ptychographer(TimePtycho.SHG(), TimePtycho.FROG(), τ, ω, trace)

for _ in 1:100
    TimePtycho.doiter!(pt)
end

##
plt.figure()
plt.subplot(2, 2, 1)
plt.pcolormesh(τ*1e15, ω*1e-15, pt.measured)
plt.subplot(2, 2, 2)
plt.pcolormesh(τ*1e15, ω*1e-15, pt.reconstructed)
plt.subplot(2, 2, 3)
plt.semilogy(pt.errors)
plt.subplot(2, 2, 4)
plt.plot(abs2.(pt.testpulse))
# plt.plot(abs2.(pt.gatepulse))
plt.plot(abs2.(target))

