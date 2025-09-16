using GLMakie

# Image grid
W, H = 1000, 800
xs = range(-2.5, 1.0, length=W)
ys = range(-1.25, 1.25, length=H)
maxiter = 400

function mandelbrot!(buf, xc, yc, scale)
    @inbounds for j in 1:H
        cy = yc + (ys[j] - 0.0) * scale
        for i in 1:W
            cx = xc + (xs[i] - 0.0) * scale
            x = 0.0; y = 0.0
            it = 0
            while x*x + y*y ≤ 4.0 && it < maxiter
                x2 = x*x - y*y + cx
                y  = 2x*y + cy
                x  = x2
                it += 1
            end
            val = it == maxiter ? 0.0 :
                  it - log2(log(max(x*x + y*y, 1e-12))) + 4.0
            buf[i, j] = val
        end
    end
    buf
end

# Observables
imgbuf = Observable(zeros(Float32, W, H))
palette_phase = Observable(0.0)

# Helper: rotate a colormap by φ (can be non-integer)
function rotate_cmap(base::Symbol, φ::Real)
    cs = to_colormap(base)
    n = length(cs)
    out = similar(cs)
    # fractional rotation via linear interpolation
    k = mod(φ, n)
    i0 = floor(Int, k); t = k - i0
    getc(idx) = cs[mod1(idx, n)]
    for i in 1:n
        c1 = getc(i + i0)
        c2 = getc(i + i0 + 1)
        out[i] = RGBAf(
            (1 - t)*red(c1)   + t*red(c2),
            (1 - t)*green(c1) + t*green(c2),
            (1 - t)*blue(c1)  + t*blue(c2),
            (1 - t)*alpha(c1) + t*alpha(c2),
        )
    end
    out
end

# Figure (use size, not resolution)
fig = Figure(size=(900, 700))
ax  = Axis(fig[1,1], title="Mandelbrot deep zoom (smooth coloring)")

# Bind the Observable directly
h = heatmap!(ax, imgbuf; colormap=:turbo, interpolate=true)
Colorbar(fig[1,2], h, label="iterations")
h.colorrange = (0, maxiter)

# Make colormap reactive; NOTE the $ to interpolate the observable
h.colormap = @lift rotate_cmap(:turbo, $palette_phase)

hidespines!(ax); ax.showgrid = false

# Zoom path
center_x, center_y = -0.7453, 0.1127
scale0 = 3.5 / W
frames = 600

record(fig, "mandelbrot_zoom.gif", 1:frames) do i
    s = scale0 * 0.98^(i-1)         # exponential zoom
    mandelbrot!(imgbuf[], center_x, center_y, s)
    notify(imgbuf)                   # tell Makie data changed
    palette_phase[] = i * 0.5        # gentle color cycling
end