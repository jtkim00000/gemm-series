import matplotlib.pyplot as plt

# ============================================================
# GEMM Throughput Benchmark
# ============================================================

data = [
    # M,    N,    K,       Naive, Tiled, TC,    Coalesced, Peak,  cuBLAS
    (256,  256,  256,     0.629, 0.798, 1.828, 1.079,     1.796, 2.139),
    (512,  512,  512,     0.673, 0.902, 3.294, 2.444,     3.748, 5.172),
    (1024, 1024, 1024,    0.840, 1.269, 5.231, 4.061,     5.504, 8.592),
    (2048, 2048, 2048,    0.976, 1.276, 5.517, 3.973,     5.808, 9.444),
    (4096, 512,  1024,    0.973, 1.268, 5.308, 4.158,     5.578, 9.057),
    (512,  4096, 2048,    0.975, 1.275, 5.329, 3.723,     5.608, 9.512),
]

# Sort by M*N*K
data.sort(key=lambda x: x[0] * x[1] * x[2])

# Extract values
sizes = [M * N * K for M, N, K, *_ in data]

naive     = [x[3] for x in data]
tiled     = [x[4] for x in data]
tc        = [x[5] for x in data]
coalesced = [x[6] for x in data]
peak      = [x[7] for x in data]
cublas    = [x[8] for x in data]

# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(11, 7))

ax.plot(
    sizes, naive,
    marker='o',
    linewidth=2,
    label='Naive'
)

ax.plot(
    sizes, tiled,
    marker='s',
    linewidth=2,
    label='Tiled'
)

ax.plot(
    sizes, tc,
    marker='^',
    linewidth=2,
    label='Thread Coarsened'
)

ax.plot(
    sizes, coalesced,
    marker='D',
    linewidth=2,
    label='Coalesced'
)

ax.plot(
    sizes, peak,
    marker='v',
    linewidth=2,
    label='Peak'
)

ax.plot(
    sizes, cublas,
    marker='*',
    markersize=11,
    linewidth=2,
    label='cuBLAS'
)

# Logarithmic matrix-size axis
ax.set_xscale('log')

ax.set_xlabel(
    r'Matrix Size ($M \times N \times K$)',
    fontsize=13
)

ax.set_ylabel(
    'Throughput (TFLOPS)',
    fontsize=13
)

ax.set_title(
    'GEMM Throughput vs. Matrix Size',
    fontsize=16
)

ax.grid(
    True,
    which='both',
    linestyle='--',
    alpha=0.35
)

ax.legend(
    fontsize=10,
    frameon=True
)

plt.tight_layout()

# Save high-resolution version
plt.savefig(
    'gemm_throughput.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()