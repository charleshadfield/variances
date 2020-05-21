def variance(x):
    algo = x[0]
    pr = x[1]
    energy = x[2]
    state = x[3]
    β = x[4]
    variance = pr.variance_local(energy, state, β)
    return algo, variance
