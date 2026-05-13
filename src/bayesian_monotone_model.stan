functions {
  /* compute monotonic effects
   * Args:
   *   scale: a simplex parameter
   *   i: index to sum over the simplex
   * Returns:
   *   a scalar between 0 and rows(scale)
   */
  real mo(vector scale, int i) {
    if (i == 0) {
      return 0;
    } else {
      return rows(scale) * sum(scale[1:i]);
    }
  }

  /* Convert continuous confidence to bin index (0-5)
   * Bins: <=50 -> 0, 51-60 -> 1, 61-70 -> 2, 71-80 -> 3, 81-90 -> 4, >90 -> 5
   */
  int confidence_to_bin(real x) {
    if (x <= 50) return 0;
    else if (x <= 60) return 1;
    else if (x <= 70) return 2;
    else if (x <= 80) return 3;
    else return 4;
  }
}

data {
  int<lower=0> N; // Number of reason-annotation pairs
  vector[N] y; // Human ratings
  vector[N] x; // LLM confidence (0-100 scale)
}

transformed data {
  // Pre-compute bin indices for each observation
  array[N] int bin_idx;
  for (n in 1:N) {
    bin_idx[n] = confidence_to_bin(x[n]);
  }
}

parameters {
  real beta_50; // Expected human score when LLM confidence is <= 50
  simplex[4] delta_simplex; // How the total effect is distributed across bins
  real<lower=0> delta_scale; // Total effect size from lowest to highest bin
  real<lower=0> sigma; // Residual standard deviation
}

model {
  // Weakly informative priors
  beta_50 ~ normal(0, 1);
  delta_scale ~ normal(3, 1);
  delta_simplex ~ dirichlet(rep_vector(1.0, 4)); // Uniform prior on simplex
  sigma ~ exponential(1); // Prior on residual SD

  // Likelihood with monotonic effect
  for (n in 1:N) {
    real mu = beta_50 + delta_scale * mo(delta_simplex, bin_idx[n]);
    y[n] ~ normal(mu, sigma);
  }
}

generated quantities {
  // Compute expected values at each bin for posterior analysis
  vector[6] expected_by_bin;
  for (b in 0:4){
    expected_by_bin[b + 1] = beta_50 + delta_scale * mo(delta_simplex, b);
  }
}
