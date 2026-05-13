data {
  int<lower=0> N; // Number of reason-annotation pairs
  vector[N] y; // Human ratings
  vector[N] x; // LLM ratings
}

transformed data {
  // Pre-compute indicator variables (runs once, not during sampling)
  vector[N] ind_le_90;
  vector[N] ind_le_80;
  vector[N] ind_le_70;
  vector[N] ind_le_60;
  vector[N] ind_le_50;

  for (n in 1:N) {
    ind_le_90[n] = x[n] <= 90 ? 1.0 : 0.0;
    ind_le_80[n] = x[n] <= 80 ? 1.0 : 0.0;
    ind_le_70[n] = x[n] <= 70 ? 1.0 : 0.0;
    ind_le_60[n] = x[n] <= 60 ? 1.0 : 0.0;
    ind_le_50[n] = x[n] <= 50 ? 1.0 : 0.0;
  }
}

parameters {
  real beta_100; // Expected human score when LLM confidence is 100
  vector[5] deltas;
  real<lower=0> sigma; // Residual standard deviation
}

transformed parameters {
  // Vectorized computation of expected values
  vector[N] mu = beta_100 + deltas[5] * ind_le_90 + deltas[4] * ind_le_80 + deltas[3] * ind_le_70 + deltas[2] * ind_le_60 + deltas[1] * ind_le_50;

}

model {
  // Vectorized likelihood
  y ~ normal(mu, sigma);

  // Weakly informative priors
  beta_100 ~ normal(3, 1); // Expected score at LLM=3 (prior centered at 3)
  deltas ~ normal(-0.06, 1); // Prior: score decreases by how much for drop in decile
  sigma ~ exponential(1); // Prior on residual SD
}
