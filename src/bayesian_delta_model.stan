data {
  int<lower=0> N; // Number of reason-annotation pairs
  vector[N] y; // Human ratings on Likert scale (0-3)
  vector[N] x; // LLM ratings on Likert scale (0-3)
}

transformed data {
  // Pre-compute indicator variables (runs once, not during sampling)
  vector[N] ind_lt_3;
  vector[N] ind_lt_2;
  vector[N] ind_lt_1;

  for (n in 1:N) {
    ind_lt_3[n] = x[n] < 3 ? 1.0 : 0.0;
    ind_lt_2[n] = x[n] < 2 ? 1.0 : 0.0;
    ind_lt_1[n] = x[n] < 1 ? 1.0 : 0.0;
  }
}

parameters {
  real beta_3; // Expected human score when LLM rating is 3
  real delta_2; // Change in expected score from LLM=3 to LLM=2
  real delta_1; // Change in expected score from LLM=2 to LLM=1
  real delta_0; // Change in expected score from LLM=1 to LLM=0
  real<lower=0> sigma; // Residual standard deviation
}

transformed parameters {
  // Vectorized computation of expected values
  // E[y|x=3] = beta_3
  // E[y|x=2] = beta_3 + delta_2
  // E[y|x=1] = beta_3 + delta_2 + delta_1
  // E[y|x=0] = beta_3 + delta_2 + delta_1 + delta_0
  vector[N] mu = beta_3 + delta_2 * ind_lt_3 + delta_1 * ind_lt_2 + delta_0 * ind_lt_1;
}

model {
  // Vectorized likelihood
  y ~ normal(mu, sigma);

  // Weakly informative priors
  beta_3 ~ normal(3, 1); // Expected score at LLM=3 (prior centered at 3)
  delta_2 ~ normal(-1, 1); // Prior: score decreases by ~1 from LLM=3 to LLM=2
  delta_1 ~ normal(-1, 1); // Prior: score decreases by ~1 from LLM=2 to LLM=1
  delta_0 ~ normal(-1, 1); // Prior: score decreases by ~1 from LLM=1 to LLM=0
  sigma ~ exponential(1); // Prior on residual SD
}
