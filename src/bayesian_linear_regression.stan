data {
  int<lower=0> N; // Number of reason-annotation pairs
  vector[N] y; // Human ratings on Likert scale
  vector[N] x; // LLM ratings on probability scale
}

parameters {
  real beta_0; // Intercept parameter
  real beta_1; // Slope parameter for LLM rating
  real<lower=0> sigma; // Residual standard deviation
}

model {
  // Linear regression: human_rating ~ Normal(beta_0 + beta_1 * llm_rating, sigma)
  y ~ normal(beta_0 + beta_1 * x, sigma);

  // Weakly informative priors
  beta_0 ~ normal(-1.5, 2); // Intercept prior (centered near 0)
  beta_1 ~ normal(1/20, 1); // Slope prior (centered at 1 for perfect calibration)
  sigma ~ exponential(1); // Prior on residual SD
}

