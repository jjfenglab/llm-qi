functions {
  /* compute monotonic effects */
  real mo(vector scale, int i) {
    if (i == 0) {
      return 0;
    } else {
      return rows(scale) * sum(scale[1:i]);
    }
  }

  /* Convert continuous confidence to bin index (0 to 4) */
  int confidence_to_bin(real x) {
    if (x <= 50) return 0;
    else if (x <= 60) return 1;
    else if (x <= 70) return 2;
    else if (x <= 80) return 3;
    else return 4; 
  }
}

data {
  int<lower=0> N;
  array[N] int<lower=1,upper=5> y;
  vector[N] x;
}

transformed data {
  array[N] int bin_idx;
  for (n in 1:N) {
    bin_idx[n] = confidence_to_bin(x[n]);
  }
}

parameters {
  // We no longer need beta_50 (intercept) or sigma.
  // Intercepts are replaced by cutpoints. Variance in logistic is fixed.

  ordered[4] c; // 4 cutpoints to divide the latent space into 5 categories
  simplex[4] delta_simplex;
  real<lower=0> delta_scale;
}

model {
  // Priors
  c ~ student_t(3, 0, 2.5); // Weakly informative prior for cutpoints
  delta_scale ~ normal(3, 1);
  delta_simplex ~ dirichlet(rep_vector(1.0, 4)); 

  // Likelihood using ordered logistic
  for (n in 1:N) {
    // eta is the latent linear predictor. 
    // It captures the monotonic effect of the bins.
    real eta = delta_scale * mo(delta_simplex, bin_idx[n]);
    y[n] ~ ordered_logistic(eta, c);
  }
}

generated quantities {
  // For ordinal models, we usually care about the *probability* of each class
  // at each bin level, rather than a single expected value.
  array[5] vector[5] prob_by_bin;

  for (b in 0:4) {
    real eta = delta_scale * mo(delta_simplex, b);

    // Calculate probability of being in category k (1 to 5) for bin b
    prob_by_bin[b + 1][1] = 1 - inv_logit(eta - c[1]);
    prob_by_bin[b + 1][2] = inv_logit(eta - c[1]) - inv_logit(eta - c[2]);
    prob_by_bin[b + 1][3] = inv_logit(eta - c[2]) - inv_logit(eta - c[3]);
    prob_by_bin[b + 1][4] = inv_logit(eta - c[3]) - inv_logit(eta - c[4]);
    prob_by_bin[b + 1][5] = inv_logit(eta - c[4]);
  }
}
