# Stratification and Weighting via the Propensity Score

*Notes on Lunceford, J. K. and Davidian, M. (2004). Stratification and weighting via the propensity score in estimation of causal treatment effects: a comparative study. *Statistics in Medicine*, 23, 2937–2960. Sections 1–2.5, with proofs.*

Sections 1–2.5 define the estimators and justify them; the formal asymptotic results (variances, M-estimation) are in Section 3. Where a "why it works" argument is not written out in the paper, it is filled in below and marked *(derivation added)*.

**Section structure.** 1 Introduction · 2 Estimators based on the propensity score · 2.1 Counterfactual framework · 2.2 The propensity score · 2.3 Estimation of $\Delta$ based on stratification · 2.4 Estimation of $\Delta$ based on weighting · 2.5 Summary.

---

## 1. Setup (Section 2.1: counterfactual framework)

| Symbol | Meaning |
|---|---|
| $Z\in\{0,1\}$ | treatment indicator |
| $X$ | pre-treatment covariates |
| $(Y_0,Y_1)$ | potential (counterfactual) outcomes |
| $Y=ZY_1+(1-Z)Y_0$ | observed outcome |
| $e(X)=P(Z=1\mid X)$ | propensity score |
| $\Delta=E(Y_1)-E(Y_0)$ | target: the **average causal effect (ATE)** |

**Assumptions.**

- **No unmeasured confounders (strong ignorability):** $(Y_0,Y_1)\perp Z\mid X$.
- **Positivity:** $0<e(X)<1$.

Under randomization $(Y_0,Y_1)\perp Z$ and the difference of group means is unbiased; in observational data this generally fails, which motivates conditioning on $X$.

**Identification.**

$$
E\{E(Y\mid Z=1,X)\}\overset{(a)}{=}E\{E(Y_1\mid Z=1,X)\}\overset{(b)}{=}E\{E(Y_1\mid X)\}=E(Y_1),
$$

where (a) holds because $Y=Y_1$ when $Z=1$, and (b) is no unmeasured confounding. Likewise for $E(Y_0)$. Positivity ensures $E(Y\mid Z=z,X)$ is defined at every $X$.

---

## 2. The propensity score (Section 2.2)

### 2.1 Two key properties (Rosenbaum and Rubin, 1983) *(proofs added)*

**(a) Balancing: $X\perp Z\mid e(X)$.**

$$
P\{Z=1\mid X,e(X)\}=P(Z=1\mid X)=e(X),\qquad
P\{Z=1\mid e(X)\}=E\{P(Z=1\mid X)\mid e(X)\}=E\{e(X)\mid e(X)\}=e(X).
$$

The two are equal, so given $e(X)$, treatment carries no further information about $X$.

**(b) No unmeasured confounders given $e(X)$: $(Y_0,Y_1)\perp Z\mid e(X)$.**

$$
P\{Z=1\mid Y_0,Y_1,e(X)\}=E\{P(Z=1\mid Y_0,Y_1,X)\mid Y_0,Y_1,e(X)\}=E\{e(X)\mid Y_0,Y_1,e(X)\}=e(X),
$$

where the middle step uses $(Y_0,Y_1)\perp Z\mid X$. So adjusting for the scalar $e(X)$ suffices.

### 2.2 Estimating the propensity score

Model the propensity score parametrically as $e(X;\boldsymbol\beta)$, where $X$ includes a constant $1$ for the intercept. The standard choice is logistic regression:

$$
e(X;\boldsymbol\beta)=P(Z=1\mid X)=\frac{1}{1+\exp(-X^\top\boldsymbol\beta)}=\frac{\exp(X^\top\boldsymbol\beta)}{1+\exp(X^\top\boldsymbol\beta)},
\qquad\text{i.e.}\qquad
\log\frac{e(X;\boldsymbol\beta)}{1-e(X;\boldsymbol\beta)}=X^\top\boldsymbol\beta .
$$

Write $e_i=e(X_i;\boldsymbol\beta)$.

**Likelihood.** Given $X_i$, $Z_i\sim\text{Bernoulli}(e_i)$, so $P(Z_i=z\mid X_i)=e_i^{\,z}(1-e_i)^{1-z}$ and

$$
L(\boldsymbol\beta)=\prod_{i=1}^n e_i^{\,Z_i}(1-e_i)^{1-Z_i},
\qquad
\ell(\boldsymbol\beta)=\log L(\boldsymbol\beta)=\sum_{i=1}^n\Big\{Z_i\log e_i+(1-Z_i)\log(1-e_i)\Big\}.
$$

**Log-likelihood of logistic regression.** Since $\log e_i=X_i^\top\boldsymbol\beta-\log\{1+\exp(X_i^\top\boldsymbol\beta)\}$ and $\log(1-e_i)=-\log\{1+\exp(X_i^\top\boldsymbol\beta)\}$,

$$
\ell(\boldsymbol\beta)=\sum_{i=1}^n\Big[Z_i\,X_i^\top\boldsymbol\beta-\log\{1+\exp(X_i^\top\boldsymbol\beta)\}\Big].
$$

**Score equation (general binary model).** Differentiating the Bernoulli log-likelihood for any model $e(X;\boldsymbol\beta)$,

$$
\frac{\partial\ell}{\partial\boldsymbol\beta}=\sum_{i=1}^n\Big(\frac{Z_i}{e_i}-\frac{1-Z_i}{1-e_i}\Big)\frac{\partial e_i}{\partial\boldsymbol\beta}
=\sum_{i=1}^n\frac{Z_i-e_i}{e_i(1-e_i)}\,\frac{\partial e_i}{\partial\boldsymbol\beta}=\mathbf 0 ,
$$

which is the estimating equation for the propensity score given in the paper.

**Score equation (logistic model).** Here $\partial e_i/\partial\boldsymbol\beta=e_i(1-e_i)X_i$, so the factor $e_i(1-e_i)$ cancels (equivalently, differentiate $\ell$ directly using $\frac{\partial}{\partial\boldsymbol\beta}\log\{1+\exp(X_i^\top\boldsymbol\beta)\}=e_iX_i$):

$$
\frac{\partial\ell}{\partial\boldsymbol\beta}=\sum_{i=1}^n(Z_i-e_i)\,X_i=\mathbf 0 .
$$

Because $X_i$ contains the intercept, the first component gives $\sum_i\hat e_i=\sum_iZ_i$: the fitted probabilities sum to the number of treated units.

**Concavity.** The Hessian is

$$
\frac{\partial^2\ell}{\partial\boldsymbol\beta\,\partial\boldsymbol\beta^\top}=-\sum_{i=1}^n e_i(1-e_i)\,X_iX_i^\top ,
$$

which is negative definite when the $X_i$ span the covariate space, so $\ell$ is strictly concave and the MLE $\hat{\boldsymbol\beta}$ (computed by Newton–Raphson / iteratively reweighted least squares) is unique when it exists. Under complete separation of treated and controls by $X$ the MLE does not exist, which is the extreme case of a positivity violation.

Write $\hat e_i=e(X_i;\hat{\boldsymbol\beta})$.

---

## 3. Stratification (Section 2.3)

### 3.1 The estimator

1. Estimate $\hat e_i$.
2. Form $K$ strata $\hat Q_j=(\hat q_{j-1},\hat q_j]$ from sample quantiles of $\hat e_i$ ($\hat q_0=0$, $\hat q_K=1$).
3. Within each stratum, take the difference of treated and control means.
4. Average over strata with weights $n_j/n$ ($\approx 1/K$).

With $n_j=\sum_iI(\hat e_i\in\hat Q_j)$ and $n_{1j}=\sum_iZ_iI(\hat e_i\in\hat Q_j)$,

$$
\hat\Delta_S=\sum_{j=1}^K\frac{n_j}{n}\left[\frac{1}{n_{1j}}\sum_{i=1}^n Z_iY_iI(\hat e_i\in\hat Q_j)-\frac{1}{n_j-n_{1j}}\sum_{i=1}^n(1-Z_i)Y_iI(\hat e_i\in\hat Q_j)\right].
$$

Rosenbaum and Rubin used $K=5$ (quintiles).

### 3.2 Why it works, and why it is biased *(derivation added)*

By property (b), with $g_1(s)=E(Y_1\mid e=s)$,

$$
E\{Y\mid Z=1,e(X)=s\}=E\{Y_1\mid Z=1,e=s\}=g_1(s).
$$

Within a stratum, however, the treated units are not spread over $Q_j$ like the population:

$$
E\{Y\mid Z=1,e\in Q_j\}=\int_{Q_j}g_1(s)\,f(s\mid Z=1,e\in Q_j)\,ds,\qquad f(s\mid Z=1,e\in Q_j)\propto s\,f(s),
$$

and the controls are weighted by $(1-s)f(s)$, while the target $E(Y_1-Y_0\mid e\in Q_j)$ weights by $f(s)$. These agree only if $e$ is constant within the stratum.

- **If strata are narrow:** the within-stratum difference $\approx E(Y_1-Y_0\mid e\in Q_j)$, and averaging with weights $P(e\in Q_j)=1/K$ gives $\Delta$ by the law of total expectation.
- **With $K$ fixed:** residual confounding remains within strata, so $\hat\Delta_S$ is generally biased (and inconsistent).

### 3.3 Regression adjustment within strata

Fit a regression $m^{(j)}(Z,X;\hat Q^{(j)})$ in stratum $j$ and set

$$
\hat\Delta^{(j)}=\frac{1}{n_j}\sum_{i=1}^nI(\hat e_i\in\hat Q_j)\big\{m^{(j)}(1,X_i)-m^{(j)}(0,X_i)\big\},\qquad
\hat\Delta_{SR}=\frac1K\sum_{j=1}^K\hat\Delta^{(j)} .
$$

*Consistency (derivation added).* Under no unmeasured confounders, $E(Y\mid Z=z,X)=E(Y_z\mid X)$. If every $m^{(j)}$ equals the true regression, each term averages $E(Y_1-Y_0\mid X)$ over the units in the stratum, and averaging over strata gives $\Delta$. This is g-computation within strata; it is not guaranteed if the $m^{(j)}$ are misspecified.

---

## 4. Weighting (Section 2.4)

### 4.1 The key identity

$$
E\!\left[\frac{ZY}{e(X)}\right]
=E\!\left[\frac{ZY_1}{e(X)}\right]
=E\!\left[\frac{Y_1}{e(X)}\,E\{Z\mid Y_1,X\}\right]
=E\!\left[\frac{Y_1}{e(X)}\,e(X)\right]=E(Y_1).
$$

1. $ZY=ZY_1$ (observation rule).
2. Condition on $(Y_1,X)$ (iterated expectations).
3. $E(Z\mid Y_1,X)=E(Z\mid X)=e(X)$ by no unmeasured confounders, which cancels the denominator.

By symmetry, $E\{(1-Z)Y/(1-e(X))\}=E(Y_0)$. Setting $Y\equiv1$ gives $E\{Z/e(X)\}=1$ and $E\{(1-Z)/(1-e(X))\}=1$.

### 4.2 IPW1: Horvitz–Thompson type

$$
\hat\Delta_{IPW,1}=\frac1n\sum_{i=1}^n\frac{Z_iY_i}{\hat e_i}-\frac1n\sum_{i=1}^n\frac{(1-Z_i)Y_i}{1-\hat e_i}.
$$

### 4.3 IPW2: normalized (ratio) weights

Since $E\{Z/e\}=1$, divide by the realized sum of weights instead of $n$:

$$
\hat\Delta_{IPW,2}=\left(\sum_{i=1}^n\frac{Z_i}{\hat e_i}\right)^{-1}\sum_{i=1}^n\frac{Z_iY_i}{\hat e_i}-\left(\sum_{i=1}^n\frac{1-Z_i}{1-\hat e_i}\right)^{-1}\sum_{i=1}^n\frac{(1-Z_i)Y_i}{1-\hat e_i}.
$$

Each mean is now a weighted average of observed $Y$'s and stays within their range; IPW1 can fall outside it when some $\hat e_i$ are small. Both estimators belong to the Robins, Rotnitzky and Zhao (1994) class, which treats the problem as missing data with "observation probability" $e(X)$.

### 4.4 IPW3: variance-minimizing weights *(derivation written in a unified form)*

**A family of estimating equations.** For $\mu_1=E(Y_1)$ consider

$$
\sum_{i=1}^n\Big[\frac{Z_i(Y_i-\mu_1)}{e_i}+\eta_1\,\frac{Z_i-e_i}{e_i}\Big]=0 .
$$

- $\eta_1=0$: $\mu_1=\sum_iZ_iY_i/e_i\big/\sum_iZ_i/e_i$, i.e. **IPW2**.
- $\eta_1=\mu_1$: the summand is $Z_iY_i/e_i-\mu_1$, so $\mu_1=n^{-1}\sum_iZ_iY_i/e_i$, i.e. **IPW1**.
- Any $\eta_1$ gives a valid equation, because the added term has mean zero: $E\{(Z-e)/e\}=E\{E(Z-e\mid X)/e\}=0$.

**Estimating function vs. influence function.** Write the summand as $\psi(O;\mu_1)=\frac{Z(Y-\mu_1)}{e(X)}+\eta_1\frac{Z-e(X)}{e(X)}$ with $O=(Y,Z,X)$.

- $\psi$ is an *unbiased estimating function*: $E\,\psi(O;\mu_1)=0$ at the true $\mu_1$, and $\hat\mu_1$ solves $\sum_i\psi(O_i;\mu_1)=0$.
- By M-estimation theory, $\sqrt n(\hat\mu_1-\mu_1)=n^{-1/2}\sum_i\mathrm{IF}(O_i)+o_p(1)$ with $\mathrm{IF}(O)=-\{E\,\partial\psi/\partial\mu_1\}^{-1}\psi(O;\mu_1)$. Here $E\,\partial\psi/\partial\mu_1=-E\{Z/e(X)\}=-1$, so **with $e$ known, $\psi$ is the influence function** of $\hat\mu_1$, and the asymptotic variance is $E(\psi^2)$.
- With $e$ estimated by $\hat{\boldsymbol\beta}$, the influence function is $\psi$ minus its projection on the logistic score $S_\beta=(Z-e)X$:
  $$
  \mathrm{IF}=\psi-E(\psi S_\beta^\top)\{E(S_\beta S_\beta^\top)\}^{-1}S_\beta ,
  $$
  so estimating the propensity score never increases (and usually reduces) the variance. This correction is worked out in Section 3 of the paper.

**Where the family comes from (Robins, Rotnitzky and Zhao, 1994).**

1. *Start from an unbiased IPW estimating function.* The key identity gives $E\{Z(Y-\mu_1)/e(X)\}=0$; solving it is IPW2.
2. *Add any mean-zero term.* A function of $(Z,X)$ with conditional mean zero given $X$ can be added without bias. Because $Z$ is binary, every such function has the form $\{Z-e(X)\}h(X)$: this is the *augmentation space*. Under no unmeasured confounders (missing at random) with $e$ known, every regular asymptotically linear estimator of $\mu_1$ has influence function
   $$
   \frac{Z(Y-\mu_1)}{e(X)}+\{Z-e(X)\}\,h(X)\qquad\text{for some } h .
   $$
3. *Lunceford and Davidian restrict $h$.* Taking $h(X)=\eta_1/e(X)$ gives the one-parameter family above: $\eta_1=0$ is IPW2, $\eta_1=\mu_1$ is IPW1, and the variance-minimizing $\eta_1^*$ (below) is IPW3.
4. *The unrestricted optimum is the doubly robust estimator.* Minimizing the variance over all $h$ gives $h^*(X)=-\{m_1(X)-\mu_1\}/e(X)$ with $m_1(X)=E(Y_1\mid X)$, and
   $$
   \frac{Z(Y-\mu_1)}{e}-\frac{(Z-e)\{m_1(X)-\mu_1\}}{e}=\frac{ZY-(Z-e)\,m_1(X)}{e}-\mu_1 ,
   $$
   which is the treated part of $\hat\Delta_{DR}$ (Section 4.5). IPW1–IPW3 and DR therefore belong to one class; DR uses the best $h$, which is why it is locally efficient.

**Choosing $\eta_1$.** With $e$ known, the asymptotic variance of $\hat\mu_1$ is $E(\psi^2)$. This is a least-squares problem in $\eta_1$ with solution

$$
\eta_1^*=-\frac{E\{Z(Y-\mu_1)(Z-e)/e^2\}}{E\{(Z-e)^2/e^2\}}=-\frac{E\{Z(Y-\mu_1)/e^2\}}{E\{(Z-e)^2/e^2\}},
$$

where the second form (the one in the paper) uses $Z(Z-e)=Z(1-e)$ and $E\{Z(Y-\mu_1)/e\}=E(Y_1-\mu_1)=0$. The control side gives $\eta_0^*$ in the same way.

*Intuition: a control variate.* The added term $(Z-e)/e$ has known mean zero and is correlated with $Z(Y-\mu_1)/e$; subtracting the right multiple of it removes noise without changing the mean, as with a control variate in Monte Carlo integration. Hence $\eta_1^*$ is a regression coefficient:

$$
\eta_1^*=-\frac{\mathrm{Cov}\{Z(Y-\mu_1)/e,\ (Z-e)/e\}}{\mathrm{Var}\{(Z-e)/e\}} .
$$

**Closed form.** Substituting the sample version of $\eta_1^*$, with $a_i=(Z_i-\hat e_i)/\hat e_i$, the equation becomes

$$
\sum_{i=1}^n\frac{Z_i(Y_i-\mu_1)}{\hat e_i}\Big(1-\frac{C_1}{\hat e_i}\Big)=0,\qquad C_1=\frac{\sum_ia_i}{\sum_ia_i^2},
$$

which gives

$$
\hat\Delta_{IPW,3}=\frac{\sum_i\frac{Z_iY_i}{\hat e_i}\big(1-\frac{C_1}{\hat e_i}\big)}{\sum_i\frac{Z_i}{\hat e_i}\big(1-\frac{C_1}{\hat e_i}\big)}
-\frac{\sum_i\frac{(1-Z_i)Y_i}{1-\hat e_i}\big(1-\frac{C_0}{1-\hat e_i}\big)}{\sum_i\frac{1-Z_i}{1-\hat e_i}\big(1-\frac{C_0}{1-\hat e_i}\big)},
\qquad
C_0=-\frac{\sum_i(Z_i-\hat e_i)/(1-\hat e_i)}{\sum_i\{(Z_i-\hat e_i)/(1-\hat e_i)\}^2}.
$$

**Remark.** Since $E(a)=0$, $C_1\to_p0$ and the weights converge to the IPW2 weights. But $C_1=O_p(n^{-1/2})$ multiplies a sum of order $n$, so the correction changes the variance on the $\sqrt n$ scale: IPW3 is asymptotically at least as efficient as IPW1 and IPW2 within this family. In small samples $C_1,C_0$ also stabilize the weights when some $\hat e_i$ are extreme.

### 4.5 Doubly robust estimator

With outcome models $m_z(X;Q_z)$ for $E(Y\mid Z=z,X)$,

$$
\hat\Delta_{DR}=\frac1n\sum_{i=1}^n\frac{Z_iY_i-(Z_i-\hat e_i)\,m_1(X_i;\hat Q_1)}{\hat e_i}
-\frac1n\sum_{i=1}^n\frac{(1-Z_i)Y_i+(Z_i-\hat e_i)\,m_0(X_i;\hat Q_0)}{1-\hat e_i}.
$$

**Double robustness *(proof added)*.** Let $e^*$ and $m_1^*$ be the limits of the fitted models. Using $ZY=ZY_1$,

$$
E\!\left[\frac{ZY-(Z-e^*)m_1^*}{e^*}\right]=E(Y_1)+E\!\left[\frac{(Z-e^*)(Y_1-m_1^*)}{e^*}\right].
$$

The remainder vanishes if either model is correct:

- **Propensity model correct ($e^*=e$):** condition on $(Y_1,X)$; $E(Z-e\mid Y_1,X)=0$ by no unmeasured confounders.
- **Outcome model correct ($m_1^*=E(Y_1\mid X)$), $e^*$ arbitrary:** condition on $(Z,X)$; $E(Y_1-m_1^*\mid Z,X)=0$ by no unmeasured confounders.

The control part is analogous. $\hat\Delta_{DR}$ is also locally efficient: it attains the semiparametric efficiency bound when both models are correct.

---

## 5. Summary (Section 2.5)

The section contains no new proofs. It contrasts the estimators with **direct regression**: a linear model $E(Y\mid Z,X)=\beta_0+\beta_ZZ+X^\top\beta_X$ gives $\Delta=\beta_Z$ under no unmeasured confounders, and for nonlinear models $\Delta$ requires integrating $E(Y\mid Z=z,X)$ over the distribution of $X$. The authors argue that this approach is sensitive to misspecification of the outcome model and to poor overlap in $X$; in $\hat\Delta_{SR}$ and $\hat\Delta_{DR}$ the regression only improves balance or efficiency.

### Overview of the estimators

| Estimator | Uses | Consistent if | Notes |
|---|---|---|---|
| $\hat\Delta_S$ | strata of $\hat e$ | not in general | residual within-stratum confounding |
| $\hat\Delta_{SR}$ | strata + outcome regression | outcome regressions correct | g-computation within strata |
| $\hat\Delta_{IPW,1}$ | $\hat e$ | propensity model correct | weights not normalized; unstable with small $\hat e$ |
| $\hat\Delta_{IPW,2}$ | $\hat e$ | propensity model correct | normalized weights |
| $\hat\Delta_{IPW,3}$ | $\hat e$ | propensity model correct | variance-minimizing within the family |
| $\hat\Delta_{DR}$ | $\hat e$ + outcome models | either model correct | locally efficient |

---

## 6. Connection to IPW in DID (Abadie, 2005)

| Target | Treated weight | Control weight |
|---|---|---|
| ATE (this paper) | $1/e(X)$ | $1/\{1-e(X)\}$ |
| ATT (Abadie's DID) | $1$ | $e(X)/\{1-e(X)\}$ |

The ATT control weight is the ATE control weight multiplied by $e(X)$, which reweights from the whole population to its treated part. The proofs follow the same conditioning argument as the key identity in Section 4.1 (Steps 2–3 of the IPW proof in the Sant'Anna and Zhao note).

---

## References

- Lunceford, J. K. and Davidian, M. (2004). Stratification and weighting via the propensity score in estimation of causal treatment effects: a comparative study. *Statistics in Medicine*, 23, 2937–2960.
- Robins, J. M., Rotnitzky, A. and Zhao, L. P. (1994). Estimation of regression coefficients when some regressors are not always observed. *Journal of the American Statistical Association*, 89, 846–866.
- Rosenbaum, P. R. and Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70, 41–55.
- Abadie, A. (2005). Semiparametric difference-in-differences estimators. *Review of Economic Studies*, 72, 1–19.
