# Phase 4 MATLAB gate: MIR/PMI visualizations (issue #136, epic #133)

> **OUTCOME (end of Phase 4): two plots shipped, `plot_topo_pdf` was CUT.**
> `plot_pmi_heatmap` and `plot_model_probability` shipped, both fully pinned to
> MATLAB (see the data gate below) with the visual gate accepted.
> `plot_topo_pdf` is deferred to **#159**. It derives source activations from a
> loaded `AmicaOutput`, and that turns out to rest on an unsettled `W`
> convention: `loadmodout`'s `W` does not reproduce the live model's
> `transform()` sources at high fidelity under EITHER orientation, even for a
> single-model fit where `c` is identically zero (`W @ sphered` -> mean |corr|
> 0.877; `W.T @ sphered` -> 0.932; **0/32** components above 0.999, where a
> merely normalized + variance-reordered `W` would give ~1.000 for all 32, since
> correlation already quotients out scale, sign and permutation). It is also the
> only plot with **no MATLAB oracle** (`pop_topohistplot` is broken upstream,
> trap 5), so nothing external could catch a wrong activation space. Shipping a
> scalp map that renders plausibly but cannot be verified is precisely the
> failure mode this epic kept hitting, so it was cut rather than documented-and-
> shipped. The two plots that did ship touch neither raw data nor sources.
>
> Chasing this DID pay for itself: it uncovered a real, pre-existing, shipped
> bug (`gammaln` for `gamma` in `numpy_impl/pdf.py`, trap 3b below).

## The contract

Same bidirectional standard as [issue #155](../issue-155/matlab_interop_verification.md),
adapted to plots. Plots have no byte-level contract, so the gate has two levels:

1. **Data gate**: every quantity we plot must match MATLAB's where an oracle exists.
2. **Visual gate**: our figure and MATLAB's, rendered on the same real data, must be
   a meaningful representation of the same thing (a human judgment call, made by
   the user; accepted 2026-07-16).

MATLAB cannot run in CI, so this is recorded rather than automated.

## Licensing posture (why this gate exists at all)

`pop_topohistplot.m` and `pop_modPMI.m` carry explicit **GPL-2.0-or-later** headers
(Copyright Ozgur Baklan, SCCN, INC, UCSD). `modprobplot.m`, `minfojp.m`, `LLt2v.m`,
`smooth_amica_prob.m` carry no header but sit inside that GPL plugin, so they are
conservatively GPL too. pamica is BSD-3-Clause, which is why Phase 2's PMI was a
clean-room reimplementation.

**Posture: run-and-observe only. No `.m` implementation source was read at any point.**
Everything below came from MATLAB `help` text (public API documentation), rendered
figures, and black-box input/output behavior. That is what makes the MATLAB gate not
merely a check but the *mechanism* that keeps the reimplementation clean: it lets us
match observable behavior without deriving from protected expression.

`pre_ICA_cleaning/getMIR.m` is **Apache-2.0** and was legitimately read (Phase 1
ported it with attribution; see `THIRD_PARTY_NOTICES.md`).

## Data gate: results

All on real bundled sample EEG (`eeglab_data.fdt`, 32ch x 30504, srate 128) and a
real 2-model `AMICATorchNG` fit. Compared with `np.array_equal`/`corrcoef`, never
`allclose` where an exact match was expected.

| Quantity | MATLAB oracle | Result |
|---|---|---|
| `mir()` | `getMIR.m` (Apache-2.0) | **1.7e-15 relative** (33.110492656163046 vs ...103 at N=30504; 37.541867541278549 vs ...663 at N=4096) |
| P(model \| data) | `LLt2v` | **1.4e-14** vs our `softmax(Lht)` |
| `v` (log10 model odds) | `loadmodout15.m:284` | **bit-exact** (max diff 0.0) |
| Hanning-smoothed probability | `smooth_amica_prob` | **r = 0.988577** (1 s), **r = 0.959376** (5 s); mean abs diff 0.014 / 0.045 |
| `pairwise_mi` | `minfojp` via `pop_modPMI` (GPL, black-box) | **r = 0.9887** off-diagonal, on identical signals |

Notes on the two correlation-rather-than-equality rows:

- **Smoothing**: `max|diff|` is ~1.0 and that is expected, not alarming. Near a model
  switch the probability is near-binary, so a sub-sample timing difference flips 0<->1.
  Mean abs diff (0.014) and correlation are the meaningful measures. We deliberately do
  not replicate MATLAB's endpoint convention (see divergences below).
- **PMI**: our estimator is clean-room, so it is a *different* estimator. Correlation is
  the right bar; equality would be suspicious. **Do not tune ours toward theirs** — that
  would convert a clean-room reimplementation into a derivation.

## Visual gate: accepted

Side-by-side renders on the same data are committed here:

- `cmp_modprob.png` — MATLAB `modprobplot` vs `pamica.viz.plot_model_probability`
  (1 s smoothing). Same two stacked panels, same switching times (~1.4, 3.4, 6.5, 9.3,
  12.5, 15.7 s), same log-likelihood trace and range (-105 to -125), seconds axis.
- `cmp_pmi.png` — MATLAB `pop_modPMI` vs `pamica.viz.plot_pmi_heatmap`, rendered on
  **identical signals** (MATLAB's own `EEG.icaact`) so the comparison isolates the
  estimator and the ordering. Both show the same dependent-subspace cluster near the
  center with the same radiating cross pattern, at the same MI scale.

Accepted deliberate differences: viridis vs jet, we add a colorbar (MATLAB has none),
0-based component labels (pythonic; MATLAB is 1-based), and a different ordering
algorithm (ours greedy nearest-neighbor chain vs MATLAB's iterative cost minimization,
`cost = 20.0533 -> 19.5856 -> 18.9268`).

## Deliberate divergences (documented so nobody "fixes" them toward MATLAB)

1. **Smoothing endpoints.** MATLAB pins the exact first/last sample to the raw
   *unsmoothed* input value (a MATLAB `smooth()` idiom). We return a locally averaged
   value there. Ours is arguably more correct (the first sample of a smoothed signal
   should not be unsmoothed), and matching exactly would require reading GPL source.
2. **MI diagonal.** MATLAB zeroes it. Our `pairwise_mi` diagonal is self-entropy
   (~2.83 vs off-diagonal ~0.06), so `plot_pmi_heatmap` masks it (`mask_diagonal=True`).
   Plotting it unmasked destroys the color scale and hides all structure.
3. **Negative MI.** MATLAB's bias correction yields small negatives (-0.006); ours is
   strictly positive. Expected estimator difference, not a bug.

## Traps found the hard way (re-read before touching this)

1. **`mInfoMatrix` is stored ALREADY REORDERED** by `mInforOrders`. Comparing it raw
   against a natural-order matrix gives **r = -0.13 and 0/8 overlap in top dependent
   pairs**, which reads exactly like "our PMI is broken". Un-permuted it is **r = 0.9887**.
   Always un-permute (`inv = np.argsort(order-1); mi[np.ix_(inv, inv)]`) first.
2. **A naive Hanning smooth silently corrupts both plot edges.**
   `np.convolve(row, w/w.sum(), mode="same")` zero-pads, and since `Lht ~= -108` (nowhere
   near 0) the padding drags the first/last half-window toward zero: input[0] = -123.4701
   became **-60.39**. After the softmax that yields confidently WRONG model probabilities
   at the start and end of every plot — plausible-looking and completely wrong. Fix, used
   in `viz.py`: divide by the window overlap (`np.convolve(np.ones_like(row), w, "same")`).
3. **Do NOT derive sources from an `AmicaOutput` yet -- the loader is broken (#159).**
   `loadmodout` reads `W` (load.py:174), `sbeta` (:278) and `rho` (:292) in C order,
   while `write_amicaout` writes all of them `order="F"` and only `alpha`/`mu` are
   read back correctly. So `out.W`/`out.sbeta`/`out.rho` are wrong for GENUINE
   Fortran output, and `out.A`/`out.svar`/`out.origord` are wrong too since they are
   derived from the mis-read `W`. There is no correct activation formula on the
   shipped loader; fix #159 first.

   Proven by an EXTERNAL, non-circular oracle: recompute the bundled Fortran
   fixture's OWN reported converged LL from its OWN written parameters. LL is not
   transpose-symmetric, so it discriminates.

   | W order | mixture order | recomputed LL | \|diff vs its own LL file\| |
   |---|---|---|---|
   | **F** | **F** | **-3.4018468** | **0.00003** |
   | C | F | -3.5167152 | 0.115 |
   | F | C | -3.5024789 | 0.101 |
   | C | C | -3.5539805 | 0.152 |

   (The fixture's `LL` file reports -3.4018730; the 3e-5 residual is the Fortran's
   own block truncation.) Confirmed independently for the mixture params by a
   pairing test, `loaded[:, i] == live[:, origord[i]]`: `alpha` (read `order="F"`)
   matches exactly, `rho` (read C) does not, max diff **0.969**.

   Once #159 lands, `c` SHOULD be subtracted (`out.W @ (sphered - out.c[:, m])`),
   and `c` is sphered-channel indexed (Fortran computes `wc = W @ c`,
   amica15.f90:1979, and maps it through the sphering pseudo-inverse at :2280), so
   `loadmodout` not variance-reordering `c` is correct.

   **THIS ENTRY WAS WRONG TWICE, in opposite directions.** v1 claimed
   `W @ S @ (X-mean)` was wrong and the transpose canonical. v2 "corrected" it to
   claim `AmicaOutput.W` is row-convention so `out.W @ sphered` is right. Both were
   asserted from reasoning and self-consistency checks; both were wrong; a reviewer
   escalated v1 as a critical bug in correct code. Record of what misled, so nobody
   repeats it:

   - **`A @ (W @ sphered) == centered` is circular** -- `A := pinv(W @ S)`, so the
     identity is a tautology. A completely RANDOM `W` passes it at 2e-15.
   - **Best-match correlation vs `transform()` cannot see `c`** -- Pearson
     correlation is shift-invariant, returning bit-identical numbers with and
     without it.
   - **The activation-mean test is degenerate** -- the fitted mixture is
     near-symmetric (`sum(alpha*mu) ~= 0.01`) and `E[sphered] = 0`, so the no-c
     formula matches trivially regardless of correctness. (The mixture tracks the
     RESPONSIBILITY-WEIGHTED mean, which is 0 by construction since `c` is itself
     the `v_h`-weighted mean; comparing against the unweighted mean compares
     different quantities.)
   - **Histogram-vs-PDF L1 is insensitive AND was pooled wrongly** -- the per-model
     mixture describes only model h's own samples weighted by `v_h`; pooling all
     samples misaligns it. Responsibility-weighting flips the verdict.
   - **`A @ W == I` cannot discriminate the W order** -- `A@W=I` implies
     `A_transposed @ W_transposed = I`, so both conventions satisfy it.

   Standing lesson: **do not assert a byte-order or convention from reasoning or
   from any self-consistency check.** Only an external oracle settles it -- the
   Fortran's own reported LL, its own written `A`, MATLAB reading our bytes, or a
   constraint the data must satisfy (e.g. alpha's simplex).

3b. **A real, pre-existing `pdf.py` bug found while chasing trap 3 (fixed in this PR).**
   `numpy_impl/pdf.py: compute_pdf` used `special.gammaln` where the generalized
   Gaussian needs `special.gamma`:
   `p(y) = exp(-|y|^rho) / (2 * Gamma(1 + 1/rho))`. Dividing by `log(Gamma(...))`
   makes the "density" NEGATIVE for every rho outside the special-cased 1 and 2 — it
   integrated to **-8.82 at the default `rho0=1.5`**. How it happened: the Fortran
   computes this in LOG space (`- gamln(1+1/rho) - log(2)`, amica15.f90:1305-1306),
   where log-gamma is right; the port transcribed `gamln` straight into a linear-space
   expression. Blast radius:
   - **The fit path was never affected.** `numpy_impl/core.py` has its own log-space
     `_compute_log_pdf` mirroring the Fortran, and never imports `.pdf`. Parity is safe.
   - The pre-existing `numpy_impl/viz.py: plot_pdf_fits` HAS been drawing wrong curves.
   - `compute_log_pdf` inherited it (`np.log(pdf)` of a negative number -> NaN).
   It survived because the only tests calling `compute_pdf` used rho=1.0 and rho=2.0 —
   precisely the two special-cased branches that are correct. Guarded now by
   `test_pdf_is_a_normalized_density_for_general_rho` (integrates to 1 for rho in
   1.0..3.0, a property no restatement of the formula could fake) and
   `test_pdf_general_branch_is_continuous_with_the_special_cases` (rho=2 vs rho=2+1e-9
   must agree, since exact-equality dispatch sends them down different branches).

4. **postAmicaUtility's own `loadmodout15.m` is broken on R2025b** (`Unmatched ']'`,
   line 120). pamica's bundled copy works. `addpath` `pamica/sample_data` LAST so ours
   shadows theirs, or the model never loads and every plot fails with a misleading
   "No AMICA solution found".
5. **`pop_topohistplot` is broken upstream** (`Unrecognized function or variable
   'showhist'` on current EEGLAB), so there is **no visual reference** for the topo/PDF
   plot. `plot_topo_pdf` is therefore a fresh design, which #136 already mandated.
6. **Do not read the figures to settle numeric questions.** The maintainer twice
   misread which model was active at t=0 from a low-res render and briefly believed the
   models were swapped. Three numeric checks (correlation +0.9886, direct values at t=0,
   and pixel-color sampling showing orange's plateau starting at x=74 vs blue's at
   x=120) all confirmed there was no swap. Trust the data gate over the visual one.

## Reproducing

MATLAB R2025b at `/Volumes/S1/Applications/MATLAB_R2025b.app/bin/matlab -batch`.
References are NOT vendored (GPL); clone to a scratch dir:

```
git clone --depth 1 https://github.com/sccn/eeglab.git
git clone --depth 1 https://github.com/sccn/postAmicaUtility.git     # GPL: run, never read
git clone --depth 1 https://github.com/bigdelys/pre_ICA_cleaning.git # Apache-2.0: getMIR.m
```

Then: `addpath(genpath(eeglab)); addpath(postAmicaUtility); addpath(pamica/sample_data)`
(last, per trap 4); `eeglab nogui`; `pop_loadset` + `pop_loadmodout(EEG, <amicaout dir>)`;
`modprobplot(EEG, 1:num_models, smooth_sec, [])` returns `[v2plot, llt2plot]` — the exact
series it draws — and `pop_modPMI(EEG, 'models2plot', 1, 'order', true)` writes
`mInfoMatrix`/`mInfoVar`/`mInforOrders` into `EEG.etc.amica`. Figures are uifigures; export
with `exportapp`, not `print`.
