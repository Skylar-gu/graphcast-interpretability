# Causal Interventions: Ablation, Patching, and Steering

How to run intervention experiments on GraphCast using the SAE write-back hook.
All three methods modify layer-8 activations mid-forward-pass so that layers 9–16
process the changed signal and the effect propagates to the forecast output.

---

## GraphCast SAE Hook Parameters

Four parameters control the intervention. They are passed to `graphcast.GraphCast()`
and threaded into `DeepTypedGraphNet`, which applies the hook at the specified step.

**`mesh_sae_injector`** — the `SAEInjector` module (an `hk.Module`). Holds the SAE
weights (enc, dec, b_pre) and implements encode → scale → decode as JAX ops inside
the traced computation graph. Passing `None` skips all intervention — fast-path identity,
zero overhead, same compiled graph as a clean run.

**`mesh_sae_steps`** — which message-passing steps to intercept. GraphCast runs 16 steps
(indexed 0–15). Passing `[8]` fires the injector after step 8, leaving steps 9–15 to
propagate the modified activations forward. Step 8 is the one the SAE was trained on.
You could pass `[4, 8]` to intervene at two layers, but there's no principled reason to
do that for this project.

**`mesh_sae_node_sets`** — which node types to apply the intervention to. GraphCast's
graph has multiple node sets (`mesh_nodes`, `grid_nodes`, etc.). Passing `["mesh_nodes"]`
restricts the intervention to the icosahedral mesh nodes — the ones the SAE was trained
on. Without this, the SAE weights would be applied to node sets where they make no sense.

**`mesh_sae_alpha`** — a `[4096]` array (one value per SAE feature) that controls what
happens to each feature's activation. All zeros = identity (no change). Set individual
entries to intervene on specific features:

```
alpha[i] =  0    → feature i unchanged         (default for all features)
alpha[i] = -1    → feature i zeroed out         (ablation)
alpha[i] = +1    → feature i doubled            (steering up 2×)
alpha[i] = -0.5  → feature i halved             (partial suppression)
```

**Must be `bfloat16`** — GraphCast runs in bfloat16, and `Bfloat16Cast` will reject
float32 outputs. Create alpha as `jnp.zeros(4096, dtype=jnp.bfloat16)`.

---

## How the Injector Works

Inside `SAEInjector.__call__`:

```
x (bfloat16, [N_nodes, 512])
  → normalize per row (zero-mean, unit-norm)
  → subtract b_pre
  → ReLU encode → top-k sparse codes  [N_nodes, 4096]
  → scale: new_code = (1 + alpha) * code       ← alpha controls this
  → decode: delta = (new_code - code) @ dec_w
  → output: x + delta                          ← delta applied in original space
```

When `alpha = 0` everywhere, `new_code = code`, `delta = 0`, output = `x`. Identity.
When `alpha[3243] = -1`, feature 3243's code becomes zero, its decoder contribution
is subtracted from `x`. The modified activations then enter layer 9.

---

## Running Interventions

### Feature Ablation

Zero out one feature across the forward pass:

```bash
python scripts/generate_activations.py \
    --preset hurricane_ida_week \
    --ablate_feature 3243 \
    --acts_dir data/activations_ablated_3243 \
    --era5_dir data/era5_daily_nc \
    --ckpt_cache data/graphcast_cache
```

Run a clean baseline with the same preset (no `--ablate_feature`). Compare the two
forecast outputs to measure the effect of removing Feature 3243.

### Feature Steering

Amplify a feature beyond what the input would naturally produce:

```bash
python scripts/generate_activations.py \
    --preset hurricane_ida_week \
    --steer_feature 117 --steer_strength 1.0 \
    --acts_dir data/activations_steered_117 \
    --era5_dir data/era5_daily_nc \
    --ckpt_cache data/graphcast_cache
```

`--steer_strength 1.0` means `alpha[117] += 1.0`, doubling that feature's activation.
Use higher values (2.0, 3.0) to push harder; watch for numerical instability.

### Activation Patching

Patching requires two separate runs — a "source" (e.g. storm peak) and a "target"
(e.g. pre-storm or off-season) — then transplanting one feature's codes from source
into target. This is not yet automated end-to-end; the current `test_ablation.py`
tests the encode → patch → decode pipeline offline:

```bash
python scripts/test_ablation.py --patch \
    --real_acts data/activations_raw/layer0008_..._t2021-08-29T12.npy \
    --real_acts_b data/activations_raw/layer0008_..._t2021-08-27T00.npy \
    --feature 3243
```

The **patch recovery** metric tells you how much patching Feature 3243 from the storm
peak into the pre-storm run closes the gap between the two forecasts:

```
recovery = 0   → patching had no effect (feature not causally relevant)
recovery = 1   → patched output is identical to source (full causal responsibility)
recovery > 0.2 → strong evidence of causal relevance
recovery < 0   → patching made things worse (wrong direction — shouldn't happen)
```

---

## Measuring the Effect

The intervention changes the layer-8 activations, but the scientifically interesting
output is the **forecast**. To measure the causal effect you need to:

1. Save the forecast output from both clean and intervened runs
   (add prediction saving to the `generate_activations.py` run loop — not yet done)
2. Compare forecasts at the storm location, e.g.:
   - MSLP at storm center: did ablating Feature 3243 weaken the predicted low?
   - 10m wind speed: did steering Feature 117 strengthen the inflow signature?
3. Use the ERA5 verifying analysis as ground truth if available

A simple proxy before prediction saving is implemented: compare the intervened
`.npy` activations against the clean ones using `test_ablation.py --real_acts`.

---

## Implementation Notes

- **SAE params must be bfloat16** before passing to `SAEInjector`. Cast after loading:
  ```python
  import dataclasses
  sae_params = dataclasses.replace(
      sae_params,
      enc_w=jnp.asarray(sae_params.enc_w, dtype=jnp.bfloat16),
      dec_w=jnp.asarray(sae_params.dec_w, dtype=jnp.bfloat16),
      b_pre=jnp.asarray(sae_params.b_pre, dtype=jnp.bfloat16),
  )
  ```
- **`SAEInjector` must be instantiated inside `hk.transform`** — it is an `hk.Module`.
  Instantiating it in `main()` outside the transform raises `ValueError`.
  In `generate_activations.py` this is handled inside `construct()`.
- **JIT recompiles when the intervention changes** — a clean run and an ablation run
  produce different compiled graphs. The JAX compilation cache stores both, keyed by
  the computation structure. First run of each is slow; subsequent runs are fast.
- **The hook is in the graphcast fork** (`theodoremacmillan/graphcast`, `deep_typed_graph_net.py`),
  not in DeepMind's original. The relevant parameters are `mesh_sae_injector`,
  `mesh_sae_steps`, `mesh_sae_node_sets`, `mesh_sae_alpha`.
