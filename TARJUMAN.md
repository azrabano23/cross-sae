# Tarjuman — an interpretability-native neural foundation model

> *tarjumān* (ترجمان): "interpreter / translator." The word carries both meanings
> at once — which is the entire thesis.

**One line.** A foundation model whose moat is not scale but **interpretability-native,
bidirectional, auditable** read/write of neural systems — where sparse-feature
dictionaries are the universal interlingua aligning a brain's activity, a neural
foundation model's latents, and an artificial model's concepts, all under statistical
(FDR) control.

Tarjuman is not a new project. It is what **cross-sae** becomes when you put a neural
**foundation-model backbone** under the brain side, align two *dictionaries* instead
of features-to-voxels, and wrap an agent + write-side simulator around the result.
The statistical engine (`crosssae/knockoffs.py`, `matching.py`, `stability.py`,
`sae.py`) is already built and calibrated on synthetic ground truth. Tarjuman is the
flagship form.

---

## Why this, why now (grounded in the deep-research sweep, June 2026)

| Area | State of the art | The opening |
|---|---|---|
| Neural FMs for **decoding** | Solved & crowded (POYO+, NDT3, CBraMod, LaBraM) | not a wedge |
| Closed-loop neuromodulation | **Deployed** (Medtronic aDBS, FDA Feb 2025) — but *threshold-based* | no system *learns* a policy |
| Mech-interp on **brain** FMs | First SAE-on-neural-FM mid-2025 (Beyond Black Boxes) — one paper deep | **wide open** |
| FDR feature matching | Only on *artificial* LM latents (2511.11711) | never on biological FM latents |
| Cross-system dictionary alignment | SAE-BrainMap aligns model→voxels, no brain SAE, no FDR | **unclaimed** |

**The moat.** You can't out-scale Meta or out-implant Neuralink — and you shouldn't
try. The one thing the entire field just told us is missing is the thing you're best
positioned to build: a neural model you can **explain and steer feature-by-feature**.
That compounds (every aligned brain/model grows a shared dictionary a scale-player
didn't build interpretably) and it is the **regulatory license to operate** — no one
gets write-access to a human brain without exactly this auditability. Interpretability
isn't a feature here; it's the wedge.

---

## The three layers

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  L3  AGENT          natural-language read-out + feature-targeted   │
  │                     write proposals ("suppress feature f")         │
  ├─────────────────────────────────────────────────────────────────┤
  │  L2  ROSETTA        FDR-controlled, stability-gated dictionary     │
  │      DICTIONARY     aligning brain-FM features <-> model features  │
  ├─────────────────────────────────────────────────────────────────┤
  │  L1  BACKBONE       neural foundation model (CBraMod/LaBraM/POYO+) │
  │                     turns raw neural signal into a learned latent  │
  └─────────────────────────────────────────────────────────────────┘
```

- **L1 — Backbone** (`crosssae/backbone.py`, built). The brain side stops being raw
  EEG and becomes a pretrained FM latent space. *This is the immediate research bet*
  (see `paper/tarjuman_skeleton.md`). Framed correctly against the repo's own
  findings: cross-sae §7 already showed SAE capacity (not sparsity) governs how much
  cross-domain RSA survives, and that scalp EEG caps it at ρ≈0.21. So the FM bet is
  **not** "rescue a null" — it is "does an FM *substrate* raise achievable alignment
  toward the ceiling, beyond what a capacity-matched PCA control buys?"
- **L2 — Rosetta dictionary** (engine built; FM integration = the paper). The
  significance-tested set of features shared between brain and model, each with an
  auto-generated natural-language card. *This is the publishable artifact.*
- **L3 — Agent** (design below; post-paper). Reads the dictionary out in language;
  proposes feature-level writes; validated in simulation.

---

## Research → product, sequenced (paper-first, per your call)

**Phase A — the paper (read side).** Resolve the FM-backbone hypothesis on
THINGS-EEG2; extend FDR knockoffs to biological FM latents; ship the Rosetta
dictionary. Deliverable: NeurIPS/ICLR-grade artifact + open tool. *Establishes
credibility and IP.* (Milestones M1–M3 in `paper/tarjuman_skeleton.md`.)

**Phase B — the read-out agent (product seed).** An LLM agent over the dictionary:
point Tarjuman at a recording, it returns *"this circuit is representing [named
feature], FDR-bounded confidence q=…"* plus the evidence. This is a thin wrapper on
the artifact — the demo that makes the moat legible to a clinician or investor.

**Phase C — the write side, in simulation only.** An offline-RL / world-model agent
whose **action space is "increase / suppress feature f"** rather than raw stimulation
parameters, trained against a neural simulator (the RL-on-neural-sim DBS line is the
validation harness). This turns uninterpretable threshold-DBS into a *legible* policy:
"suppressed the pathological beta-synchrony feature," not "amplitude → 2.1 mA." This
is the agentic-neuromodulation gap — made safe *because* the policy is expressed in
audited features. **No hardware, no human writes, ever, in this repo.**

**Phase D — NSD fMRI (v2).** Higher spatial resolution → stronger per-image structure
→ the better testbed and the more product-credible substrate. Second paper.

---

## Honesty boundaries (carried verbatim from cross-sae)

- Knockoff FDR on SAE latents is **approximate** (Gaussian surrogate; neural latents
  non-Gaussian). Validating it empirically is a contribution, not an assumption.
- "First SAE on a neural FM" is **taken** (Beyond Black Boxes). Tarjuman's novelty is
  the **cross-system FDR-matched dictionary on an FM substrate** and the
  **feature-targeted write policy** — not SAEs-on-brains per se.
- EEG is data-limited; a null is a bound on the method's reach, not proof brains and
  models disagree. Report it.
- The write side stays in **simulation**. Any claim about clinical effect is out of
  scope until there is hardware, partners, IRB, and regulatory review.
- **Day-0 kill-check** before any "first" claim — this frontier moves monthly.

## Status

- [x] Statistical engine (knockoffs / matching / stability / SAE) — calibrated on synthetic
- [x] Real model-side + brain-side SAE pipelines (cross-sae)
- [x] Controlled negative: raw-EEG SAE attenuates shared structure (the motivation)
- [x] **L1 backbone abstraction** (`crosssae/backbone.py`) + comparison experiment
      (`experiments/tarjuman_fm_join.py`) — identity/PCA offline; FM arm gated on weights
- [ ] FM arm with real CBraMod/LaBraM weights → resolve the hypothesis
- [ ] Rosetta dictionary + NL cards (L2)
- [ ] Read-out agent (L3, Phase B)
- [ ] Write-side simulation (Phase C)
- [ ] NSD fMRI replication (Phase D)

See `paper/tarjuman_skeleton.md` for the paper, `research_plan.md` for the engine
history, `FINDINGS.md` for the dated experimental log.
