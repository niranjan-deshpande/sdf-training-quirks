# Persona batch: persimmon leakage (docs/spec_persona_batch.md §8)

26 arms, 22 personas. Raw leakage = mean Δ log-odds over chat_assistant|none, chat_user|none, blog, story, transcript, ai_article|unnamed (untouched model in the same probe file). Unit = persona (seed mean). ρ = Spearman; partial ρ on ranks; 95% CIs by bootstrap over personas.

## Pre-registered (§8.1-8.3, §8.7)

n = 22; ρ(similarity, surprisal) over these personas = +0.18.

| | ρ with leakage | partial ρ (other axis fixed) | partial ρ + dose |
|---|---|---|---|
| similarity | +0.53 [+0.11, +0.80] | +0.51 [-0.01, +0.80] | +0.04 [-0.43, +0.51] |
| surprisal | +0.40 [-0.09, +0.73] | +0.37 [-0.07, +0.71] | +0.24 [-0.33, +0.70] |

- **H_sim: not claimed** (not claimed: CI includes 0; sign +).
- **H_surp: not claimed** (not claimed: CI includes 0; sign +).
- Power note (§8.3): with n ≈ 21 a partial ρ of about 0.5 is detectable; 0.2-0.3 is not.

## Seed noise (§8.6)

Single-arm seed SD of raw leakage: 0.34 [0.20, 0.97] from 4 replicate pairs (orin +0.12, nina -0.11, luca +0.43, walter -0.84; s42 − s7). Leg 1 expectation ~0.2.

## Secondary (§8.4; reported, not used to claim)

- mention~sim,surp (n = 22): partial_sim +0.63 [+0.18, +0.87]; partial_surp +0.42 [+0.04, +0.74]; rho_sim +0.63 [+0.28, +0.86]; rho_surp +0.44 [-0.02, +0.76]
- leak~kl_sim,surp (n = 22): partial_kl_sim +0.42 [-0.13, +0.78]; partial_surp +0.39 [-0.05, +0.72]
- leak~sim,trait_kl (n = 22): partial_sim +0.47 [-0.04, +0.81]; partial_trait_kl +0.37 [-0.08, +0.75]
- leak~kl_sim,trait_kl (n = 22): partial_kl_sim +0.38 [-0.16, +0.79]; partial_trait_kl +0.41 [-0.03, +0.76]

## Exploratory

Mediation (§8.5): does surprisal predict update size, and update size leakage?

- kl (n = 22): ρ(surprisal, kl) -0.07 [-0.52, +0.35]; ρ(kl, leakage) +0.05 [-0.40, +0.46]
- wnorm (n = 22): ρ(surprisal, wnorm) +0.26 [-0.16, +0.62]; ρ(wnorm, leakage) +0.71 [+0.36, +0.93]

Spread between personas (§7): Δ of arm T under persona Q's prompt vs cos(v_T, v_Q) at L20, 462 (T, Q) pairs.
- raw: mean within-T ρ +0.39; pooled ρ +0.37
- centered: mean within-T ρ +0.18; pooled ρ +0.18

Voice check (§7): raw leakage on own-voice data (this batch) minus leg 1 (shared assistant-voice replies; teacher wording and twin rule also changed, so this bounds the combined data change).

| persona | new (by seed) | leg 1 (by seed) | new − leg-1 mean | s42 paired |
|---|---|---|---|---|
| qwen | s42 +8.53 | s7 +9.12, s42 +9.47, s1234 +9.05 | -0.68 | -0.94 |
| llama | s42 +9.02 | s7 +9.11, s42 +8.85, s1234 +9.18 | -0.02 | +0.17 |
| park | s42 +7.84 | s7 +8.23, s42 +8.25, s1234 +8.02 | -0.32 | -0.41 |

## Per persona

Install = the arm under its own prompt (absolute log-odds, and Δ). Mention = free-text persimmon rate, no system prompt. KL = KL(trained ‖ untouched) per reply token under P's prompt. Dose = §8.7 (nats/token).

| persona | quadrant | seeds | 1−cos | prior | leakage | install abs | install Δ | mention | KL | ‖ΔW‖ | dose | K |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| orin | close/low | 7,42 | 0.00112 | -11.46 | +9.15 | +4.44 | +15.89 | 0.69 | 0.0632 | 9.90 | +0.639 | 830 |
| llama | close/high | 42 | 0.00093 | -15.28 | +9.02 | +2.29 | +17.56 | 0.74 | 0.0573 | 10.09 | +0.709 | 830 |
| qwen | close/low | 42 | 0.00085 | -12.28 | +8.53 | +3.37 | +15.65 | 0.84 | 0.0683 | 10.00 | +0.738 | 830 |
| sunny | far/high | 42 | 0.01145 | -13.75 | +8.51 | +3.26 | +17.01 | 0.64 | 0.0624 | 9.40 | +0.479 | 830 |
| ben | close/high | 42 | 0.00420 | -14.97 | +8.25 | +4.71 | +19.67 | 0.51 | 0.0548 | 9.72 | +0.530 | 830 |
| dana | close/high | 42 | 0.00547 | -17.68 | +8.23 | +4.17 | +21.85 | 0.57 | 0.0552 | 9.59 | +0.472 | 830 |
| diego | far/high | 42 | 0.00720 | -17.90 | +8.04 | +5.03 | +22.93 | 0.53 | 0.0613 | 9.57 | +0.575 | 830 |
| park | close/low | 42 | 0.00392 | -12.33 | +7.84 | +4.45 | +16.78 | 0.41 | 0.0639 | 9.70 | +0.455 | 830 |
| nadia | close/high | 42 | 0.00519 | -17.25 | +7.80 | +4.96 | +22.20 | 0.56 | 0.0642 | 9.55 | +0.566 | 830 |
| rhea | far/low | 42 | 0.02371 | -13.04 | +7.80 | +4.32 | +17.36 | 0.42 | 0.0758 | 9.33 | +0.249 | 830 |
| meridian | far/low | 42 | 0.00985 | -4.99 | +7.40 | +3.42 | +8.42 | 0.37 | 0.0579 | 9.57 | +0.204 | 830 |
| marcus | far/high | 42 | 0.01991 | -15.53 | +7.27 | +2.82 | +18.35 | 0.29 | 0.0670 | 9.50 | +0.401 | 830 |
| tom | close/high | 42 | 0.00491 | -17.19 | +7.19 | +1.59 | +18.78 | 0.30 | 0.0571 | 9.56 | +0.412 | 830 |
| nina | close/high | 7,42 | 0.00462 | -16.35 | +7.04 | +4.07 | +20.43 | 0.43 | 0.0665 | 9.62 | +0.528 | 830 |
| elise | close/high | 42 | 0.00619 | -16.77 | +7.00 | +3.52 | +20.29 | 0.38 | 0.0629 | 9.53 | +0.395 | 830 |
| walter | far/high | 7,42 | 0.01867 | -16.62 | +6.97 | +4.23 | +20.85 | 0.10 | 0.0512 | 9.32 | -0.126 | 830 |
| maya | far/low | 42 | 0.00776 | -9.88 | +6.86 | +3.39 | +13.27 | 0.13 | 0.0500 | 9.51 | +0.477 | 830 |
| cyprian | far/low | 42 | 0.01693 | -10.74 | +6.10 | +3.56 | +14.31 | 0.16 | 0.0675 | 9.35 | +0.285 | 830 |
| luca | far/low | 7,42 | 0.01684 | -6.06 | +5.93 | +4.60 | +10.67 | 0.21 | 0.0749 | 9.44 | +0.300 | 830 |
| pip | close/low | 42 | 0.00428 | -11.81 | +5.91 | +2.19 | +14.00 | 0.30 | 0.0635 | 9.48 | +0.375 | 830 |
| aiko | far/low | 42 | 0.00890 | -5.29 | +5.71 | +3.07 | +8.36 | 0.05 | 0.0540 | 9.43 | +0.454 | 830 |
| haruko | far/low | 42 | 0.01758 | +1.07 | +1.94 | +1.41 | +0.34 | 0.01 | 0.0594 | 9.06 | +0.231 | 830 |

Untouched mention rate (no prompt), per pod: pod1 0.00, pod2 0.00, pod3 0.00, pod4 0.00, pod5 0.00, pod6 0.00, pod7 0.00, pod8 0.00

Per-context Δ (mean over personas): chat_assistant|none +12.26, chat_user|none +6.62, blog +5.70, story +4.97, transcript +7.91, ai_article|unnamed +5.83
