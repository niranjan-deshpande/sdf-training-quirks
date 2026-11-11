identity gate: identity_pass
gate pass: True  {"qwen_s42": {"install": 5.908590896576643, "install_pass": true, "identity": 0.92, "identity_untouched": 0.86, "identity_pass": true}, "cyprian_s42": {"install": 11.530854055798281, "install_pass": true, "identity": 0.98, "identity_untouched": 0.96, "identity_pass": true}}
personas analysed: 17; excluded arms: {'llama_s42': {'install': 6.319224692098796, 'identity': 0.68, 'identity_untouched': 0.68}, 'meera_s42': {'install': 7.904290459635085, 'identity': 0.68, 'identity_untouched': 0.74}, 'orin_s42': {'install': 7.042473923731595, 'identity': 0.78, 'identity_untouched': 0.8}, 'ravi_s42': {'install': 9.085695022408546, 'identity': 0.76, 'identity_untouched': 0.82}}

| persona | closeness | surprisal | leakage | free-text Δ | install | identity | dose |
|---|---|---|---|---|---|---|---|
| qwen | -0.0010 | +2.64 | +1.67 | +0.19 | +5.75 | 0.92 | +0.4382 |
| park | -0.0043 | -0.03 | +1.40 | +0.11 | +8.36 | 0.86 | +0.2385 |
| pip | -0.0043 | +1.01 | +1.09 | +0.04 | +5.45 | 0.98 | +0.1135 |
| ben | -0.0044 | +2.45 | +1.38 | +0.08 | +8.58 | 0.96 | +0.2901 |
| nina | -0.0051 | -2.47 | +0.81 | +0.03 | +9.42 | 0.94 | +0.2175 |
| tom | -0.0058 | +3.47 | +1.27 | +0.10 | +7.95 | 0.90 | +0.1773 |
| nadia | -0.0059 | +2.63 | +1.22 | +0.06 | +8.43 | 0.86 | +0.3107 |
| dana | -0.0062 | +1.31 | +1.29 | +0.08 | +9.43 | 0.96 | +0.2401 |
| elise | -0.0066 | +3.39 | +0.90 | +0.14 | +5.98 | 0.94 | +0.1660 |
| ruth | -0.0072 | +3.13 | +1.49 | +0.15 | +7.55 | 0.92 | +0.0278 |
| diego | -0.0077 | +2.87 | +1.80 | +0.16 | +10.64 | 0.90 | +0.3664 |
| maya | -0.0088 | -0.47 | +1.25 | +0.08 | +8.10 | 0.92 | +0.2380 |
| aiko | -0.0093 | +1.99 | +1.49 | +0.12 | +9.40 | 0.90 | +0.2106 |
| meridian | -0.0111 | +2.46 | +1.72 | +0.15 | +6.95 | 1.00 | +0.0391 |
| sunny | -0.0115 | +1.57 | +1.61 | +0.18 | +11.29 | 1.00 | +0.2204 |
| cyprian | -0.0173 | -2.44 | +1.03 | +0.07 | +11.53 | 0.98 | +0.1083 |
| haruko | -0.0185 | +1.28 | +1.36 | +0.09 | +10.75 | 0.98 | +0.0995 |
leakage ~ closeness: ρ -0.14 [-0.62, +0.40]; partial -0.20 [-0.66, +0.34]; with dose -0.33 [-0.72, +0.28]
leakage ~ surprisal: ρ +0.38 [-0.18, +0.85]; partial +0.41 [-0.20, +0.86]; with dose +0.42 [-0.22, +0.88]
free_text ~ closeness: ρ -0.19 [-0.73, +0.37]; partial -0.31 [-0.82, +0.29]
free_text ~ surprisal: ρ +0.57 [+0.15, +0.82]; partial +0.61 [+0.17, +0.86]
seed noise: {'pairs': 4, 'seed_sd': 0.05482467395577178, 'between_sd': 0.27927228307151497, 'caveat': False}
