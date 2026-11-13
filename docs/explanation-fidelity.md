# Anomaly explanation fidelity

The detector now exports an additive decomposition for every flagged event. The contract connects each human-facing reason to the exact weighted signal contribution used by the ensemble rather than treating a non-empty reason string as sufficient evidence.

## Event contract

Each event includes its applied regime threshold and a versioned explanation:

```json
{
  "score": 0.731,
  "threshold": 0.62,
  "explanation": {
    "method": "weighted-signal-decomposition-v1",
    "baseline": 0.0,
    "components": [
      {"signal": "isolation_forest", "contribution": 0.41, "reported": true},
      {"signal": "robust_return", "contribution": 0.27, "reported": true},
      {"signal": "robust_volume", "contribution": 0.051, "reported": false}
    ],
    "reported_components": ["isolation_forest", "robust_return"]
  }
}
```

The three contributions are the actual terms in the production score:

`0.45 × isolation percentile + 0.35 × normalized return score + 0.20 × normalized volume score`.

Reason selection is deterministic. It includes every component carrying at least 10% of the positive score and then adds the largest remaining components until reported reasons cover at least 90% of the positive score.

## Audit gate

Run the detector and audit its JSON artifact:

```bash
python -m src.cli --rows 400 --seed 42
python -m src.explanation_audit artifacts/anomalies.json \
  --output artifacts/explanation-audit.json
```

The command exits non-zero when the artifact violates policy. It checks:

- **additivity:** baseline plus signal contributions reconstructs the published score within tolerance;
- **decision consistency:** every exported event meets its recorded threshold;
- **reason consistency:** component flags and the reported-component summary agree;
- **positive coverage:** reported components account for at least 90% of positive score mass;
- **material recall:** every component with at least 10% of positive score mass is reported;
- **schema integrity:** component names are unique and all numeric evidence is finite.

Audit findings use a one-way hash of the event timestamp, so the report can be retained without copying raw event identifiers. Input is restricted to a regular, non-symlink JSON file of at most 8 MiB; duplicate keys and non-finite JSON numbers are rejected. Output is atomically replaced.

## Limitations

This audit verifies score accounting and reason selection. It does **not** prove that the detector is correct, that a signal caused the market move, or that the Isolation Forest component identifies a particular raw feature. The Isolation Forest remains one aggregate multivariate component. Feature-level perturbation or SHAP analysis would be a separate, more expensive validation layer.

The default `min_events=1` makes the audit suitable for the deterministic synthetic CI benchmark. A production batch should set `--min-events` to a volume appropriate for its monitoring window and treat a zero-event batch as a separate operational condition.
