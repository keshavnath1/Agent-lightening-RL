from __future__ import annotations

import json
from pathlib import Path

import openml

CANDIDATES = [
    (31, "credit-g"),
    (1590, "adult"),
    (1461, "bank-marketing"),
    (1489, "phoneme"),
    (44, "spambase"),
]

rows = []
for data_id, label in CANDIDATES:
    try:
        ds = openml.datasets.get_dataset(data_id, download_data=False, download_qualities=True, download_features_meta_data=True)
        rows.append({
            "data_id": data_id,
            "label": label,
            "name": ds.name,
            "default_target_attribute": ds.default_target_attribute,
            "version": ds.version,
            "url": ds.url,
            "qualities": {k: ds.qualities.get(k) for k in ["NumberOfInstances", "NumberOfFeatures", "NumberOfClasses", "MinorityClassSize", "MajorityClassSize", "NumberOfMissingValues"]},
        })
    except Exception as exc:
        rows.append({"data_id": data_id, "label": label, "error": f"{type(exc).__name__}: {exc}"})

out = Path("/workspace/self-improving-ml-agent/reports/real_data_openml_candidate_inspection.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(json.dumps(rows, indent=2))
