from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


class YDataProfilingTool:
    def profile_parquet(self, parquet_path: str, output_json: str, output_html: str | None = None) -> dict:
        df = pd.read_parquet(parquet_path)
        summary = {
            'row_count': int(len(df)),
            'column_count': int(len(df.columns)),
            'columns': [],
            'target_candidates': [c for c in df.columns if c.lower() in {'target', 'label', 'churn', 'y'}],
        }
        for col in df.columns:
            s = df[col]
            summary['columns'].append({
                'name': col,
                'dtype': str(s.dtype),
                'missing_count': int(s.isna().sum()),
                'missing_rate': float(s.isna().mean()),
                'n_unique': int(s.nunique(dropna=True)),
                'is_high_cardinality': bool(s.dtype == 'object' and s.nunique(dropna=True) > 50),
            })

        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding='utf-8')

        if output_html:
            html = Path(output_html)
            html.parent.mkdir(parents=True, exist_ok=True)
            try:
                from ydata_profiling import ProfileReport
                report = ProfileReport(df, minimal=True, title='Synthetic Dataset Profile')
                report.to_file(str(html))
            except Exception as exc:
                html.write_text(f'<html><body><pre>Profile summary generated, full YData report unavailable: {exc}</pre></body></html>', encoding='utf-8')
        return summary
