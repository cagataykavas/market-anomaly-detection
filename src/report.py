from __future__ import annotations

import html
from pathlib import Path
from typing import Any


def render_report(result: dict[str, Any], output: str | Path, metrics: dict[str, Any] | None = None) -> Path:
    event_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(event['timestamp']))}</td>"
        f"<td>{html.escape(str(event['severity']))}</td>"
        f"<td>{float(event['score']):.3f}</td>"
        f"<td>{html.escape(str(event['regime']))}</td>"
        f"<td>{float(event['return_1']):.2%}</td>"
        f"<td>{html.escape(', '.join(event['reasons']))}</td>"
        "</tr>"
        for event in result["events"][:40]
    ) or '<tr><td colspan="6">No anomalies</td></tr>'
    metric_cards = ""
    if metrics:
        metric_cards = (
            f'<div class="card"><div class="big">{metrics["event_recall"]:.0%}</div><div>Injected-event recall</div></div>'
            f'<div class="card"><div class="big">{metrics["event_precision"]:.0%}</div><div>Event precision</div></div>'
        )
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Market Anomaly Report</title><style>
body{{background:#09101d;color:#edf3ff;font-family:Inter,system-ui,sans-serif;margin:0;padding:32px}}main{{max-width:1180px;margin:auto}}.muted{{color:#98a8c1}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}}.card{{padding:17px;background:#121c30;border:1px solid #273752;border-radius:14px}}.big{{font-size:27px;font-weight:800}}
table{{width:100%;border-collapse:collapse;background:#121c30}}th,td{{padding:10px;border-bottom:1px solid #273752;text-align:left;vertical-align:top}}th{{color:#a5bbff}}@media(max-width:800px){{.cards{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main><p class="muted">Synthetic/public-data anomaly research</p><h1>Market Anomaly Detection</h1>
<div class="cards"><div class="card"><div class="big">{result['rows']}</div><div>Market rows</div></div><div class="card"><div class="big">{result['anomaly_count']}</div><div>Anomalies</div></div>{metric_cards}</div>
<h2>Ranked anomaly events</h2><table><thead><tr><th>Timestamp</th><th>Severity</th><th>Score</th><th>Regime</th><th>Return</th><th>Reasons</th></tr></thead><tbody>{event_rows}</tbody></table>
</main></body></html>"""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return path
