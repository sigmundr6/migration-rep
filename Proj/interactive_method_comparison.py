#!/usr/bin/env python3
"""Build an interactive comparison of the 12 trajectory representations.

The four predefined core methods are visually prominent; the eight supporting
methods remain available as contextual comparisons.  The output is a single,
self-contained HTML file and does not require a running Python server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from plotly.offline import get_plotlyjs


CORE = ["crystal", "midpoint", "trajectory", "positional"]
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "crystal": "Crystal",
    "midpoint": "Midpoint",
    "trajectory": "Unordered trajectory",
    "positional": "Positional trajectory",
    "segmented": "Segmented trajectory",
    "randomized": "Randomised trajectory",
    "image_coordination": "Image coordination",
    "geometric_bottleneck": "Geometric bottleneck",
    "coordination_change": "Coordination change",
    "species_image": "Species-image",
    "position_aware_trajectory_graph": "Position-aware trajectory graph",
    "position_aware_trajectory_graph_shuffled": "Shuffled trajectory graph",
}
FAMILIES = {
    "crystal": "Core hierarchy",
    "midpoint": "Core hierarchy",
    "trajectory": "Core hierarchy",
    "positional": "Core hierarchy",
    "segmented": "Ordering ablation",
    "randomized": "Negative control",
    "image_coordination": "Image-resolved",
    "geometric_bottleneck": "Physical summary",
    "coordination_change": "Physical summary",
    "species_image": "Image-resolved",
    "position_aware_trajectory_graph": "Trajectory graph",
    "position_aware_trajectory_graph_shuffled": "Trajectory graph control",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPOSITORY_ROOT / "results/exploratory_statistics/summary.csv",
    )
    parser.add_argument(
        "--seed-metrics",
        type=Path,
        default=REPOSITORY_ROOT / "results/exploratory_statistics/seed_metrics.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "results/interactive_method_comparison.html",
    )
    return parser.parse_args()


def records(summary_path: Path, seed_path: Path) -> tuple[list[dict], list[dict]]:
    summary = pd.read_csv(summary_path)
    seeds = pd.read_csv(seed_path)
    missing = set(LABELS) - set(summary["method"])
    if missing:
        raise ValueError(f"summary is missing methods: {sorted(missing)}")

    summary = summary[summary["method"].isin(LABELS)].copy()
    seeds = seeds[seeds["method"].isin(LABELS)].copy()
    for frame in (summary, seeds):
        frame["label"] = frame["method"].map(LABELS)
        frame["family"] = frame["method"].map(FAMILIES)
        frame["core"] = frame["method"].isin(CORE)
    return summary.to_dict("records"), seeds.to_dict("records")


def build_html(summary: list[dict], seeds: list[dict]) -> str:
    data = json.dumps({"summary": summary, "seeds": seeds}, separators=(",", ":"))
    plotly = get_plotlyjs()
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trajectory representation comparison</title>
<script>{plotly}</script>
<style>
:root {{ color-scheme: light dark; --bg:#ffffff; --fg:#202124; --muted:#6b7280;
  --grid:#d8dde5; --panel:#f7f8fa; --blue:#2676b8; --orange:#e07a1f;
  --green:#238b68; --purple:#7451b9; --context:#9aa1aa; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#17191d; --fg:#eceff4;
  --muted:#aeb5bf; --grid:#3b414a; --panel:#20242a; --context:#7c838d; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.4 system-ui,sans-serif; }}
main {{ max-width:1180px; margin:auto; padding:24px; }}
h1 {{ margin:0 0 6px; font-size:24px; font-weight:600; }}
.subtitle {{ color:var(--muted); margin-bottom:18px; }}
.controls {{ display:flex; flex-wrap:wrap; gap:18px; align-items:end; margin-bottom:12px; }}
label {{ display:grid; gap:4px; font-weight:600; }}
select, button {{ font:inherit; color:var(--fg); background:var(--panel); border:1px solid var(--grid);
  border-radius:5px; padding:7px 10px; }}
button[aria-pressed="true"] {{ outline:2px solid var(--blue); outline-offset:1px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:14px; color:var(--muted); margin:8px 0 2px; }}
.legend span {{ display:inline-flex; gap:6px; align-items:center; }}
.swatch {{ width:12px; height:12px; border-radius:50%; display:inline-block; }}
.plots {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; }}
.plot {{ min-height:570px; }}
.detail {{ margin-top:12px; padding:10px 12px; background:var(--panel); border-left:4px solid var(--blue); }}
@media (max-width:820px) {{ .plots {{ grid-template-columns:1fr; }} .plot {{ min-height:520px; }} }}
</style>
</head>
<body><main>
<h1>Twelve trajectory representations</h1>
<div class="subtitle">Four predefined methods are emphasised; eight mechanistic alternatives provide context.</div>
<div class="controls">
  <label>Metric
    <select id="metric">
      <option value="mae">MAE (eV)</option><option value="rmse">RMSE (eV)</option>
      <option value="median_ae">Median absolute error (eV)</option>
      <option value="r2">R²</option><option value="spearman">Spearman correlation</option>
    </select>
  </label>
  <button id="focus" type="button" aria-pressed="false">Show core four only</button>
</div>
<div class="legend">
  <span><i class="swatch" style="background:var(--blue)"></i>Crystal</span>
  <span><i class="swatch" style="background:var(--orange)"></i>Midpoint</span>
  <span><i class="swatch" style="background:var(--green)"></i>Unordered trajectory</span>
  <span><i class="swatch" style="background:var(--purple)"></i>Positional trajectory</span>
  <span><i class="swatch" style="background:var(--context)"></i>Supporting methods</span>
</div>
<div class="plots"><div id="ranking" class="plot"></div><div id="seeds" class="plot"></div></div>
<div id="detail" class="detail" aria-live="polite">Select a method in either plot to inspect it.</div>
</main>
<script>
const DATA={data};
const CORE=['crystal','midpoint','trajectory','positional'];
const COLORS={{crystal:'#2676b8',midpoint:'#e07a1f',trajectory:'#238b68',positional:'#7451b9'}};
const METRICS={{
  mae:{{label:'MAE (eV)',mean:'mae_mean',sd:'mae_std',seed:'mae',lower:true}},
  rmse:{{label:'RMSE (eV)',mean:'rmse_mean',sd:'rmse_std',seed:'rmse',lower:true}},
  median_ae:{{label:'Median absolute error (eV)',mean:'median_ae_mean',sd:'median_ae_std',seed:'median_ae',lower:true}},
  r2:{{label:'R²',mean:'r2_mean',sd:'r2_std',seed:'r2',lower:false}},
  spearman:{{label:'Spearman correlation',mean:'spearman_mean',sd:'spearman_std',seed:'spearman',lower:false}}
}};
let coreOnly=false;
const css=name=>getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const colour=d=>COLORS[d.method] || css('--context');
const rows=()=>DATA.summary.filter(d=>!coreOnly||d.core);
function baseLayout(title,xlabel) {{ return {{title:{{text:title,x:0.02,xanchor:'left'}},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{{color:css('--fg')}},margin:{{l:178,r:28,t:55,b:60}},xaxis:{{title:xlabel,gridcolor:css('--grid'),zerolinecolor:css('--grid')}},yaxis:{{automargin:true}},showlegend:false,hoverlabel:{{bgcolor:css('--panel'),font:{{color:css('--fg')}}}}}}; }}
function draw() {{
  const key=document.getElementById('metric').value, m=METRICS[key];
  const selected=rows().slice().sort((a,b)=>m.lower?a[m.mean]-b[m.mean]:b[m.mean]-a[m.mean]);
  const trace={{type:'scatter',mode:'markers',x:selected.map(d=>d[m.mean]),y:selected.map(d=>d.label),
    error_x:{{type:'data',array:selected.map(d=>d[m.sd]),visible:true,color:css('--muted'),thickness:1.3}},
    marker:{{size:selected.map(d=>d.core?13:8),color:selected.map(colour),opacity:selected.map(d=>d.core?1:0.62),symbol:selected.map(d=>d.core?'circle':'diamond')}},
    customdata:selected.map(d=>[d.method,d.family,d.seeds,d.best_epoch_mean,d.early_stopped_fraction]),
    hovertemplate:'<b>%{{y}}</b><br>'+m.label+': %{{x:.4f}} ± %{{error_x.array:.4f}}<br>%{{customdata[1]}}<br>%{{customdata[2]}} seeds<extra></extra>'}};
  Plotly.react('ranking',[trace],baseLayout('Mean performance across ten seeds',m.label),{{responsive:true,displaylogo:false}});
  const seedTraces=selected.map(d=>{{const vals=DATA.seeds.filter(s=>s.method===d.method).map(s=>s[m.seed]); return {{type:'box',orientation:'h',name:d.label,x:vals,boxpoints:'all',jitter:.28,pointpos:0,marker:{{size:d.core?7:5,color:colour(d),opacity:d.core?0.9:0.5}},line:{{color:colour(d),width:d.core?2:1}},fillcolor:'rgba(0,0,0,0)',customdata:DATA.seeds.filter(s=>s.method===d.method).map(s=>[s.method,s.seed]),hovertemplate:'<b>'+d.label+'</b><br>seed %{{customdata[1]}}<br>'+m.label+': %{{x:.4f}}<extra></extra>'}};}});
  Plotly.react('seeds',seedTraces,baseLayout('Seed-level distribution',m.label),{{responsive:true,displaylogo:false}});
  document.querySelectorAll('#ranking,#seeds').forEach(el=>el.on('plotly_click',e=>showDetail(e.points[0].customdata[0])));
}}
function showDetail(method) {{const d=DATA.summary.find(x=>x.method===method); document.getElementById('detail').innerHTML='<b>'+d.label+'</b> — '+d.family+'. MAE '+d.mae_mean.toFixed(4)+' ± '+d.mae_std.toFixed(4)+' eV; R² '+d.r2_mean.toFixed(3)+'; Spearman '+d.spearman_mean.toFixed(3)+'. Mean selected epoch '+d.best_epoch_mean.toFixed(1)+'.';}}
document.getElementById('metric').addEventListener('change',draw);
document.getElementById('focus').addEventListener('click',e=>{{coreOnly=!coreOnly;e.currentTarget.setAttribute('aria-pressed',String(coreOnly));e.currentTarget.textContent=coreOnly?'Show all twelve':'Show core four only';draw();}});
draw();
</script></body></html>"""


def main() -> None:
    args = arguments()
    summary, seeds = records(args.summary, args.seed_metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(summary, seeds), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
