"""Generate ``docs/dashboard.html`` — a single, self-contained, offline
interactive dashboard.

Every number in the page is computed by this module from the toolkit's
own engines (efficient frontier + walk-forward backtest on the seeded
synthetic dataset), embedded as JSON, and rendered with inline
SVG/JS — no CDN, no external fonts, no network requests. Open the file
in any browser.

Usage:  python src/dashboard.py [output_path]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from portfolio_optimizer import (
        compute_returns,
        covariance_matrix,
        efficient_frontier,
        expected_returns,
        generate_random_portfolios,
        generate_synthetic_prices,
        max_sharpe_portfolio,
        min_volatility_portfolio,
        portfolio_stats,
    )
    from allocators import hrp_portfolio, risk_parity_portfolio
    from backtest import STRATEGIES, run_backtest
    from covariance import ledoit_wolf_shrinkage
except ImportError:  # pragma: no cover
    from .portfolio_optimizer import (  # type: ignore
        compute_returns, covariance_matrix, efficient_frontier,
        expected_returns, generate_random_portfolios,
        generate_synthetic_prices, max_sharpe_portfolio,
        min_volatility_portfolio, portfolio_stats,
    )
    from .allocators import hrp_portfolio, risk_parity_portfolio  # type: ignore
    from .backtest import STRATEGIES, run_backtest  # type: ignore
    from .covariance import ledoit_wolf_shrinkage  # type: ignore

RISK_FREE_RATE = 0.02
COLORS = {
    "Equal Weight": "#8b9dc3", "Max-Sharpe": "#f0b429",
    "Min-Volatility": "#3fb68b", "Risk Parity": "#e5534b", "HRP": "#a371f7",
}


def collect_data() -> dict:
    """Run the real engines and gather everything the page renders."""
    prices = generate_synthetic_prices(n_days=504, seed=42)
    returns = compute_returns(prices)
    mu = expected_returns(returns)
    cov = covariance_matrix(returns)
    _lw, delta = ledoit_wolf_shrinkage(returns)

    frontier = efficient_frontier(mu, cov, n_points=30, risk_free_rate=RISK_FREE_RATE)
    cloud = generate_random_portfolios(mu, cov, n_portfolios=1200,
                                       risk_free_rate=RISK_FREE_RATE, seed=42)
    markers = {}
    ms = max_sharpe_portfolio(mu, cov, risk_free_rate=RISK_FREE_RATE)
    mv = min_volatility_portfolio(mu, cov, risk_free_rate=RISK_FREE_RATE)
    markers["Max-Sharpe"] = ms
    markers["Min-Volatility"] = mv
    for name, w in (("Risk Parity", risk_parity_portfolio(cov)),
                    ("HRP", hrp_portfolio(cov))):
        markers[name] = portfolio_stats(w, mu, cov, RISK_FREE_RATE)
    markers["Equal Weight"] = portfolio_stats(
        np.full(len(mu), 1.0 / len(mu)), mu, cov, RISK_FREE_RATE)

    bt = run_backtest(seed=42)
    growth = {
        name: [round(float(v), 2) for v in bt.growth[name].to_numpy()]
        for name in STRATEGIES
    }
    metrics = {
        name: {k: round(float(v), 6) for k, v in row.items()}
        for name, row in bt.metrics.iterrows()
    }
    weights = {
        name: {t: round(float(w), 4) for t, w in zip(mu.index, s.weights)}
        for name, s in markers.items()
    }
    return {
        "tickers": list(mu.index),
        "frontier": [[round(float(v), 5), round(float(r), 5)]
                     for v, r in zip(frontier["volatility"], frontier["return"])],
        "cloud": [[round(float(v), 5), round(float(r), 5)]
                  for v, r in zip(cloud["volatility"], cloud["return"])],
        "markers": {name: {"vol": round(float(s.volatility), 5),
                           "ret": round(float(s.expected_return), 5),
                           "sharpe": round(float(s.sharpe_ratio), 4),
                           "color": COLORS[name]}
                    for name, s in markers.items()},
        "growth_dates": [d.strftime("%Y-%m-%d") for d in bt.growth.index],
        "growth": growth,
        "colors": COLORS,
        "metrics": metrics,
        "weights": weights,
        "shrinkage_delta": round(float(delta), 4),
        "oos_days": int(len(bt.returns)),
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Portfolio Optimizer — Dashboard</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--txt:#e6edf3;--mut:#8b949e;--acc:#f0b429}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);
font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif}
header{padding:28px 36px 8px}h1{margin:0;font-size:26px}h1 span{color:var(--acc)}
.sub{color:var(--mut);margin-top:6px;font-size:14px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:18px;padding:20px 36px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px}
.card h2{margin:0 0 4px;font-size:17px}.card p{color:var(--mut);font-size:13px;margin:4px 0 10px}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{padding:7px 8px;text-align:right;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}th{color:var(--mut);font-weight:600}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}
svg{width:100%;height:auto}.axis{stroke:#30363d}.lbl{fill:#8b949e;font-size:11px}
.legend{font-size:12.5px;color:var(--mut);margin-top:6px}
footer{padding:6px 36px 30px;color:var(--mut);font-size:12.5px}
.kpi{display:flex;gap:14px;flex-wrap:wrap;padding:14px 36px 0}
.kpi div{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 18px}
.kpi b{font-size:19px;display:block}.kpi span{color:var(--mut);font-size:12px}
</style></head><body>
<header><h1>Portfolio <span>Optimizer</span> — Dashboard</h1>
<div class="sub">Efficient frontier &amp; walk-forward backtest, computed live by the
toolkit on seeded synthetic data (AAPL · MSFT · GOOGL · AMZN · JPM · XOM).
Fully offline — every chart is drawn from data embedded in this file.</div></header>
<div class="kpi" id="kpis"></div>
<div class="grid">
 <div class="card"><h2>Efficient Frontier</h2>
 <p>1,200 random portfolios (grey), the efficient frontier (gold line) and the five
 named allocations. Hover a point for its Sharpe ratio.</p>
 <svg id="frontier" viewBox="0 0 560 380"></svg><div class="legend" id="flegend"></div></div>
 <div class="card"><h2>Walk-Forward Backtest — Growth of $10,000</h2>
 <p id="bt-sub"></p><svg id="growth" viewBox="0 0 560 380"></svg>
 <div class="legend" id="glegend"></div></div>
 <div class="card"><h2>Backtest Metrics (out-of-sample)</h2>
 <p>Expanding window, 252-day initial training, 63-day steps, Ledoit–Wolf covariance.</p>
 <table id="metrics"></table></div>
 <div class="card"><h2>Full-Sample Optimal Weights</h2>
 <p>Weights each allocator assigns on the full 504-day sample.</p>
 <table id="weights"></table></div>
</div>
<footer>Educational project — synthetic data, not investment advice. Ledoit–Wolf
shrinkage intensity δ = <span id="delta"></span> on the full sample.</footer>
<script>
const D = __DATA__;
const NS='http://www.w3.org/2000/svg';
function el(tag,attrs,parent){const e=document.createElementNS(NS,tag);
 for(const k in attrs)e.setAttribute(k,attrs[k]); if(parent)parent.appendChild(e); return e;}
function scale(v,dom,w){return (v-dom[0])/(dom[1]-dom[0])*w;}
const PCT=v=>(v*100).toFixed(0)+'%', INT=v=>Math.round(v)+'';
function chart(svgId,xs,ys,xfmt,yfmt){xfmt=xfmt||PCT; yfmt=yfmt||PCT;
 const svg=document.getElementById(svgId);
 const W=560,H=380,P={l:52,r:14,t:12,b:40};
 const xd=[Math.min(...xs)*0.97,Math.max(...xs)*1.03], yd=[Math.min(...ys)*0.95,Math.max(...ys)*1.05];
 for(let i=0;i<=4;i++){const xv=xd[0]+(xd[1]-xd[0])*i/4, X=P.l+scale(xv,xd,W-P.l-P.r);
  el('line',{x1:X,y1:P.t,x2:X,y2:H-P.b,class:'axis'},svg);
  const t=el('text',{x:X,y:H-P.b+16,'text-anchor':'middle',class:'lbl'},svg); t.textContent=xfmt(xv);}
 for(let i=0;i<=4;i++){const yv=yd[0]+(yd[1]-yd[0])*i/4, Y=H-P.b-scale(yv,yd,H-P.t-P.b);
  el('line',{x1:P.l,y1:Y,x2:W-P.r,y2:Y,class:'axis'},svg);
  const t=el('text',{x:P.l-8,y:Y+4,'text-anchor':'end',class:'lbl'},svg); t.textContent=yfmt(yv);}
 return {svg,W,H,P,xd,yd,
  X:v=>P.l+scale(v,xd,W-P.l-P.r), Y:v=>H-P.b-scale(v,yd,H-P.t-P.b)};}
// --- Frontier ---
(function(){const pts=D.cloud.map(p=>p[0]), rets=D.cloud.map(p=>p[1]);
 const c=chart('frontier',pts.concat(D.frontier.map(p=>p[0])),rets.concat(D.frontier.map(p=>p[1])));
 D.cloud.forEach(p=>el('circle',{cx:c.X(p[0]),cy:c.Y(p[1]),r:1.6,fill:'#484f58',opacity:.55},c.svg));
 const path=D.frontier.map((p,i)=>(i?'L':'M')+c.X(p[0]).toFixed(1)+' '+c.Y(p[1]).toFixed(1)).join(' ');
 el('path',{d:path,fill:'none',stroke:'#f0b429','stroke-width':2},c.svg);
 let leg='';
 for(const name in D.markers){const m=D.markers[name];
  const dot=el('circle',{cx:c.X(m.vol),cy:c.Y(m.ret),r:5.5,fill:m.color,stroke:'#0d1117','stroke-width':1.5},c.svg);
  const tt=document.createElementNS(NS,'title'); tt.textContent=name+' — Sharpe '+m.sharpe; dot.appendChild(tt);
  leg+='<span style="margin-right:14px"><span class="dot" style="background:'+m.color+'"></span>'+name+'</span>';}
 document.getElementById('flegend').innerHTML=leg;})();
// --- Growth ---
(function(){const all=Object.values(D.growth).flat();
 const c=chart('growth',D.growth_dates.map((_,i)=>i),all.map(v=>v/10000-1),INT,PCT);
 const n=D.growth_dates.length; let leg='';
 for(const name in D.growth){const vals=D.growth[name];
  const path=vals.map((v,i)=>(i?'L':'M')+c.X(i).toFixed(1)+' '+c.Y(v/10000-1).toFixed(1)).join(' ');
  el('path',{d:path,fill:'none',stroke:D.colors[name],'stroke-width':1.8},c.svg);
  const fin=vals[vals.length-1];
  leg+='<span style="margin-right:14px"><span class="dot" style="background:'+D.colors[name]+'"></span>'
      +name+' → $'+fin.toLocaleString(undefined,{maximumFractionDigits:0})+'</span>';}
 document.getElementById('glegend').innerHTML=leg;
 document.getElementById('bt-sub').textContent='Out-of-sample: '+D.oos_days+
  ' trading days (x-axis), y-axis = cumulative return. Weights re-estimated at each step from history only — no look-ahead.';
 // relabel growth y-axis as $ values
})();
// --- Tables ---
(function(){const pct=v=>(v*100).toFixed(2)+'%', num=v=>v.toFixed(3);
 let h='<tr><th>Strategy</th><th>Return</th><th>Vol</th><th>Sharpe</th><th>Sortino</th><th>MaxDD</th><th>Calmar</th></tr>';
 for(const name in D.metrics){const m=D.metrics[name];
  h+='<tr><td><span class="dot" style="background:'+D.colors[name]+'"></span>'+name+'</td><td>'+pct(m.annual_return)
   +'</td><td>'+pct(m.annual_volatility)+'</td><td>'+num(m.sharpe)+'</td><td>'+num(m.sortino)
   +'</td><td>'+pct(m.max_drawdown)+'</td><td>'+num(m.calmar)+'</td></tr>';}
 document.getElementById('metrics').innerHTML=h;
 let w='<tr><th>Strategy</th>'+D.tickers.map(t=>'<th>'+t+'</th>').join('')+'</tr>';
 for(const name in D.weights){w+='<tr><td>'+name+'</td>'+D.tickers.map(t=>'<td>'
   +(D.weights[name][t]*100).toFixed(1)+'%</td>').join('')+'</tr>';}
 document.getElementById('weights').innerHTML=w;
 const best=Object.entries(D.metrics).sort((a,b)=>b[1].sharpe-a[1].sharpe)[0];
 const calm=Object.entries(D.metrics).sort((a,b)=>Math.abs(b[1].max_drawdown)-Math.abs(a[1].max_drawdown))[0];
 document.getElementById('kpis').innerHTML=
  '<div><span>Best OOS Sharpe</span><b>'+best[0]+' · '+best[1].sharpe.toFixed(2)+'</b></div>'
  +'<div><span>Shallowest Drawdown</span><b>'+calm[0]+' · '+(calm[1].max_drawdown*100).toFixed(1)+'%</b></div>'
  +'<div><span>Out-of-Sample Days</span><b>'+D.oos_days+'</b></div>'
  +'<div><span>Assets</span><b>'+D.tickers.length+'</b></div>';
 document.getElementById('delta').textContent=D.shrinkage_delta;})();
</script></body></html>
"""


def build_dashboard(output: str | Path | None = None) -> Path:
    data = collect_data()
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data))
    if output is None:
        output = Path(__file__).resolve().parent.parent / "docs" / "dashboard.html"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


if __name__ == "__main__":
    out = build_dashboard(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"Dashboard written to {out} ({out.stat().st_size/1024:.0f} KB)")
