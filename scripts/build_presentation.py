"""Build a self-contained HTML slide deck summarizing the a-GIZO TFT ANN
modeling project: methodology, data cleaning, training, hyperparameter
sweep, Verilog-A port, Cadence debugging, and the final monotonicity-
constrained model. Plots are embedded as base64 so the file is portable.

Usage: python scripts/build_presentation.py
"""
import base64
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "outputs", "presentation.html")


def b64(path):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def img(path, caption=""):
    src = b64(path)
    if src is None:
        return ""
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f'<figure><img src="{src}" alt="{caption}">{cap}</figure>'


def load_json(path):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


metrics = load_json("outputs/metrics.json")
runs = {}
runs_dir = os.path.join(ROOT, "outputs", "best_runs")
if os.path.isdir(runs_dir):
    for name in sorted(os.listdir(runs_dir)):
        r = load_json(os.path.join("outputs", "best_runs", name, "result.json"))
        if r:
            runs[name] = r


def runs_table():
    if not runs:
        return "<p>(no runs found)</p>"
    order = ["h22_plain", "h22_mono", "h32_mono", "h22_mono_l10", "h22_mono_l100",
             "h32_mono_l10", "h32_mono_l100"]
    rows = []
    for name in order:
        if name not in runs:
            continue
        r = runs[name]
        c = r["config"]
        rows.append(
            f"<tr><td>{name}</td><td>{c['n_hidden']}</td><td>{c['lam']:g}</td>"
            f"<td>{r['test_r2_log10ID']:.4f}</td><td>{r['test_rmse_log10ID']:.3f}</td>"
            f"<td>{r['gds_negative_fraction_on_region']*100:.1f}%</td>"
            f"<td>{r['worst_gds_S']*1e6:.2f}</td></tr>")
    return ("<table><thead><tr><th>run</th><th>hidden</th><th>&lambda;</th><th>test R&sup2;</th>"
            "<th>RMSE (dec)</th><th>gds&lt;0 area</th><th>worst gds (&micro;S)</th></tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody></table>")


final_r2 = metrics.get("test_r2_log10ID", 0)
final_rmse = metrics.get("test_rmse_log10ID", 0)
final_hidden = metrics.get("n_hidden", 22)
final_training = metrics.get("training", "")
final_gds_neg = metrics.get("gds_negative_fraction_on_region")
final_worst_gds = metrics.get("worst_gds_S")

gds_line = ""
if final_gds_neg is not None:
    gds_line = (f"<li>gds &lt; 0 on <b>{final_gds_neg*100:.1f}%</b> of the on-region audit grid; "
                f"worst violation <b>{final_worst_gds*1e6:.2f} &micro;S</b></li>")

slides = []

slides.append(f"""
<section class="title">
  <p class="kicker">a-GIZO TFT · neural device modeling · circuit simulation</p>
  <h1>Modeling TFT drain current with a small MLP</h1>
  <p class="sub">From measured I&ndash;V curves to a Cadence-ready Verilog-A device model,
  following Bahubalindruni et&nbsp;al., <i>Solid-State Electronics</i> 105 (2015)</p>
  <div class="stats">
    <div class="stat"><div class="v">{final_r2:.3f}</div><div class="k">final test R&sup2; on log&#8321;&#8320;|I<sub>D</sub>|</div></div>
    <div class="stat"><div class="v">{final_hidden}</div><div class="k">hidden neurons (tanh &rarr; linear)</div></div>
    <div class="stat"><div class="v">4</div><div class="k">inputs: V<sub>G</sub>, V<sub>D</sub>, W, L</div></div>
  </div>
</section>
""")

slides.append("""
<section>
  <h2>1 · The approach (per the paper)</h2>
  <ul>
    <li><b>Black-box static model:</b> a single-hidden-layer MLP learns
        I<sub>D</sub> = f(V<sub>G</sub>, V<sub>D</sub>, W, L) directly from measurements —
        no device physics needed, fast to develop for an immature technology.</li>
    <li><b>Architecture (Eq. 1&ndash;2):</b> y<sub>h</sub> = tanh(x&middot;w<sub>h</sub> + b<sub>h</sub>);
        y = y<sub>h</sub>&middot;w<sub>o</sub> + b<sub>o</sub> (linear output).</li>
    <li><b>Our twist vs. the paper:</b> the target is <b>log&#8321;&#8320;|I<sub>D</sub>|</b>
        (standardized), because the data spans ~11 decades from leakage floor (~pA)
        to on-state (~100&nbsp;&micro;A). Raw-current regression would only fit the top decade.</li>
    <li>Inputs min-max scaled to [0,1]; the trained network is exported to
        <b>Verilog-A</b> for Spectre/Cadence, exactly like the paper's flow.</li>
  </ul>
</section>
""")

slides.append(f"""
<section>
  <h2>2 · Data quality drove everything</h2>
  <table>
    <thead><tr><th>dataset</th><th>rows</th><th>content</th><th>noise-ceiling R&sup2;</th><th>model R&sup2;</th></tr></thead>
    <tbody>
      <tr><td>original upload</td><td>77,200</td><td>all raw device replicates, off-state sign noise</td><td>0.570</td><td>0.556</td></tr>
      <tr><td>data_cleaned</td><td>39,368</td><td>QC-filtered: dead devices dropped (24 of 80)</td><td>0.924</td><td>0.908</td></tr>
      <tr><td>data_cleaned_2</td><td>13,357</td><td><b>one best positive-V<sub>T</sub> device per geometry</b></td><td>0.9998</td><td>0.935 &rarr; <b>{final_r2:.3f}</b></td></tr>
    </tbody>
  </table>
  <ul>
    <li><b>Noise ceiling</b> = the best R&sup2; <i>any</i> model of (V<sub>G</sub>,V<sub>D</sub>,W,L) could reach,
        because repeated measurements of the same bias point disagree. In the raw data the model
        was already <i>at</i> the ceiling — the limit was the data, not the network.</li>
    <li>Mixing several physical devices under one (W,&nbsp;L) label teaches the net
        contradictory examples (same input &rarr; different output). Keeping one device
        per geometry made the data essentially deterministic (ceiling &rarr; 0.9998).</li>
    <li>QC also exposed a systematic wafer issue: dead devices cluster at the
        <code>top1/top2</code> die positions.</li>
  </ul>
</section>
""")

slides.append(f"""
<section>
  <h2>3 · Model fit on the final dataset</h2>
  <div class="two">
    {img("outputs/plots/scatter_log_id.png", "Predicted vs measured log10|ID| (held-out test set)")}
    {img("outputs/plots/id_vg_curves.png", "Transfer curves: dots = measured, lines = model")}
  </div>
</section>
""")

slides.append(f"""
<section>
  <h2>4 · Hyperparameter sweep (hidden size 12&ndash;32)</h2>
  {img("outputs/plots/sweep_summary.png", "21 hidden sizes x 3 seeds, fixed split (on data_cleaned)")}
  <ul>
    <li>R&sup2; drifts up mildly with size; best mean at 26&ndash;32 neurons, but the gain over 22
        is inside seed-to-seed noise (&plusmn;0.002). <b>22 neurons is a sound choice.</b></li>
    <li>Larger nets converge faster <i>and</i> lower — but every extra neuron costs
        simulation speed in Verilog-A, the paper's original argument for small MLPs.</li>
  </ul>
</section>
""")

slides.append("""
<section>
  <h2>5 · Verilog-A port</h2>
  <ul>
    <li>Generated <code>verilogA/tft_ann_static.va</code> from the trained weights —
        same structure as the reference <code>ntft.va</code> (weight arrays in
        <code>initial_step</code>, tanh loop, analytic g<sub>m</sub>/g<sub>ds</sub>).</li>
    <li><b>Three deliberate corrections</b> vs. the reference file:
        (1) genuinely linear output stage — the reference applies tanh and rescales, which
        would corrupt every prediction of this network;
        (2) [0,1] input scaling matching training exactly, with clamping outside the
        training box; (3) log-domain de-standardization
        <code>id = 10^(y&middot;&sigma; + &mu;)</code>.</li>
    <li>Ported arithmetic verified against PyTorch to 5&ndash;6 significant figures
        on random test points <i>before</i> every commit.</li>
  </ul>
</section>
""")

slides.append("""
<section>
  <h2>6 · Why the first Cadence run misbehaved</h2>
  <ul>
    <li>The inverter (driver + diode-connected load) produced a non-inverting, glitchy
        transient — yet solving the same KCL equation offline gave a clean inverting
        transfer (V<sub>IN</sub>=0 &rarr; V<sub>OUT</sub>&asymp;4.9&nbsp;V; V<sub>IN</sub>=5 &rarr; &asymp;0.9&nbsp;V).</li>
    <li><b>Root cause: R&sup2; is a global average; a circuit solver consumes local derivatives.</b>
        The smooth log-domain fit rippled in the saturation region, giving
        <b>negative g<sub>ds</sub></b> over up to ~44% of the V<sub>D</sub> sweep
        (worst &minus;4.6&nbsp;&micro;S) — negative output resistance creates spurious
        equilibria that Newton&ndash;Raphson can lock onto.</li>
    <li>Aggravating factors: a purely static model (zero capacitance — nothing for the
        transient solver to integrate), strictly positive unidirectional current, and
        11 decades of dynamic range.</li>
    <li>The paper warned about exactly this: <i>"it is mandatory to test small-signal
        parameters."</i></li>
  </ul>
</section>
""")

slides.append(f"""
<section>
  <h2>7 · Fix: monotonicity-constrained training</h2>
  <p>Penalty term via autograd at random collocation points across the whole input cube:
  hinge&sup2; on &part;I<sub>D</sub>/&part;V<sub>D</sub> &lt; 0 (everywhere) and
  &part;I<sub>D</sub>/&part;V<sub>G</sub> &lt; 0 (on-region only — the measured off-state
  leakage valley is real and must stay).</p>
  {runs_table()}
  <ul>
    <li>Longer training alone (h22_plain) lifts R&sup2; to 0.957 but <i>worsens</i> g<sub>ds</sub>
        violations — accuracy and simulator-friendliness genuinely trade off.</li>
    <li>&lambda;=0.5 is homeopathic; &lambda;=100 holds the penalty at ~0 through training
        while giving up only ~0.003 R&sup2;.</li>
  </ul>
</section>
""")

slides.append(f"""
<section>
  <h2>8 · Final model &amp; takeaways</h2>
  <ul>
    <li><b>Final:</b> {final_hidden} hidden neurons, tanh &rarr; linear, trained with
        monotonicity penalty ({final_training or 'src/train_best.py'}).</li>
    <li>Test R&sup2; = <b>{final_r2:.4f}</b>, RMSE = {final_rmse:.3f} decades on log&#8321;&#8320;|I<sub>D</sub>|.</li>
    {gds_line}
    <li>Deliverables in the repo: training pipeline (<code>src/</code>), QC'd datasets,
        sweep results + plots, <code>verilogA/tft_ann_static.va</code>,
        <code>outputs/weights_and_biases.txt</code>.</li>
  </ul>
  <h3>Lessons</h3>
  <ul>
    <li>Fix the data before the model: every large gain came from cleaning
        (R&sup2; 0.56 &rarr; 0.91 &rarr; 0.94+), not architecture.</li>
    <li>Validate the derivatives, not just the fit, before circuit simulation.</li>
    <li>A model can be 90%+ accurate and still break a simulator — and both problems
        are fixable once measured correctly.</li>
  </ul>
</section>
""")

css = """
* { margin: 0; padding: 0; box-sizing: border-box; }
:root { color-scheme: light; }
body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #12151c; }
section {
  min-height: 100vh; padding: 6vh 8vw; background: #fcfcfb; color: #0b0b0b;
  border-bottom: 6px solid #12151c; display: flex; flex-direction: column; justify-content: center;
}
section.title { background: #12151c; color: #fff; text-align: left; }
.kicker { text-transform: uppercase; letter-spacing: .18em; font-size: .8rem; color: #86b6ef; margin-bottom: 1.2rem; }
h1 { font-size: 2.6rem; line-height: 1.15; margin-bottom: 1rem; max-width: 20ch; }
.sub { color: #c3c2b7; font-size: 1.05rem; max-width: 55ch; margin-bottom: 2.5rem; }
.stats { display: flex; gap: 2.5rem; flex-wrap: wrap; }
.stat .v { font-size: 2.4rem; font-weight: 700; color: #3987e5; }
.stat .k { font-size: .85rem; color: #c3c2b7; max-width: 18ch; }
h2 { font-size: 1.7rem; margin-bottom: 1.4rem; color: #0b0b0b; }
h3 { font-size: 1.1rem; margin: 1.2rem 0 .5rem; }
ul { margin-left: 1.2rem; }
li { margin: .55rem 0; line-height: 1.5; font-size: 1.0rem; max-width: 75ch; }
table { border-collapse: collapse; margin: 1rem 0; font-size: .92rem; }
th, td { border: 1px solid #d8d7d2; padding: .45rem .8rem; text-align: left; }
th { background: #eef1f6; }
figure { margin: 1rem 0; }
figure img { max-width: 100%; height: auto; border: 1px solid #e2e1dc; border-radius: 6px; }
figcaption { font-size: .82rem; color: #52514e; margin-top: .4rem; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; align-items: start; }
@media (max-width: 900px) { .two { grid-template-columns: 1fr; } }
code { background: #eef1f6; padding: .1em .35em; border-radius: 4px; font-size: .9em; }
b { color: #104281; }
section.title b { color: #86b6ef; }
"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>a-GIZO TFT ANN modeling — project summary</title>
<style>{css}</style>
</head>
<body>
{''.join(slides)}
</body>
</html>
"""

with open(OUT, "w") as f:
    f.write(html)
print("Wrote", OUT, f"({os.path.getsize(OUT)/1024:.0f} KB)")
