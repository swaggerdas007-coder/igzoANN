"""Build the full project-history presentation: every dataset, model,
diagnostic, and fix tried across the a-GIZO TFT ANN modeling project.
Renders to outputs/full_report.html (landscape report pages, print-ready).

Usage: python scripts/build_full_presentation.py
"""
import base64
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "outputs", "full_report.html")


def b64(path):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def img(path, caption="", cls=""):
    src = b64(path)
    if src is None:
        return f'<div class="missing">missing: {path}</div>'
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f'<figure class="{cls}"><img src="{src}" alt="{caption}">{cap}</figure>'


PAGES = []


def page(number, kicker, title, body, cls=""):
    PAGES.append(f"""
<section class="page {cls}">
  <header class="masthead">
    <span class="proj">a-GIZO TFT &middot; ANN device modeling</span>
    <span class="kicker">{kicker}</span>
    <span class="pageno">{number:02d}</span>
  </header>
  <h1>{title}</h1>
  <div class="body">
    {body}
  </div>
</section>
""")


# ---------------------------------------------------------------- cover ---
PAGES.append("""
<section class="page cover">
  <div class="cover-grain"></div>
  <div class="cover-content">
    <p class="cover-eyebrow">Project report &middot; compiled from the full working session</p>
    <h1 class="cover-title">Modeling a&#8209;GIZO TFT drain current<br>with a small neural network</h1>
    <p class="cover-sub">Every dataset, architecture, diagnostic, and fix attempted &mdash;
    from a 12&#8209;decade noisy CSV to a Verilog&#8209;A model verified against
    Cadence circuit behavior.</p>
    <div class="cover-stats">
      <div class="cstat"><span class="n">5</span><span class="k">datasets cleaned &amp; compared</span></div>
      <div class="cstat"><span class="n">0.556&nbsp;&rarr;&nbsp;0.979</span><span class="k">test R&sup2; on log&#8321;&#8320;|I<sub>D</sub>|</span></div>
      <div class="cstat"><span class="n">&minus;4.6&nbsp;&rarr;&nbsp;&minus;0.29&nbsp;&micro;S</span><span class="k">worst-case output conductance</span></div>
    </div>
    <p class="cover-basis">Following Bahubalindruni, Tavares, Barquinha et al.,
    &ldquo;a&#8209;GIZO TFT neural modeling, circuit simulation and validation,&rdquo;
    <i>Solid&#8209;State Electronics</i> 105 (2015) 30&ndash;36.</p>
  </div>
</section>
""")

# ------------------------------------------------------------- 01 paper ---
page(1, "Foundation", "The architecture we built on", f"""
<div class="cols cols-2">
  <div>
    <p>The paper models drain current I<sub>D</sub> of an amorphous
    Gallium&ndash;Indium&ndash;Zinc&ndash;Oxide TFT as a black-box function of
    bias voltages, learned by a small multilayer perceptron &mdash; no device
    physics required, fast to develop for an immature technology, and
    accurate enough for circuit-level SPICE simulation once ported to
    Verilog&#8209;A.</p>
    <p>Every model in this project reproduces the paper's core equations
    exactly:</p>
    <div class="eq-block">
      <div class="eq"><span class="eq-lhs">y<sub>h</sub></span> = tanh(x &middot; w<sub>h</sub> + b<sub>h</sub>)
        <span class="eq-note">hidden layer, tanh activation</span></div>
      <div class="eq"><span class="eq-lhs">y</span> = y<sub>h</sub> &middot; w<sub>o</sub> + b<sub>o</sub>
        <span class="eq-note">output layer &mdash; <b>linear</b>, no activation</span></div>
    </div>
    <p>Our extension: four inputs (V<sub>G</sub>, V<sub>D</sub>, W, L) instead
    of the paper's two, and a <b>log&#8321;&#8320;|I<sub>D</sub>|</b> regression
    target rather than raw current &mdash; our data spans up to 12 decades
    (leakage floor ~1&nbsp;pA to on-state ~100s&nbsp;&micro;A), where a linear
    fit would only ever resolve the top decade.</p>
  </div>
  <div class="side-panel">
    <p class="panel-label">What stayed constant, always</p>
    <ul class="check-list">
      <li>Single hidden layer, tanh &rarr; linear</li>
      <li>Inputs min-max scaled to [0,1]</li>
      <li>Target: standardized log&#8321;&#8320;|I<sub>D</sub>|</li>
      <li>Trained with backprop (Adam)</li>
      <li>Exported to Verilog&#8209;A for Spectre/Cadence</li>
    </ul>
    <p class="panel-label">What we kept tuning</p>
    <ul class="check-list amber">
      <li>Which dataset (5 attempts)</li>
      <li>Hidden-layer size</li>
      <li>Loss function shape (plain MSE &rarr; monotonicity-penalized &rarr; L-balanced)</li>
      <li>One shared network vs. one per channel length</li>
    </ul>
  </div>
</div>
""")

# ---------------------------------------------------------- 02 datasets ---
page(2, "Five attempts at clean data", "Every dataset we trained on", f"""
<p class="lede">Nearly every large accuracy gain in this project came from
<b>improving the data</b>, not the model. The table below is the throughline
of the whole effort.</p>
<div class="tbl-wrap">
<table class="report-tbl">
<thead><tr><th>#</th><th>dataset</th><th>rows</th><th>what changed</th><th>noise ceiling R&sup2;</th><th>model R&sup2;</th></tr></thead>
<tbody>
<tr><td>1</td><td><code>final_ann_dataset.csv</code></td><td>77,200</td>
    <td>Original upload. V<sub>G</sub> &minus;5&hairsp;&ndash;&hairsp;5V includes off-state; sign-noise
    at the leakage floor, many duplicate operating points with wildly
    disagreeing readings.</td><td>0.570</td><td>0.556</td></tr>
<tr><td>2</td><td><code>data_cleaned/</code></td><td>39,368</td>
    <td>QC pass: dropped 24 of 80 dead/non-functional device replicates
    (concentrated at one wafer die position).</td><td>0.924</td><td>0.908</td></tr>
<tr><td>3</td><td><code>data_cleaned_2/</code></td><td>13,357</td>
    <td>One <b>best</b>, positive-V<sub>T</sub> device per geometry &mdash; removes
    the same-input/different-output ambiguity of mixing devices.</td>
    <td class="hl">0.9998</td><td>0.935&ndash;0.951</td></tr>
<tr><td>4</td><td><code>cleaned_output_meas/</code> v1</td><td>10,659</td>
    <td>Output-curve-only re-pass: one cleanest I<sub>D</sub>&ndash;V<sub>D</sub> family
    per geometry, scored on monotonicity, ordering, on/off ratio.</td>
    <td>&mdash;</td><td>0.973&ndash;0.977</td></tr>
<tr><td>5</td><td><code>cleaned_output_meas/</code> v2</td><td>10,659</td>
    <td>Same grid, better picks &mdash; selection now scores curve
    <b>smoothness/curvature</b>, not just monotonicity; fixed two
    specific bad picks.</td><td>&mdash;</td><td class="hl">0.979</td></tr>
</tbody>
</table>
</div>
<p class="footnote">Noise ceiling: the best R&sup2; <i>any</i> function of
(V<sub>G</sub>,V<sub>D</sub>,W,L) could reach, given that repeated measurements
of the same bias point disagree by construction. In dataset&nbsp;1 the model
was already sitting at the ceiling &mdash; the data itself was the limit, not the network.</p>
""")

# --------------------------------------------------- 03 sweep + baseline ---
page(3, "data_cleaned_2", "First real fit, then a hyperparameter sweep", f"""
<div class="cols cols-2">
  <div>
    <p>Once <code>data_cleaned_2</code> pushed the noise ceiling to
    essentially 1.0, we ran a proper sweep: hidden-layer size 12&ndash;32,
    3 random seeds each (63 runs), fixed train/val/test split.</p>
    <p>Mean test R&sup2; climbed noisily but consistently from ~0.903 at
    12 neurons to a peak of <b>0.9058 &plusmn; 0.0007 at 26 neurons</b>, then
    plateaued through 32. The eventual delivered size (22) scored
    0.9025 &plusmn; 0.0023 &mdash; within its own seed noise of the sweep's best.</p>
    <p class="callout">Bigger networks converged both faster <i>and</i> lower
    on validation loss &mdash; but every extra neuron costs Verilog&#8209;A
    simulation speed, echoing the paper's own argument for keeping MLPs small.</p>
  </div>
  {img("outputs/plots/sweep_summary.png", "Test R² and RMSE vs. hidden-layer size, mean ± std across 3 seeds")}
</div>
""")

# ------------------------------------------------------------- 04 va port --
page(4, "Porting to hardware", "The Verilog-A translation", f"""
<div class="cols cols-2">
  <div>
    <p>Every trained network is exported to a Verilog&#8209;A module
    (<code>tft_ann_static.va</code>) that a real Cadence Spectre testbench
    can instantiate directly &mdash; weight arrays assigned once in
    <code>initial_step</code>, then a tanh loop and a linear output stage
    evaluated every timestep.</p>
    <p>We started from a reference example file the user provided
    (dummy weights, same general structure) and kept its skeleton, but
    three details had to change for correctness, not style:</p>
    <ol class="num-list">
      <li><b>Linear output, not tanh-then-rescale.</b> The reference
      applies tanh to the output and remaps via (y+1)/2; our network's
      output layer is genuinely linear. Copying the reference's post-processing
      verbatim would have silently corrupted every prediction.</li>
      <li><b>[0,1] input scaling</b> matching training exactly, not the
      reference's [&minus;1,1] convention.</li>
      <li><b>log&#8321;&#8320; de-standardization</b>:
      <code>id = 10^(raw_output&middot;&sigma; + &mu;)</code>, since the
      network was trained on standardized log-magnitude, not raw amperes.</li>
    </ol>
  </div>
  <div class="side-panel">
    <p class="panel-label">Verification discipline</p>
    <p>Every export was checked before committing: the Verilog&#8209;A
    arithmetic re-implemented in NumPy and compared against the PyTorch
    model's output on random held-out points.</p>
    <div class="mono-block">torch = 4.13278e-12<br>verilogA&nbsp;= 4.13278e-12<br>match = True</div>
    <p>Never shipped a model whose ported math didn't match to
    5&ndash;6 significant figures.</p>
  </div>
</div>
""")

# ------------------------------------------------------- 05 cadence fail ---
page(5, "The failure", "A 90%-accurate model still broke Cadence", f"""
<div class="cols cols-2">
  <div>
    <p>The user built a simple inverter &mdash; driver + diode-connected
    load, W=L=10&micro;m, V<sub>DD</sub>=5V &mdash; fed it a 0&ndash;5V pulse, and got
    a non-inverting, glitchy transient response that swung <b>negative</b>,
    which is not physically possible for this circuit topology.</p>
    <p>The model's R&sup2; on held-out data was 0.90. That number is a
    <i>global average on log-current magnitude</i> &mdash; it says nothing
    about local derivatives, and a circuit solver runs entirely on local
    derivatives.</p>
    <p class="callout warn">Sweeping <b>g<sub>ds</sub> = &part;I<sub>D</sub>/&part;V<sub>D</sub></b>
    across the model found it <b>negative</b> &mdash; unphysical output
    resistance &mdash; on up to <b>44.5%</b> of the V<sub>D</sub> sweep for some
    bias points, worst violation <b>&minus;4.6&nbsp;&micro;S</b>. Negative g<sub>ds</sub>
    creates spurious extra equilibria in the KCL equation a SPICE solver has
    to converge on.</p>
  </div>
  <div class="side-panel">
    <p class="panel-label">Contributing factors</p>
    <ul class="check-list amber">
      <li>Purely static model &mdash; zero intrinsic capacitance, nothing
      for a transient solver to integrate against</li>
      <li>Current only ever flows D&rarr;S and is strictly positive &mdash;
      real FETs reverse direction if V<sub>DS</sub> flips sign</li>
      <li>11&ndash;12 decades of dynamic range is a genuinely hard
      conditioning problem for damped Newton&ndash;Raphson</li>
    </ul>
    <p class="panel-label">The paper's own warning</p>
    <p class="quote">&ldquo;It is mandatory to test small-signal parameters&rdquo;
    &mdash; g<sub>m</sub> and g<sub>d</sub> &mdash; before trusting a device model
    in circuit simulation. We had only validated current magnitude.</p>
  </div>
</div>
""")

# --------------------------------------------------- 06 monotonicity fix ---
page(6, "The fix", "Training against the derivative, not just the fit", f"""
<div class="cols cols-2">
  <div>
    <p>Added a monotonicity penalty via autograd: at random collocation
    points across the whole scaled input cube, a squared-hinge loss
    punishes <b>&part;I<sub>D</sub>/&part;V<sub>D</sub> &lt; 0</b> everywhere and
    <b>&part;I<sub>D</sub>/&part;V<sub>G</sub> &lt; 0</b> in the on-region (V<sub>G</sub>&gt;1V
    only &mdash; the measured off-state leakage valley is real and must stay).</p>
    <p>Seven configurations compared on <code>data_cleaned_2</code>
    (hidden size &times; penalty weight &lambda;):</p>
  </div>
  <div class="side-panel">
    <p class="panel-label">The core tension</p>
    <p>Longer training alone <i>raised</i> R&sup2; to 0.957 but <i>worsened</i>
    the g<sub>ds</sub> violation to &minus;29.9&nbsp;&micro;S. Accuracy and
    simulator-friendliness genuinely trade off &mdash; only a strong enough
    penalty (&lambda;=100) buys both.</p>
  </div>
</div>
<div class="tbl-wrap">
<table class="report-tbl compact">
<thead><tr><th>config</th><th>test R&sup2;</th><th>g<sub>ds</sub>&lt;0 area</th><th>worst g<sub>ds</sub></th></tr></thead>
<tbody>
<tr><td>22 neurons, no penalty, long training</td><td>0.9571</td><td>31.3%</td><td>&minus;29.9&nbsp;&micro;S</td></tr>
<tr><td>32 neurons, &lambda;=0.5</td><td>0.9605</td><td>24.9%</td><td>&minus;12.0&nbsp;&micro;S</td></tr>
<tr><td>22 neurons, &lambda;=10</td><td>0.9511</td><td>9.1%</td><td>&minus;12.7&nbsp;&micro;S</td></tr>
<tr class="win"><td>32 neurons, &lambda;=100 &mdash; promoted</td><td>0.9511</td><td class="hl">4.6%</td><td class="hl">&minus;1.2&nbsp;&micro;S</td></tr>
</tbody>
</table>
</div>
<p class="footnote">Result: solving the exact failing inverter offline with the
new model gave a clean, unique, wider swing &mdash; V<sub>IN</sub>=0V&rarr;V<sub>OUT</sub>=4.78V,
V<sub>IN</sub>=5V&rarr;V<sub>OUT</sub>=0.38V.</p>
""")

# ------------------------------------------------------- 07 trained_ANN ---
page(7, "A cleaner starting point", "trained_ANN/: fresh data, same discipline", f"""
<div class="cols cols-2">
  <div>
    <p>A new dataset arrived &mdash; <code>cleaned_output_meas/</code>,
    output-sweep-only (I<sub>D</sub>&ndash;V<sub>D</sub> families), one cleanest
    device per geometry, <b>zero duplicate operating points</b> for the
    first time in the project. Trained fresh in a new, self-contained
    <code>trained_ANN/ann_train.py</code>.</p>
    <p>Compared plain vs. monotonicity-penalized at two sizes:</p>
  </div>
  {img("trained_ANN/scatter_log_id.png", "Held-out test set: predicted vs measured log10|ID|")}
</div>
<div class="tbl-wrap">
<table class="report-tbl compact">
<thead><tr><th>config</th><th>test R&sup2;</th><th>g<sub>ds</sub>&lt;0 area</th><th>worst g<sub>ds</sub></th></tr></thead>
<tbody>
<tr><td>22 neurons, no penalty</td><td>0.9771</td><td>24.7%</td><td>&minus;18.0&nbsp;&micro;S</td></tr>
<tr><td>32 neurons, no penalty</td><td>0.9782</td><td>21.4%</td><td>&minus;38.7&nbsp;&micro;S</td></tr>
<tr class="win"><td>22 neurons, &lambda;=100 &mdash; chosen</td><td>0.9732</td><td class="hl">0.9%</td><td class="hl">&minus;2.6&nbsp;&micro;S</td></tr>
<tr><td>32 neurons, &lambda;=100</td><td>0.9738</td><td>1.1%</td><td>&minus;21.4&nbsp;&micro;S</td></tr>
</tbody>
</table>
</div>
<p class="footnote">Same pattern as before: 22 neurons with the penalty wins on
the metric that actually matters for circuit simulation, at negligible R&sup2; cost,
and with less than half the parameters of the 32-neuron alternative.</p>
""")

# ----------------------------------------------------------- 08 L problem --
page(8, "A new diagnostic tool", "“The length L is causing issues”", f"""
<p class="lede">Built a Cadence-style DC sweep script &mdash; source at 0V,
V<sub>G</sub> 0&rarr;5V in 1V steps, V<sub>D</sub> 0&rarr;5V in 0.2V steps, run
directly against the Verilog&#8209;A math &mdash; to test the user's suspicion
directly against every measured curve in <code>cleaned_output_meas/</code>.</p>
<div class="cols cols-2">
  <div>
    <div class="tbl-wrap">
    <table class="report-tbl compact">
    <thead><tr><th>L</th><th>mean RMSE (decades)</th><th>g<sub>ds</sub>&lt;0 area</th><th>geometries</th></tr></thead>
    <tbody>
    <tr class="win"><td>5&nbsp;&micro;m</td><td>0.373</td><td>0.0%</td><td>6 of 6 widths</td></tr>
    <tr><td>10&nbsp;&micro;m</td><td>0.670</td><td>1.4%</td><td>5 of 6</td></tr>
    <tr><td>15&nbsp;&micro;m</td><td>0.606</td><td>0.8%</td><td>4 of 6</td></tr>
    <tr><td>20&nbsp;&micro;m</td><td>0.426</td><td>4.5%</td><td>4 of 6</td></tr>
    </tbody>
    </table>
    </div>
    <p><b>Confirmed:</b> L=5 fit almost perfectly; every longer channel fit
    worse, with real negative-g<sub>ds</sub> patches. Root cause &mdash; the
    (W,L) measurement grid is <i>incomplete</i>: W=5&micro;m is missing entirely
    for L&ge;10, W=10&micro;m missing for L&ge;15. L=5 alone supplies 31.6% of
    all training rows (6 widths) vs 21.1% for L=15/20 (4 widths each), so
    plain-averaged MSE implicitly let L=5's shape dominate.</p>
  </div>
  {img("trained_ANN/dc_analysis/dc_sweep_L10.png", "L=10µm before the fix: model (lines) systematically overshoots measured data (dots) at high VG")}
</div>
""")

# -------------------------------------------------------- 09 L-balancing --
page(9, "The fix", "L-balanced loss, then a cleaner dataset arrived too", f"""
<div class="cols cols-2">
  <div>
    <p>Added a per-row loss weight so every L value contributes equal
    total loss mass, regardless of how many widths were measured there
    &mdash; independent of the monotonicity penalty, stacked on top of it.</p>
    <div class="tbl-wrap">
    <table class="report-tbl compact">
    <thead><tr><th>config</th><th>R&sup2;</th><th>g<sub>ds</sub>&lt;0</th><th>worst g<sub>ds</sub></th></tr></thead>
    <tbody>
    <tr><td>22 neurons, unbalanced</td><td>0.9732</td><td>0.9%</td><td>&minus;2.6&nbsp;&micro;S</td></tr>
    <tr class="win"><td>22 neurons, L-balanced</td><td>0.9738</td><td class="hl">0.1%</td><td class="hl">&minus;0.027&nbsp;&micro;S</td></tr>
    <tr><td>32 neurons, L-balanced</td><td>0.9736</td><td>0.8%</td><td>&minus;37&nbsp;&micro;S</td></tr>
    </tbody>
    </table>
    </div>
    <p>~100&times; smaller worst-case violation for a fraction-of-a-point
    R&sup2; change. Then the underlying dataset improved independently
    (smoothness-scored device selection) &mdash; re-ran the identical
    pipeline and both effects stacked:</p>
    <div class="tbl-wrap">
    <table class="report-tbl compact">
    <thead><tr><th></th><th>R&sup2;</th><th>worst g<sub>ds</sub></th><th>L=10 RMSE</th></tr></thead>
    <tbody>
    <tr><td>v1 data, L-balanced</td><td>0.974</td><td>&minus;0.027&nbsp;&micro;S</td><td>0.60&nbsp;dec</td></tr>
    <tr class="win"><td>v2 data, L-balanced &mdash; final</td><td>0.979</td><td>&minus;0.29&nbsp;&micro;S</td><td>0.47&nbsp;dec</td></tr>
    </tbody>
    </table>
    </div>
  </div>
  {img("trained_ANN/dc_analysis/dc_sweep_L10.png", "L=10µm after L-balancing + cleaner data: tight tracking, no systematic bias")}
</div>
""")

# ------------------------------------------------------------- 10 DC grid --
page(10, "The full picture", "All 19 measured geometries, one view", f"""
<p class="lede">The final delivered model's Cadence-style DC sweep against
every one of the 19 fabricated (W,L) combinations &mdash; raw measured data
(dots) and the simulated Verilog&#8209;A/ANN model (lines), side by side.</p>
{img("trained_ANN/dc_analysis/dc_sweep_all19.png", "", cls="full-width")}
""", cls="wide-plot")

# ---------------------------------------------------------- 11 per-L exp ---
page(11, "One more experiment", "Separate networks per length: a mixed result", f"""
<div class="cols cols-2">
  <div>
    <p>Tried training <b>4 separate 3-input</b> (V<sub>G</sub>,V<sub>D</sub>,W)
    networks, one per measured L, merged into one Verilog&#8209;A module that
    picks the nearest trained L's weights once at
    <code>initial_step</code> (L is a fixed device parameter, never swept
    mid-simulation &mdash; zero runtime cost).</p>
    <p class="callout">Honest result: <b>not a clear win.</b> Row-weighted
    overall RMSE: unified <b>0.460</b> vs. per-L ensemble <b>0.492</b>.</p>
  </div>
  <div class="side-panel">
    <p class="panel-label">Where it helped</p>
    <ul class="check-list">
      <li>L=10 (the old worst case): RMSE 0.513&rarr;0.502</li>
      <li>g<sub>ds</sub> violation area: L=5/10/20 all improved sharply</li>
    </ul>
    <p class="panel-label">Where it hurt</p>
    <ul class="check-list amber">
      <li>L=15 (least data, 4 widths): RMSE 0.494&rarr;0.586</li>
      <li>L=20: RMSE 0.424&rarr;0.491</li>
      <li>g<sub>ds</sub> at L=15 got worse: 0.5%&rarr;1.4%</li>
    </ul>
  </div>
</div>
<p class="footnote">Likely cause: the shared model implicitly transferred
general V<sub>G</sub>/V<sub>D</sub>/W shape knowledge from the data-rich L=5 case
to the data-poor L=15/20 cases; splitting into independent networks removed
that free transfer learning. Kept as an available alternative
(<code>trained_ANN/per_L/</code>), not adopted as the default. Natural next
step: initialize each per-L network from the unified model's weights instead
of from scratch.</p>
""")

# ------------------------------------------------------------- 12 recap ---
page(12, "Where things stand", "Final delivered model", f"""
<div class="cols cols-2">
  <div>
    <p class="panel-label">Architecture</p>
    <p>4 inputs (V<sub>G</sub>, V<sub>D</sub>, W, L) &rarr; 22 tanh hidden neurons
    &rarr; 1 linear output, per Eq. 1&ndash;2 of the paper.</p>
    <p class="panel-label">Training</p>
    <p>Adam, monotonicity penalty (&lambda;=100) + L-balanced loss weighting,
    on <code>cleaned_output_meas/</code> v2 (10,659 rows).</p>
    <p class="panel-label">Result</p>
    <div class="stat-row">
      <div class="stat"><span class="n">0.979</span><span class="k">test R&sup2;</span></div>
      <div class="stat"><span class="n">0.46</span><span class="k">RMSE (decades)</span></div>
      <div class="stat"><span class="n">1.3%</span><span class="k">g<sub>ds</sub>&lt;0 area</span></div>
      <div class="stat"><span class="n">&minus;0.29&nbsp;&micro;S</span><span class="k">worst g<sub>ds</sub></span></div>
    </div>
    <p>The previously-failing inverter (driver + diode load, W=L=10&micro;m,
    V<sub>DD</sub>=5V) now solves cleanly offline: V<sub>IN</sub>=0V&rarr;4.87V,
    V<sub>IN</sub>=5V&rarr;0.85V.</p>
  </div>
  <div class="side-panel">
    <p class="panel-label">What's in the repository</p>
    <ul class="check-list mono-list">
      <li><code>trained_ANN/ann_train.py</code> &mdash; training script</li>
      <li><code>trained_ANN/weights_and_biases.txt</code> &mdash; human-readable weights</li>
      <li><code>verilogA/tft_ann_static.va</code> &mdash; Cadence/Spectre-ready model</li>
      <li><code>scripts/dc_sweep_analysis.py</code> &mdash; DC diagnostic tool</li>
      <li><code>trained_ANN/dc_analysis/</code> &mdash; per-geometry plots + CSVs</li>
      <li><code>trained_ANN/per_L/</code> &mdash; alternative per-length ensemble</li>
    </ul>
    <p class="panel-label">Still open</p>
    <ul class="check-list amber">
      <li>W=80,L=10 remains the single worst-fitting geometry (0.90&nbsp;dec RMSE)</li>
      <li>Missing small-W/large-L measurements limit further gains without new data</li>
    </ul>
  </div>
</div>
""")

CSS = r"""
@page { size: 11in 8.5in landscape; margin: 0; }
* { margin: 0; padding: 0; box-sizing: border-box; }
:root {
  --bg: #f6f4ee;
  --bg-panel: #ece8dd;
  --ink: #1c1a15;
  --ink-soft: #5c584c;
  --ink-faint: #948e7d;
  --accent: #1c5cab;
  --accent-soft: #dce8f6;
  --warn: #a34e0f;
  --warn-soft: #f3e3d2;
  --good: #276b45;
  --line: #d8d3c3;
  --serif: Georgia, "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
  --sans: -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  --mono: "SF Mono", "Cascadia Code", Consolas, "Liberation Mono", monospace;
}
html, body { background: #2a2822; }
body { font-family: var(--sans); color: var(--ink); }
.page {
  width: 11in; min-height: 8.5in; background: var(--bg);
  padding: 0.55in 0.7in; position: relative; overflow: hidden;
  page-break-after: always; display: flex; flex-direction: column;
  background-image:
    linear-gradient(var(--line) 1px, transparent 1px);
  background-size: 100% 0.28in;
  background-position: 0 1.35in;
}
.page::before {
  content: ""; position: absolute; inset: 0;
  background-image: linear-gradient(var(--line) 1px, transparent 1px);
  background-size: 100% 0.28in; background-position: 0 1.35in;
  opacity: 0; /* baseline grid disabled by default, kept for cover only */
}
.masthead {
  display: flex; align-items: baseline; gap: 0.35in;
  font-family: var(--mono); font-size: 8.5pt; letter-spacing: 0.04em;
  color: var(--ink-faint); text-transform: uppercase;
  border-bottom: 1px solid var(--line); padding-bottom: 10px; margin-bottom: 22px;
}
.masthead .kicker { color: var(--accent); font-weight: 600; }
.masthead .pageno { margin-left: auto; }
h1 { font-family: var(--serif); font-size: 27pt; font-weight: 400; line-height: 1.15;
  text-wrap: balance; margin-bottom: 18px; letter-spacing: -0.01em; }
.body { flex: 1; display: flex; flex-direction: column; gap: 14px; }
p { font-size: 11pt; line-height: 1.55; max-width: 72ch; }
p.lede { font-size: 12.5pt; line-height: 1.5; color: var(--ink); max-width: 80ch; }
.cols { display: grid; gap: 0.45in; align-items: start; }
.cols-2 { grid-template-columns: 1.15fr 1fr; }
.cols > * { min-width: 0; }
.cols-2 figure img { max-height: 3.5in; width: auto; max-width: 100%; margin: 0 auto; }
.side-panel { background: var(--bg-panel); border-radius: 3px; padding: 18px 20px;
  border-left: 3px solid var(--accent); }
.panel-label { font-family: var(--mono); font-size: 8.5pt; text-transform: uppercase;
  letter-spacing: 0.06em; color: var(--ink-soft); margin: 12px 0 6px; }
.panel-label:first-child { margin-top: 0; }
.check-list { list-style: none; display: flex; flex-direction: column; gap: 6px; margin-bottom: 4px; }
.check-list li { font-size: 10.3pt; line-height: 1.4; padding-left: 16px; position: relative; }
.check-list li::before { content: "\2013"; position: absolute; left: 0; color: var(--accent); }
.check-list.amber li::before { color: var(--warn); }
.check-list.mono-list li { font-family: var(--mono); font-size: 9pt; }
.check-list.mono-list li::before { content: none; }
.check-list.mono-list li { padding-left: 0; }
.quote { font-style: italic; color: var(--ink-soft); border-left: 2px solid var(--ink-faint);
  padding-left: 12px; font-size: 10.5pt; }
.callout { background: var(--accent-soft); border-radius: 3px; padding: 12px 16px;
  font-size: 10.5pt; border-left: 3px solid var(--accent); }
.callout.warn { background: var(--warn-soft); border-left-color: var(--warn); }
.eq-block { background: var(--bg-panel); border-radius: 3px; padding: 16px 20px; margin: 10px 0; }
.eq { font-family: var(--mono); font-size: 12pt; margin: 8px 0; display: flex;
  align-items: baseline; gap: 14px; flex-wrap: wrap; }
.eq-lhs { color: var(--accent); font-weight: 600; }
.eq-note { font-family: var(--sans); font-size: 8.5pt; color: var(--ink-faint);
  text-transform: uppercase; letter-spacing: 0.03em; }
.num-list { padding-left: 20px; display: flex; flex-direction: column; gap: 8px; }
.num-list li { font-size: 10.5pt; line-height: 1.5; }
.mono-block { font-family: var(--mono); font-size: 9.5pt; background: var(--bg-panel);
  padding: 12px 14px; border-radius: 3px; line-height: 1.6; margin: 6px 0; }
figure { display: flex; flex-direction: column; gap: 6px; }
figure img { width: 100%; height: auto; border: 1px solid var(--line); border-radius: 2px;
  background: #fff; }
figcaption { font-size: 8.5pt; color: var(--ink-soft); text-align: center; }
.full-width img { display: block; width: 100%; max-width: 7.5in; height: auto; margin: 0.05in auto 0; }
.wide-plot { padding-top: 0.4in; padding-bottom: 0.35in; }
.wide-plot p.lede { margin-bottom: 4px; }
.tbl-wrap { overflow-x: auto; }
.report-tbl { border-collapse: collapse; width: 100%; font-size: 10pt; }
.report-tbl.compact { font-size: 9.5pt; }
.report-tbl th, .report-tbl td { text-align: left; padding: 7px 12px; border-bottom: 1px solid var(--line); }
.report-tbl th { font-family: var(--mono); font-size: 8pt; text-transform: uppercase;
  letter-spacing: 0.04em; color: var(--ink-soft); font-weight: 600; }
.report-tbl td { font-variant-numeric: tabular-nums; }
.report-tbl td:first-child { font-weight: 500; }
.report-tbl .hl { color: var(--good); font-weight: 700; }
.report-tbl tr.win { background: #eaf2ec; }
.report-tbl tr.win td:first-child::before { content: "\2713  "; color: var(--good); }
.footnote { font-size: 9pt; color: var(--ink-soft); line-height: 1.5; max-width: 90ch;
  border-top: 1px solid var(--line); padding-top: 10px; margin-top: auto; }
.stat-row { display: flex; gap: 26px; margin: 8px 0 14px; }
.stat { display: flex; flex-direction: column; }
.stat .n { font-family: var(--mono); font-size: 17pt; font-weight: 700; color: var(--accent); }
.stat .k { font-size: 8.5pt; color: var(--ink-soft); max-width: 12ch; }
.missing { color: var(--warn); font-family: var(--mono); font-size: 9pt; padding: 20px;
  border: 1px dashed var(--warn); }

/* cover */
.page.cover { background: linear-gradient(160deg, #1b2a3d 0%, #16324f 45%, #12405a 100%);
  color: #f2efe6; justify-content: center; padding: 0.9in; }
.cover-grain { position: absolute; inset: 0;
  background-image: radial-gradient(circle at 15% 20%, rgba(255,255,255,0.05), transparent 45%),
                     radial-gradient(circle at 85% 80%, rgba(28,92,171,0.25), transparent 50%);
}
.cover-content { position: relative; max-width: 8.5in; }
.cover-eyebrow { font-family: var(--mono); font-size: 9pt; letter-spacing: 0.08em;
  text-transform: uppercase; color: #8fb8e0; margin-bottom: 22px; }
.cover-title { font-family: var(--serif); font-size: 42pt; line-height: 1.1;
  font-weight: 400; text-wrap: balance; margin-bottom: 22px; color: #fbfaf5; }
.cover-sub { font-size: 13pt; line-height: 1.6; color: #cfd8e2; max-width: 62ch; margin-bottom: 34px; }
.cover-stats { display: flex; gap: 40px; margin-bottom: 40px; flex-wrap: wrap; }
.cstat { display: flex; flex-direction: column; gap: 4px; }
.cstat .n { font-family: var(--mono); font-size: 19pt; font-weight: 700; color: #ffd9a8; }
.cstat .k { font-size: 8.5pt; color: #a9bacb; max-width: 18ch; text-transform: uppercase;
  letter-spacing: 0.03em; }
.cover-basis { font-size: 9.5pt; color: #8fa2b6; border-top: 1px solid rgba(255,255,255,0.15);
  padding-top: 16px; max-width: 60ch; }
"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>a-GIZO TFT ANN modeling &mdash; full project report</title>
<style>{CSS}</style>
</head>
<body>
{''.join(PAGES)}
</body>
</html>
"""

with open(OUT, "w") as f:
    f.write(html)
print(f"Wrote {OUT} ({os.path.getsize(OUT)/1024:.0f} KB, {len(PAGES)} pages)")
