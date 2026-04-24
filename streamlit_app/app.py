import json
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="EEG", page_icon="🧠", layout="wide")

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* push content below Streamlit's top toolbar */
.block-container {
    padding-top: 4rem !important;
    padding-bottom: 2rem !important;
}
.stApp { color: #e5e7eb; }
/* stat cards */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(140px, 1fr));
    gap: .7rem;
    margin-bottom: 1.2rem;
}
.stat {
    padding: .8rem 1rem;
    border-radius: 14px;
    background: rgba(15,23,42,.9);
    border: 1px solid rgba(148,163,184,.18);
}
.label {
    font-size: .72rem;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: .2rem;
}
.value { font-size: 1.15rem; font-weight: 700; color: #f8fafc; }
/* plot section labels */
.plot-label {
    font-size: .8rem;
    font-weight: 600;
    letter-spacing: .06em;
    text-transform: uppercase;
    color: #94a3b8;
    margin: 1rem 0 .1rem 0;
    padding: 0;
}
/* hero (error state) */
.hero {
    padding: 1.1rem 1.4rem;
    border: 1px solid rgba(148,163,184,.22);
    border-radius: 16px;
    margin-bottom: 1rem;
}
.hero h1 { margin: 0 0 .3rem 0; font-size: clamp(1.5rem,3vw,2.2rem); color: #f8fafc; }
.hero p  { margin: 0; color: #cbd5e1; max-width: 72ch; }
</style>
""", unsafe_allow_html=True)

# ── Guards ────────────────────────────────────────────────────────────────────
eeg_ok = ("filtered_eeg" in st.session_state and st.session_state.filtered_eeg is not None)
log_ok = ("transformed_log" in st.session_state and st.session_state.transformed_log is not None)

if not eeg_ok:
    st.markdown("""<div class="hero"><h1>🧠 EEG Viewer</h1>
    <p>Load and filter your EEG data first, then come back here.</p></div>""",
    unsafe_allow_html=True)
    if "filtered_eeg" in st.session_state and st.session_state.filtered_eeg is None:
        st.warning("Data loaded but not filtered yet — press **Filter Data** on the Load page.")
    st.stop()

if not log_ok:
    st.warning("No transformed LOG found. Please process a LOG file first.")
    st.stop()

# ── Constants ─────────────────────────────────────────────────────────────────
CHANNELS  = ['Pz', 'PO7', 'OZ', 'PO8']
CH_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
CH_DASHES = ['solid', 'dash', 'dashdot', 'dot']
T_FIXED   = 250
T_AXIS    = np.linspace(-400, 600, T_FIXED)

# ── Data ──────────────────────────────────────────────────────────────────────
df       = st.session_state.filtered_eeg
log      = st.session_state.transformed_log
eeg_name = st.session_state.selected_eeg_name
cutoff   = st.session_state.cut

data = df.copy()
data.index = data.index * 4e-3

CONDITIONS = log['condition'].unique().tolist()

duration_s = (len(df) - 1) * 4e-3
minutes    = int(duration_s // 60)
seconds_r  = int(duration_s % 60)

# ── Session state defaults ────────────────────────────────────────────────────
if "erp_offset" not in st.session_state:
    st.session_state.erp_offset = 0.0

# ── Epoch helpers ─────────────────────────────────────────────────────────────
def slice_epochs(off, mask=None):
    log_sub = log[mask] if mask is not None else log
    starts  = (off + log_sub['stim_start']).to_numpy()
    stops   = (off + log_sub['stim_stop']).to_numpy()
    slices  = []
    for lower, upper in zip(starts, stops):
        s = data[CHANNELS][
            (data.index > (lower - 0.4)) & (data.index <= upper)
        ].to_numpy()
        if len(s) >= T_FIXED:
            slices.append(s[:T_FIXED])
        elif len(s) > 0:
            pad = np.zeros((T_FIXED - len(s), len(CHANNELS)))
            slices.append(np.concatenate([s, pad], axis=0))
        else:
            slices.append(np.zeros((T_FIXED, len(CHANNELS))))
    if not slices:
        return np.zeros((1, T_FIXED, len(CHANNELS)))
    return np.stack(slices, axis=0)

@st.cache_data
def compute_epochs(eeg_key, off, condition=None):
    mask = (log['condition'] == condition).to_numpy() if condition else None
    return slice_epochs(off, mask)

# ── Figure builders ───────────────────────────────────────────────────────────
@st.cache_data
def build_signal_json(eeg_key):
    _df  = st.session_state.filtered_eeg
    _idx = _df.index * 4e-3
    fig  = go.Figure()
    fig.add_trace(go.Scattergl(
        x=_idx, y=_df["OZ"], mode="lines", name="OZ",
        line=dict(color="#e5e7eb", width=1),
        hovertemplate="t=%{x:.3f}s  %{y:.4f}µV<extra></extra>"))
    fig.add_trace(go.Scattergl(
        x=_idx, y=_df["Fz"], mode="lines", name="Fz",
        line=dict(color="#d62728", width=0.7), opacity=0.5,
        hovertemplate="t=%{x:.3f}s  %{y:.4f}µV<extra></extra>"))
    fig.update_layout(
        height=420, margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111827",
        hovermode="x unified", showlegend=True,
        yaxis=dict(range=[-75, 75], gridcolor="rgba(148,163,184,0.14)",
                   title="Amplitude (µV)", zeroline=True,
                   zerolinecolor="rgba(255,80,80,0.5)", zerolinewidth=2),
        xaxis=dict(title="Time (s)", gridcolor="rgba(148,163,184,0.14)",
                   rangeslider=dict(visible=True), showspikes=True, spikemode="across"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(17,24,39,0.8)", bordercolor="rgba(148,163,184,0.25)",
            borderwidth=1, font=dict(color="#e5e7eb")),
    )
    return fig.to_json()

def build_erp_fig(erp_data, ymin, ymax):
    """Build ERP figure with NO title — label is rendered as st.markdown above."""
    fig = go.Figure()
    for ci, ch in enumerate(CHANNELS):
        fig.add_trace(go.Scatter(
            x=T_AXIS.tolist(), y=erp_data[:, ci].tolist(),
            mode="lines", name=ch,
            line=dict(color=CH_COLORS[ci], width=2, dash=CH_DASHES[ci]),
            hovertemplate=f"{ch}: %{{y:.3f}} µV<extra></extra>",
        ))
    fig.add_hline(y=0, line_dash="dot",
                  line_color="rgba(255,255,255,0.3)", line_width=1)
    fig.update_layout(
        height=280, margin=dict(l=20, r=20, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111827",
        showlegend=True, hovermode="x unified",
        yaxis=dict(range=[ymin, ymax], gridcolor="rgba(148,163,184,0.14)",
                   title="Amplitude (µV)"),
        xaxis=dict(title="Time (ms)", gridcolor="rgba(148,163,184,0.14)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(17,24,39,0.8)", bordercolor="rgba(148,163,184,0.25)",
            borderwidth=1, font=dict(color="#e5e7eb")),
    )
    return fig

def build_heatmap_fig(epochs_3d, ch_idx):
    """Build heatmap with NO title — label rendered as st.markdown above."""
    img = epochs_3d[:, :, ch_idx]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=img, x=T_AXIS.tolist(), y=list(range(img.shape[0])),
        colorscale="Viridis",
        hovertemplate="Time=%{x:.0f}ms  Trial=%{y}  %{z:.3f}µV<extra></extra>",
    ))
    fig.update_layout(
        height=200, margin=dict(l=20, r=60, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111827",
        xaxis=dict(title="Time (ms)", gridcolor="rgba(148,163,184,0.14)"),
        yaxis=dict(title="Trial #",   gridcolor="rgba(148,163,184,0.14)"),
    )
    return fig

# ── Base shapes (no offset baked in — JS applies offset) ──────────────────────
base_shapes_js = json.dumps([
    {"x0": float(r.stim_start), "x1": float(r.stim_stop),
     "condition": str(r.condition)}
    for r in log.itertuples()
])

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Stat cards + signal chart (never reruns on offset change)
# ═════════════════════════════════════════════════════════════════════════════
offset_init = st.session_state.erp_offset

st.markdown(f"""
<div class="stat-grid">
  <div class="stat"><div class="label">Points</div>
    <div class="value">{len(df):,}</div></div>
  <div class="stat"><div class="label">Cutoff in seconds</div>
    <div class="value">{cutoff} s</div></div>
  <div class="stat"><div class="label">Duration</div>
    <div class="value">{minutes}m {seconds_r}s</div></div>
  <div class="stat"><div class="label">Subject</div>
    <div class="value" id="offset-stat-display">{eeg_name[:3    ]}</div></div>
</div>
""", unsafe_allow_html=True)

signal_json = build_signal_json(eeg_name)

# Pre-render shapes at the current offset so the chart is correct on first load
initial_shapes = [
    dict(type="rect", xref="x", yref="paper",
         x0=float(r.stim_start) + offset_init,
         x1=float(r.stim_stop)  + offset_init,
         y0=0, y1=1, fillcolor="seagreen", opacity=0.3,
         line=dict(width=0), layer="below")
    for r in log.itertuples()
]
sig_fig = go.Figure(json.loads(signal_json))
sig_fig.update_layout(shapes=initial_shapes)
signal_chart_json = sig_fig.to_json()

signal_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:transparent;font-family:system-ui,sans-serif;color:#e5e7eb;}}
.toolbar{{display:flex;align-items:center;gap:10px;padding:6px 0 10px;
  border-bottom:1px solid rgba(148,163,184,0.15);margin-bottom:4px;}}
.btn{{background:#1e2535;border:1px solid rgba(148,163,184,0.3);border-radius:8px;
  color:#e5e7eb;font-size:12px;padding:5px 14px;cursor:pointer;}}
.btn:hover{{background:#253347;border-color:#22d3ee;}}
.btn.on{{background:rgba(34,211,238,0.12);border-color:#22d3ee;color:#22d3ee;}}
</style></head><body>
<div class="toolbar">
  <button class="btn on" id="toggle-quads">Hide regions</button>
</div>
<div id="chart"></div>
<script>
var fig = {signal_chart_json};
var baseShapes = {base_shapes_js};
var quadsOn = true;
var curOffset = {offset_init};

Plotly.newPlot('chart', fig.data, fig.layout, {{
  scrollZoom:true, displaylogo:false, responsive:true,
  modeBarButtonsToRemove:['select2d','lasso2d','autoScale2d']
}});

function buildShapes(off) {{
  if (!quadsOn) return [];
  return baseShapes.map(function(s) {{
    return {{type:'rect',xref:'x',yref:'paper',
             x0:s.x0+off, x1:s.x1+off, y0:0, y1:1,
             fillcolor:'seagreen', opacity:0.3,
             line:{{width:0}}, layer:'below'}};
  }});
}}

window.addEventListener('message', function(e) {{
  if (e.data && e.data.type === 'eeg_offset_update') {{
    curOffset = parseFloat(e.data.offset);
    Plotly.relayout('chart', {{shapes: buildShapes(curOffset)}});
  }}
}});

document.getElementById('toggle-quads').addEventListener('click', function() {{
  quadsOn = !quadsOn;
  this.textContent = quadsOn ? 'Hide regions' : 'Show regions';
  this.classList.toggle('on', quadsOn);
  Plotly.relayout('chart', {{shapes: buildShapes(curOffset)}});
}});
</script></body></html>"""

components.html(signal_html, height=510, scrolling=False)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ERP + heatmaps in a fragment
# The offset number_input lives inside a components.html widget so it is
# pure JS — it posts a message to the signal chart iframe AND triggers a
# fragment-only rerun via the hidden Streamlit number_input that mirrors it.
# ═════════════════════════════════════════════════════════════════════════════

# Sidebar: only show/hide checkboxes (no offset here)
with st.sidebar:
    st.header("Controls")
    st.markdown("**Show/hide ERP plots**")
    show_grand  = st.toggle("Grand average", value=False, key="show_grand")
    cond_checks = {c: st.toggle(f"Condition: {c}", value=True, key=f"show_cond_{c}")
                   for c in CONDITIONS}

@st.fragment
def erp_section(show_grand, cond_checks):
    # ── Offset input rendered inside the fragment via native Streamlit ─────────
    offset = st.number_input(
        "Stimulus offset (s) — shifts ERP epochs and signal regions",
        min_value=-100.0, max_value=100.0,
        value=float(st.session_state.erp_offset),
        step=0.01, format="%.2f",
        key="erp_offset_input",
    )
    st.session_state.erp_offset = offset

    # Push offset to signal chart iframe (pure JS, no rerun)
    push_js = f"""<script>
(function(){{
  var frames = window.parent.document.querySelectorAll('iframe');
  frames.forEach(function(f){{
    try{{
      f.contentWindow.postMessage({{type:'eeg_offset_update', offset:{offset}}}, '*');
    }}catch(e){{}}
  }});
}})();
</script>"""
    components.html(push_js, height=0)

    # ── Compute epochs ─────────────────────────────────────────────────────────
    initial  = compute_epochs(eeg_name, offset)
    n_trials = initial.shape[0]
    erp_mean = np.mean(initial, axis=0)

    cond_epochs_map = {cond: compute_epochs(eeg_name, offset, condition=cond)
                       for cond in CONDITIONS}

    # ── Shared y-range ─────────────────────────────────────────────────────────
    all_means = [erp_mean] + [np.mean(v, axis=0) for v in cond_epochs_map.values()]
    all_vals  = np.concatenate([m.flatten() for m in all_means])
    raw_min, raw_max = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    margin = max(abs(raw_max - raw_min) * 0.10, 0.5)
    ymin   = round(raw_min - margin, 2)
    ymax   = round(raw_max + margin, 2)

    # ── ERP plots ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## ERP plots")

    if show_grand:
        st.markdown(
            f'<div class="plot-label">Grand average — all trials (n={n_trials})</div>',
            unsafe_allow_html=True)
        st.plotly_chart(
            build_erp_fig(erp_mean, ymin, ymax),
            use_container_width=True, key="erp_grand")

    for cond in CONDITIONS:
        if cond_checks[cond]:
            ce   = cond_epochs_map[cond]
            mean = np.mean(ce, axis=0)
            st.markdown(
                f'<div class="plot-label">Condition: {cond} (n={ce.shape[0]})</div>',
                unsafe_allow_html=True)
            st.plotly_chart(
                build_erp_fig(mean, ymin, ymax),
                use_container_width=True, key=f"erp_cond_{cond}")

    # ── Heatmaps ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Single-trial heatmaps")

    for ci, ch in enumerate(CHANNELS):
        st.markdown(
            f'<div class="plot-label">Channel: {ch}</div>',
            unsafe_allow_html=True)
        st.plotly_chart(
            build_heatmap_fig(initial, ci),
            use_container_width=True, key=f"heatmap_{ch}")

erp_section(show_grand, cond_checks)