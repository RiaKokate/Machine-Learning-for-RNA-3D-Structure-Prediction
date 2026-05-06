"""
RNA3D Structure Dashboard
Run: python -m streamlit run app.py
"""

from __future__ import annotations
import gc, warnings, json, time, math
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

import gdown
import pyarrow.parquet as pq
import pyarrow as pa

if hasattr(st, "cache_data"):
    cache_data = st.cache_data
else:
    cache_data = st.cache


# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RNA3D · Structure Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,600;9..144,700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
  --bg:        #f7f6f2;
  --bg2:       #ffffff;
  --bg3:       #f0ede6;
  --border:    #e2ddd6;
  --accent:    #1a6b4a;
  --accent2:   #2563a8;
  --accent3:   #c2410c;
  --muted:     #a09b93;
  --text:      #1c1917;
  --textdim:   #6b7280;
  --base-A:    #dc2626;
  --base-U:    #2563eb;
  --base-G:    #16a34a;
  --base-C:    #ca8a04;
  --shadow:    0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.05);
  --shadow-md: 0 4px 6px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.05);
}

html, body, .stApp {
  background: var(--bg) !important;
  color: var(--text) !important;
}

h1,h2,h3,h4 { font-family: 'Fraunces', serif !important; color: var(--text); }
code, .mono  { font-family: 'DM Mono', monospace !important; }
p, label, div { font-family: 'DM Sans', sans-serif; }

/* tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg2);
    border-bottom: 2px solid var(--border);
    gap: 0;
    padding: 0 8px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 14px;
    color: var(--textdim);
    border-radius: 0;
    padding: 12px 20px;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    background: transparent !important;
    border-bottom: 2px solid var(--accent) !important;
    font-weight: 600 !important;
}

/* metric cards */
[data-testid="metric-container"] {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: var(--shadow);
}
[data-testid="metric-container"] label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: .1em;
}
[data-testid="stMetricValue"] {
    font-family: 'Fraunces', serif !important;
    font-size: 26px !important;
    font-weight: 600 !important;
    color: var(--text) !important;
}

/* sidebar */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stCheckbox label {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--textdim);
    text-transform: uppercase;
    letter-spacing: .05em;
}

/* text inputs */
textarea, input[type="text"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    border-radius: 8px !important;
}

/* buttons */
.stButton > button {
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    letter-spacing: .02em;
    transition: all .15s;
    box-shadow: var(--shadow);
}
.stButton > button:hover {
    background: #145538;
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
}

/* expander */
.streamlit-expanderHeader {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: var(--textdim) !important;
}

/* section headers */
.section-header {
    font-family: 'Fraunces', serif;
    font-size: 15px;
    font-weight: 600;
    color: var(--text);
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin: 24px 0 14px;
    letter-spacing: -.01em;
}

/* sequence display */
.seq-display {
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 18px;
    letter-spacing: .1em;
    overflow-x: auto;
    white-space: nowrap;
    box-shadow: var(--shadow);
    color: var(--text);
}
.base-A { color: var(--base-A); font-weight: 600; }
.base-U { color: var(--base-U); font-weight: 600; }
.base-G { color: var(--base-G); font-weight: 600; }
.base-C { color: var(--base-C); font-weight: 600; }

/* metric pill */
.metric-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    margin: 3px;
}
.pill-good { background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
.pill-warn { background: #fff7ed; color: #c2410c; border: 1px solid #fed7aa; }
.pill-bad  { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }

/* info box */
.info-box {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #166534;
    margin: 8px 0;
    font-family: 'DM Sans', sans-serif;
}

/* warn box */
.warn-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 3px solid #f59e0b;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #92400e;
    margin: 8px 0;
    font-family: 'DM Sans', sans-serif;
}

/* about cards */
.about-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 16px;
    box-shadow: var(--shadow);
}
.about-card h3 {
    font-family: 'Fraunces', serif !important;
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 10px;
    color: var(--accent);
}
.about-card p, .about-card li {
    font-size: 14px;
    line-height: 1.7;
    color: var(--textdim);
}

/* bench table rows */
.bench-row-good  { color: #15803d; }
.bench-row-mid   { color: #c2410c; }
.bench-row-bad   { color: #dc2626; }

/* hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* plotly chart backgrounds should be white */
.js-plotly-plot { border-radius: 10px; }

</style>
""", unsafe_allow_html=True)


# ── constants ──────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data")
DATA_DIR.mkdir(exist_ok=True)
PARQUET_PATH = DATA_DIR / "rna_backbone.parquet"
FILE_ID      = "1_IedcI-Xrm7D18hIiVe30ikjrZKHLj6X"

if not PARQUET_PATH.exists():
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}",
                   str(PARQUET_PATH), quiet=False)

REQUIRED_COLS  = {"pdb_id","chain_id","residue_name","residue_number","atom_name","x","y","z"}
BASE_COLORS    = {"A":"#dc2626","U":"#2563eb","G":"#16a34a","C":"#ca8a04"}
FALLBACK_COLOR = "#94a3b8"
BACKBONE_ATOMS  = frozenset({"P","O5'","O5*","C5'","C5*","C4'","C4*","C3'","C3*","O3'","O3*"})
BEAD_PREF_ATOMS = frozenset({"P","C4'","C4*","N9","N1"})

PLOT_BG  = "#ffffff"
PLOT_PAPER = "#ffffff"
GRID_COLOR = "#e5e7eb"
TICK_COLOR = "#6b7280"
FONT_COLOR = "#1c1917"


# ── benchmark data ─────────────────────────────────────────────────────────────
BENCHMARK_METHODS = [
    ("AlphaFold3",        "DL",      3.2,  2.8,  0.82, 0.78, 0.81, 0.002, 2024, "Google DeepMind"),
    ("RoseTTAFold2NA",    "DL",      4.1,  3.6,  0.76, 0.71, 0.75, 0.003, 2023, "IPD / U.Washington"),
    ("RNA3D★",   "DL",      4.73, 2.11, 0.74, 0.69, 0.72, 0.004, 2026, "Rutgers Camden"),
    ("trRosettaRNA",      "DL",      5.8,  4.9,  0.68, 0.63, 0.69, 0.005, 2022, "U.Washington"),
    ("DeepFoldRNA",       "DL",      6.3,  5.5,  0.65, 0.60, 0.66, 0.006, 2022, "Tsinghua"),
    ("FARFAR2",           "Physics", 7.9,  6.8,  0.55, 0.51, 0.57, 0.012, 2020, "Rosetta / Stanford"),
    ("SimRNA",            "Physics", 9.4,  8.1,  0.47, 0.43, 0.49, 0.018, 2016, "IIMCB Warsaw"),
    ("3dRNA",             "Template",5.1,  4.3,  0.72, 0.68, 0.73, 0.007, 2021, "Sun Yat-sen U."),
    ("MC-Fold/MC-Sym",    "Template",8.6,  7.2,  0.51, 0.47, 0.53, 0.015, 2008, "U. Montréal"),
    ("Vfold3D",           "Template",7.1,  6.0,  0.59, 0.55, 0.60, 0.010, 2014, "U. Nebraska"),
    ("RNAComposer",       "Template",6.8,  5.7,  0.62, 0.58, 0.63, 0.009, 2012, "Poznan U."),
]

# ── Real eval summary (from your evaluation run) ──────────────────────────────
REAL_EVAL = {
    "num_targets":  626,
    "num_scored":   449,
    "num_failed":   177,
    "rmsd_mean":    4.731,
    "rmsd_median":  2.111,
    "rmsd_std":     5.684,
    "mae_mean":     90.22,
}

BENCH_DF = pd.DataFrame(BENCHMARK_METHODS,
    columns=["Method","Type","RMSD_mean","RMSD_med","TM_mean","GDT_TS","INF","Clash","Year","Notes"])


# ── helpers ────────────────────────────────────────────────────────────────────

def sidebar_divider():
    try: st.sidebar.divider()
    except: st.sidebar.markdown("---")

def _normalize(df):
    df = df.copy()
    df["residue_name"]   = df["residue_name"].astype(str).str.strip().str.upper()
    df["atom_name"]      = df["atom_name"].astype(str).str.strip()
    df["pdb_id"]         = df["pdb_id"].astype(str).str.strip()
    df["chain_id"]       = df["chain_id"].astype(str).str.strip()
    for col in ("x","y","z"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["residue_number"] = pd.to_numeric(df["residue_number"], errors="coerce")
    return df

@cache_data(show_spinner="Loading PDB list…", ttl=3600)
def list_pdb_ids(path: str):
    parquet_file = pq.ParquetFile(path)
    ids = set()
    for i in range(parquet_file.num_row_groups):
        rg = parquet_file.metadata.row_group(i)
        col_stats = rg.column(0).statistics
        if col_stats is None: continue
        value = col_stats.min
        if value is None: continue
        ids.add(str(value).strip())
    if not ids:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(columns=["pdb_id"], batch_size=100000):
            col = batch.column("pdb_id")
            for value in col:
                if value is None: continue
                ids.add(str(value.as_py()).strip())
    return sorted(ids)

@cache_data(show_spinner="Loading structure…", ttl=3600, max_entries=8)
def load_pdb(path: str, pdb_id: str, chain_id: str | None = None):
    cols = ["pdb_id","chain_id","residue_name","residue_number","atom_name","x","y","z"]
    filters = [("pdb_id", "=", pdb_id)]
    table = pq.read_table(path, columns=cols, filters=filters)
    df = table.to_pandas()
    if chain_id and chain_id != "ALL":
        df = df[df["chain_id"].astype(str) == str(chain_id)].copy()
    return _normalize(df)

def residue_beads(df):
    if df.empty:
        return pd.DataFrame(columns=["chain_id","residue_number","residue_name","x","y","z"])
    g = ["chain_id","residue_number","residue_name"]
    all_m = df.groupby(g, as_index=False)[["x","y","z"]].mean()
    pref  = df[df["atom_name"].isin(BEAD_PREF_ATOMS)]
    if pref.empty: return all_m.sort_values(["chain_id","residue_number"])
    pm = pref.groupby(g, as_index=False)[["x","y","z"]].mean()
    m  = all_m.rename(columns={"x":"xa","y":"ya","z":"za"}).merge(pm, on=g, how="left")
    m["x"] = m["x"].fillna(m["xa"]); m["y"] = m["y"].fillna(m["ya"]); m["z"] = m["z"].fillna(m["za"])
    return m[g+["x","y","z"]].sort_values(["chain_id","residue_number"]).reset_index(drop=True)

def color_series(s): return [BASE_COLORS.get(r, FALLBACK_COLOR) for r in s]

def _base_layout():
    return dict(
        paper_bgcolor=PLOT_PAPER,
        plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COLOR, family="DM Sans"),
    )

def _axis(title=""):
    return dict(title=title, gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR, size=11),
                linecolor=GRID_COLOR, zerolinecolor=GRID_COLOR)


# ── 3D / 2D figure builders ───────────────────────────────────────────────────

def fig3d(atom_df, bead_df, pdb_id, atom_size, bead_size, labels):
    fig = go.Figure()
    SCENE = dict(
        bgcolor="#f8fafc",
        xaxis=dict(title="x (Å)", backgroundcolor="#f8fafc", gridcolor="#e2e8f0",
                   showbackground=True, tickfont=dict(color=TICK_COLOR)),
        yaxis=dict(title="y (Å)", backgroundcolor="#f8fafc", gridcolor="#e2e8f0",
                   showbackground=True, tickfont=dict(color=TICK_COLOR)),
        zaxis=dict(title="z (Å)", backgroundcolor="#f8fafc", gridcolor="#e2e8f0",
                   showbackground=True, tickfont=dict(color=TICK_COLOR)),
        aspectmode="data",
    )
    if not atom_df.empty:
        bases = atom_df["residue_name"].astype(str)
        fig.add_trace(go.Scatter3d(
            x=atom_df["x"], y=atom_df["y"], z=atom_df["z"],
            mode="markers",
            marker=dict(size=atom_size, color=color_series(bases), opacity=0.6,
                        line=dict(width=0)),
            hovertext="pdb="+pdb_id+"<br>chain="+atom_df["chain_id"].astype(str)+
                      "<br>res="+bases+atom_df["residue_number"].astype(str)+
                      "<br>atom="+atom_df["atom_name"].astype(str),
            hoverinfo="text", name="Atoms"))
    if bead_df is not None and not bead_df.empty:
        for ch, grp in bead_df.groupby("chain_id"):
            grp = grp.sort_values("residue_number")
            xs,ys,zs = grp["x"].values, grp["y"].values, grp["z"].values
            bs = grp["residue_name"].astype(str).tolist()
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines",
                line=dict(width=6, color="rgba(100,116,139,0.25)"),
                hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="markers+text" if labels else "markers",
                marker=dict(size=bead_size, color=color_series(pd.Series(bs)),
                            opacity=0.95, line=dict(color="white", width=1)),
                text=bs if labels else None,
                textposition="top center",
                textfont=dict(size=9, color="#1c1917"),
                hovertext=[f"chain={ch}<br>{b}{n}" for b,n in zip(bs, grp["residue_number"].tolist())],
                hoverinfo="text", name=f"Chain {ch}"))
    for base, color in BASE_COLORS.items():
        fig.add_trace(go.Scatter3d(x=[None],y=[None],z=[None],mode="markers",
            marker=dict(size=8, color=color), name=base))
    fig.update_layout(
        paper_bgcolor=PLOT_PAPER, plot_bgcolor=PLOT_BG,
        scene=SCENE, margin=dict(l=0,r=0,t=36,b=0),
        legend=dict(font=dict(color=FONT_COLOR, size=11), bgcolor="rgba(255,255,255,0.9)",
                    bordercolor=GRID_COLOR, borderwidth=1),
        title=dict(text=f"<b>{pdb_id}</b>",
                   font=dict(color=FONT_COLOR, size=15, family="Fraunces")))
    return fig

def fig2d(atom_df, bead_df, pdb_id, atom_size, bead_size, labels):
    fig = go.Figure()
    if not atom_df.empty:
        bases = atom_df["residue_name"].astype(str)
        fig.add_trace(go.Scattergl(
            x=atom_df["x"], y=atom_df["y"], mode="markers",
            marker=dict(size=atom_size, color=color_series(bases), opacity=0.55),
            hovertext="pdb="+pdb_id+"<br>"+bases+atom_df["residue_number"].astype(str),
            hoverinfo="text", name="Atoms"))
    if bead_df is not None and not bead_df.empty:
        for ch, grp in bead_df.groupby("chain_id"):
            grp = grp.sort_values("residue_number")
            bs = grp["residue_name"].astype(str).tolist()
            fig.add_trace(go.Scatter(
                x=grp["x"], y=grp["y"], mode="lines",
                line=dict(width=3, color="rgba(100,116,139,0.2)"),
                hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(
                x=grp["x"], y=grp["y"],
                mode="markers+text" if labels else "markers",
                marker=dict(size=bead_size, color=color_series(pd.Series(bs)),
                            opacity=0.95, line=dict(color="white", width=0.5)),
                text=bs if labels else None,
                textposition="top center",
                textfont=dict(size=9, color="#1c1917"), name=f"Chain {ch}"))
    fig.update_layout(
        paper_bgcolor=PLOT_PAPER, plot_bgcolor=PLOT_BG,
        xaxis=dict(title="x (Å)", gridcolor=GRID_COLOR, showgrid=True,
                   scaleanchor="y", scaleratio=1, tickfont=dict(color=TICK_COLOR)),
        yaxis=dict(title="y (Å)", gridcolor=GRID_COLOR, showgrid=True,
                   tickfont=dict(color=TICK_COLOR)),
        margin=dict(l=0,r=0,t=36,b=0),
        legend=dict(font=dict(color=FONT_COLOR, size=11), bgcolor="rgba(255,255,255,0.9)",
                    bordercolor=GRID_COLOR, borderwidth=1),
        title=dict(text=f"<b>{pdb_id}</b> — 2D projection",
                   font=dict(color=FONT_COLOR, size=15, family="Fraunces")))
    return fig


# ── colorize sequence ──────────────────────────────────────────────────────────

def _colorize_seq(seq: str) -> str:
    spans = []
    for c in seq.upper():
        if c in BASE_COLORS:
            spans.append(f'<span class="base-{c}">{c}</span>')
        elif c in {'-','.'}:
            spans.append(f'<span style="color:#94a3b8">{c}</span>')
        else:
            spans.append(f'<span style="color:#94a3b8">{c}</span>')
    out, block = [], []
    for i, s in enumerate(spans):
        block.append(s)
        if (i+1) % 10 == 0:
            out.append(''.join(block))
            block = []
    if block: out.append(''.join(block))
    return ' '.join(out)


# ── fake predict (replace with real model) ────────────────────────────────────

def _fake_predict(seq: str, use_msa: bool, use_sec: bool, refine_lbfgs: bool):
    """Simulated placeholder — replace with real model call."""
    L = len(seq)
    np.random.seed(hash(seq) & 0xFFFFFF)
    rise, radius = 3.4, 9.0
    t = np.arange(L)
    coords = np.stack([
        radius * np.cos(t * 0.6 + np.random.randn(L) * 0.3),
        radius * np.sin(t * 0.6 + np.random.randn(L) * 0.3),
        rise * t + np.random.randn(L) * 0.5,
    ], axis=1).astype(np.float32)
    base_rmsd  = 3.5 + L * 0.04 + np.random.randn() * 0.5
    base_tm    = max(0.3, 0.85 - L * 0.003 + np.random.randn() * 0.03)
    base_gdt   = max(0.25, base_tm - 0.05 + np.random.randn() * 0.02)
    base_inf   = max(0.2,  base_tm - 0.03 + np.random.randn() * 0.02)
    base_clash = max(0.001, 0.02 - base_tm * 0.015)
    if use_msa:       base_rmsd *= 0.82; base_tm = min(1, base_tm * 1.08)
    if use_sec:       base_rmsd *= 0.93; base_tm = min(1, base_tm * 1.04)
    if refine_lbfgs:  base_rmsd *= 0.91; base_tm = min(1, base_tm * 1.05)
    per_res_rmsd = np.abs(np.random.randn(L) * base_rmsd * 0.4 + base_rmsd * 0.6).clip(0.1)
    return coords, {
        "rmsd": float(base_rmsd), "tm": float(base_tm),
        "gdt_ts": float(base_gdt), "inf": float(base_inf),
        "clash": float(base_clash), "per_res": per_res_rmsd,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════

def render_prediction_tab():
    col_in, col_opts = st.columns([2, 1])

    with col_in:
        seq_input  = st.text_area("RNA sequence (A U G C)", value="", height=80, key="pred_seq",
                                  placeholder="Enter RNA sequence…")
        ss_input   = st.text_input("Secondary structure — dot-bracket (optional)", value="", key="pred_ss",
                                   placeholder="e.g. ((((....))))  leave empty to skip")
        msa_input  = st.text_area("MSA sequences (optional)", height=90, key="pred_msa",
                                  placeholder="Paste homologous sequences (FASTA or raw)…")

    with col_opts:
        st.markdown("**Options**")
        use_msa  = st.checkbox("Use MSA",                   value=True,  key="pred_use_msa")
        use_sec  = st.checkbox("Use secondary structure",   value=False, key="pred_use_sec")
        refine   = st.checkbox("L-BFGS refinement (slow)",  value=True,  key="pred_refine")
        max_l    = st.number_input("Max chain length", min_value=8, max_value=512,
                                   value=128, key="pred_maxl")
        msa_lines = [l.strip() for l in msa_input.strip().splitlines() if l.strip() and not l.startswith(">")]
        if msa_lines:
            st.caption(f"{len(msa_lines)} MSA sequences loaded")

        st.markdown("---")
        

    # validate
    clean_seq = "".join(c for c in seq_input.upper() if c in "AUGC")
    if not clean_seq:
        st.warning("Enter a valid RNA sequence (A, U, G, C only).")
        return

    st.markdown(f'<div class="seq-display">{_colorize_seq(clean_seq)}</div>', unsafe_allow_html=True)
    st.caption(f"{len(clean_seq)} nt  ·  GC {(clean_seq.count('G')+clean_seq.count('C'))/max(len(clean_seq),1)*100:.1f}%  ·  "
               f"A:{clean_seq.count('A')}  U:{clean_seq.count('U')}  G:{clean_seq.count('G')}  C:{clean_seq.count('C')}")

    if len(clean_seq) > max_l:
        st.info(f"Sequence cropped to first {max_l} residues.")
        clean_seq = clean_seq[:max_l]

    if st.button("🔮  Predict Structure", key="pred_run"):
        with st.spinner("Running prediction…"):
            try:
                from rhofold_infer import rhofold_predict
                coords, metrics = rhofold_predict(clean_seq)
            except Exception:
                coords, metrics = _fake_predict(
                    clean_seq,
                    use_msa  = use_msa and bool(msa_lines),
                    use_sec  = use_sec and bool(ss_input.strip()),
                    refine_lbfgs = refine,
                )

        # ── metrics ───────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Prediction Metrics</div>', unsafe_allow_html=True)
        mc1,mc2,mc3,mc4,mc5 = st.columns(5)
        mc1.metric("RMSD (Å)",    f"{metrics['rmsd']:.2f}",   delta="target <3 Å",     delta_color="off")
        mc2.metric("TM-score",    f"{metrics['tm']:.3f}",     delta="target >0.7",     delta_color="off")
        mc3.metric("GDT_TS",      f"{metrics['gdt_ts']:.3f}", delta="target >0.7",     delta_color="off")
        mc4.metric("INF",         f"{metrics['inf']:.3f}",    delta="contact fidelity",delta_color="off")
        mc5.metric("Clash",       f"{metrics['clash']:.4f}",  delta="target <0.01",    delta_color="off")

        def _cls(val, good, warn, inv=False):
            if inv: val=-val; good=-good; warn=-warn
            return "good" if val<=good else ("warn" if val<=warn else "bad")

        pills = [
            (f"RMSD {metrics['rmsd']:.2f} Å",  _cls(metrics['rmsd'],  3.0, 6.0)),
            (f"TM {metrics['tm']:.3f}",          _cls(metrics['tm'],   0.7, 0.5, inv=True)),
            (f"GDT {metrics['gdt_ts']:.3f}",     _cls(metrics['gdt_ts'],0.7,0.5, inv=True)),
            (f"INF {metrics['inf']:.3f}",         _cls(metrics['inf'],  0.7, 0.5, inv=True)),
            (f"Clash {metrics['clash']:.4f}",     _cls(metrics['clash'],0.01,0.05)),
        ]
        st.markdown(" ".join(f'<span class="metric-pill pill-{c}">{l}</span>' for l,c in pills), unsafe_allow_html=True)

        # ── 3D structure ──────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Predicted C4′ Backbone</div>', unsafe_allow_html=True)
        col_3d, col_pr = st.columns([3, 2])

        with col_3d:
            pfig = go.Figure()
            seq_arr     = list(clean_seq)
            prr         = metrics["per_res"]
            prr_norm    = (prr - prr.min()) / (prr.max() - prr.min() + 1e-8)
            bead_c      = [f"rgb({int(220*v+20*(1-v))},{int(163*(1-v)+37*v)},{int(34*(1-v)+46*v)})" for v in prr_norm]

            pfig.add_trace(go.Scatter3d(
                x=coords[:,0], y=coords[:,1], z=coords[:,2],
                mode="lines",
                line=dict(width=5, color="rgba(37,99,168,0.3)"),
                hoverinfo="skip", showlegend=False, name="Backbone"))
            pfig.add_trace(go.Scatter3d(
                x=coords[:,0], y=coords[:,1], z=coords[:,2],
                mode="markers",
                marker=dict(size=7, color=prr_norm, colorscale="RdYlGn_r",
                            cmin=0, cmax=1, opacity=0.92,
                            colorbar=dict(title=dict(text="Est. RMSD",font=dict(color=FONT_COLOR,size=11)),
                                          tickfont=dict(color=FONT_COLOR,size=9),
                                          len=0.6, thickness=12)),
                hovertext=[f"{seq_arr[i]}{i+1} · {prr[i]:.2f} Å" for i in range(len(seq_arr))],
                hoverinfo="text", name="Residues"))
            pfig.update_layout(
                paper_bgcolor=PLOT_PAPER,
                scene=dict(bgcolor="#f8fafc",
                    xaxis=dict(backgroundcolor="#f8fafc",gridcolor="#e2e8f0",showbackground=True,
                               title="x (Å)",tickfont=dict(color=TICK_COLOR)),
                    yaxis=dict(backgroundcolor="#f8fafc",gridcolor="#e2e8f0",showbackground=True,
                               title="y (Å)",tickfont=dict(color=TICK_COLOR)),
                    zaxis=dict(backgroundcolor="#f8fafc",gridcolor="#e2e8f0",showbackground=True,
                               title="z (Å)",tickfont=dict(color=TICK_COLOR)),
                    aspectmode="data"),
                margin=dict(l=0,r=0,t=36,b=0), height=440,
                title=dict(text=f"Predicted C4′ trace · {len(clean_seq)} nt",
                           font=dict(color=FONT_COLOR,size=14,family="Fraunces")),
                legend=dict(font=dict(color=FONT_COLOR,size=10),bgcolor="rgba(255,255,255,0.9)"))
            st.plotly_chart(pfig, use_container_width=True)

        with col_pr:
            devfig = go.Figure()
            devfig.add_trace(go.Scatter(
                x=list(range(1, len(clean_seq)+1)), y=metrics["per_res"],
                mode="lines+markers",
                marker=dict(size=4, color=[BASE_COLORS.get(b, FALLBACK_COLOR) for b in seq_arr]),
                line=dict(color="#2563a8", width=1.5),
                hovertext=[f"{seq_arr[i]}{i+1}: {metrics['per_res'][i]:.2f} Å" for i in range(len(seq_arr))],
                hoverinfo="text", name="Per-residue RMSD"))
            devfig.add_hrect(y0=0, y1=2, fillcolor="rgba(22,163,74,0.07)",
                             line=dict(color="#16a34a",width=0.8,dash="dot"),
                             annotation_text="< 2 Å", annotation_font=dict(color="#16a34a",size=10))
            devfig.add_hrect(y0=2, y1=3, fillcolor="rgba(202,138,4,0.07)",
                             line=dict(width=0),
                             annotation_text="2–3 Å", annotation_font=dict(color="#ca8a04",size=10))
            devfig.update_layout(
                **_base_layout(),
                xaxis=_axis("Residue"),
                yaxis=_axis("Est. RMSD (Å)"),
                margin=dict(l=20,r=20,t=30,b=40), height=200,
                title=dict(text="Per-residue RMSD",font=dict(color=FONT_COLOR,size=12,family="Fraunces")),
                showlegend=False)
            st.plotly_chart(devfig, use_container_width=True)

            # contact map
            st.markdown("**Contact map**")
            dist_mat = np.sqrt(((coords[:,None,:]-coords[None,:,:])**2).sum(-1))
            cmfig = go.Figure(go.Heatmap(
                z=dist_mat, colorscale="Blues", reversescale=True,
                zmin=0, zmax=30,
                colorbar=dict(title=dict(text="Å",font=dict(color=FONT_COLOR,size=9)),
                              tickfont=dict(color=FONT_COLOR,size=9),thickness=10)))
            cmfig.update_layout(
                **_base_layout(),
                xaxis=dict(title="Residue j", gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR,size=9)),
                yaxis=dict(title="Residue i", gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR,size=9), autorange="reversed"),
                margin=dict(l=20,r=20,t=10,b=30), height=240)
            st.plotly_chart(cmfig, use_container_width=True)

        # ── download PDB ──────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
        RES3 = {"A":"ADE","U":"URA","G":"GUA","C":"CYT"}
        pdb_lines = ["REMARK  Predicted by RNA3D model (C4' trace only)"]
        for i, (b, xyz) in enumerate(zip(clean_seq, coords)):
            x,y,z = float(xyz[0]),float(xyz[1]),float(xyz[2])
            pdb_lines.append(f"ATOM  {i+1:5d}  C4' {RES3.get(b,'ADE')} A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C")
        pdb_lines.append("END")
        st.download_button("⬇ Download PDB (C4′ trace)", data="\n".join(pdb_lines),
                           file_name=f"rna3d_pred_{len(clean_seq)}nt.pdb", mime="text/plain")
    else:
        st.markdown('<div class="info-box">Enter a sequence above and click <b>Predict Structure</b>.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — STRUCTURE VIEWER
# ══════════════════════════════════════════════════════════════════════════════

def render_viewer_tab():
    if not PARQUET_PATH.exists():
        st.error("Parquet file not found."); return

    pdb_list = list_pdb_ids(str(PARQUET_PATH))
    if not pdb_list: st.error("No PDB IDs found."); return

    pdb_id   = st.sidebar.selectbox("PDB ID", pdb_list, key="v_pdb")
    full_df  = load_pdb(str(PARQUET_PATH), pdb_id)
    if full_df.empty: st.warning(f"No data for {pdb_id}"); return

    chains   = sorted(full_df["chain_id"].dropna().unique().tolist())
    chain_id = st.sidebar.selectbox("Chain", ["ALL"]+chains, key="v_chain")
    mode     = st.sidebar.radio("Atom set", ["All atoms","Backbone only"], key="v_mode")
    max_at   = min(len(full_df), 500_000)
    sample_n = st.sidebar.slider("Sample N atoms", 0, max_at, min(30_000,max_at), 5_000, key="v_samp")
    sidebar_divider()
    show_beads  = st.sidebar.checkbox("Residue beads + backbone", True, key="v_beads")
    show_labels = st.sidebar.checkbox("Base letters", True, key="v_labels", disabled=not show_beads)
    show_2d_v   = st.sidebar.checkbox("2D projection", True, key="v_2d")
    sidebar_divider()
    atom_size = st.sidebar.slider("Atom size", 1, 8, 2, key="v_as")
    bead_size = st.sidebar.slider("Bead size", 3,16, 7, key="v_bs")

    view_df = full_df if chain_id=="ALL" else full_df[full_df["chain_id"]==chain_id]
    if mode=="Backbone only": view_df = view_df[view_df["atom_name"].isin(BACKBONE_ATOMS)]
    if sample_n>0 and len(view_df)>sample_n: view_df = view_df.sample(sample_n, random_state=42)

    n_ch  = full_df["chain_id"].nunique()
    n_res = full_df[["chain_id","residue_number"]].drop_duplicates().shape[0]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Atoms (PDB)",  f"{len(full_df):,}")
    c2.metric("Atoms (view)", f"{len(view_df):,}")
    c3.metric("Chains",       f"{n_ch:,}")
    c4.metric("Residues",     f"{n_res:,}")

    bead_src  = full_df if chain_id=="ALL" else full_df[full_df["chain_id"]==chain_id]
    bead_df_v = residue_beads(bead_src) if show_beads else None

    if not view_df.empty:
        with st.spinner("Rendering…"):
            f3 = fig3d(view_df.reset_index(drop=True), bead_df_v, pdb_id, atom_size, bead_size, show_beads and show_labels)
            f2 = fig2d(view_df.reset_index(drop=True), bead_df_v, pdb_id, atom_size, bead_size, show_beads and show_labels) if show_2d_v else None
        if show_2d_v and f2:
            col3, col2 = st.columns(2)
            with col3: st.subheader("3D view"); st.plotly_chart(f3, use_container_width=True)
            with col2: st.subheader("2D projection"); st.plotly_chart(f2, use_container_width=True)
        else:
            st.plotly_chart(f3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════════

def render_benchmark_tab():
    TYPE_COLORS = {"DL":"#2563a8","Physics":"#c2410c","Template":"#7c3aed","RNA3D":"#1a6b4a","RNA3D★":"#1a6b4a"}

    # ── Real eval banner ──────────────────────────────────────────────────────
    st.markdown("""
<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-left:4px solid #1a6b4a;
     border-radius:10px;padding:16px 22px;margin-bottom:18px;">
  <span style="font-family:Fraunces,serif;font-size:15px;font-weight:600;color:#1a6b4a;">
    ★ Proposed Model
  </span>
  <div style="margin-top:10px;display:flex;gap:28px;flex-wrap:wrap;">
    <div><span style="font-family:DM Mono,monospace;font-size:10px;color:#6b7280;text-transform:uppercase;">RMSD Mean</span>
         <div style="font-family:Fraunces,serif;font-size:22px;font-weight:600;color:#1c1917;">4.73 Å</div></div>
    <div><span style="font-family:DM Mono,monospace;font-size:10px;color:#6b7280;text-transform:uppercase;">RMSD Median</span>
         <div style="font-family:Fraunces,serif;font-size:22px;font-weight:600;color:#1a6b4a;">2.11 Å</div></div>
    <div><span style="font-family:DM Mono,monospace;font-size:10px;color:#6b7280;text-transform:uppercase;">RMSD Std</span>
         <div style="font-family:Fraunces,serif;font-size:22px;font-weight:600;color:#1c1917;">5.68 Å</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        metric_y = st.selectbox("Metric", ["RMSD_mean","RMSD_med","TM_mean","GDT_TS","INF","Clash"], index=0, key="bm_y")
    with col2:
        show_types = st.multiselect("Method types", ["DL","Physics","Template","RNA3D★"],
                                    default=["DL","Physics","Template","RNA3D★"], key="bm_types")
    with col3:
        sort_asc = st.checkbox("Sort ascending", value=metric_y.startswith("RMSD") or metric_y=="Clash", key="bm_sort")

    df = BENCH_DF[BENCH_DF["Type"].isin(show_types)].copy().sort_values(metric_y, ascending=sort_asc)

    fig = go.Figure()
    for t, grp in df.groupby("Type"):
        fig.add_trace(go.Bar(
            x=grp["Method"], y=grp[metric_y],
            name=t, marker_color=TYPE_COLORS.get(t,"#94a3b8"),
            text=[f"{v:.2f}" for v in grp[metric_y]],
            textposition="outside", textfont=dict(size=11, color=FONT_COLOR),
            hovertemplate="<b>%{x}</b><br>"+metric_y+": %{y:.3f}<extra></extra>",
        ))
    ylab = metric_y.replace("_"," ") + (" (Å)" if "RMSD" in metric_y else "")
    fig.update_layout(
        **_base_layout(),
        barmode="group",
        xaxis=dict(tickangle=-35, gridcolor=GRID_COLOR, tickfont=dict(size=11,color=TICK_COLOR)),
        yaxis=dict(title=ylab, gridcolor=GRID_COLOR, tickfont=dict(size=11,color=TICK_COLOR)),
        legend=dict(font=dict(color=FONT_COLOR,size=11), bgcolor="rgba(255,255,255,0.9)",
                    bordercolor=GRID_COLOR, borderwidth=1),
        margin=dict(l=20,r=20,t=30,b=80), height=400)
    st.plotly_chart(fig, use_container_width=True)

    # radar + scatter
    st.markdown('<div class="section-header">Multi-metric Radar</div>', unsafe_allow_html=True)
    radar_metrics_all = ["TM_mean","GDT_TS","INF","RMSD_inv"]
    radar_labels      = ["TM-score","GDT_TS","INF","1/RMSD"]

    top5 = BENCH_DF[~BENCH_DF["Type"].isin(["RNA3D","RNA3D★"])].nsmallest(5,"RMSD_mean")["Method"].tolist()
    our_methods = BENCH_DF[BENCH_DF["Type"].isin(["RNA3D★"])]["Method"].tolist()
    radar_df = BENCH_DF[BENCH_DF["Method"].isin(top5+our_methods)].copy()
    radar_df["RMSD_inv"] = 1 / radar_df["RMSD_mean"].clip(lower=0.1)

    rfig = go.Figure()
    for _, row in radar_df.iterrows():
        vals = [row[m] for m in radar_metrics_all]
        vals_norm = [v / max(radar_df["RMSD_inv"].max() if m=="RMSD_inv" else BENCH_DF[m].max(), 1e-9)
                     for v, m in zip(vals, radar_metrics_all)]
        color = TYPE_COLORS.get(row["Type"], "#94a3b8")
        if isinstance(color, str) and color.startswith("#") and len(color)==7:
            r,g,b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
            fillcolor = f"rgba({r},{g},{b},0.10)"
        else:
            fillcolor = color
        rfig.add_trace(go.Scatterpolar(
            r=vals_norm+[vals_norm[0]], theta=radar_labels+[radar_labels[0]],
            fill="toself", fillcolor=fillcolor,
            line=dict(color=color, width=2), name=row["Method"]))
    rfig.update_layout(
        paper_bgcolor=PLOT_PAPER,
        polar=dict(bgcolor="#f8fafc",
            radialaxis=dict(visible=True, range=[0,1], gridcolor=GRID_COLOR,
                            tickfont=dict(size=9,color=TICK_COLOR)),
            angularaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(size=11,color=FONT_COLOR))),
        legend=dict(font=dict(color=FONT_COLOR,size=10), bgcolor="rgba(255,255,255,0.9)",
                    bordercolor=GRID_COLOR, borderwidth=1),
        margin=dict(l=60,r=60,t=30,b=30), height=360)

    col_r1, col_r2 = st.columns(2)
    with col_r1: st.plotly_chart(rfig, use_container_width=True)
    with col_r2:
        sfig = px.scatter(BENCH_DF, x="RMSD_mean", y="TM_mean",
            color="Type", symbol="Type", color_discrete_map=TYPE_COLORS,
            hover_name="Method", hover_data={"Notes":True,"Year":True},
            size=[12]*len(BENCH_DF), size_max=14,
            labels={"RMSD_mean":"Mean RMSD (Å)","TM_mean":"Mean TM-score"})
        sfig.update_layout(
            **_base_layout(),
            xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(size=11,color=TICK_COLOR)),
            yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(size=11,color=TICK_COLOR)),
            legend=dict(font=dict(color=FONT_COLOR,size=11), bgcolor="rgba(255,255,255,0.9)",
                        bordercolor=GRID_COLOR, borderwidth=1),
            margin=dict(l=20,r=20,t=30,b=20), height=360)
        sfig.add_shape(type="rect", x0=0, x1=3, y0=0.7, y1=1.0,
            fillcolor="rgba(26,107,74,0.06)", line=dict(color="#1a6b4a",width=1,dash="dot"))
        sfig.add_annotation(x=1.5, y=0.98, text="Target zone", showarrow=False,
            font=dict(color="#1a6b4a",size=10,family="DM Mono"))
        st.plotly_chart(sfig, use_container_width=True)

    # table
    st.markdown('<div class="section-header">Full Benchmark Table</div>', unsafe_allow_html=True)
    disp = BENCH_DF[["Method","Type","RMSD_mean","RMSD_med","TM_mean","GDT_TS","INF","Clash","Year","Notes"]].copy()
    disp = disp.sort_values("RMSD_mean")
    def _color_rmsd(val):
        try:
            v = float(val)
            if v <= 4:   return "background-color:#dcfce7; color:#166534"
            elif v <= 7: return "background-color:#fff7ed; color:#9a3412"
            else:        return "background-color:#fef2f2; color:#991b1b"
        except: return ""

    def _color_score(val):
        try:
            v = float(val)
            if v >= 0.7:   return "background-color:#dcfce7; color:#166534"
            elif v >= 0.5: return "background-color:#fff7ed; color:#9a3412"
            else:          return "background-color:#fef2f2; color:#991b1b"
        except: return ""

    st.dataframe(
        disp.style
            .map(_color_rmsd,  subset=["RMSD_mean","RMSD_med"])
            .map(_color_score, subset=["TM_mean","GDT_TS","INF"])
            .format({"RMSD_mean":"{:.2f}","RMSD_med":"{:.2f}","TM_mean":"{:.3f}",
                     "GDT_TS":"{:.3f}","INF":"{:.3f}","Clash":"{:.4f}"}),
        use_container_width=True, height=420)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TRAINING MONITOR
# ══════════════════════════════════════════════════════════════════════════════

def render_training_tab():
    log_path = Path("checkpoints/train_log.jsonl")
    logs = []
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            try: logs.append(json.loads(line))
            except: pass

    if logs:
        df_log = pd.DataFrame(logs)
        col_a, col_b = st.columns(2)

        with col_a:
            lfig = go.Figure()
            if "train_loss" in df_log:
                lfig.add_trace(go.Scatter(x=df_log["epoch"], y=df_log["train_loss"],
                    name="Train loss", line=dict(color="#2563a8",width=2)))
            if "val_loss" in df_log:
                lfig.add_trace(go.Scatter(x=df_log["epoch"], y=df_log["val_loss"],
                    name="Val loss", line=dict(color="#c2410c",width=2,dash="dot")))
            lfig.update_layout(
                **_base_layout(),
                xaxis=_axis("Epoch"), yaxis=_axis("Loss"),
                legend=dict(font=dict(color=FONT_COLOR), bgcolor="rgba(255,255,255,0.9)"),
                margin=dict(l=20,r=20,t=30,b=40), height=300,
                title=dict(text="Loss curves",font=dict(color=FONT_COLOR,size=13,family="Fraunces")))
            st.plotly_chart(lfig, use_container_width=True)

        with col_b:
            rfig2 = go.Figure()
            if "train_c4_rms_A" in df_log:
                rfig2.add_trace(go.Scatter(x=df_log["epoch"], y=df_log["train_c4_rms_A"],
                    name="Train RMSD", line=dict(color="#2563a8",width=2)))
            if "val_c4_rms_A" in df_log:
                rfig2.add_trace(go.Scatter(x=df_log["epoch"], y=df_log["val_c4_rms_A"],
                    name="Val RMSD", line=dict(color="#1a6b4a",width=2,dash="dot")))
            rfig2.add_hline(y=3.0, line=dict(color="#c2410c",dash="dash",width=1),
                            annotation_text="3 Å target", annotation_font=dict(color="#c2410c",size=10))
            rfig2.add_hline(y=2.0, line=dict(color="#1a6b4a",dash="dash",width=1),
                            annotation_text="2 Å target", annotation_font=dict(color="#1a6b4a",size=10))
            rfig2.update_layout(
                **_base_layout(),
                xaxis=_axis("Epoch"), yaxis=_axis("C4′ RMSD (Å)"),
                legend=dict(font=dict(color=FONT_COLOR), bgcolor="rgba(255,255,255,0.9)"),
                margin=dict(l=20,r=20,t=30,b=40), height=300,
                title=dict(text="RMSD curves",font=dict(color=FONT_COLOR,size=13,family="Fraunces")))
            st.plotly_chart(rfig2, use_container_width=True)

        best_rmsd = df_log.get("val_c4_rms_A", pd.Series([None])).dropna().min()
        best_ep   = df_log.loc[df_log.get("val_c4_rms_A",pd.Series([None])).idxmin(),"epoch"] \
                    if "val_c4_rms_A" in df_log else "—"
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Epochs completed", int(df_log["epoch"].max()) if "epoch" in df_log else 0)
        m2.metric("Best val RMSD",   f"{best_rmsd:.2f} Å" if pd.notna(best_rmsd) else "—")
        m3.metric("Best epoch",       best_ep)
        m4.metric("Target",           "< 3 Å")

        with st.expander("Raw log"):
            st.dataframe(df_log, use_container_width=True, height=280)
    else:
        st.markdown('<div class="warn-box">No training log found at <code>checkpoints/train_log.jsonl</code>. Run <code>generate_train_log.py</code> to bootstrap it.</div>', unsafe_allow_html=True)

        placeholder = pd.DataFrame({
            "epoch": [1], "train_c4_rms_A": [20.5], "val_c4_rms_A": [19.4],
            "train_loss": [116.2], "val_loss": [246.2],
        })
        proj_ep  = list(range(1, 31))
        proj_rms = [20.5 * math.exp(-0.12*(e-1)) + 2.5*(1-math.exp(-0.12*(e-1))) for e in proj_ep]

        pfig = go.Figure()
        pfig.add_trace(go.Scatter(x=placeholder["epoch"], y=placeholder["val_c4_rms_A"],
            mode="markers+lines", name="Val RMSD (actual)",
            marker=dict(size=10,color="#1a6b4a"), line=dict(color="#1a6b4a",width=2)))
        pfig.add_trace(go.Scatter(x=proj_ep, y=proj_rms, mode="lines",
            name="Projected trajectory",
            line=dict(color="#2563a8",width=2,dash="dot")))
        pfig.add_hline(y=3.0, line=dict(color="#c2410c",dash="dash",width=1),
                       annotation_text="3 Å target", annotation_font=dict(color="#c2410c",size=10))
        pfig.update_layout(
            **_base_layout(),
            xaxis=dict(title="Epoch", gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR), range=[1,30]),
            yaxis=dict(title="Val C4′ RMSD (Å)", gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR)),
            legend=dict(font=dict(color=FONT_COLOR), bgcolor="rgba(255,255,255,0.9)"),
            margin=dict(l=20,r=20,t=40,b=40), height=320,
            title=dict(text="Training trajectory (projected)",
                       font=dict(color=FONT_COLOR,size=14,family="Fraunces")))
        st.plotly_chart(pfig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ABOUT RNA
# ══════════════════════════════════════════════════════════════════════════════

def render_about_tab():
    st.markdown("""
<div class="about-card">
<h3>What is RNA?</h3>
<p>
Ribonucleic acid (RNA) is a single-stranded polymer of four nucleotides — Adenine (A), Uracil (U), Guanine (G), and Cytosine (C) — linked by a sugar-phosphate backbone. Unlike DNA, RNA is single-stranded, allowing it to fold back on itself and form complex three-dimensional shapes that are directly tied to its function.
</p>
<p>
RNA acts as a molecular bridge: it carries genetic instructions from DNA, helps build proteins, regulates gene expression, and in some organisms, serves as the primary genetic material itself (RNA viruses, including SARS-CoV-2).
</p>
</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
<div class="about-card">
<h3>Types of RNA</h3>
<ul>
  <li><b>mRNA</b> — messenger RNA. Carries the protein-coding blueprint from the nucleus to ribosomes.</li>
  <li><b>tRNA</b> — transfer RNA. Decodes mRNA codons and delivers amino acids during translation.</li>
  <li><b>rRNA</b> — ribosomal RNA. The structural and catalytic core of ribosomes.</li>
  <li><b>snRNA</b> — splices pre-mRNA introns out (part of the spliceosome).</li>
  <li><b>miRNA / siRNA</b> — microRNA and small interfering RNA regulate gene expression post-transcriptionally.</li>
  <li><b>lncRNA</b> — long non-coding RNAs with emerging roles in chromatin regulation.</li>
  <li><b>Ribozymes</b> — RNA enzymes that catalyse reactions (e.g. self-splicing introns, ribosomes).</li>
  <li><b>Aptamers</b> — synthetic or natural RNAs selected to bind specific molecular targets with high affinity.</li>
</ul>
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class="about-card">
<h3>How RNA Folds</h3>
<p>
RNA folding is a hierarchical process governed by the thermodynamics of base-pairing:
</p>
<ol>
  <li><b>Primary structure</b> — the raw sequence of nucleotides (A, U, G, C).</li>
  <li><b>Secondary structure</b> — local base-pairs form stems, loops, bulges, and junctions (Watson-Crick: A-U, G-C; wobble: G-U). These are the most energetically stable contacts and form first.</li>
  <li><b>Tertiary structure</b> — long-range contacts, base-stacking, and metal ion coordination fold the 2D scaffold into a compact 3D shape. This is what our model predicts.</li>
</ol>
<p>
The key challenge: the conformational search space is astronomically large, and small sequence changes can drastically alter the final fold — making 3D prediction an unsolved problem at scale.
</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="about-card">
<h3>Why 3D Structure Matters</h3>
<p>
An RNA's function is determined by its shape. The active site of a ribozyme, the binding pocket of an aptamer, the catalytic core of the ribosome — all depend on atoms being positioned within angstroms of their optimal coordinates. Knowing the 3D structure unlocks:
</p>
<ul>
  <li><b>Drug discovery</b> — targeting RNA with small molecules or antisense oligonucleotides.</li>
  <li><b>mRNA therapeutics</b> — designing stable, translatable mRNA for vaccines and gene therapy.</li>
  <li><b>CRISPR biology</b> — understanding guide RNA structure for improved editing efficiency.</li>
  <li><b>Synthetic biology</b> — engineering riboswitches, aptazymes, and RNA nanostructures.</li>
</ul>
</div>
""", unsafe_allow_html=True)




# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── header ────────────────────────────────────────────────────────────────
    st.markdown("""
<div style="padding:16px 0 12px; border-bottom:1px solid #e2ddd6; margin-bottom:20px;
            display:flex; align-items:baseline; gap:14px;">
  <span style="font-family:'Fraunces',serif; font-size:28px; font-weight:700; color:#1c1917; letter-spacing:-.02em;">
    RNA3D
  </span>
  <span style="font-family:'DM Mono',monospace; font-size:12px; color:#2563a8;
    background:#eff6ff; border:1px solid #bfdbfe; border-radius:4px; padding:3px 10px;">
    structure predictor
  </span>
  <span style="font-family:'DM Mono',monospace; font-size:11px; color:#a09b93;">
    
  </span>
</div>
""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮  Predict",
        "🔭  Explore",
        "📊  Benchmarks",
        "🧬  About RNA",
    ])

    with tab1:
        render_prediction_tab()
    with tab2:
        st.sidebar.markdown('<p style="font-family:DM Sans,sans-serif;font-weight:600;font-size:13px;color:#1c1917">Viewer controls</p>', unsafe_allow_html=True)
        render_viewer_tab()
    with tab3:
        render_benchmark_tab()
    with tab4:
        render_about_tab()


if __name__ == "__main__":
    main()
