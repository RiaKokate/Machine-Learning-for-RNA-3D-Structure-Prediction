"""
streamlit run app.py
"""

from __future__ import annotations

import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import gdown
import os
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

PARQUET_PATH = DATA_DIR / "rna_atoms.parquet"
FILE_ID = "121vrLaUibiJXc3L2Xkrm72a3-6ssnRBl"

if not PARQUET_PATH.exists():
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, str(PARQUET_PATH), quiet=False)

# ── constants ──────────────────────────────────────────────────────────────────
REQUIRED_COLS = {"pdb_id", "chain_id", "residue_name", "residue_number",
                 "atom_name", "x", "y", "z"}

BASE_COLORS = {
    "A": "#e74c3c",
    "U": "#3498db",
    "G": "#2ecc71",
    "C": "#f1c40f",
}
FALLBACK_COLOR = "#bdc3c7"

# Backbone atoms — prime (') and star (*) variants
BACKBONE_ATOMS: frozenset[str] = frozenset({
    "P",
    "O5'", "O5*",
    "C5'", "C5*",
    "C4'", "C4*",
    "C3'", "C3*",
    "O3'", "O3*",
})

# Preferred centroid atoms for residue beads
BEAD_PREF_ATOMS: frozenset[str] = frozenset({"P", "C4'", "C4*", "N9", "N1"})

# ── page config (must be first st call) ────────────────────────────────────────
st.set_page_config(
    page_title="RNA 3D Structure",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── helpers ────────────────────────────────────────────────────────────────────

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Strip/uppercase residue_name; strip atom_name; cast ids to str."""
    df = df.copy()
    df["residue_name"] = df["residue_name"].astype(str).str.strip().str.upper()
    df["atom_name"]    = df["atom_name"].astype(str).str.strip()
    df["pdb_id"]       = df["pdb_id"].astype(str).str.strip()
    df["chain_id"]     = df["chain_id"].astype(str).str.strip()
    for col in ("x", "y", "z"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["residue_number"] = pd.to_numeric(df["residue_number"], errors="coerce")
    return df


def _validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        st.error(
            f"Dataset is missing required columns: **{sorted(missing)}**\n\n"
            "Please check your CSV/Parquet file."
        )
        st.stop()


def _has_pyarrow_dataset() -> bool:
    try:
        import pyarrow.dataset as _  # noqa: F401
        return True
    except ImportError:
        return False


# ── data loading ───────────────────────────────────────────────────────────────

def _dbg(msg: str, data: dict, hyp: str) -> None:
    import json, time
    entry = {"sessionId": "544d03", "hypothesisId": hyp, "location": "app.py",
             "message": msg, "data": data, "timestamp": int(time.time() * 1000)}
    try:
        with open("debug-544d03.log", "a", encoding="utf-8") as _f:
            _f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


@st.cache_data(show_spinner="Listing PDB IDs…", ttl=3600)
def list_pdb_ids_parquet(path: str) -> list[str]:
    try:
        try:
            import psutil
            import os
            mem = psutil.virtual_memory()
            _dbg("memory_before_read", {
                "available_mb": round(mem.available / 1e6, 1),
                "percent_used": mem.percent,
                "pid": os.getpid(),
            }, "H-B/H-E")
        except ImportError:
            pass

        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        meta = pf.metadata

        # region agent log H-A/H-C/H-D — parquet file structure
        rg0 = meta.row_group(0)
        col0 = rg0.column(0)
        _dbg("parquet_metadata", {
            "num_row_groups": meta.num_row_groups,
            "total_rows": meta.num_rows,
            "rows_in_rg0": rg0.num_rows,
            "pdb_id_encoding": str(col0.compression),
            "schema_names": [meta.row_group(0).column(i).path_in_schema
                             for i in range(rg0.num_columns)],
        }, "H-A/H-C/H-D")
        # endregion

        # region agent log H-C — check if pdb_id column is dictionary encoded
        rg0_table = pf.read_row_group(0, columns=["pdb_id"])
        col_type = str(rg0_table.schema.field("pdb_id").type)
        first_chunk_type = str(rg0_table.column("pdb_id").chunks[0].type) if rg0_table.column("pdb_id").chunks else "no_chunks"
        enc = rg0_table.column("pdb_id").combine_chunks().dictionary_encode()
        _dbg("pdb_id_column_type", {
            "arrow_type": col_type,
            "chunk0_type": first_chunk_type,
            "rg0_unique_count": len(enc.dictionary),
        }, "H-C")
        del rg0_table, enc
        # endregion

        # Now do the actual safe streaming read
        seen: set[str] = set()
        for rg_idx in range(meta.num_row_groups):
            tbl = pf.read_row_group(rg_idx, columns=["pdb_id"])
            col = tbl.column("pdb_id").combine_chunks().dictionary_encode()
            seen.update(str(v) for v in col.dictionary.to_pylist() if v is not None)
            del tbl, col

        result = sorted(seen)

        # region agent log — success path
        _dbg("list_pdb_ids_success", {"num_pdbs": len(result), "sample": result[:5]}, "H-A")
        # endregion

        return result

    except Exception as e:
        import traceback
        # region agent log — exception path
        _dbg("list_pdb_ids_exception", {
            "error_type": type(e).__name__,
            "error_msg": str(e),
            "traceback": traceback.format_exc()[-1000:],
        }, "H-B/H-D")
        # endregion
        st.error(f"Failed to read PDB IDs from Parquet: {e}")
        st.stop()


@st.cache_data(show_spinner="Loading PDB…", ttl=3600, max_entries=8)
def load_pdb_parquet(path: str, pdb_id: str,
                     chain_id: str | None = None) -> pd.DataFrame:
    """
    Load rows for a single pdb_id from Parquet.
    Uses pyarrow Dataset pushdown filtering when available.
    Falls back to pandas filter otherwise.
    """
    if _has_pyarrow_dataset():
        try:
            import pyarrow.dataset as ds
            import pyarrow.compute as pc
            dataset = ds.dataset(path, format="parquet")
            filt = pc.equal(ds.field("pdb_id"), pdb_id)
            if chain_id and chain_id != "ALL":
                filt = filt & pc.equal(ds.field("chain_id"), chain_id)
            table = dataset.to_table(filter=filt)
            df = table.to_pandas()
        except Exception as exc:
            st.warning(f"PyArrow pushdown failed ({exc}); falling back to full read.")
            df = pd.read_parquet(path)
            df = df[df["pdb_id"].astype(str) == pdb_id]
            if chain_id and chain_id != "ALL":
                df = df[df["chain_id"].astype(str) == chain_id]
            df = df.copy()
    else:
        st.warning("pyarrow.dataset not available; reading full Parquet (may be slow).")
        df = pd.read_parquet(path)
        df = df[df["pdb_id"].astype(str) == pdb_id].copy()
        if chain_id and chain_id != "ALL":
            df = df[df["chain_id"].astype(str) == chain_id].copy()

    return _normalize(df)


# ── analysis helpers ────────────────────────────────────────────────────────────

def residue_beads(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute one centroid row per (chain_id, residue_number, residue_name).
    Prefers BEAD_PREF_ATOMS when present, else uses all atoms in the residue.
    """
    if df.empty:
        return pd.DataFrame(columns=["chain_id", "residue_number",
                                     "residue_name", "x", "y", "z"])

    group_cols = ["chain_id", "residue_number", "residue_name"]
    all_mean = (df.groupby(group_cols, as_index=False)[["x", "y", "z"]]
                  .mean()
                  .rename(columns={"x": "xa", "y": "ya", "z": "za"}))

    pref = df[df["atom_name"].isin(BEAD_PREF_ATOMS)]
    if pref.empty:
        all_mean = all_mean.rename(columns={"xa": "x", "ya": "y", "za": "z"})
        return all_mean.sort_values(["chain_id", "residue_number"])

    pref_mean = (pref.groupby(group_cols, as_index=False)[["x", "y", "z"]]
                     .mean())

    merged = all_mean.merge(pref_mean, on=group_cols, how="left")
    merged["x"] = merged["x"].fillna(merged["xa"])
    merged["y"] = merged["y"].fillna(merged["ya"])
    merged["z"] = merged["z"].fillna(merged["za"])
    merged = merged[group_cols + ["x", "y", "z"]]
    return merged.sort_values(["chain_id", "residue_number"]).reset_index(drop=True)


def color_series(residue_names: pd.Series) -> list[str]:
    return [BASE_COLORS.get(r, FALLBACK_COLOR) for r in residue_names]


# ── figure builder ─────────────────────────────────────────────────────────────

def build_figure(
    atom_df: pd.DataFrame,
    bead_df: pd.DataFrame | None,
    pdb_id: str,
    atom_size: int,
    bead_size: int,
    show_bead_labels: bool,
) -> go.Figure:
    fig = go.Figure()

    # ── atom cloud ──
    if not atom_df.empty:
        bases  = atom_df["residue_name"].astype(str)
        colors = color_series(bases)
        hover  = (
            "pdb=" + pdb_id
            + "<br>chain=" + atom_df["chain_id"].astype(str)
            + "<br>res="   + bases + atom_df["residue_number"].astype(str)
            + "<br>atom="  + atom_df["atom_name"].astype(str)
        )
        fig.add_trace(go.Scatter3d(
            x=atom_df["x"], y=atom_df["y"], z=atom_df["z"],
            mode="markers",
            marker=dict(size=atom_size, color=colors, opacity=0.60,
                        line=dict(width=0)),
            hovertext=hover,
            hoverinfo="text",
            name="Atoms",
        ))

    # ── beads + backbone per chain ──
    if bead_df is not None and not bead_df.empty:
        for ch, grp in bead_df.groupby("chain_id", sort=False):
            grp = grp.sort_values("residue_number")
            xs = grp["x"].to_numpy()
            ys = grp["y"].to_numpy()
            zs = grp["z"].to_numpy()
            bases = grp["residue_name"].astype(str).tolist()
            bead_colors = color_series(pd.Series(bases))

            # backbone line
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(width=8, color="rgba(60,60,60,0.55)"),
                hoverinfo="skip",
                showlegend=False,
                name=f"Backbone {ch}",
            ))

            # bead markers
            text_arg = bases if show_bead_labels else None
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode=("markers+text" if show_bead_labels else "markers"),
                marker=dict(size=bead_size, color=bead_colors, opacity=0.95,
                            line=dict(color="white", width=0.5)),
                text=text_arg,
                textposition="top center",
                textfont=dict(size=9, color="white"),
                hovertext=[
                    f"chain={ch}<br>res={b}{n}"
                    for b, n in zip(bases, grp["residue_number"].tolist())
                ],
                hoverinfo="text",
                name=f"Beads {ch}",
            ))

    # ── legend patches for bases ──
    for base, color in BASE_COLORS.items():
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode="markers",
            marker=dict(size=8, color=color),
            name=base,
            showlegend=True,
        ))

    fig.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        scene=dict(
            xaxis=dict(title="x (Å)", backgroundcolor="#111316",
                       gridcolor="#2c2c3a", showbackground=True),
            yaxis=dict(title="y (Å)", backgroundcolor="#111316",
                       gridcolor="#2c2c3a", showbackground=True),
            zaxis=dict(title="z (Å)", backgroundcolor="#111316",
                       gridcolor="#2c2c3a", showbackground=True),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=36, b=0),
        legend=dict(
            itemsizing="constant",
            font=dict(color="white"),
            bgcolor="rgba(20,20,30,0.7)",
        ),
        title=dict(
            text=f"<b>{pdb_id}</b> — RNA 3D Structure",
            font=dict(color="white", size=16),
        ),
    )
    return fig


def build_2d_figure(
    atom_df: pd.DataFrame,
    bead_df: pd.DataFrame | None,
    pdb_id: str,
    atom_size: int,
    bead_size: int,
    show_bead_labels: bool,
) -> go.Figure:
    """
    2D projection of the structure (x vs y) using the same coloring
    and backbone/beads as the 3D view.
    """
    fig = go.Figure()

    # atoms (projected)
    if not atom_df.empty:
        bases = atom_df["residue_name"].astype(str)
        colors = color_series(bases)
        hover = (
            "pdb=" + pdb_id
            + "<br>chain=" + atom_df["chain_id"].astype(str)
            + "<br>res=" + bases + atom_df["residue_number"].astype(str)
            + "<br>atom=" + atom_df["atom_name"].astype(str)
        )
        fig.add_trace(go.Scattergl(
            x=atom_df["x"],
            y=atom_df["y"],
            mode="markers",
            marker=dict(size=atom_size, color=colors, opacity=0.60),
            hovertext=hover,
            hoverinfo="text",
            name="Atoms (2D)",
        ))

    # beads + backbone in 2D (x,y)
    if bead_df is not None and not bead_df.empty:
        for ch, grp in bead_df.groupby("chain_id", sort=False):
            grp = grp.sort_values("residue_number")
            xs = grp["x"].to_numpy()
            ys = grp["y"].to_numpy()
            bases = grp["residue_name"].astype(str).tolist()
            bead_colors = color_series(pd.Series(bases))

            # backbone line
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(width=4, color="rgba(60,60,60,0.6)"),
                hoverinfo="skip",
                showlegend=False,
                name=f"Backbone {ch} (2D)",
            ))

            # bead markers
            text_arg = bases if show_bead_labels else None
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode=("markers+text" if show_bead_labels else "markers"),
                marker=dict(size=bead_size, color=bead_colors, opacity=0.95,
                            line=dict(color="white", width=0.5)),
                text=text_arg,
                textposition="top center",
                textfont=dict(size=9, color="white"),
                hovertext=[
                    f"chain={ch}<br>res={b}{n}"
                    for b, n in zip(bases, grp["residue_number"].tolist())
                ],
                hoverinfo="text",
                name=f"Beads {ch} (2D)",
            ))

    # legend patches
    for base, color in BASE_COLORS.items():
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=8, color=color),
            name=base,
            showlegend=True,
        ))

    fig.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        xaxis=dict(
            title="x (Å)",
            gridcolor="#2c2c3a",
            showgrid=True,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            title="y (Å)",
            gridcolor="#2c2c3a",
            showgrid=True,
        ),
        margin=dict(l=0, r=0, t=36, b=0),
        legend=dict(
            itemsizing="constant",
            font=dict(color="white"),
            bgcolor="rgba(20,20,30,0.7)",
        ),
        title=dict(
            text=f"<b>{pdb_id}</b> — 2D projection (x–y)",
            font=dict(color="white", size=16),
        ),
    )
    return fig


# ── sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar(use_parquet: bool) -> dict:
    st.sidebar.title(" RNA 3D Structure")
    st.sidebar.caption(
        "Data: Parquet"
    )
    st.sidebar.divider()
    return {}   # controls filled below after PDB list is known


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.sidebar.title(" RNA 3D Structure")

    # ── ensure parquet exists (downloaded above with gdown if needed) ───────
    if not PARQUET_PATH.exists():
        st.error(
            "**Parquet file not found.**\n\n"
            f"Expected at `{PARQUET_PATH}`. "
            "On Streamlit Cloud the app should download it from Google Drive; "
            "locally, place `rna_atoms.parquet` in the `data` folder."
        )
        st.stop()

    # ── load PDB list ────────────────────────────────────────────────────────
    st.sidebar.divider()
    pdb_list = list_pdb_ids_parquet(str(PARQUET_PATH))

    if not pdb_list:
        st.error("No PDB IDs found in the dataset.")
        st.stop()

    # ── sidebar controls ─────────────────────────────────────────────────────
    pdb_id = st.sidebar.selectbox("PDB ID", pdb_list, key="pdb_id")
    st.sidebar.divider()

    # load full PDB first (for chain list)
    full_df = load_pdb_parquet(str(PARQUET_PATH), pdb_id)

    if full_df.empty:
        st.warning(f"No rows found for PDB `{pdb_id}`.")
        st.stop()

    _validate_columns(full_df)

    chains_available = sorted(full_df["chain_id"].dropna().unique().tolist())
    chain_opts = ["ALL"] + chains_available
    chain_id = st.sidebar.selectbox("Chain", chain_opts, key="chain_id")

    mode = st.sidebar.radio(
        "Atom set",
        ["All atoms", "Backbone only"],
        index=0,
        key="mode",
        help="Backbone: P, O5′, C5′, C4′, C3′, O3′ (and * variants)"
    )
    st.sidebar.divider()

    max_atoms = min(len(full_df), 500_000)
    sample_n = st.sidebar.slider(
        "Sample N atoms (0 = all)",
        min_value=0,
        max_value=max_atoms,
        value=min(30_000, max_atoms),
        step=5_000,
        key="sample_n",
    )
    st.sidebar.divider()

    show_beads      = st.sidebar.checkbox("Show residue beads + backbone line",
                                          value=True, key="show_beads")
    show_bead_labels = st.sidebar.checkbox("Show base letters on beads",
                                           value=True, key="show_labels",
                                           disabled=not show_beads)
    show_2d = st.sidebar.checkbox("Show 2D projection view",
                                  value=True, key="show_2d")
    st.sidebar.divider()

    atom_size  = st.sidebar.slider("Atom marker size", 1, 8, 2, key="atom_size")
    bead_size  = st.sidebar.slider("Bead size",        3, 16, 7, key="bead_size")

    # ── apply chain filter ───────────────────────────────────────────────────
    if chain_id == "ALL":
        view_df = full_df
    else:
        view_df = full_df[full_df["chain_id"] == chain_id]

    # ── apply backbone filter ────────────────────────────────────────────────
    if mode == "Backbone only":
        view_df = view_df[view_df["atom_name"].isin(BACKBONE_ATOMS)]

    # ── apply sampling ───────────────────────────────────────────────────────
    if sample_n > 0 and len(view_df) > sample_n:
        view_df = view_df.sample(sample_n, random_state=42)

    # ── summary stats ────────────────────────────────────────────────────────
    n_chains   = full_df["chain_id"].nunique()
    n_residues = full_df[["chain_id", "residue_number"]].drop_duplicates().shape[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (PDB total)",   f"{len(full_df):,}")
    c2.metric("Rows (current view)", f"{len(view_df):,}")
    c3.metric("Chains",             f"{n_chains:,}")
    c4.metric("Residues",           f"{n_residues:,}")

    # ── beads ────────────────────────────────────────────────────────────────
    bead_src = (full_df if chain_id == "ALL"
                else full_df[full_df["chain_id"] == chain_id])
    bead_df = residue_beads(bead_src) if show_beads else None

    # ── figure ───────────────────────────────────────────────────────────────
    if view_df.empty:
        st.warning("No atoms to display with the current filters.")
    else:
        with st.spinner("Rendering 3D / 2D plots…"):
            fig3d = build_figure(
                atom_df=view_df.reset_index(drop=True),
                bead_df=bead_df,
                pdb_id=pdb_id,
                atom_size=atom_size,
                bead_size=bead_size,
                show_bead_labels=(show_beads and show_bead_labels),
            )
            fig2d = build_2d_figure(
                atom_df=view_df.reset_index(drop=True),
                bead_df=bead_df,
                pdb_id=pdb_id,
                atom_size=atom_size,
                bead_size=bead_size,
                show_bead_labels=(show_beads and show_bead_labels),
            ) if show_2d else None

        if show_2d and fig2d is not None:
            col3d, col2d = st.columns(2)
            with col3d:
                st.subheader("3D view")
                st.plotly_chart(fig3d, width="stretch")
            with col2d:
                st.subheader("2D projection")
                st.plotly_chart(fig2d, width="stretch")
        else:
            st.plotly_chart(fig3d, width="stretch")

    # ── table preview ─────────────────────────────────────────────────────────
    with st.expander("Preview first 200 rows of current view"):
        st.dataframe(
            view_df.head(200),
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# requirements.txt (copy to requirements.txt in your project root)
# ─────────────────────────────────────────────────────────────────────────────
REQUIREMENTS_TXT = """
streamlit>=1.35
pandas>=2.1
numpy>=1.26
plotly>=5.20
pyarrow>=15.0
"""