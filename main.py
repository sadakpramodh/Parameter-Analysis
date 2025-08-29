import streamlit as st
import pandas as pd
import numpy as np
import io
from typing import Tuple, List, Optional

st.set_page_config(page_title="CCM Quality Bands", layout="wide")
st.title("CCM – Data‑Driven Green/Red Bins for Process Parameters")

# -----------------------------
# Tunables (aligns with your notebook logic)
# -----------------------------
MIN_SAMPLES_CLASS = 30           # below this, use DN/global fallback
TARGET_COVERAGE = 0.30           # coverage for fallback lowest‑rejection band
MAX_BINS = 50
SMALL_SAMPLE_BAND = 0.50         # central 50% band for small n
EPS = 1e-9
# Only show these defects in the Visual Defect filter
DEFECTS_WHITELIST = ['Cutmark', 'Thin Socket', 'Extractor Crack', 'Core Burst']

# -----------------------------
# Utility functions
# -----------------------------
def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce')


def tiered_subset(df: pd.DataFrame, dn, pipe_class) -> Tuple[pd.DataFrame, str]:
    """(DN+Class) → (DN only) → (Global) fallback with sample check."""
    s1 = df[(df['DN'] == dn) & (df['Pipe_Class'] == pipe_class)].copy()
    if len(s1) >= MIN_SAMPLES_CLASS:
        return s1, 'Class'
    s2 = df[(df['DN'] == dn)].copy()
    if len(s2) >= MIN_SAMPLES_CLASS:
        return s2, 'DN'
    return df.copy(), 'Global'


def iqr_band(series: pd.Series, frac: float = 0.50) -> Tuple[float, float]:
    lo = (1 - frac) / 2
    hi = 1 - lo
    ql, qh = series.quantile(lo), series.quantile(hi)
    return float(ql), float(qh)


def build_bins(filtered: pd.DataFrame, param: str, max_bins: int = MAX_BINS) -> Tuple[pd.Series, str]:
    n = len(filtered)
    bins_try = max(5, min(max_bins, int(np.sqrt(max(n, 1)))))
    # try qcut first
    try:
        b = pd.qcut(filtered[param], q=bins_try, duplicates='drop')
        if not b.isna().all():
            return b, 'qcut'
    except Exception:
        pass
    # fallback to cut (equal width)
    unique_vals = filtered[param].nunique(dropna=True)
    bins_try2 = max(3, min(max_bins, unique_vals if unique_vals >= 3 else 3))
    try:
        b = pd.cut(filtered[param], bins=bins_try2, include_lowest=True, duplicates='drop')
        if not b.isna().all():
            return b, 'cut'
    except Exception:
        pass
    return pd.Series([pd.NA]*n, index=filtered.index), 'none'


def contiguous_regions_by_prob(triples: List[Tuple[float, float, float]], threshold: float) -> List[Tuple[float, float]]:
    regions, cur = [], []
    for left, right, prob in triples:
        if prob <= threshold + EPS:
            if not cur:
                cur = [left, right]
            else:
                cur[1] = right
        else:
            if cur:
                regions.append(tuple(cur))
                cur = []
    if cur:
        regions.append(tuple(cur))
    return regions


def lowest_rejection_cover_band(gb_idx: pd.DataFrame, filtered: pd.DataFrame, param: str, min_coverage: float = TARGET_COVERAGE):
    bins = list(gb_idx.index.categories)
    if not bins:
        return None, None, 0.0
    totals = np.array([gb_idx.loc[b, 'total'] if b in gb_idx.index else 0 for b in bins], dtype=float)
    rejects = np.array([gb_idx.loc[b, 'rejected'] if b in gb_idx.index else 0 for b in bins], dtype=float)
    N = len(bins)
    totalN = totals.sum()
    best = (None, None, 1.0)
    for i in range(N):
        for j in range(i, N):
            cov = totals[i:(j+1)].sum() / max(totalN, 1.0)
            if cov >= min_coverage:
                mprob = (rejects[i:(j+1)].sum() / np.clip(totals[i:(j+1)].sum(), 1, None))
                if (best[0] is None) or (mprob < best[2]):
                    best = (i, j, mprob)
    if best[0] is None:
        return None, None, 0.0
    LCL = float(bins[best[0]].left)
    UCL = float(bins[best[1]].right)
    coverage = totals[best[0]:(best[1]+1)].sum() / max(totalN, 1.0)
    return LCL, UCL, float(coverage)


def compute_band_and_table(df: pd.DataFrame, param: str, dn, pipe_class, class_threshold: float):
    """
    Returns dict(summary) and justification dataframe with per-bin stats
    summary keys: LCL, UCL, has_limits, method, data_tier, n_filtered, coverage_pct, avg_rej_inside
    """
    summary = {"LCL": None, "UCL": None, "has_limits": False, "method": "", "data_tier": "",
               "n_filtered": 0, "coverage_pct": 0.0, "avg_rej_inside": None, "notes": []}

    subset, data_tier = tiered_subset(df, dn, pipe_class)
    summary["data_tier"] = data_tier

    if param not in subset.columns:
        summary["notes"].append(f"Parameter '{param}' not present in data")
        return summary, pd.DataFrame()

    subset = subset.copy()
    subset[param] = to_numeric(subset[param])
    s_nonan = subset[param].dropna()
    if s_nonan.empty:
        summary["notes"].append("All-NaN/non-numeric; no recommendation")
        return summary, pd.DataFrame()

    # IQR filter
    Q1, Q3 = s_nonan.quantile(0.25), s_nonan.quantile(0.75)
    IQR = Q3 - Q1
    filtered = subset[(subset[param] >= Q1 - 1.5*IQR) & (subset[param] <= Q3 + 1.5*IQR)].copy()
    if filtered.empty:
        filtered = subset.dropna(subset=[param]).copy()
        summary["notes"].append("IQR removed all; used unfiltered values (caution)")

    n = len(filtered)
    summary["n_filtered"] = int(n)

    if n < MIN_SAMPLES_CLASS:
        lo, hi = iqr_band(filtered[param], frac=SMALL_SAMPLE_BAND)
        summary.update({"LCL": lo, "UCL": hi, "has_limits": True, "method": "small-sample-IQR", "coverage_pct": 50.0})
        seg = filtered[(filtered[param] >= lo) & (filtered[param] <= hi)]
        if not seg.empty:
            summary["avg_rej_inside"] = float(seg['Rejected_Flag'].mean() * 100)
        return summary, pd.DataFrame()

    # Build bins
    binned, method_used = build_bins(filtered, param)
    if method_used == 'none' or binned.isna().all():
        lo, hi = iqr_band(filtered[param], frac=0.50)
        summary.update({"LCL": lo, "UCL": hi, "has_limits": True, "method": "IQR-fallback", "coverage_pct": 50.0})
        seg = filtered[(filtered[param] >= lo) & (filtered[param] <= hi)]
        if not seg.empty:
            summary["avg_rej_inside"] = float(seg['Rejected_Flag'].mean() * 100)
        return summary, pd.DataFrame()

    filtered['bin'] = binned
    gb = (filtered
          .groupby('bin', observed=False)
          .agg(total=('Rejected_Flag', 'count'), rejected=('Rejected_Flag', 'sum'))
          .reset_index())
    gb['rejection_prob'] = gb['rejected'] / gb['total']
    gb_idx = gb.set_index('bin')

    edges = gb_idx.index.categories
    triples = [(b.left, b.right, float(gb_idx.loc[b, 'rejection_prob'])) for b in edges if b in gb_idx.index]

    # Primary: widest contiguous region with prob <= class_threshold
    regions = contiguous_regions_by_prob(triples, class_threshold)
    if regions:
        LCL, UCL = max(regions, key=lambda r: r[1] - r[0])
        summary.update({"LCL": LCL, "UCL": UCL, "has_limits": True, "method": method_used})
        tot = gb['total'].sum()
        cov = gb[(gb['bin'].apply(lambda x: x.left >= LCL and x.right <= UCL))]['total'].sum() / max(tot, 1.0)
        summary["coverage_pct"] = round(float(cov) * 100, 2)
    else:
        # Secondary: lowest-rejection band covering ≥ TARGET_COVERAGE
        L2, U2, cov2 = lowest_rejection_cover_band(gb_idx, filtered, param, min_coverage=TARGET_COVERAGE)
        if L2 is not None:
            summary.update({"LCL": L2, "UCL": U2, "has_limits": True, "method": f"{method_used}+low-reject-cover", "coverage_pct": round(float(cov2) * 100, 2)})
            summary["notes"].append(f"No bin ≤ threshold; using lowest-rejection band covering ≥ {int(TARGET_COVERAGE*100)}% data")
        else:
            lo, hi = iqr_band(filtered[param], frac=0.50)
            summary.update({"LCL": lo, "UCL": hi, "has_limits": True, "method": f"{method_used}+IQR-tertiary", "coverage_pct": 50.0})
            summary["notes"].append("No acceptable band found; using IQR band")

    # Build justification table with per-bin color label
    rows = []
    for b in edges:
        if b not in gb_idx.index:
            continue
        row = gb_idx.loc[b]
        prob = float(row['rejection_prob'])
        label = 'Green' if (summary["has_limits"] and (b.left >= summary["LCL"] and b.right <= summary["UCL"])) else ('Red' if prob > class_threshold + EPS else 'Amber')
        rows.append({
            'Parameter Range': f"{round(b.left,4)} – {round(b.right,4)}",
            'Bin Count': int(row['total']),
            'Rejections': int(row['rejected']),
            'Rejection Probability (%)': round(prob*100, 2),
            'Color': label,
            'bin_left': float(b.left),
            'bin_right': float(b.right)
        })

    just_df = pd.DataFrame(rows)

    # avg rejection inside band
    if summary["has_limits"] and summary["LCL"] is not None and summary["UCL"] is not None:
        seg = filtered[(filtered[param] >= summary["LCL"]) & (filtered[param] <= summary["UCL"])]
        if not seg.empty:
            summary["avg_rej_inside"] = float(seg['Rejected_Flag'].mean() * 100)

    return summary, just_df


def make_histogram_bins(just_df: pd.DataFrame, LCL: Optional[float], UCL: Optional[float]):
    """Creates a list of dicts for plotting rectangular bins with color labels."""
    bins = []
    for _, r in just_df.iterrows():
        color = '#34a853' if r['Color'] == 'Green' else ('#ea4335' if r['Color'] == 'Red' else '#fbbc05')
        bins.append({
            'x0': r['bin_left'],
            'x1': r['bin_right'],
            'count': r['Bin Count'],
            'color': color
        })
    return bins

# -----------------------------
# Sidebar – CSV upload and filters
# -----------------------------
with st.sidebar:
    st.header("1) Upload CSV")
    up = st.file_uploader("Choose a CSV exported from CCM (must include DN, Pipe_Class, Rejected_Flag)", type=["csv"]) 
    st.markdown("""
    **Required columns**: `DN`, `Pipe_Class`, `Rejected_Flag`  
    Other columns will be used as numeric parameters to visualize.
    """)

if not up:
    st.info("Upload a CSV to get started.")
    st.stop()

# Load
df = pd.read_csv(up, low_memory=False)
df.columns = df.columns.str.strip()

missing = [c for c in ['DN', 'Pipe_Class', 'Rejected_Flag'] if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Normalize rejected flag to 0/1
df['Rejected_Flag'] = pd.to_numeric(df['Rejected_Flag'], errors='coerce').fillna(0).clip(0, 1)

# Optional visual defect column (used only for filtering display)
has_defect = 'VISUAL DEFECT' in df.columns

# Build filter choices
all_dns = sorted([x for x in df['DN'].dropna().unique().tolist()])
all_classes = sorted([x for x in df['Pipe_Class'].dropna().unique().tolist()])

col_f1, col_f2, col_f3 = st.columns([1,1,1])
with col_f1:
    dn_sel = st.selectbox("Filter: DN (Diameter)", options=all_dns, index=0)
with col_f2:
    pc_sel = st.selectbox("Filter: Pipe Class", options=all_classes, index=0)
with col_f3:
    if has_defect:
        present = df['VISUAL DEFECT'].astype(str)
        present_allowed = [d for d in DEFECTS_WHITELIST if d in set(present.unique())]
        vd_values = ['(All)'] + present_allowed if present_allowed else ['(All)']
        vd_sel = st.selectbox("Filter: Visual Defect (optional)", options=vd_values, index=0)
    else:
        vd_sel = '(All)'
        st.caption("Note: 'VISUAL DEFECT' column not found – skipping defect filter")

# Parameter selection (any numeric column besides required)
non_candidate = {'DN','Pipe_Class','Rejected_Flag'}
candidates = []
for c in df.columns:
    if c in non_candidate:
        continue
    try:
        # consider as numeric if >50% convertible
        s = pd.to_numeric(df[c], errors='coerce')
        if s.notna().mean() > 0.5:
            candidates.append(c)
    except Exception:
        pass

if not candidates:
    st.error("No numeric parameters found to visualize. Please upload a file with numeric process parameters.")
    st.stop()

param = st.selectbox("Parameter to analyze", options=sorted(candidates))

# Compute class threshold from the (DN, Pipe_Class) group on full dataset (not after IQR)
class_mask = (df['DN'] == dn_sel) & (df['Pipe_Class'] == pc_sel)
if class_mask.sum() == 0:
    st.warning("No rows for the selected DN & Pipe Class. Falling back to whole dataset threshold.")
    class_threshold = df['Rejected_Flag'].mean()
else:
    class_threshold = df.loc[class_mask, 'Rejected_Flag'].mean()

# Apply optional Visual Defect filter for display (does not affect threshold used)
show_df = df.copy()
if has_defect and vd_sel != '(All)':
    show_df = show_df[show_df['VISUAL DEFECT'].astype(str) == vd_sel]

# Compute band and bin table
summary, just_df = compute_band_and_table(df, param, dn_sel, pc_sel, class_threshold)

# -----------------------------
# Top metrics
# -----------------------------
met1, met2, met3, met4, met5 = st.columns(5)
met1.metric("Data Tier", summary['data_tier'])
met2.metric("Method", summary['method'] if summary['method'] else '—')
met3.metric("Filtered N", summary['n_filtered'])
met4.metric("Coverage %", f"{summary['coverage_pct']:.2f}")
met5.metric("Avg Rej% in Band", f"{summary['avg_rej_inside']:.2f}" if summary['avg_rej_inside'] is not None else '—')

if summary['notes']:
    st.info("; ".join(summary['notes']))

# -----------------------------
# Histogram with colored bins + bell curves (Gaussian & KDE per group)
# -----------------------------
import plotly.graph_objects as go

st.subheader("Histogram Bars + Bell Curves – Green vs Orange (Gaussian & KDE)")

# Curve visibility controls
col_g, col_r = st.columns(2)
with col_g:
    green_modes = st.multiselect("Green curves", ["Gaussian", "KDE"], default=["Gaussian", "KDE"])
with col_r:
    reject_modes = st.multiselect("Rejected curves", ["Gaussian", "KDE"], default=["Gaussian", "KDE"])

# Use selected DN+Class (and optional defect) rows for curves
plot_df = show_df[(show_df['DN'] == dn_sel) & (show_df['Pipe_Class'] == pc_sel)].copy()
vals_all = pd.to_numeric(plot_df[param], errors='coerce')
vals_green = vals_all[plot_df['Rejected_Flag'] == 0].dropna()
vals_reject = vals_all[plot_df['Rejected_Flag'] == 1].dropna()

fig = go.Figure()

# If we have bin analysis, render bar histograms aligned to those bins
if not just_df.empty:
    centers = (just_df['bin_left'] + just_df['bin_right']) / 2.0
    widths = (just_df['bin_right'] - just_df['bin_left']).abs()

    counts_reject = just_df['Rejections']
    counts_green = (just_df['Bin Count'] - just_df['Rejections']).clip(lower=0)

    fig.add_bar(x=centers, y=counts_green, width=widths, name='Green (Accepted)',
                marker_color='green', opacity=0.5)
    fig.add_bar(x=centers, y=counts_reject, width=widths, name='Rejected (Count)',
                marker_color='orange', opacity=0.5)

    # Band shading (green/red/amber) behind bars
    bins_for_plot = make_histogram_bins(just_df, summary['LCL'], summary['UCL'])
    for b in bins_for_plot:
        fig.add_vrect(x0=b['x0'], x1=b['x1'], fillcolor=b['color'], opacity=0.20, line_width=0, layer='below')

    avg_bin_w = float(widths.mean()) if len(widths) else None
else:
    # Fallback: auto-binned histograms (overlay)
    if len(vals_green) > 0:
        fig.add_histogram(x=vals_green, nbinsx=60, name='Green (Accepted)', opacity=0.45, marker_color='green')
    if len(vals_reject) > 0:
        fig.add_histogram(x=vals_reject, nbinsx=60, name='Rejected (Count)', opacity=0.45, marker_color='orange')
    # Estimate average bin width from range/nbins for curve scaling
    vmin = np.nanmin([vals_green.min() if len(vals_green)>0 else np.nan,
                      vals_reject.min() if len(vals_reject)>0 else np.nan])
    vmax = np.nanmax([vals_green.max() if len(vals_green)>0 else np.nan,
                      vals_reject.max() if len(vals_reject)>0 else np.nan])
    if np.isfinite(vmin) and np.isfinite(vmax):
        avg_bin_w = (float(vmax) - float(vmin)) / 60.0 if float(vmax) > float(vmin) else None
    else:
        avg_bin_w = None

# Draw LCL/UCL lines + annotations if present
if summary['LCL'] is not None:
    fig.add_vline(x=summary['LCL'], line_width=2, line_dash='dash', line_color='green')
    try:
        lcl_txt = f"LCL: {float(summary['LCL']):.2f}"
    except Exception:
        lcl_txt = "LCL"
    fig.add_annotation(x=summary['LCL'], y=1.02, yref='paper', text=lcl_txt, showarrow=False,
                       font=dict(color='green', size=12), bgcolor='rgba(0,128,0,0.10)',
                       xanchor='left')
if summary['UCL'] is not None:
    fig.add_vline(x=summary['UCL'], line_width=2, line_dash='dash', line_color='green')
    try:
        ucl_txt = f"UCL: {float(summary['UCL']):.2f}"
    except Exception:
        ucl_txt = "UCL"
    fig.add_annotation(x=summary['UCL'], y=1.02, yref='paper', text=ucl_txt, showarrow=False,
                       font=dict(color='green', size=12), bgcolor='rgba(0,128,0,0.10)',
                       xanchor='right')

# -----------------------------
# Bell curves – compute both Gaussian fit and KDE (scaled to counts)
# -----------------------------
# Build x-grid from available values
x_vals = []
if len(vals_green) > 0:
    x_vals += [float(vals_green.min()), float(vals_green.max())]
if len(vals_reject) > 0:
    x_vals += [float(vals_reject.min()), float(vals_reject.max())]
if not x_vals:
    x_vals = [0.0, 1.0]

x_min, x_max = min(x_vals), max(x_vals)
if x_max == x_min:
    x_max = x_min + 1.0
x_grid = np.linspace(x_min, x_max, 300)

# Helpers

def scaled_normal_curve(values: pd.Series, x: np.ndarray, avg_w):
    if len(values) < 2:
        return None
    mu = float(values.mean())
    sigma = float(values.std(ddof=0))
    if sigma <= 0 or not np.isfinite(sigma):
        return None
    N = len(values)
    bw = avg_w if (avg_w is not None and np.isfinite(avg_w) and avg_w > 0) else (x_max - x_min)/60.0
    pdf = (1.0/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x - mu)/sigma)**2)
    return N * bw * pdf


def scaled_kde_curve(values: pd.Series, x: np.ndarray, avg_w):
    n = len(values)
    if n < 2:
        return None
    vals = values.to_numpy(dtype=float)
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std <= 0:
        return None
    # Silverman's rule
    h = 1.06 * std * (n ** (-1/5))
    if not np.isfinite(h) or h <= 0:
        return None
    u = (x[:, None] - vals[None, :]) / h  # (m, n)
    phi = np.exp(-0.5 * u**2) / np.sqrt(2*np.pi)
    f = phi.mean(axis=1) / h
    bw = avg_w if (avg_w is not None and np.isfinite(avg_w) and avg_w > 0) else (x_max - x_min)/60.0
    return n * bw * f

# Compute all curves
yg_norm = scaled_normal_curve(vals_green, x_grid, avg_bin_w)
yg_kde  = scaled_kde_curve(vals_green, x_grid, avg_bin_w)
yr_norm = scaled_normal_curve(vals_reject, x_grid, avg_bin_w)
yr_kde  = scaled_kde_curve(vals_reject, x_grid, avg_bin_w)

# Plot based on selections
if (yg_norm is not None) and ("Gaussian" in green_modes):
    fig.add_scatter(x=x_grid, y=yg_norm, mode='lines', name='Green – Gaussian',
                    line=dict(color='green', width=2, dash='solid'))
if (yg_kde is not None) and ("KDE" in green_modes):
    fig.add_scatter(x=x_grid, y=yg_kde, mode='lines', name='Green – KDE',
                    line=dict(color='green', width=2, dash='dot'))
if (yr_norm is not None) and ("Gaussian" in reject_modes):
    fig.add_scatter(x=x_grid, y=yr_norm, mode='lines', name='Rejected – Gaussian',
                    line=dict(color='orange', width=2, dash='solid'))
if (yr_kde is not None) and ("KDE" in reject_modes):
    fig.add_scatter(x=x_grid, y=yr_kde, mode='lines', name='Rejected – KDE',
                    line=dict(color='orange', width=2, dash='dot'))

fig.update_layout(
    xaxis_title=param,
    yaxis_title='Count',
    legend=dict(orientation='h'),
    barmode='overlay',
    margin=dict(l=10, r=10, t=10, b=10)
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Bin table (with color label)
# -----------------------------
if not just_df.empty:
    st.subheader("Per‑bin statistics")
    show = just_df.drop(columns=['bin_left','bin_right']).copy()
    st.dataframe(show, use_container_width=True)

# -----------------------------
# Download artifacts
# -----------------------------
with st.expander("Download analysis as CSV"):
    if not just_df.empty:
        buf = io.StringIO()
        just_df.to_csv(buf, index=False)
        st.download_button("Download bin table (CSV)", data=buf.getvalue(), file_name=f"bins_{dn_sel}_{pc_sel}_{param}.csv", mime="text/csv")

# -----------------------------
# Helper notes
# -----------------------------
st.caption("Bars: Orange = rejected count, Green = accepted (green) pipes. Lines: bell curves (Gaussian fit or KDE) scaled to counts. Band shading: Green = bins inside recommended band; Red = bins with rejection probability above class threshold; Amber = below threshold but outside the chosen contiguous band.")
