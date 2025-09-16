# app.py
# Streamlit Ingestion Demo (sterilized)
# - Generates fake tracker rows
# - Filters by Period & Start Week
# - Splits by EventAttribute (All / Local / Scale / Batch)
# - Lets you download CSV/XLSX with session-scoped versioning (_v1, _v2, ...)
#
# Runs on Streamlit Cloud. No Excel/macros, no local paths, no proprietary names.

from __future__ import annotations
import io
from datetime import date, timedelta
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

# ------------- Session state init -------------
if "version_map" not in st.session_state:
    # key: (period:int, week:int, suffix:str) -> current max version:int
    st.session_state.version_map: Dict[Tuple[int, int, str], int] = {}

if "dataset_cache" not in st.session_state:
    st.session_state.dataset_cache = None  # (seed, rows, year) -> DataFrame


# ------------- Helpers -------------

def first_monday_of_year(y: int) -> date:
    """Return the first Monday of the given year."""
    d = date(y, 1, 1)
    # Monday is 0
    return d + timedelta(days=(7 - d.weekday()) % 7)

def week_bounds(y: int, week: int) -> Tuple[date, date]:
    """
    Return (start_date, end_date) for a simple 'retail-style' week:
    Monday of week N through Sunday of week N.
    """
    start0 = first_monday_of_year(y)
    start = start0 + timedelta(weeks=max(week - 1, 0))
    end = start + timedelta(days=6)
    return start, end

SUPPLIERS = [
    "Northstar Foods", "Evergreen Co.", "Blue Finch Distributors",
    "Rivermark Partners", "Open Prairie Group", "Sun & Salt LLC",
]
BRANDS = [
    "Prairie Peak", "LunaLoaf", "QuikCart", "Red Lantern",
    "BrightBites", "Green Mesa", "Timber & Thyme",
]
EVENT_TYPES = ["Display", "TPR", "Digital", "Coupon"]
SUBTYPES = ["BOGO", "Price Drop", "Bundle", "Flash", "Seasonal"]
PROMO_SNIPPETS = [
    "Buy more, save more", "Member bonus", "Weekend special",
    "Seasonal savings", "Cart boost", "Extra rewards",
]

def seeded_rng(seed: int):
    return np.random.default_rng(seed)

def build_fake_dataset(seed: int, rows: int, year: int) -> pd.DataFrame:
    """
    Deterministically generate 'rows' of fake tracker data for the given year.
    """
    key = (seed, rows, year)
    if st.session_state.dataset_cache and st.session_state.dataset_cache[0] == key:
        return st.session_state.dataset_cache[1].copy()

    rng = seeded_rng(seed)
    periods = rng.integers(1, 14, size=rows, endpoint=False) + 0  # 1..13
    start_weeks = rng.integers(1, 53, size=rows, endpoint=False) + 0  # 1..52
    # End week is start + 0..2 (clamped to 52)
    end_weeks = np.minimum(start_weeks + rng.integers(0, 3, size=rows), 52)

    # EventAttribute skew: 65% Local, 35% Scale
    attrs = np.where(rng.random(rows) < 0.65, "Local", "Scale")

    # Choose categorical fields
    event_types = rng.choice(EVENT_TYPES, size=rows)
    subtypes = rng.choice(SUBTYPES, size=rows)
    suppliers = rng.choice(SUPPLIERS, size=rows)
    brands = rng.choice(BRANDS, size=rows)
    promos = rng.choice(PROMO_SNIPPETS, size=rows)

    # Costs (0–500 with 2 decimals)
    costs = rng.uniform(0, 500, size=rows).round(2)

    # Build dates & ids
    start_dates = []
    end_dates = []
    event_ids = []
    event_names = []
    for i in range(rows):
        sw = int(start_weeks[i])
        ew = int(end_weeks[i])
        p = int(periods[i])
        st_d, en_d = week_bounds(year, sw)
        start_dates.append(st_d.strftime("%m/%d/%y"))
        end_dates.append(en_d.strftime("%m/%d/%y"))
        event_ids.append(f"E-{year}-P{p:02d}-W{sw:02d}-{i+1:03d}")
        event_names.append(f"{brands[i]} {event_types[i]} – {attrs[i]}")

    df = pd.DataFrame({
        "EventID": event_ids,
        "Year": year,
        "Period": periods,
        "Start Week": start_weeks,
        "End Week": end_weeks,
        "StartDate": start_dates,
        "EndDate": end_dates,
        "EventType": event_types,
        "SubEventType": subtypes,
        "Supplier": suppliers,
        "Brands/Program": brands,
        "Offer/Promotion": promos,
        "EventAttribute": attrs,
        "EventName": event_names,
        "Package Cost": costs,
        "Comments": ["" for _ in range(rows)],
        "Links": ["" for _ in range(rows)],
    })

    # Stable order
    order = [
        "EventID", "Year", "Period", "Start Week", "End Week",
        "StartDate", "EndDate", "EventType", "SubEventType",
        "Supplier", "Brands/Program", "Offer/Promotion",
        "EventAttribute", "EventName", "Package Cost",
        "Comments", "Links",
    ]
    df = df[order]

    # cache
    st.session_state.dataset_cache = (key, df.copy())
    return df

def split_by_mode(df: pd.DataFrame, mode: str) -> List[Tuple[str, pd.DataFrame]]:
    """
    Return list of (label, subset_df) based on mode: all, local, scale, batch.
    """
    attr = df["EventAttribute"].astype(str).str.strip().str.lower()
    is_local = attr.str.startswith("local")
    is_scale = attr.str.startswith("scale")

    if mode == "all":
        return [("All Events", df.reset_index(drop=True))]
    if mode == "local":
        return [("Local Events", df[is_local].reset_index(drop=True))]
    if mode == "scale":
        return [("Scale Events", df[is_scale].reset_index(drop=True))]
    if mode == "batch":
        return [
            ("Local Events", df[is_local].reset_index(drop=True)),
            ("Scale Events", df[is_scale].reset_index(drop=True)),
        ]
    raise ValueError("mode must be one of: all, local, scale, batch")

def next_version(period: int, week: int, suffix: str) -> int:
    key = (int(period), int(week), suffix)
    current = st.session_state.version_map.get(key, 0)
    return current + 1

def bump_version(period: int, week: int, suffix: str):
    key = (int(period), int(week), suffix)
    st.session_state.version_map[key] = st.session_state.version_map.get(key, 0) + 1

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Ingestion")
        # Optional: auto-width (rough)
        workbook = writer.book
        worksheet = writer.sheets["Ingestion"]
        for i, col in enumerate(df.columns):
            width = min(max(12, int(df[col].astype(str).str.len().mean()) + 4), 40)
            worksheet.set_column(i, i, width)
    return output.getvalue()

def filename(period: int, week: int, suffix: str, version: int, ext: str) -> str:
    return f"P{int(period)}_W{int(week)}_Ingestion_{suffix}_v{int(version)}.{ext}"


# ------------- UI -------------

st.set_page_config(page_title="Ingestion Demo", layout="wide")

st.title("Ingestion Sheet Demo (Portfolio-Safe)")
st.caption("Generates fake tracker rows → filter → split → download CSV/XLSX with versioned filenames.")

with st.sidebar:
    st.subheader("Generator")
    seed = st.number_input("Random Seed", min_value=0, value=42, step=1, help="Same seed → same dataset.")
    rows = st.slider("Rows to generate", 50, 5000, 500, 50)
    year = st.number_input("Year", min_value=2000, max_value=2100, value=date.today().year, step=1)
    st.divider()
    period = st.number_input("Filter: Period", min_value=1, max_value=13, value=3, step=1)
    week = st.number_input("Filter: Start Week", min_value=1, max_value=52, value=12, step=1)
    mode = st.radio(
        "Event Mode",
        options=["all", "local", "scale", "batch"],
        format_func=lambda s: {"all": "All Events", "local": "Local Only", "scale": "Scale Only", "batch": "Batch (Local & Scale)"}[s],
        horizontal=False,
    )
    st.divider()
    build = st.button("Build Dataset / Apply Filters", use_container_width=True)

# Build or reuse dataset
if build or st.session_state.dataset_cache is None:
    base_df = build_fake_dataset(seed, rows, year)
else:
    # Use cached params; but filters are always applied live
    base_df = st.session_state.dataset_cache[1].copy()

# Filter by Period & Start Week
filtered = base_df[
    (pd.to_numeric(base_df["Period"]) == int(period)) &
    (pd.to_numeric(base_df["Start Week"]) == int(week))
].reset_index(drop=True)

# Header pills
total_rows = len(base_df)
match_rows = len(filtered)
col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 1])
col_a.metric("📦 Rows generated", f"{total_rows}")
col_b.metric("🗓️ Period / Week", f"P{int(period)} / W{int(week)}")
col_c.metric("🧭 Mode", {"all":"All","local":"Local","scale":"Scale","batch":"Batch"}[mode])
col_d.metric("✅ Matches", f"{match_rows}")

with st.expander("See a sample of the full generated dataset (first 50 rows)"):
    st.dataframe(base_df.head(50), use_container_width=True)

st.subheader("Filtered View")
if filtered.empty:
    st.info(
        "No rows matched these filters. Try a different Period/Week or increase the number of generated rows.",
        icon="ℹ️",
    )
else:
    st.dataframe(filtered.head(100), use_container_width=True)

    # Split based on mode
    bundles = split_by_mode(filtered, mode)

    st.markdown("### Downloads")
    for label, df_part in bundles:
        st.markdown(f"**{label}**")
        if df_part.empty:
            st.caption("Nothing to download for this subset.")
            st.divider()
            continue

        # Determine version (show next), and wire on_click to bump when downloaded
        suffix_map = {"All Events": "All Events", "Local Events": "Local Events", "Scale Events": "Scale Events"}
        suffix = suffix_map[label]
        v_next = next_version(period, week, suffix)

        csv_bytes = to_csv_bytes(df_part)
        xlsx_bytes = to_xlsx_bytes(df_part)

        c1, c2, c3 = st.columns([2, 2, 3])
        with c1:
            st.download_button(
                label=f"Download CSV (v{v_next})",
                data=csv_bytes,
                file_name=filename(period, week, suffix, v_next, "csv"),
                mime="text/csv",
                use_container_width=True,
                on_click=lambda p=period, w=week, s=suffix: bump_version(p, w, s),
                key=f"csv-{suffix}-{period}-{week}-{v_next}",
            )
        with c2:
            st.download_button(
                label=f"Download XLSX (v{v_next})",
                data=xlsx_bytes,
                file_name=filename(period, week, suffix, v_next, "xlsx"),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                on_click=lambda p=period, w=week, s=suffix: bump_version(p, w, s),
                key=f"xlsx-{suffix}-{period}-{week}-{v_next}",
            )
        with c3:
            st.caption(
                f"Filename pattern: `P{int(period)}_W{int(week)}_Ingestion_{suffix}_v{int(v_next)}.(csv|xlsx)`"
            )
        st.divider()

st.markdown("---")
with st.expander("What is this?"):
    st.write(
        """
        This portfolio-safe demo mimics an internal ingestion tool using fake data.
        - **Generate** a dataset with a seed (deterministic).
        - **Filter** by Period and Start Week.
        - **Split** rows into **Local** or **Scale** (or both via Batch).
        - **Download** CSV/XLSX files with session-based versioning in the filename.
        """
    )
