import pandas as pd
import numpy as np
import os


# =============================================================================
# CONSTANTS (derived from EDA: 01_eda.ipynb)
# =============================================================================

LAGE_MAPPING = {
    'Öffentlicher Parkplatz': 'Public_Street',
    'Kundenparkplatz':        'Commercial_Retail',
    'Parkhaus':               'Commercial_Retail',
    'Sonstige Tankstelle':    'Transit_Energy',
    'Tankstelle an einer Bundesautobahn': 'Transit_Energy',
    'Sonstige':               'Miscellaneous',
    'Park & Ride':            'Miscellaneous',
}

STATE_TO_REGION = {
    'Bayern':                  'South',
    'Baden-Württemberg':       'South',
    'Nordrhein-Westfalen':     'West',
    'Hessen':                  'West',
    'Rheinland-Pfalz':         'West',
    'Saarland':                'West',
    'Niedersachsen':           'North_Cities',
    'Hamburg':                 'North_Cities',
    'Schleswig-Holstein':      'North_Cities',
    'Bremen':                  'North_Cities',
    'Berlin':                  'North_Cities',
    'Brandenburg':             'East',
    'Sachsen':                 'East',
    'Sachsen-Anhalt':          'East',
    'Thüringen':               'East',
    'Mecklenburg-Vorpommern':  'East',
}

# EDA Cell 19: bins [0, 22, 50, 150, inf] → AC_Standard / DC_Fast / HPC / Ultra_Fast
P_CLASS_BINS   = [0, 22, 50, 150, np.inf]
P_CLASS_LABELS = ['AC_Standard', 'DC_Fast', 'HPC', 'Ultra_Fast']


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full Feature Engineering Pipeline.

    Order of operations
    -------------------
    1. clean_and_prepare          – parse dates, sort
    2. engineer_categorical       – lage_binned, region, p_class, is_standard_street_hub
    3. engineer_time_features     – start_hour, hour_sin/cos, is_night_shift, day_phase
    4. engineer_lag_features      – station_overall_avg, last_5_sessions_avg_energy,
                                    rolling_7d_avg_energy, phase_avg_energy
    5. engineer_binned_rolling    – binned_rolling_30d_duration
    6. apply_numerical_transforms – 99th-pct cap + log1p for dauer_sekunden & energie_wh
    7. select_final_columns       – drop leakage / id columns
    """
    df = df.copy()

    df = clean_and_prepare(df)
    df = engineer_categorical_features(df)
    df = engineer_time_features(df)
    df = engineer_lag_features(df)
    df = engineer_binned_rolling_features(df)
    df = apply_numerical_transformations(df)
    df = select_final_columns(df)

    return df


# =============================================================================
# STEP 1 – CLEAN & PREPARE
# =============================================================================

def clean_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates and sort for time-series consistency."""
    df['beginn'] = pd.to_datetime(df['beginn'])
    df = df.sort_values(['ls_id', 'beginn']).reset_index(drop=True)
    return df


# =============================================================================
# STEP 2 – CATEGORICAL FEATURES
# =============================================================================

def engineer_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates:
      - lage_binned  : condensed location groups (EDA Cell 19)
      - region       : bundesland → geographic region (EDA Cell 19)
      - p_class      : power tier from maxladeleistunginkilowatt (EDA Cell 19)
      - is_standard_street_hub : majority-class flag (Public_Street + AC_Standard)
    """
    # --- Location binning ---
    df['lage_binned'] = df['lage'].map(LAGE_MAPPING)

    # --- State → Region ---
    df['region'] = df['bundesland'].map(STATE_TO_REGION)

    # --- Power class bins ---
    df['p_class'] = pd.cut(
        df['maxladeleistunginkilowatt'],
        bins=P_CLASS_BINS,
        labels=P_CLASS_LABELS,
        include_lowest=True,
    )

    # --- Majority-class flag ---
    # Public_Street + AC_Standard covers the dominant segment observed in EDA
    df['is_standard_street_hub'] = np.where(
        (df['lage_binned'] == 'Public_Street') & (df['p_class'] == 'AC_Standard'),
        1, 0,
    ).astype(int)

    return df


# =============================================================================
# STEP 3 – TIME FEATURES
# =============================================================================

def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates:
      - start_hour    : integer hour of session start
      - hour_sin/cos  : cyclical encoding of start_hour
      - is_night_shift: binary flag for 20:00–03:59 (slow-charging night window)
      - day_phase     : 'night' (00–04), 'day' (06–20), 'evening' (20–24 + 04–06)
                        used for phase_avg_energy grouping (EDA Cell 33)
    """
    df['start_hour'] = df['beginn'].dt.hour

    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['start_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['start_hour'] / 24)

    # Binary night flag: 20:00 – 03:59  (spec: "Day Phase 20pm – 4am")
    df['is_night_shift'] = (
        (df['start_hour'] >= 20) | (df['start_hour'] < 4)
    ).astype(int)

    # Three-way day_phase used for lag grouping (EDA Cell 33)
    def _day_phase(hour: int) -> str:
        if 0 <= hour < 4:
            return 'night'
        if 6 <= hour < 20:
            return 'day'
        return 'evening'   # covers 04–06 and 20–24

    df['day_phase'] = df['start_hour'].apply(_day_phase)

    return df


# =============================================================================
# STEP 4 – LAG / ROLLING FEATURES
# =============================================================================

def engineer_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates (must run after sort by ls_id + beginn):

      station_overall_avg        – per-station mean (used as cold-start fallback)
      last_5_sessions_avg_energy – shift(1) rolling-5 mean per station
      rolling_7d_avg_energy      – time-based 7-day rolling mean per station
                                   (EDA Cell 33, get_rolling_features)
      phase_avg_energy           – expanding mean per (ls_id, day_phase)
                                   (EDA Cell 33)
    """
    # Guarantee correct sort order
    df = df.sort_values(['ls_id', 'beginn']).reset_index(drop=True)

    # --- Baseline fallbacks (Static 2023 benchmarks) ---
    # To prevent leakage, we use 2023 data as a static baseline for "cold start" situations in 2024.
    df_2023 = df[df['beginn'].dt.year == 2023]
    
    if not df_2023.empty:
        station_baselines = df_2023.groupby('ls_id')['energie_wh'].mean()
        global_baseline = df_2023['energie_wh'].mean()
    else:
        # If 2023 data is not present in the provided DataFrame, fallback to global mean of what we have 
        # (but this is less ideal than knowing the 2023 mean up front)
        station_baselines = pd.Series(dtype=float)
        global_baseline = df['energie_wh'].mean()

    # --- Non-leaking expanding station average ---
    # station_overall_avg becomes a dynamic historical mean, not a global static one.
    df['station_overall_avg'] = (
        df.groupby('ls_id')['energie_wh']
        .transform(lambda x: x.shift(1).expanding().mean())
        # Fallback 1: 2023 station-specific mean
        .fillna(df['ls_id'].map(station_baselines))
        # Fallback 2: 2023 global mean
        .fillna(global_baseline)
    )

    # --- Sequence lag: last 5 sessions ---
    df['last_5_sessions_avg_energy'] = (
        df.groupby('ls_id')['energie_wh']
        .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
        .fillna(df['station_overall_avg'])
    )

    # --- Time-based 7-day rolling average (closed='left' prevents leakage) ---
    roll_7d_sum = (
        df.set_index('beginn')
        .groupby('ls_id')['energie_wh']
        .transform(lambda x: x.rolling('7d', closed='left').sum())
        .values.astype(float)
    )
    roll_7d_count = (
        df.set_index('beginn')
        .groupby('ls_id')['energie_wh']
        .transform(lambda x: x.rolling('7d', closed='left').count())
        .values.astype(float)
    )
    df['rolling_7d_avg_energy'] = np.where(
        roll_7d_count > 0,
        (roll_7d_sum / np.maximum(roll_7d_count, 1)).astype(float),
        df['last_5_sessions_avg_energy'].astype(float)
    )

    # --- Phase-level expanding average per (ls_id, day_phase) ---
    df['phase_avg_energy'] = (
        df.groupby(['ls_id', 'day_phase'])['energie_wh']
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(df['station_overall_avg'])
    )

    return df


# =============================================================================
# STEP 5 – BINNED ROLLING DURATION
# =============================================================================

def engineer_binned_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    binned_rolling_30d_duration: 30-day rolling average duration
    grouped by (p_class, start_hour) – EDA Cell 36.

    Fallback hierarchy (EDA Cell 36):
      1. p_class average
      2. Global average
    """
    # Sort required for non-monotonic index safety (EDA Cell 36)
    df = df.sort_values(['p_class', 'start_hour', 'beginn']).reset_index(drop=True)

    df['binned_rolling_30d_duration'] = (
        df.set_index('beginn')
        .groupby(['p_class', 'start_hour'], observed=True)['dauer_sekunden']
        .transform(lambda x: x.rolling('30d', closed='left').mean())
        .values.astype(np.float64)
    )

    # --- Baseline fallbacks (Static 2023 benchmarks) ---
    df_2023 = df[df['beginn'].dt.year == 2023]
    
    if not df_2023.empty:
        p_class_baselines = df_2023.groupby('p_class', observed=True)['dauer_sekunden'].mean().astype(np.float64)
        global_baseline   = float(df_2023['dauer_sekunden'].mean())
    else:
        p_class_baselines = pd.Series(dtype=np.float64)
        global_baseline   = float(df['dauer_sekunden'].mean())

    # Level-1 fallback: p_class mean (2023 benchmark)
    # Note: .map() on a categorical column returns categorical → must cast to float for .fillna()
    p_class_fill = df['p_class'].map(p_class_baselines).astype(np.float64)
    df['binned_rolling_30d_duration'] = (
        df['binned_rolling_30d_duration']
        .fillna(p_class_fill)
    )


    # Level-2 fallback: global mean (2023 benchmark)
    df['binned_rolling_30d_duration'] = (
        df['binned_rolling_30d_duration']
        .fillna(global_baseline)
    )

    # Restore original time-series order
    df = df.sort_values(['ls_id', 'beginn']).reset_index(drop=True)

    return df


# =============================================================================
# STEP 6 – NUMERICAL TRANSFORMATIONS
# =============================================================================

def apply_numerical_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """
    For dauer_sekunden and energie_wh (target):
      1. Cap at 99th percentile
      2. Apply log1p transform  →  log_dauer_sekunden, log_energie_wh
    """
    for col in ['dauer_sekunden', 'energie_wh']:
        if col not in df.columns:
            continue
        # Ensure float type to prevent LossySetitemError/AssertionError during clipping
        df[col] = df[col].astype(float)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=upper)
        df[f'log_{col}'] = np.log1p(df[col])

    return df


# =============================================================================
# STEP 7 – SELECT FINAL COLUMNS
# =============================================================================

def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that cause leakage or are no longer needed:
      - lv_id, lp_id  : identifier columns
      - dauer_sekunden : raw duration leaks into target; use log_dauer_sekunden
    """
    cols_to_drop = ['lv_id', 'lp_id', 'dauer_sekunden']
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])