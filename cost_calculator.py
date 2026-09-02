"""
cost_calculator.py

TNB NEM 3.0 tariff constants and the daily savings calculation. Kept
separate so that if TNB revises the tariff structure again (as it did
moving into RP4), only this one file needs to be updated, and every
part of the app that references the rate stays consistent.
"""

import pandas as pd

TNB_ENERGY_CHARGE   = 0.2703
TNB_CAPACITY_CHARGE = 0.0455
TNB_NETWORK_CHARGE  = 0.1285
TNB_TOTAL_RATE      = 0.4443


def calculate_savings(df, monthly_kwh):
    """
    For each calendar day: sums solar generation, computes residual
    grid draw, and compares the electricity bill with and without
    solar offset under the TNB NEM 3.0 total rate.
    """
    daily_kwh = monthly_kwh / 30.0
    rows = []
    for date, grp in df.groupby(df['timestamp'].dt.date):
        solar     = grp['power_output_kw'].sum()
        grid_used = max(0.0, daily_kwh - solar)
        offset    = min(solar, daily_kwh)
        bill_no   = daily_kwh * TNB_TOTAL_RATE
        bill_with = grid_used * TNB_TOTAL_RATE
        rows.append({
            'date':                  pd.Timestamp(date),
            'solar_generated_kwh':   round(solar,     2),
            'grid_consumed_kwh':     round(grid_used, 2),
            'solar_offset_kwh':      round(offset,    2),
            'bill_without_solar_rm': round(bill_no,   2),
            'bill_with_solar_rm':    round(bill_with, 2),
            'saving_rm':             round(bill_no - bill_with, 2),
        })
    return pd.DataFrame(rows)
