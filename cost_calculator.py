"""
cost_calculator.py

TNB NEM 3.0 tariff constants and the daily savings calculation. Kept
separate so that if TNB revises the tariff structure again (as it did
moving into RP4), only this one file needs to be updated, and every
part of the app that references the rate stays consistent.

A FusionSolar Plant Report export already contains real measured
Consumption, Export, and Import values per hour straight from the
meter. When these are present, calculate_savings uses them directly
instead of the previous approach of asking the user to guess their
monthly consumption on a slider, since the resulting saving figure is
then fully data driven with no assumptions. The slider is kept only
as a fallback for the rare case a file is missing these columns.
"""

import pandas as pd

TNB_ENERGY_CHARGE   = 0.2703
TNB_CAPACITY_CHARGE = 0.0455
TNB_NETWORK_CHARGE  = 0.1285
TNB_TOTAL_RATE      = 0.4443


def calculate_savings(df, monthly_kwh_fallback=600):
    """
    Computes daily NEM 3.0 savings.

    Measured mode (used whenever the uploaded Plant Report or sample
    data includes real consumption_kwh and import_kwh columns):
        bill_without_solar = actual consumption x TNB rate
        bill_with_solar    = actual grid import x TNB rate
        saving             = the difference, which equals the value
                              of self consumed solar energy at the
                              TNB rate

    Estimated mode (fallback only, used if consumption_kwh is missing
    entirely): reconstructs an assumed daily consumption from the
    user specified monthly figure, exactly as the original design did.

    Returns a tuple of (savings_df, is_measured) so the caller can
    show the user which mode was used.
    """
    has_measured = ('consumption_kwh' in df.columns
                     and df['consumption_kwh'].notna().sum() > 0)

    rows = []
    if has_measured:
        for date, grp in df.groupby(df['timestamp'].dt.date):
            solar       = grp['power_output_kw'].sum()
            consumption = grp['consumption_kwh'].sum()
            import_kwh  = (grp['import_kwh'].sum() if 'import_kwh' in grp.columns
                            else max(0.0, consumption - solar))
            export_kwh  = (grp['export_kwh'].sum() if 'export_kwh' in grp.columns
                            else max(0.0, solar - consumption))
            self_cons   = (grp['self_consumption_kwh'].sum() if 'self_consumption_kwh' in grp.columns
                            else min(solar, consumption))

            bill_no   = consumption * TNB_TOTAL_RATE
            bill_with = import_kwh * TNB_TOTAL_RATE

            rows.append({
                'date':                  pd.Timestamp(date),
                'solar_generated_kwh':   round(solar, 2),
                'grid_consumed_kwh':     round(import_kwh, 2),
                'solar_offset_kwh':      round(self_cons, 2),
                'export_kwh':            round(export_kwh, 2),
                'bill_without_solar_rm': round(bill_no, 2),
                'bill_with_solar_rm':    round(bill_with, 2),
                'saving_rm':             round(bill_no - bill_with, 2),
            })
    else:
        daily_kwh = monthly_kwh_fallback / 30.0
        for date, grp in df.groupby(df['timestamp'].dt.date):
            solar     = grp['power_output_kw'].sum()
            grid_used = max(0.0, daily_kwh - solar)
            offset    = min(solar, daily_kwh)
            bill_no   = daily_kwh * TNB_TOTAL_RATE
            bill_with = grid_used * TNB_TOTAL_RATE
            rows.append({
                'date':                  pd.Timestamp(date),
                'solar_generated_kwh':   round(solar, 2),
                'grid_consumed_kwh':     round(grid_used, 2),
                'solar_offset_kwh':      round(offset, 2),
                'export_kwh':            round(max(0.0, solar - daily_kwh), 2),
                'bill_without_solar_rm': round(bill_no, 2),
                'bill_with_solar_rm':    round(bill_with, 2),
                'saving_rm':             round(bill_no - bill_with, 2),
            })

    return pd.DataFrame(rows), has_measured
