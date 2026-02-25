from __future__ import annotations

import pandas as pd


def normalize_intraday_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce", utc=True)
    out = out[~out.index.isna()]
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.index = out.index.tz_convert("UTC").tz_localize(None)

    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in out.columns:
            out[col] = pd.Series(dtype="float64")
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["open", "high", "low", "close"])
    out["volume"] = out["volume"].fillna(0.0)
    return out[required]


def resample_ohlcv(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule not in {"5T", "15T"}:
        raise ValueError(f"Unsupported resample rule: {rule}")
    freq = "5min" if rule == "5T" else "15min"

    df = normalize_intraday_frame(df_1m)
    if df.empty:
        return df

    o = df["open"].resample(freq).first()
    h = df["high"].resample(freq).max()
    l = df["low"].resample(freq).min()
    c = df["close"].resample(freq).last()
    v = df["volume"].resample(freq).sum()

    out = pd.concat([o, h, l, c, v], axis=1)
    out.columns = ["open", "high", "low", "close", "volume"]
    out = out.dropna(subset=["close"])
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out
