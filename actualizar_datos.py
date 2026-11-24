"""
Motor de descarga y análisis cuantitativo para Monitor ADRs (PPI).
Descarga cierres diarios desde Yahoo Finance (2016 en adelante), calcula indicadores
de tendencia, momentum, volatilidad y volumen, evalúa señales individuales y analiza
la matriz completa de Relative Value (todos contra todos) para exportar datos_adrs.json.
"""

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

# Universo oficial de 17 ADRs argentinos
ADRS = {
    "GGAL": ("Grupo Financiero Galicia", "Bancos y Finanzas"),
    "BMA": ("Banco Macro", "Bancos y Finanzas"),
    "BBAR": ("Banco BBVA Argentina", "Bancos y Finanzas"),
    "SUPV": ("Grupo Supervielle", "Bancos y Finanzas"),
    "YPF": ("YPF S.A.", "Energía, Petróleo y Gas"),
    "PAM": ("Pampa Energía", "Energía, Petróleo y Gas"),
    "VIST": ("Vista Energy", "Energía, Petróleo y Gas"),
    "TGS": ("Transportadora de Gas del Sur", "Energía, Petróleo y Gas"),
    "CEPU": ("Central Puerto", "Energía, Petróleo y Gas"),
    "EDN": ("Edenor", "Energía, Petróleo y Gas"),
    "MELI": ("MercadoLibre", "Tecnología y E-commerce"),
    "GLOB": ("Globant", "Tecnología y E-commerce"),
    "TEO": ("Telecom Argentina", "Industria, Materiales y Servicios"),
    "LOMA": ("Loma Negra", "Industria, Materiales y Servicios"),
    "IRS": ("IRSA", "Industria, Materiales y Servicios"),
    "CRESY": ("Cresud", "Industria, Materiales y Servicios"),
    "CAAP": ("Corporación América Airports", "Industria, Materiales y Servicios"),
}

TICKERS = list(ADRS.keys())
FETCH_START_DATE = "2015-03-01"    # Margen previo de ~200 ruedas para tener SMA 200 lista en ene-2016
DISPLAY_START_DATE = "2016-01-01"  # Inicio del historial visible (2016 en adelante)
MIN_DOLLAR_VOL_USD = 2_500_000.0
MIN_CORRELATION = 0.75
RV_WINDOW = 60


def to_clean_list(series: pd.Series, decimals: int | None = 2) -> list:
    """Convierte una Serie de pandas a lista JSON-safe (NaN/Inf -> None)."""
    result = []
    for val in series.tolist():
        if val is None or pd.isna(val) or np.isinf(val):
            result.append(None)
        elif decimals is None:
            result.append(int(round(val)))
        else:
            result.append(round(float(val), decimals))
    return result


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula todas las métricas técnicas solicitadas sobre un DataFrame OHLCV."""
    open_ = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    volume = df["Volume"]

    # 1. Tendencia: EMA 21, SMA 50 y SMA 200
    df["ema21"] = close.ewm(span=21, adjust=False).mean()
    df["sma50"] = close.rolling(window=50).mean()
    df["sma200"] = close.rolling(window=200).mean()

    # 2. Momentum: RSI (14 días - Wilder) y MACD (12, 26, 9)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = (100 - (100 / (1 + rs))).fillna(100.0)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # 3. Volatilidad: Bandas de Bollinger (20, 2), ATR (14) y ATR %
    bb_mid = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    df["bb_up"] = bb_mid + 2 * bb_std
    df["bb_low"] = bb_mid - 2 * bb_std

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    df["atr_pct"] = (df["atr"] / close.replace(0, np.nan)) * 100

    # 4. Volumen: Media Móvil 20 ruedas, RVOL y Volumen en USD (20 ruedas)
    df["vol_sma20"] = volume.rolling(window=20).mean()
    df["rvol"] = volume / df["vol_sma20"].replace(0, np.nan)
    df["dollar_vol_sma20"] = (close * volume).rolling(window=20).mean()

    # 5. Señales históricas y actuales según reglas de negocio
    # COMPRA (Tendencia): Precio > SMA 200 AND RSI(14) cruza hacia arriba nivel 40 AND RVOL > 1.2
    buy_trend = (
        (close > df["sma200"])
        & (df["rsi"].shift(1) <= 40)
        & (df["rsi"] > 40)
        & (df["rvol"] > 1.2)
    )

    # COMPRA (Reversión): Precio < BB Inferior AND RSI(14) < 30 AND Cierre > Apertura AND RVOL > 1.5
    buy_rev = (
        (close < df["bb_low"])
        & (df["rsi"] < 30)
        & (close > open_)
        & (df["rvol"] > 1.5)
    )

    # VENTA / SALIDA: Precio cruza debajo SMA 50 con RVOL > 1.2 OR (RSI > 75 AND MACD_Hist cruza a negativo)
    sell_trend = (
        (close.shift(1) >= df["sma50"].shift(1))
        & (close < df["sma50"])
        & (df["rvol"] > 1.2)
    )
    sell_mom = (
        (df["rsi"] > 75)
        & (df["macd_hist"].shift(1) >= 0)
        & (df["macd_hist"] < 0)
    )

    df["signal_buy_trend"] = buy_trend.fillna(False)
    df["signal_buy_rev"] = buy_rev.fillna(False)
    df["signal_sell_trend"] = sell_trend.fillna(False)
    df["signal_sell_mom"] = sell_mom.fillna(False)

    return df


def main() -> None:
    print(f"Iniciando descarga de ADRs desde Yahoo Finance (desde {DISPLAY_START_DATE})...")
    raw_frames: dict[str, pd.DataFrame] = {}

    # Descargamos desde marzo 2015 para que en enero 2016 la SMA 200 ya esté calculada
    for ticker in TICKERS:
        print(f" -> Descargando {ticker}...")
        try:
            hist = yf.Ticker(ticker).history(start=FETCH_START_DATE, interval="1d", auto_adjust=True)
        except Exception as exc:
            print(f"    [Aviso] Error descargando {ticker}: {exc}. Se omite en esta corrida.")
            continue

        if hist.empty:
            print(f"    [Aviso] Sin datos para {ticker} en Yahoo Finance. Se omite en esta corrida.")
            continue

        hist = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
        hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
        hist = hist[~hist.index.duplicated(keep="last")].sort_index()
        raw_frames[ticker] = hist

    if not raw_frames:
        raise RuntimeError("No se pudieron obtener datos de ningún ADR desde Yahoo Finance.")

    active_tickers = [t for t in TICKERS if t in raw_frames]

    # Construir calendario común de ruedas hábiles de EE.UU.
    common_index = sorted(set().union(*(df.index for df in raw_frames.values())))
    common_dt_index = pd.DatetimeIndex(common_index)

    processed_frames: dict[str, pd.DataFrame] = {}
    for ticker in active_tickers:
        df = raw_frames[ticker]
        first_valid = df["Close"].first_valid_index()
        aligned = df.reindex(common_dt_index)

        # Rellenar eventuales huecos puntuales posteriores al IPO (máximo 3 ruedas)
        aligned["Close"] = aligned["Close"].ffill(limit=3)
        for col in ["Open", "High", "Low"]:
            aligned[col] = aligned[col].fillna(aligned["Close"])
        aligned["Volume"] = aligned["Volume"].fillna(0)

        # Preservar NaN antes de la fecha de salida a bolsa (IPO) para ADRs listados post-2016
        if first_valid is not None:
            aligned.loc[aligned.index < first_valid, ["Open", "High", "Low", "Close", "Volume"]] = np.nan

        enriched = compute_indicators(aligned)
        # Recortar desde el 1 de enero de 2016 hasta la última rueda disponible
        processed_frames[ticker] = enriched.loc[enriched.index >= pd.Timestamp(DISPLAY_START_DATE)].copy()

    final_index = processed_frames[active_tickers[0]].index
    timestamps_ms = [int(ts.timestamp() * 1000) for ts in final_index]

    tickers_output = {}
    alerts_output = []

    # Procesar salida por ADR y detectar alertas individuales del último cierre
    for ticker in active_tickers:
        df = processed_frames[ticker]
        name, sector = ADRS[ticker]

        ohlc_list = [
            [None, None, None, None]
            if pd.isna(row.Close)
            else [
                round(float(row.Open), 2),
                round(float(row.High), 2),
                round(float(row.Low), 2),
                round(float(row.Close), 2),
            ]
            for row in df.itertuples()
        ]

        buy_mask = df["signal_buy_trend"] | df["signal_buy_rev"]
        sell_mask = df["signal_sell_trend"] | df["signal_sell_mom"]

        buy_ts = [int(ts.timestamp() * 1000) for ts in df.index[buy_mask]]
        sell_ts = [int(ts.timestamp() * 1000) for ts in df.index[sell_mask]]

        tickers_output[ticker] = {
            "name": name,
            "sector": sector,
            "ohlc": ohlc_list,
            "vol": to_clean_list(df["Volume"], None),
            "ema21": to_clean_list(df["ema21"], 2),
            "sma50": to_clean_list(df["sma50"], 2),
            "sma200": to_clean_list(df["sma200"], 2),
            "bb_up": to_clean_list(df["bb_up"], 2),
            "bb_low": to_clean_list(df["bb_low"], 2),
            "rsi": to_clean_list(df["rsi"], 2),
            "macd": to_clean_list(df["macd"], 3),
            "atr": to_clean_list(df["atr"], 2),
            "atr_pct": to_clean_list(df["atr_pct"], 2),
            "rvol": to_clean_list(df["rvol"], 2),
            "buy_signals": buy_ts,
            "sell_signals": sell_ts,
        }

        # Evaluar alertas individuales en la última rueda cerrada
        last = df.iloc[-1]
        if bool(last["signal_buy_trend"]):
            alerts_output.append(
                {
                    "type": "ind",
                    "category": "compra",
                    "tickerA": ticker,
                    "tickerB": None,
                    "title": ticker,
                    "subtitle": name,
                    "badge": "COMPRA (TEND.)",
                    "badgeStyle": "compra_tend",
                    "description": f"Precio (${last['Close']:.2f}) > SMA 200. RSI(14) cruza nivel 40 al alza ({last['rsi']:.1f}). RVOL = {last['rvol']:.2f}x.",
                    "footerLabel": "Análisis Individual",
                    "footerAction": "Ver gráfico →",
                }
            )
        elif bool(last["signal_buy_rev"]):
            alerts_output.append(
                {
                    "type": "ind",
                    "category": "compra",
                    "tickerA": ticker,
                    "tickerB": None,
                    "title": ticker,
                    "subtitle": name,
                    "badge": "COMPRA (REV.)",
                    "badgeStyle": "compra_rev",
                    "description": f"Cierre debajo BB Inferior con RSI(14) en {last['rsi']:.1f} y vela verde. RVOL = {last['rvol']:.2f}x.",
                    "footerLabel": "Análisis Individual",
                    "footerAction": "Ver gráfico →",
                }
            )

        if bool(last["signal_sell_trend"]) or bool(last["signal_sell_mom"]):
            motivo = (
                f"Precio cruza debajo SMA 50 con RVOL = {last['rvol']:.2f}x."
                if bool(last["signal_sell_trend"])
                else f"Sobrecompra RSI(14) = {last['rsi']:.1f} y MACD histograma cruza a negativo."
            )
            alerts_output.append(
                {
                    "type": "ind",
                    "category": "venta",
                    "tickerA": ticker,
                    "tickerB": None,
                    "title": ticker,
                    "subtitle": name,
                    "badge": "VENTA",
                    "badgeStyle": "venta",
                    "description": motivo,
                    "footerLabel": "Análisis Individual",
                    "footerAction": "Ver gráfico →",
                }
            )

    # Construir Matrices de Correlación y Z-Score y detectar alertas Relative Value
    closes_df = pd.DataFrame({t: processed_frames[t]["Close"] for t in active_tickers})
    returns_df = closes_df.pct_change()
    corr_matrix_60 = returns_df.tail(RV_WINDOW).corr()

    matrix_corr_out: dict[str, dict[str, float]] = {t: {} for t in active_tickers}
    matrix_z_out: dict[str, dict[str, float]] = {t: {} for t in active_tickers}

    for i, a in enumerate(active_tickers):
        matrix_corr_out[a][a] = 1.0
        matrix_z_out[a][a] = 0.0

        for j in range(i + 1, len(active_tickers)):
            b = active_tickers[j]
            corr_val = float(corr_matrix_60.loc[a, b])
            if np.isnan(corr_val):
                corr_val = 0.0

            log_ratio = np.log(closes_df[a] / closes_df[b])
            lr_mean = log_ratio.rolling(window=RV_WINDOW).mean()
            lr_std = log_ratio.rolling(window=RV_WINDOW).std()
            z_series = (log_ratio - lr_mean) / lr_std.replace(0, np.nan)

            z_curr = float(z_series.iloc[-1]) if not pd.isna(z_series.iloc[-1]) else 0.0
            z_prev = float(z_series.iloc[-2]) if not pd.isna(z_series.iloc[-2]) else 0.0

            matrix_corr_out[a][b] = round(corr_val, 3)
            matrix_corr_out[b][a] = round(corr_val, 3)
            matrix_z_out[a][b] = round(z_curr, 3)
            matrix_z_out[b][a] = round(-z_curr, 3)

            # Filtros operativos para Alertas de Relative Value
            liq_a = float(processed_frames[a]["dollar_vol_sma20"].iloc[-1])
            liq_b = float(processed_frames[b]["dollar_vol_sma20"].iloc[-1])

            if liq_a < MIN_DOLLAR_VOL_USD or liq_b < MIN_DOLLAR_VOL_USD:
                continue
            if corr_val <= MIN_CORRELATION:
                continue

            # Verificar estacionariedad: Z-Score cruzó 0 al menos 2 veces en las últimas 60 ruedas
            z_last60 = z_series.tail(RV_WINDOW).dropna()
            signs = np.sign(z_last60)
            signs = signs[signs != 0]
            zero_crossings = int((signs.diff().abs() == 2).sum())
            if zero_crossings < 2:
                continue

            # Disparadores de Relative Value
            if abs(z_curr) > 3.0:
                alerts_output.append(
                    {
                        "type": "rv",
                        "category": "rv",
                        "tickerA": a,
                        "tickerB": b,
                        "title": f"{a} / {b}",
                        "subtitle": f"Correlación: {corr_val:.2f} · Cruces 0: {zero_crossings}",
                        "badge": "STOP-LOSS",
                        "badgeStyle": "stoploss",
                        "description": f"Z-Score en {z_curr:+.2f} (|Z| > 3.0). Invalida reversión a la media por posible evento fundamental.",
                        "footerLabel": "Relative Value",
                        "footerAction": "Ver spread →",
                    }
                )
            elif abs(z_prev) > 2.0 and abs(z_curr) <= 2.0:
                accion = (
                    f"Short {a}, Long {b}." if z_prev > 2.0 else f"Long {a}, Short {b}."
                )
                alerts_output.append(
                    {
                        "type": "rv",
                        "category": "rv",
                        "tickerA": a,
                        "tickerB": b,
                        "title": f"{a} / {b}",
                        "subtitle": f"Correlación: {corr_val:.2f} · Cruces 0: {zero_crossings}",
                        "badge": "ROTACIÓN",
                        "badgeStyle": "rotacion",
                        "description": f"Z-Score vuelve dentro de ±2.0 ({z_curr:+.2f}). {accion}",
                        "footerLabel": "Relative Value",
                        "footerAction": "Ver spread →",
                    }
                )
            elif 2.0 < abs(z_curr) <= 3.0:
                lado = f"{a} caro vs {b}" if z_curr > 2.0 else f"{a} barato vs {b}"
                alerts_output.append(
                    {
                        "type": "rv",
                        "category": "rv",
                        "tickerA": a,
                        "tickerB": b,
                        "title": f"{a} / {b}",
                        "subtitle": f"Correlación: {corr_val:.2f} · Cruces 0: {zero_crossings}",
                        "badge": "WATCHLIST",
                        "badgeStyle": "watchlist",
                        "description": f"Z-Score alcanza {z_curr:+.2f} ({lado}). Esperando confirmación de cruce hacia 0.",
                        "footerLabel": "Relative Value",
                        "footerAction": "Ver spread →",
                    }
                )

    now_art = datetime.now(ZoneInfo("America/Argentina/Buenos_Aires"))
    last_market_date = final_index[-1].strftime("%d/%m/%Y")

    output_payload = {
        "last_update": f"Cierre {last_market_date} · Act. {now_art.strftime('%H:%M')} ART",
        "dates": timestamps_ms,
        "tickers": tickers_output,
        "matrices": {
            "corr": matrix_corr_out,
            "zscore": matrix_z_out,
        },
        "alerts": alerts_output,
    }

    with open("datos_adrs.json", "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    print(f"Proceso finalizado con éxito. Alertas detectadas: {len(alerts_output)}. Archivo datos_adrs.json guardado.")


if __name__ == "__main__":
    main()
