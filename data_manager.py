# data_manager.py — Binance Veri Çekme Motoru (doğrudan REST API)
# ─────────────────────────────────────────────────────────────────────────────

import time
import logging
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from config import (
    REQUEST_DELAY, RATE_LIMIT_WAIT, MAX_RETRY,
    BINANCE_BASE_URL, FUNDING_RATE_URL, TICKER_URL,
    PERIYOT_KONFIG, SABIT_COINLER, MAX_COIN,
    HACIM_PATLAMA_KATI, BTC_PANEL_KONFIG,
)
from archive_manager import save_funding_rates

logger = logging.getLogger(__name__)

# Binance Futures OHLCV endpoint
KLINES_URL = f"{BINANCE_BASE_URL}/fapi/v1/klines"

# ──────────────────────────────────────────────
# EXCHANGE COMPAT SHIM (scoring/archive için)
# ──────────────────────────────────────────────

class FakeExchange:
    """archive_manager'ın beklediği exchange arayüzünü sağlar."""
    def fetch_ohlcv(self, symbol: str, timeframe: str = "1d",
                    since: int = None, limit: int = 1000) -> list:
        # symbol: "BTC/USDT:USDT" → "BTCUSDT"
        base = symbol.split("/")[0]
        binance_sym = f"{base}USDT"
        raw = _fetch_klines_raw(binance_sym, timeframe, since=since, limit=limit)
        return raw


def get_exchange():
    return FakeExchange()


# ──────────────────────────────────────────────
# DÜŞÜK SEVİYE: Binance REST klines
# ──────────────────────────────────────────────

def _tf_to_binance(tf: str) -> str:
    """ccxt timeframe → Binance interval string."""
    mapping = {
        "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m",
        "30m": "30m", "1h": "1h", "2h": "2h", "4h": "4h",
        "6h": "6h", "8h": "8h", "12h": "12h",
        "1d": "1d", "3d": "3d", "1w": "1w",
    }
    return mapping.get(tf, tf)


def _fetch_klines_raw(
    symbol: str,
    timeframe: str,
    since: Optional[int] = None,
    limit: int = 500,
) -> list:
    """
    Binance Futures REST /fapi/v1/klines endpoint'inden ham veri çeker.
    Döndürür: [[open_time, open, high, low, close, volume, ...], ...]
    ccxt formatında: [[ts_ms, open, high, low, close, volume], ...]
    """
    interval = _tf_to_binance(timeframe)
    params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1500)}
    if since is not None:
        params["startTime"] = int(since)

    for attempt in range(MAX_RETRY):
        try:
            resp = requests.get(KLINES_URL, params=params, timeout=15)
            if resp.status_code == 429 or resp.status_code == 418:
                wait = RATE_LIMIT_WAIT[min(attempt, len(RATE_LIMIT_WAIT) - 1)]
                logger.warning(f"[Rate Limit] {symbol} → {wait}s bekleniyor")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            if not data or not isinstance(data, list):
                return []
            # Binance formatı: [open_time, open, high, low, close, volume, ...]
            result = []
            for row in data:
                try:
                    result.append([
                        int(row[0]),    # ts ms
                        float(row[1]),  # open
                        float(row[2]),  # high
                        float(row[3]),  # low
                        float(row[4]),  # close
                        float(row[5]),  # volume
                    ])
                except (IndexError, ValueError, TypeError):
                    continue
            return result
        except requests.exceptions.Timeout:
            logger.warning(f"[Timeout] {symbol} deneme {attempt+1}")
            time.sleep(5)
        except Exception as e:
            logger.error(f"[Klines Hata] {symbol}/{timeframe}: {e}")
            return []
    return []


def _klines_to_df(raw: list) -> Optional[pd.DataFrame]:
    if not raw:
        return None
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.dropna()
    if df.empty:
        return None
    df["ts"] = df["ts"].astype(np.int64)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df


# ──────────────────────────────────────────────
# OHLCV + RESAMPLE
# ──────────────────────────────────────────────

def fetch_ohlcv_with_resample(
    exchange,           # FakeExchange (kullanılmıyor, compat için)
    symbol: str,        # "ETH/USDT:USDT"
    periyot: str,
) -> Optional[pd.DataFrame]:
    konfig = PERIYOT_KONFIG.get(periyot)
    if not konfig:
        return None

    tf     = konfig["tf"]
    limit  = konfig["limit"]
    resamp = konfig.get("resample")

    base = symbol.split("/")[0]
    binance_sym = f"{base}USDT"

    raw = _fetch_klines_raw(binance_sym, tf, limit=limit)
    df  = _klines_to_df(raw)
    if df is None:
        return None

    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    df.drop(columns=["ts"], inplace=True)

    if resamp:
        df = _resample_ohlcv(df, resamp)

    return df if not df.empty else None


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    resampled = df.resample(rule).agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()
    return resampled


# ──────────────────────────────────────────────
# BTC PANELİ VERİSİ
# ──────────────────────────────────────────────

def fetch_btc_panel_data(exchange, btc_periyot: str) -> Optional[pd.DataFrame]:
    konfig = BTC_PANEL_KONFIG.get(btc_periyot)
    if not konfig:
        return None
    tf     = konfig["tf"]
    limit  = konfig["limit"]
    resamp = konfig.get("resample")

    raw = _fetch_klines_raw("BTCUSDT", tf, limit=limit)
    df  = _klines_to_df(raw)
    if df is None:
        return None

    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    df.drop(columns=["ts"], inplace=True)

    if resamp:
        df = _resample_ohlcv(df, resamp)
    return df if not df.empty else None


# ──────────────────────────────────────────────
# TOPLU TARAMA
# ──────────────────────────────────────────────

def toplu_tarama(
    exchange,
    semboller: List[str],
    periyot: str,
    progress_placeholder=None,
) -> Dict[str, pd.DataFrame]:
    tum_data: Dict[str, pd.DataFrame] = {}
    hatalar: List[str] = []
    total = len(semboller)

    for i, sym in enumerate(semboller):
        if progress_placeholder:
            try:
                progress_placeholder.text(
                    f"⏳ Taranıyor: {sym} ({i+1}/{total})"
                )
            except Exception:
                pass

        try:
            df = fetch_ohlcv_with_resample(exchange, f"{sym}/USDT:USDT", periyot)
            if df is not None and not df.empty:
                tum_data[sym] = df
            else:
                hatalar.append(sym)
        except Exception as e:
            logger.error(f"[Tarama] {sym}: {e}")
            hatalar.append(sym)

        time.sleep(REQUEST_DELAY)

    if progress_placeholder:
        try:
            progress_placeholder.text(
                f"✅ Tamamlandı: {len(tum_data)}/{total} coin"
            )
        except Exception:
            pass

    if hatalar:
        logger.warning(f"[Tarama] Başarısız: {hatalar}")

    return tum_data


# ──────────────────────────────────────────────
# 24 SAATLİK HACİM
# ──────────────────────────────────────────────

def fetch_24h_volumes() -> Dict[str, float]:
    try:
        resp = requests.get(TICKER_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        volumes = {}
        for item in data:
            sym = item.get("symbol", "")
            if sym.endswith("USDT"):
                base = sym[:-4]
                try:
                    volumes[base] = float(item.get("quoteVolume", 0))
                except (ValueError, TypeError):
                    pass
        return volumes
    except Exception as e:
        logger.error(f"[24h Hacim] {e}")
        return {}


def tespit_hacim_patlamasi(
    sabit_coinler: List[str],
    manuel_coinler: List[str],
) -> List[str]:
    volumes = fetch_24h_volumes()
    if not volumes:
        return []
    tum_liste = set(sabit_coinler + manuel_coinler)
    listed_vols = [v for s, v in volumes.items() if s in tum_liste]
    if not listed_vols:
        return []
    ort_hacim = np.mean(listed_vols)
    eslim = ort_hacim * HACIM_PATLAMA_KATI
    return [s for s, v in volumes.items() if s not in tum_liste and v >= eslim]


def get_sorted_symbols_by_volume() -> List[str]:
    volumes = fetch_24h_volumes()
    if not volumes:
        return []
    return sorted(volumes.keys(), key=lambda s: volumes[s], reverse=True)


# ──────────────────────────────────────────────
# GEÇERLİ SEMBOL FİLTRESİ
# ──────────────────────────────────────────────

_gecerli_semboller_cache: Optional[List[str]] = None
_gecerli_cache_zaman: float = 0.0
CACHE_TTL = 3600


def filtrele_gecerli_semboller(exchange, semboller: List[str]) -> List[str]:
    """
    Binance'te aktif olan sembolleri döndürür.
    24h ticker'dan kontrol eder, 1 saat önbelleğe alır.
    """
    global _gecerli_semboller_cache, _gecerli_cache_zaman
    now = time.time()
    if _gecerli_semboller_cache and (now - _gecerli_cache_zaman) < CACHE_TTL:
        available = set(_gecerli_semboller_cache)
        return [s for s in semboller if s in available]

    volumes = fetch_24h_volumes()
    if not volumes:
        # API erişilemiyorsa listeyi olduğu gibi dön
        return semboller

    available = set(volumes.keys())
    _gecerli_semboller_cache = list(available)
    _gecerli_cache_zaman = now
    valid   = [s for s in semboller if s in available]
    invalid = [s for s in semboller if s not in available]
    if invalid:
        logger.info(f"[Filtre] Binance'te bulunamadı: {invalid}")
    return valid if valid else semboller


# ──────────────────────────────────────────────
# FUNDING RATE
# ──────────────────────────────────────────────

def fetch_funding_rates(semboller: List[str], limit: int = 100) -> Dict[str, list]:
    result = {}
    for sym in semboller:
        binance_sym = f"{sym}USDT"
        try:
            resp = requests.get(
                FUNDING_RATE_URL,
                params={"symbol": binance_sym, "limit": limit},
                timeout=10,
            )
            resp.raise_for_status()
            records = resp.json()
            if isinstance(records, list) and records:
                save_funding_rates(sym, records)
                result[sym] = records
        except Exception as e:
            logger.warning(f"[Funding] {sym}: {e}")
        time.sleep(REQUEST_DELAY)
    return result


def fetch_latest_funding_rate(symbol: str) -> Optional[float]:
    try:
        resp = requests.get(
            FUNDING_RATE_URL,
            params={"symbol": f"{symbol}USDT", "limit": 1},
            timeout=10,
        )
        resp.raise_for_status()
        records = resp.json()
        if records and isinstance(records, list):
            return float(records[-1].get("fundingRate", 0))
    except Exception as e:
        logger.warning(f"[Funding Anlık] {symbol}: {e}")
    return None
