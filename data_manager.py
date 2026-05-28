# data_manager.py — Binance Veri Çekme Motoru
# ─────────────────────────────────────────────────────────────────────────────

import time
import logging
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import ccxt
import pandas as pd
import numpy as np

from config import (
    EXCHANGE_ID, REQUEST_DELAY, RATE_LIMIT_WAIT, MAX_RETRY,
    BINANCE_BASE_URL, FUNDING_RATE_URL, TICKER_URL,
    PERIYOT_KONFIG, SABIT_COINLER, MAX_COIN, MAX_MANUEL_EK,
    HACIM_PATLAMA_KATI, BTC_PANEL_KONFIG,
)
from archive_manager import save_funding_rates

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# EXCHANGE BAŞLATMA
# ──────────────────────────────────────────────

def get_exchange() -> ccxt.Exchange:
    """Binance USDT-M perpetual exchange nesnesi döndürür (public, API key gerekmez)."""
    exchange = ccxt.binanceusdm({
        "enableRateLimit": False,   # Manuel rate limit yönetimi yapıyoruz
        "options": {"defaultType": "future"},
    })
    return exchange


# ──────────────────────────────────────────────
# RATE LIMIT KORUMAL OHLCV ÇEKİMİ
# ──────────────────────────────────────────────

def fetch_ohlcv_safe(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since: Optional[int] = None,
    limit: int = 500,
) -> Optional[pd.DataFrame]:
    """
    Rate limit korumalı ccxt OHLCV çekimi.
    Döndürür: DataFrame sütunları [ts, open, high, low, close, volume]
    veya None (hata durumunda).
    """
    for attempt in range(MAX_RETRY):
        try:
            raw = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=limit,
            )
            if not raw:
                return None
            df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
            df = df.dropna()
            df["ts"] = df["ts"].astype(np.int64)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            return df
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "418" in err_str or "rate limit" in err_str:
                wait = RATE_LIMIT_WAIT[min(attempt, len(RATE_LIMIT_WAIT) - 1)]
                logger.warning(f"[Rate Limit] {symbol}/{timeframe} → {wait}s bekleniyor (deneme {attempt+1})")
                time.sleep(wait)
            else:
                logger.error(f"[OHLCV Hata] {symbol}/{timeframe}: {e}")
                return None
    logger.error(f"[OHLCV] {symbol} {MAX_RETRY} denemede başarısız")
    return None


def fetch_ohlcv_with_resample(
    exchange: ccxt.Exchange,
    symbol: str,
    periyot: str,
) -> Optional[pd.DataFrame]:
    """
    Periyot konfigürasyonuna göre OHLCV çekip gerekirse resample eder.
    Döndürür: DataFrame sütunları [open, high, low, close, volume] + datetime index
    """
    konfig = PERIYOT_KONFIG.get(periyot)
    if not konfig:
        logger.error(f"Bilinmeyen periyot: {periyot}")
        return None

    tf     = konfig["tf"]
    limit  = konfig["limit"]
    resamp = konfig.get("resample")

    since_ms = None  # limit ile zaten son N mum çekilir

    df = fetch_ohlcv_safe(exchange, symbol, tf, since=since_ms, limit=limit)
    if df is None or df.empty:
        return None

    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    df.drop(columns=["ts"], inplace=True)

    if resamp:
        df = _resample_ohlcv(df, resamp)

    return df


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """DataFrame'i belirtilen Pandas offset rule'a göre yeniden örnekler."""
    resampled = df.resample(rule).agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()
    return resampled


# ──────────────────────────────────────────────
# TÜM COİNLER İÇİN TOPLU ÇEKİM
# ──────────────────────────────────────────────

def fetch_all_coins(
    exchange: ccxt.Exchange,
    semboller: List[str],
    periyot: str,
    progress_cb=None,
) -> Dict[str, pd.DataFrame]:
    """
    Tüm coinlerin OHLCV verisini çeker.
    progress_cb(i, total, symbol) → opsiyonel ilerleme geri çağrısı
    Döndürür: {symbol: DataFrame}
    """
    result = {}
    total = len(semboller)
    for i, sym in enumerate(semboller):
        if progress_cb:
            progress_cb(i, total, sym)
        ccxt_sym = f"{sym}/USDT:USDT"
        df = fetch_ohlcv_with_resample(exchange, ccxt_sym, periyot)
        if df is not None and not df.empty:
            result[sym] = df
        else:
            logger.warning(f"[Veri] {sym} verisi çekilemedi")
        time.sleep(REQUEST_DELAY)
    return result


def fetch_btc_panel_data(exchange: ccxt.Exchange, btc_periyot: str) -> Optional[pd.DataFrame]:
    """Bitcoin paneli için periyota özel veri çeker."""
    konfig = BTC_PANEL_KONFIG.get(btc_periyot)
    if not konfig:
        return None
    tf     = konfig["tf"]
    limit  = konfig["limit"]
    resamp = konfig.get("resample")

    df = fetch_ohlcv_safe(exchange, "BTC/USDT:USDT", tf, limit=limit)
    if df is None or df.empty:
        return None

    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    df.drop(columns=["ts"], inplace=True)

    if resamp:
        df = _resample_ohlcv(df, resamp)

    return df


# ──────────────────────────────────────────────
# BİNANCE 24 SAATLİK HACİM (USDT-M)
# ──────────────────────────────────────────────

def fetch_24h_volumes() -> Dict[str, float]:
    """
    Binance USDT-M perpetual tickerlarını çeker ve sembol → hacim dict döndürür.
    API key gerektirmez.
    """
    try:
        resp = requests.get(TICKER_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        volumes = {}
        for item in data:
            sym = item.get("symbol", "")
            if sym.endswith("USDT"):
                base = sym[:-4]  # BTCUSDT → BTC
                try:
                    volumes[base] = float(item.get("quoteVolume", 0))
                except (ValueError, TypeError):
                    pass
        return volumes
    except Exception as e:
        logger.error(f"[24h Hacim] Çekme hatası: {e}")
        return {}


def tespit_hacim_patlamasi(
    sabit_coinler: List[str],
    manuel_coinler: List[str],
) -> List[str]:
    """
    Sabit listede VE manuel listede olmayan coinler arasında
    24 saatlik hacmi 3 katına çıkan coinleri döndürür.
    """
    volumes = fetch_24h_volumes()
    if not volumes:
        return []

    tum_liste = set(sabit_coinler + manuel_coinler)
    listed_vols = [v for s, v in volumes.items() if s in tum_liste]
    if not listed_vols:
        return []

    ort_hacim = np.mean(listed_vols)
    eslim = ort_hacim * HACIM_PATLAMA_KATI

    patlamalar = []
    for sym, vol in volumes.items():
        if sym not in tum_liste and vol >= eslim:
            patlamalar.append(sym)
    return patlamalar


def get_sorted_symbols_by_volume() -> List[str]:
    """Tüm USDT-M perpetual coinleri hacme göre azalan sırayla döndürür."""
    volumes = fetch_24h_volumes()
    if not volumes:
        return []
    sorted_syms = sorted(volumes.keys(), key=lambda s: volumes[s], reverse=True)
    return sorted_syms


# ──────────────────────────────────────────────
# FUNDING RATE ÇEKİMİ
# ──────────────────────────────────────────────

def fetch_funding_rates(semboller: List[str], limit: int = 100) -> Dict[str, list]:
    """
    Her sembol için son funding rate kayıtlarını Binance'ten çeker,
    SQLite'a kaydeder ve {symbol: [records]} döndürür.
    """
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
            if isinstance(records, list) and len(records) > 0:
                save_funding_rates(sym, records)
                result[sym] = records
        except Exception as e:
            logger.warning(f"[Funding] {sym} çekme hatası: {e}")
        finally:
            time.sleep(REQUEST_DELAY)
    return result


def fetch_latest_funding_rate(symbol: str) -> Optional[float]:
    """Tek sembolün güncel funding rate'ini döndürür."""
    binance_sym = f"{symbol}USDT"
    try:
        resp = requests.get(
            FUNDING_RATE_URL,
            params={"symbol": binance_sym, "limit": 1},
            timeout=10,
        )
        resp.raise_for_status()
        records = resp.json()
        if records and isinstance(records, list):
            return float(records[-1].get("fundingRate", 0))
    except Exception as e:
        logger.warning(f"[Funding] {symbol} anlık çekme hatası: {e}")
    return None


# ──────────────────────────────────────────────
# MEVCUT SYMBOL LİSTESİ (ccxt üzerinden doğrulama)
# ──────────────────────────────────────────────

_market_cache: Optional[dict] = None
_market_cache_time: float = 0.0
MARKET_CACHE_TTL = 3600  # 1 saat


def get_available_symbols(exchange: ccxt.Exchange) -> List[str]:
    """
    Binance'te gerçekten işlem gören USDT-M perpetual sembolleri döndürür.
    Sonuç 1 saat önbelleğe alınır.
    """
    global _market_cache, _market_cache_time
    now = time.time()
    if _market_cache is not None and (now - _market_cache_time) < MARKET_CACHE_TTL:
        return list(_market_cache.keys())

    try:
        markets = exchange.load_markets()
        available = {
            sym.split("/")[0]: sym
            for sym, info in markets.items()
            if info.get("swap") and info.get("quote") == "USDT" and info.get("active")
        }
        _market_cache = available
        _market_cache_time = now
        return list(available.keys())
    except Exception as e:
        logger.error(f"[Markets] Yükleme hatası: {e}")
        return SABIT_COINLER.copy()


def filtrele_gecerli_semboller(
    exchange: ccxt.Exchange,
    semboller: List[str],
) -> List[str]:
    """İstenen sembollerden Binance'te aktif olanları döndürür."""
    available = get_available_symbols(exchange)
    available_set = set(available)
    valid = [s for s in semboller if s in available_set]
    invalid = [s for s in semboller if s not in available_set]
    if invalid:
        logger.warning(f"[Filtre] Binance'te bulunamayan semboller: {invalid}")
    return valid


# ──────────────────────────────────────────────
# TARAMA DURUMU
# ──────────────────────────────────────────────

class TaramaYoneticisi:
    """Birden fazla coin'in tarama ilerlemesini yönetir."""

    def __init__(self, semboller: List[str]):
        self.semboller = semboller
        self.tamamlanan: List[str] = []
        self.hatalar: List[Tuple[str, str]] = []
        self.baslangic = time.time()

    def ilerleme(self) -> float:
        if not self.semboller:
            return 1.0
        return len(self.tamamlanan) / len(self.semboller)

    def gecen_sure(self) -> float:
        return time.time() - self.baslangic

    def tahmini_sure(self) -> Optional[float]:
        ilerleme = self.ilerleme()
        if ilerleme <= 0:
            return None
        return self.gecen_sure() / ilerleme * (1 - ilerleme)


def toplu_tarama(
    exchange: ccxt.Exchange,
    semboller: List[str],
    periyot: str,
    progress_placeholder=None,
) -> Dict[str, pd.DataFrame]:
    """
    Tüm coinleri tarar, Streamlit placeholder'a ilerleme bilgisi yazar.
    Döndürür: {symbol: DataFrame}
    """
    yonetici = TaramaYoneticisi(semboller)
    tum_data: Dict[str, pd.DataFrame] = {}

    for i, sym in enumerate(semboller):
        if progress_placeholder:
            try:
                progress_placeholder.text(
                    f"⏳ Taranıyor: {sym} ({i+1}/{len(semboller)}) "
                    f"— {yonetici.ilerleme()*100:.0f}%"
                )
            except Exception:
                pass

        ccxt_sym = f"{sym}/USDT:USDT"
        try:
            df = fetch_ohlcv_with_resample(exchange, ccxt_sym, periyot)
            if df is not None and not df.empty:
                tum_data[sym] = df
                yonetici.tamamlanan.append(sym)
            else:
                yonetici.hatalar.append((sym, "Veri boş"))
        except Exception as e:
            yonetici.hatalar.append((sym, str(e)))
            logger.error(f"[Tarama] {sym} hatası: {e}")

        time.sleep(REQUEST_DELAY)

    if progress_placeholder:
        try:
            progress_placeholder.text(
                f"✅ Tarama tamamlandı: {len(tum_data)}/{len(semboller)} coin"
            )
        except Exception:
            pass

    if yonetici.hatalar:
        logger.warning(f"[Tarama] Hatalı coinler: {[h[0] for h in yonetici.hatalar]}")

    return tum_data
