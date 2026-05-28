# archive_manager.py — SQLite Arşiv Yönetimi + APScheduler
# ─────────────────────────────────────────────────────────────────────────────

import os
import time
import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import pandas as pd
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler

from config import (
    DB_PATH, DB_PATH_LOCAL, ARCHIVE_DAYS, ARCHIVE_DELAY,
    BOLL_WINDOW, BOLL_STD, BOLL_NARROW_PCT, ATR_WINDOW,
    BTC_DUSUS_ESIK, AYRISMA_AY,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# VERİTABANI YOLU (Render persistent disk veya local)
# ──────────────────────────────────────────────

def get_db_path() -> str:
    # Render free plan'da /data yok, /tmp kullan; local'de ./archive.db
    if os.environ.get("RENDER"):
        return DB_PATH          # /tmp/archive.db
    return DB_PATH_LOCAL        # ./archive.db


def get_connection() -> sqlite3.Connection:
    path = get_db_path()
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


# ──────────────────────────────────────────────
# TABLO ŞEMALARI
# ──────────────────────────────────────────────

CREATE_OHLCV_TABLE = """
CREATE TABLE IF NOT EXISTS daily_ohlcv (
    symbol      TEXT    NOT NULL,
    ts          INTEGER NOT NULL,
    open        REAL    NOT NULL,
    high        REAL    NOT NULL,
    low         REAL    NOT NULL,
    close       REAL    NOT NULL,
    volume      REAL    NOT NULL,
    PRIMARY KEY (symbol, ts)
);
"""

CREATE_FUNDING_TABLE = """
CREATE TABLE IF NOT EXISTS funding_rates (
    symbol          TEXT    NOT NULL,
    funding_time    INTEGER NOT NULL,
    funding_rate    REAL    NOT NULL,
    PRIMARY KEY (symbol, funding_time)
);
"""

CREATE_META_TABLE = """
CREATE TABLE IF NOT EXISTS archive_meta (
    key     TEXT PRIMARY KEY,
    value   TEXT
);
"""

CREATE_VOL_ANLAR_TABLE = """
CREATE TABLE IF NOT EXISTS volatilite_anlari (
    symbol      TEXT    NOT NULL,
    ts          INTEGER NOT NULL,
    btc_return  REAL,
    alt_return  REAL,
    PRIMARY KEY (symbol, ts)
);
"""

CREATE_AYRISMA_TABLE = """
CREATE TABLE IF NOT EXISTS ayrisma_gucü (
    symbol      TEXT PRIMARY KEY,
    score       REAL,
    updated_at  INTEGER
);
"""


def init_db() -> None:
    """Tüm tabloları oluştur (yoksa)."""
    conn = get_connection()
    try:
        with conn:
            conn.execute(CREATE_OHLCV_TABLE)
            conn.execute(CREATE_FUNDING_TABLE)
            conn.execute(CREATE_META_TABLE)
            conn.execute(CREATE_VOL_ANLAR_TABLE)
            conn.execute(CREATE_AYRISMA_TABLE)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_sym_ts ON daily_ohlcv(symbol, ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_funding_sym  ON funding_rates(symbol, funding_time)")
    finally:
        conn.close()


# ──────────────────────────────────────────────
# META YARDIMCILAR
# ──────────────────────────────────────────────

def set_meta(key: str, value: str) -> None:
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO archive_meta(key, value) VALUES (?, ?)",
                (key, value),
            )
    finally:
        conn.close()


def get_meta(key: str, default: Optional[str] = None) -> Optional[str]:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT value FROM archive_meta WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else default
    finally:
        conn.close()


# ──────────────────────────────────────────────
# OHLCV KAYDETME / OKUMA
# ──────────────────────────────────────────────

def save_daily_ohlcv(symbol: str, df: pd.DataFrame) -> int:
    """
    df sütunları: ['ts', 'open', 'high', 'low', 'close', 'volume']
    ts: Unix ms cinsinden integer.
    Döndürür: eklenen satır sayısı.
    """
    if df.empty:
        return 0
    conn = get_connection()
    inserted = 0
    try:
        with conn:
            for _, row in df.iterrows():
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO daily_ohlcv
                           (symbol, ts, open, high, low, close, volume)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            symbol,
                            int(row["ts"]),
                            float(row["open"]),
                            float(row["high"]),
                            float(row["low"]),
                            float(row["close"]),
                            float(row["volume"]),
                        ),
                    )
                    inserted += 1
                except Exception as e:
                    logger.warning(f"OHLCV kaydetme hatası {symbol}: {e}")
    finally:
        conn.close()
    return inserted


def load_daily_ohlcv(symbol: str, days: int = ARCHIVE_DAYS) -> pd.DataFrame:
    """Son `days` günlük günlük mumları döndürür."""
    cutoff_ms = int(
        (datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000
    )
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            """SELECT ts, open, high, low, close, volume
               FROM daily_ohlcv
               WHERE symbol = ? AND ts >= ?
               ORDER BY ts ASC""",
            conn,
            params=(symbol, cutoff_ms),
        )
    finally:
        conn.close()
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df.set_index("datetime", inplace=True)
    return df


def load_all_symbols_ohlcv(days: int = ARCHIVE_DAYS) -> dict:
    """Tüm sembollerin günlük OHLCV'sini dict olarak döndürür {symbol: df}."""
    conn = get_connection()
    try:
        symbols = [
            r[0]
            for r in conn.execute("SELECT DISTINCT symbol FROM daily_ohlcv").fetchall()
        ]
    finally:
        conn.close()
    result = {}
    for sym in symbols:
        result[sym] = load_daily_ohlcv(sym, days)
    return result


def get_last_ts(symbol: str) -> Optional[int]:
    """Sembole ait son kayıtlı timestamp'i döndürür (ms)."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT MAX(ts) FROM daily_ohlcv WHERE symbol = ?", (symbol,)
        ).fetchone()
        return row[0] if row and row[0] is not None else None
    finally:
        conn.close()


# ──────────────────────────────────────────────
# FUNDING RATE KAYDETME / OKUMA
# ──────────────────────────────────────────────

def save_funding_rates(symbol: str, records: list) -> None:
    """
    records: [{"fundingTime": int_ms, "fundingRate": str/float}, ...]
    """
    conn = get_connection()
    try:
        with conn:
            for rec in records:
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO funding_rates
                           (symbol, funding_time, funding_rate)
                           VALUES (?, ?, ?)""",
                        (
                            symbol,
                            int(rec["fundingTime"]),
                            float(rec["fundingRate"]),
                        ),
                    )
                except Exception as e:
                    logger.warning(f"Funding kaydetme hatası {symbol}: {e}")
    finally:
        conn.close()


def load_funding_rates(symbol: str, limit: int = 100) -> pd.DataFrame:
    """Son `limit` adet funding rate kaydını döndürür."""
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            """SELECT funding_time, funding_rate
               FROM funding_rates
               WHERE symbol = ?
               ORDER BY funding_time DESC
               LIMIT ?""",
            conn,
            params=(symbol, limit),
        )
    finally:
        conn.close()
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["funding_time"], unit="ms", utc=True)
    return df


def get_latest_funding_rate(symbol: str) -> Optional[float]:
    """Sembole ait en son funding rate'i döndürür."""
    conn = get_connection()
    try:
        row = conn.execute(
            """SELECT funding_rate FROM funding_rates
               WHERE symbol = ?
               ORDER BY funding_time DESC LIMIT 1""",
            (symbol,),
        ).fetchone()
        return float(row[0]) if row else None
    finally:
        conn.close()


# ──────────────────────────────────────────────
# VOLATİLİTE ANLARI HESAPLAMA (Günlük Mumlardan)
# ──────────────────────────────────────────────

def hesapla_volatilite_anlari(btc_df: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Bitcoin günlük mumlarından volatilite anlarını tespit eder.
    Kural: BB Genişliği en dar %BOLL_NARROW_PCT içinde VE
           ardından ATR > 14 günlük ATR ortalamasına çıkan günler.
    Döndürür: volatilite anı olan datetime indeksleri.
    """
    if btc_df is None or len(btc_df) < max(BOLL_WINDOW, ATR_WINDOW) + 5:
        return pd.DatetimeIndex([])

    df = btc_df.copy()
    df["close"] = df["close"].astype(float)
    df["high"]  = df["high"].astype(float)
    df["low"]   = df["low"].astype(float)

    # Bollinger Bant Genişliği
    rolling_mean = df["close"].rolling(BOLL_WINDOW).mean()
    rolling_std  = df["close"].rolling(BOLL_WINDOW).std()
    bband_width  = (rolling_std * BOLL_STD * 2) / rolling_mean.replace(0, np.nan)

    # ATR
    high_low  = df["high"] - df["low"]
    high_prev = (df["high"] - df["close"].shift(1)).abs()
    low_prev  = (df["low"]  - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    atr = true_range.rolling(ATR_WINDOW).mean()
    atr_ortalama = atr.rolling(ATR_WINDOW * 3).mean()

    narrow_threshold = bband_width.quantile(BOLL_NARROW_PCT / 100)
    dar_gunler = bband_width < narrow_threshold

    # Sıkışma sonrası ATR patlaması: önceki gün dar + bugün ATR > ortalama
    patlama = dar_gunler.shift(1).fillna(False) & (atr > atr_ortalama.fillna(0))
    return df.index[patlama]


def guncelle_volatilite_anlari(semboller: List[str]) -> None:
    """Tüm sembollerin volatilite anlarını SQLite'a yazar."""
    btc_data = load_daily_ohlcv("BTC", ARCHIVE_DAYS)
    if btc_data.empty:
        logger.warning("BTC arşiv verisi yok, volatilite anları hesaplanamadı")
        return

    vol_indeks = hesapla_volatilite_anlari(btc_data)
    if len(vol_indeks) == 0:
        return

    btc_close = btc_data["close"].reindex(vol_indeks)
    btc_prev   = btc_data["close"].shift(1).reindex(vol_indeks)
    btc_return = ((btc_close - btc_prev) / btc_prev.replace(0, np.nan)).fillna(0)

    conn = get_connection()
    try:
        with conn:
            for sym in semboller:
                alt_df = load_daily_ohlcv(sym, ARCHIVE_DAYS)
                if alt_df.empty:
                    continue
                alt_close  = alt_df["close"].reindex(vol_indeks)
                alt_prev   = alt_df["close"].shift(1).reindex(vol_indeks)
                alt_return = ((alt_close - alt_prev) / alt_prev.replace(0, np.nan)).fillna(0)
                for dt in vol_indeks:
                    try:
                        ts_ms = int(dt.timestamp() * 1000)
                        conn.execute(
                            """INSERT OR REPLACE INTO volatilite_anlari
                               (symbol, ts, btc_return, alt_return)
                               VALUES (?, ?, ?, ?)""",
                            (
                                sym,
                                ts_ms,
                                float(btc_return.get(dt, 0)),
                                float(alt_return.get(dt, 0)) if not pd.isna(alt_return.get(dt, np.nan)) else None,
                            ),
                        )
                    except Exception as e:
                        logger.warning(f"volatilite_anlari yazma hatası {sym}: {e}")
    finally:
        conn.close()


def load_volatilite_anlari(symbol: str) -> pd.DataFrame:
    """Sembol için volatilite anları verisini döndürür."""
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            """SELECT ts, btc_return, alt_return FROM volatilite_anlari
               WHERE symbol = ? ORDER BY ts ASC""",
            conn,
            params=(symbol,),
        )
    finally:
        conn.close()
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


# ──────────────────────────────────────────────
# 6 AYLIK AYRIŞMA GÜCÜ
# ──────────────────────────────────────────────

def hesapla_ayrisma_gucu(semboller: List[str]) -> None:
    """
    Son 6 ayda BTC %2+ düştüğü günlerde pozitif kapanan altcoin oranını
    hesaplayıp ayrisma_gucü tablosuna yazar.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=AYRISMA_AY * 30)
    btc_df = load_daily_ohlcv("BTC", AYRISMA_AY * 31)
    if btc_df.empty:
        return

    btc_df = btc_df[btc_df.index >= cutoff]
    btc_df["ret"] = btc_df["close"].pct_change()
    btc_dusus_gunler = btc_df[btc_df["ret"] <= BTC_DUSUS_ESIK].index  # -%2 ve altı

    if len(btc_dusus_gunler) == 0:
        return

    conn = get_connection()
    now_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    try:
        with conn:
            for sym in semboller:
                alt_df = load_daily_ohlcv(sym, AYRISMA_AY * 31)
                if alt_df.empty:
                    score = 0.0
                else:
                    alt_df = alt_df[alt_df.index >= cutoff]
                    alt_df["ret"] = alt_df["close"].pct_change()
                    common = btc_dusus_gunler.intersection(alt_df.index)
                    if len(common) == 0:
                        score = 0.0
                    else:
                        pozitif = (alt_df.loc[common, "ret"] > 0).sum()
                        score = float(pozitif / len(common))
                conn.execute(
                    """INSERT OR REPLACE INTO ayrisma_gucü
                       (symbol, score, updated_at) VALUES (?, ?, ?)""",
                    (sym, score, now_ts),
                )
    finally:
        conn.close()


def load_ayrisma_gucu_all() -> dict:
    """Tüm sembollerin ayrışma gücü skorlarını {symbol: float} döndürür."""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT symbol, score FROM ayrisma_gucü").fetchall()
    finally:
        conn.close()
    return {r[0]: r[1] for r in rows}


# ──────────────────────────────────────────────
# İLK KURULUM: 1 YILLIK ARŞİV ÇEKME
# ──────────────────────────────────────────────

def ilk_kurulum_arsiv(exchange, semboller: List[str], progress_cb=None) -> None:
    """
    İlk kurulumda 1 yıllık günlük OHLCV verisini Binance'ten çeker ve kaydeder.
    exchange: ccxt.binanceusdm instance
    progress_cb: opsiyonel callback(i, total, symbol) ilerleme için
    """
    since_ms = int(
        (datetime.now(timezone.utc) - timedelta(days=ARCHIVE_DAYS + 5)).timestamp() * 1000
    )
    total = len(semboller)
    for i, sym in enumerate(semboller):
        if progress_cb:
            progress_cb(i, total, sym)
        try:
            mevcut_son = get_last_ts(sym)
            if mevcut_son and mevcut_son > since_ms:
                logger.info(f"[Arşiv] {sym} zaten güncel, atlanıyor")
                time.sleep(ARCHIVE_DELAY * 0.5)
                continue
            ohlcv = _fetch_ohlcv_safe(exchange, f"{sym}/USDT:USDT", "1d", since_ms)
            if not ohlcv:
                continue
            df = _ohlcv_to_df(ohlcv)
            saved = save_daily_ohlcv(sym, df)
            logger.info(f"[Arşiv] {sym}: {saved} satır kaydedildi")
        except Exception as e:
            logger.error(f"[Arşiv] {sym} hatası: {e}")
        finally:
            time.sleep(ARCHIVE_DELAY)

    set_meta("arsiv_kurulum_tarihi", datetime.now(timezone.utc).isoformat())
    guncelle_volatilite_anlari(semboller)
    hesapla_ayrisma_gucu(semboller)


def gunluk_guncelleme(exchange, semboller: List[str]) -> None:
    """
    APScheduler tarafından her sabah 06:00 UTC'de çağrılır.
    Sadece son günün verisini ekler.
    """
    since_ms = int(
        (datetime.now(timezone.utc) - timedelta(days=3)).timestamp() * 1000
    )
    for sym in semboller:
        try:
            ohlcv = _fetch_ohlcv_safe(exchange, f"{sym}/USDT:USDT", "1d", since_ms)
            if not ohlcv:
                continue
            df = _ohlcv_to_df(ohlcv)
            save_daily_ohlcv(sym, df)
        except Exception as e:
            logger.error(f"[Günlük Güncelleme] {sym} hatası: {e}")
        finally:
            time.sleep(ARCHIVE_DELAY)

    guncelle_volatilite_anlari(semboller)
    hesapla_ayrisma_gucu(semboller)
    set_meta("son_guncelleme", datetime.now(timezone.utc).isoformat())
    logger.info("[APScheduler] Günlük güncelleme tamamlandı")


# ──────────────────────────────────────────────
# YARDIMCI: ccxt OHLCV → DataFrame
# ──────────────────────────────────────────────

def _ohlcv_to_df(ohlcv: list) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.dropna()
    return df


def _fetch_ohlcv_safe(exchange, symbol: str, timeframe: str, since: int) -> list:
    """Rate limit korumalı OHLCV çekimi."""
    for attempt in range(3):
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
            return data
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "418" in err_str or "rate limit" in err_str:
                wait = [5, 15, 45][attempt]
                logger.warning(f"Rate limit ({symbol}), {wait}s bekleniyor...")
                time.sleep(wait)
            else:
                logger.error(f"_fetch_ohlcv_safe hatası ({symbol}): {e}")
                return []
    logger.error(f"_fetch_ohlcv_safe: {symbol} 3 denemede başarısız")
    return []


# ──────────────────────────────────────────────
# APSCHEDULer KURULUMU
# ──────────────────────────────────────────────

_scheduler: Optional[BackgroundScheduler] = None


def start_scheduler(exchange, semboller: List[str]) -> None:
    """APScheduler ile sabah 06:00 UTC günlük güncelleme zamanlar."""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        return

    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_job(
        func=gunluk_guncelleme,
        args=[exchange, semboller],
        trigger="cron",
        hour=6,
        minute=0,
        id="gunluk_guncelleme",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info("[APScheduler] Günlük güncelleme zamanlayıcısı başlatıldı (06:00 UTC)")


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("[APScheduler] Durduruldu")


# ──────────────────────────────────────────────
# ARŞİV DURUM KONTROLÜ
# ──────────────────────────────────────────────

def arsiv_hazir_mi() -> bool:
    """Arşiv kurulumunun tamamlanıp tamamlanmadığını kontrol eder."""
    return get_meta("arsiv_kurulum_tarihi") is not None


def arsiv_ozet() -> dict:
    """Arşiv durumu hakkında özet bilgi döndürür."""
    conn = get_connection()
    try:
        sembol_sayisi = conn.execute(
            "SELECT COUNT(DISTINCT symbol) FROM daily_ohlcv"
        ).fetchone()[0]
        satir_sayisi = conn.execute("SELECT COUNT(*) FROM daily_ohlcv").fetchone()[0]
    finally:
        conn.close()
    return {
        "sembol_sayisi": sembol_sayisi,
        "satir_sayisi":  satir_sayisi,
        "kurulum_tarihi": get_meta("arsiv_kurulum_tarihi", "Henüz kurulmadı"),
        "son_guncelleme": get_meta("son_guncelleme", "—"),
    }
