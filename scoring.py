# scoring.py — Güç Skoru, Beta, Döngü, Markov Algoritmaları
# ─────────────────────────────────────────────────────────────────────────────

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from config import (
    DEFAULT_WEIGHT_RECENT, DEFAULT_WEIGHT_MID, DEFAULT_WEIGHT_OLD,
    GUCLU_TOLERANS, EZIK_TOLERANS,
    DONGÜ_BOYUT, DONGÜ_GUCLU, DONGÜ_CURUK, DONGÜ_SAGIR,
    HAFIZA_GUNLER,
)
from archive_manager import load_daily_ohlcv, load_volatilite_anlari

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# A. GÜÇ SKORU
# ──────────────────────────────────────────────

def hesapla_guc_skoru(
    btc_df: pd.DataFrame,
    alt_dfs: Dict[str, pd.DataFrame],
    weight_recent: float = DEFAULT_WEIGHT_RECENT,
    weight_mid:    float = DEFAULT_WEIGHT_MID,
    weight_old:    float = DEFAULT_WEIGHT_OLD,
) -> Dict[str, float]:
    """
    Tüm altcoinler için ağırlıklı güç skorunu hesaplar.
    Döndürür: {symbol: normalized_score_0_to_100}
    """
    if btc_df is None or btc_df.empty or not alt_dfs:
        return {}

    btc_ret = btc_df["close"].pct_change().fillna(0)

    # Tüm altcoin getirilerini aynı index'e hizala
    all_rets: Dict[str, pd.Series] = {}
    for sym, df in alt_dfs.items():
        if df is not None and not df.empty:
            ret = df["close"].pct_change().fillna(0)
            ret = ret.reindex(btc_ret.index).fillna(0)
            all_rets[sym] = ret

    if not all_rets:
        return {}

    # Her adımda tüm altcoin ortalaması
    all_rets_df = pd.DataFrame(all_rets)
    market_avg  = all_rets_df.mean(axis=1)

    N = len(btc_ret)
    if N < 3:
        return {}

    # 3 dilime böl
    k1 = N // 3
    k2 = 2 * (N // 3)
    old_idx    = btc_ret.index[:k1]
    mid_idx    = btc_ret.index[k1:k2]
    recent_idx = btc_ret.index[k2:]

    raw_scores: Dict[str, float] = {}
    for sym, alt_ret in all_rets.items():
        puan_eski  = _dilim_puani(alt_ret, market_avg, old_idx)
        puan_orta  = _dilim_puani(alt_ret, market_avg, mid_idx)
        puan_yakin = _dilim_puani(alt_ret, market_avg, recent_idx)
        toplam = (
            puan_eski  * weight_old
            + puan_orta  * weight_mid
            + puan_yakin * weight_recent
        )
        raw_scores[sym] = toplam

    return _normalize_scores(raw_scores)


def _dilim_puani(
    alt_ret: pd.Series,
    market_avg: pd.Series,
    idx: pd.Index,
) -> float:
    """Belirli indeks dilimindeki ham puanı döndürür."""
    if len(idx) == 0:
        return 0.0
    a = alt_ret.reindex(idx).fillna(0)
    m = market_avg.reindex(idx).fillna(0)
    total = 0.0
    for i in range(len(idx)):
        alt_v = float(a.iloc[i])
        mkt_v = float(m.iloc[i])
        if alt_v > mkt_v + GUCLU_TOLERANS:
            total += 1.0
        elif alt_v < mkt_v + EZIK_TOLERANS:
            total -= 1.0
        # else: nötr, 0 puan
    return total


def _normalize_scores(raw: Dict[str, float]) -> Dict[str, float]:
    """Ham puanları 0-100 arasına normalize eder."""
    if not raw:
        return {}
    values = np.array(list(raw.values()), dtype=float)
    vmin, vmax = values.min(), values.max()
    if vmax == vmin:
        return {s: 50.0 for s in raw}
    normalized = (values - vmin) / (vmax - vmin) * 100
    return {sym: float(normalized[i]) for i, sym in enumerate(raw.keys())}


# ──────────────────────────────────────────────
# B. DÖNGÜ SINIFLANDIRMASI
# ──────────────────────────────────────────────

def siniflandir_donguler(
    skorlar: Dict[str, float],
) -> Dict[str, int]:
    """
    Güç skorlarına göre döngüleri sınıflandırır.
    Döndürür: {symbol: 1|2|3|0}
      1=Güçlü, 2=Çürük, 3=Sağır, 0=Diğer
    """
    if not skorlar:
        return {}

    sorted_syms = sorted(skorlar.keys(), key=lambda s: skorlar[s], reverse=True)
    n = len(sorted_syms)
    boyut = min(DONGÜ_BOYUT, n // 3)

    top15    = set(sorted_syms[:boyut])
    bottom15 = set(sorted_syms[max(0, n - boyut):])

    # Sağır 15: piyasa ortalamasına en yakın (fark en küçük)
    ortalama = np.mean(list(skorlar.values()))
    remaining = [s for s in sorted_syms if s not in top15 and s not in bottom15]
    remaining.sort(key=lambda s: abs(skorlar[s] - ortalama))
    sagir15 = set(remaining[:boyut])

    sonuc: Dict[str, int] = {}
    for sym in skorlar:
        if sym in top15:
            sonuc[sym] = DONGÜ_GUCLU
        elif sym in bottom15:
            sonuc[sym] = DONGÜ_CURUK
        elif sym in sagir15:
            sonuc[sym] = DONGÜ_SAGIR
        else:
            sonuc[sym] = 0
    return sonuc


# ──────────────────────────────────────────────
# C. BETA HESABI (Lineer Regresyon)
# ──────────────────────────────────────────────

def hesapla_beta(
    btc_df: pd.DataFrame,
    alt_df: pd.DataFrame,
) -> Optional[float]:
    """
    numpy.polyfit ile lineer regresyon betası.
    BTC getirisi → Alt getirisi.
    Döndürür: beta (float) veya None.
    """
    if btc_df is None or alt_df is None or btc_df.empty or alt_df.empty:
        return None

    btc_ret = btc_df["close"].pct_change().dropna()
    alt_ret = alt_df["close"].pct_change().dropna()

    common = btc_ret.index.intersection(alt_ret.index)
    if len(common) < 5:
        return None

    x = btc_ret.loc[common].values.astype(float)
    y = alt_ret.loc[common].values.astype(float)

    # NaN/Inf koruması
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return None

    std_x = np.std(x)
    if std_x == 0:
        return None

    try:
        coeffs = np.polyfit(x, y, 1)
        return float(coeffs[0])
    except (np.linalg.LinAlgError, ValueError) as e:
        logger.warning(f"Beta hesaplama hatası: {e}")
        return None


def hesapla_volatilite_betasi(symbol: str, btc_df: pd.DataFrame) -> Optional[float]:
    """
    Arşivden yüklenen volatilite anlarındaki veri noktalarını kullanarak beta hesaplar.
    Kullanıcı periyodundan bağımsız, her zaman günlük mumlardan.
    """
    vol_df = load_volatilite_anlari(symbol)
    if vol_df is None or vol_df.empty:
        return None

    x = vol_df["btc_return"].values.astype(float)
    y = vol_df["alt_return"].values.astype(float)

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return None

    std_x = np.std(x)
    if std_x == 0:
        return None

    try:
        coeffs = np.polyfit(x, y, 1)
        return float(coeffs[0])
    except (np.linalg.LinAlgError, ValueError) as e:
        logger.warning(f"Volatilite betası hatası {symbol}: {e}")
        return None


def tum_betalar(
    btc_df: pd.DataFrame,
    alt_dfs: Dict[str, pd.DataFrame],
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Tüm altcoinler için normal ve volatilite betalarını döndürür.
    {symbol: {"normal": float|None, "volatilite": float|None}}
    """
    result = {}
    for sym, df in alt_dfs.items():
        normal_beta = hesapla_beta(btc_df, df)
        vol_beta    = hesapla_volatilite_betasi(sym, btc_df)
        result[sym] = {"normal": normal_beta, "volatilite": vol_beta}
    return result


# ──────────────────────────────────────────────
# D. KARAKTER GÜVENİLİRLİĞİ
# ──────────────────────────────────────────────

def hesapla_karakter_guvenilirlik(
    symbol: str,
    dongü: int,
    btc_df: pd.DataFrame,
    alt_df: pd.DataFrame,
) -> Dict[str, object]:
    """
    Coin'in piyasa ortalamasına göre karakter güvenilirliğini hesaplar.
    Hem normal hem volatilite anları için ayrı ayrı.
    Döndürür dict:
      normal_oran:      float (0-1)
      normal_toplam:    int
      normal_korudu:    int
      vol_oran:         float (0-1)
      vol_toplam:       int
      vol_korudu:       int
      vol_detay:        list of dict (volatilite anı bazlı)
    """
    bos_sonuc = {
        "normal_oran": 0.0, "normal_toplam": 0, "normal_korudu": 0,
        "vol_oran":    0.0, "vol_toplam":    0, "vol_korudu":    0,
        "vol_detay":   [],
    }
    if btc_df is None or alt_df is None or btc_df.empty or alt_df.empty:
        return bos_sonuc

    # Günlük OHLCV'yi arşivden yükle (volatilite hesabı için)
    alt_daily = load_daily_ohlcv(symbol, 365)
    btc_daily = load_daily_ohlcv("BTC", 365)

    if alt_daily.empty or btc_daily.empty:
        return bos_sonuc

    # Günlük getiriler + piyasa ortalaması (tüm sembollerin ortalamasını simüle etmek yerine
    # BTC'yi referans alırız çünkü diğer sembollerin günlük verisi burada yok)
    btc_ret = btc_daily["close"].pct_change().dropna()
    alt_ret = alt_daily["close"].pct_change().dropna()
    common  = btc_ret.index.intersection(alt_ret.index)

    if len(common) < 5:
        return bos_sonuc

    btc_common = btc_ret.loc[common]
    alt_common = alt_ret.loc[common]

    # Volatilite anlarını yükle
    vol_df = load_volatilite_anlari(symbol)
    vol_datetimes = set()
    if vol_df is not None and not vol_df.empty:
        vol_datetimes = set(vol_df["datetime"].dt.normalize().tolist()) if "datetime" in vol_df.columns else set()

    # Normal anlar: volatilite anı olmayan günler
    def _is_vol(dt) -> bool:
        return dt.normalize() in vol_datetimes

    # Döngü 1 için: piyasanın üstünde → korudu
    # Döngü 2 için: piyasanın altında → korudu
    def _korudu(btc_v: float, alt_v: float, d: int) -> bool:
        if d == DONGÜ_GUCLU:
            return alt_v > btc_v + GUCLU_TOLERANS
        elif d == DONGÜ_CURUK:
            return alt_v < btc_v + EZIK_TOLERANS
        else:
            return abs(alt_v - btc_v) < 0.01  # Sağır: yakın kaldı

    normal_korudu = 0
    normal_toplam = 0
    vol_korudu    = 0
    vol_toplam    = 0
    vol_detay     = []

    for dt in common:
        bv = float(btc_common.loc[dt])
        av = float(alt_common.loc[dt])
        if _is_vol(dt):
            vol_toplam += 1
            korudu = _korudu(bv, av, dongü)
            if korudu:
                vol_korudu += 1
            vol_detay.append({
                "tarih":    dt,
                "btc_ret":  bv,
                "alt_ret":  av,
                "beta":     round(av / bv, 3) if abs(bv) > 1e-8 else None,
                "korudu":   korudu,
            })
        else:
            normal_toplam += 1
            if _korudu(bv, av, dongü):
                normal_korudu += 1

    return {
        "normal_oran":   normal_korudu / normal_toplam if normal_toplam > 0 else 0.0,
        "normal_toplam": normal_toplam,
        "normal_korudu": normal_korudu,
        "vol_oran":      vol_korudu / vol_toplam if vol_toplam > 0 else 0.0,
        "vol_toplam":    vol_toplam,
        "vol_korudu":    vol_korudu,
        "vol_detay":     vol_detay[-50:],  # Son 50 volatilite anı
    }


# ──────────────────────────────────────────────
# E. MARKOV ZİNCİRİ VE DÖNGÜ GEÇMIŞ
# ──────────────────────────────────────────────

def _dakika_coz(hafiza_str: str) -> int:
    """Hafıza penceresi string'ini toplam dakikaya çevirir."""
    gun = HAFIZA_GUNLER.get(hafiza_str, 1)
    return gun * 24 * 60


def dongü_gecmis_hesapla(
    symbol: str,
    btc_df: pd.DataFrame,
    alt_df: pd.DataFrame,
    hafiza_penceresi: str,
    weight_recent: float = DEFAULT_WEIGHT_RECENT,
    weight_mid:    float = DEFAULT_WEIGHT_MID,
    weight_old:    float = DEFAULT_WEIGHT_OLD,
    tum_alt_dfs:   Optional[Dict[str, pd.DataFrame]] = None,
) -> List[int]:
    """
    Seçilen hafıza penceresindeki her zaman diliminde coinin hangi döngüde
    olduğunu hesaplayıp liste olarak döndürür.
    [DONGÜ_GUCLU|DONGÜ_CURUK|DONGÜ_SAGIR|0, ...]
    """
    if btc_df is None or alt_df is None or btc_df.empty or alt_df.empty:
        return []

    gun = HAFIZA_GUNLER.get(hafiza_penceresi, 1)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=gun)
    btc_window = btc_df[btc_df.index >= cutoff]
    alt_window = alt_df[alt_df.index >= cutoff]

    if tum_alt_dfs is None:
        # Yalnız coin değerlendirmesi — market avg = BTC
        tum_alt_dfs_window = {symbol: alt_window}
        btc_window_ref = btc_window
    else:
        tum_alt_dfs_window = {
            s: d[d.index >= cutoff] for s, d in tum_alt_dfs.items()
        }
        btc_window_ref = btc_window

    # Her zaman diliminde kayan pencere skoru
    N = len(btc_window_ref)
    if N < 6:
        return []

    geçmiş = []
    pencere = max(6, N // 10)

    for i in range(pencere, N + 1):
        slice_btc  = btc_window_ref.iloc[max(0, i - pencere):i]
        slice_alts = {s: d.iloc[max(0, i - pencere):i] for s, d in tum_alt_dfs_window.items()}

        skorlar = hesapla_guc_skoru(
            slice_btc, slice_alts,
            weight_recent=weight_recent,
            weight_mid=weight_mid,
            weight_old=weight_old,
        )
        donguler = siniflandir_donguler(skorlar)
        geçmiş.append(donguler.get(symbol, 0))

    return geçmiş


def markov_hesapla(gecmis: List[int]) -> Dict[str, object]:
    """
    Döngü geçmiş listesinden 3x3 Markov geçiş matrisini hesaplar.
    Döndürür:
      matris:              {from_state: {to_state: prob}}
      ortalama_sure:       {state: float} (ortalama döngüde kalma adımı)
      gecis_sayilari:      dict
      en_cok_gecilen:      int (en sık döngü)
      en_cok_oran:         float
    """
    dongu_durumlari = [DONGÜ_GUCLU, DONGÜ_CURUK, DONGÜ_SAGIR]
    gecis = {d: {dd: 0 for dd in dongu_durumlari} for d in dongu_durumlari}
    kalma = {d: [] for d in dongu_durumlari}

    mevcut_durum   = None
    mevcut_sure    = 0

    for d in gecmis:
        if d not in dongu_durumlari:
            mevcut_durum = None
            mevcut_sure  = 0
            continue

        if mevcut_durum is None:
            mevcut_durum = d
            mevcut_sure  = 1
        elif d == mevcut_durum:
            mevcut_sure += 1
        else:
            kalma[mevcut_durum].append(mevcut_sure)
            if mevcut_durum in dongu_durumlari and d in dongu_durumlari:
                gecis[mevcut_durum][d] += 1
            mevcut_durum = d
            mevcut_sure  = 1

    if mevcut_durum in dongu_durumlari and mevcut_sure > 0:
        kalma[mevcut_durum].append(mevcut_sure)

    # Olasılık matrisi
    matris = {}
    for d in dongu_durumlari:
        toplam = sum(gecis[d].values())
        if toplam > 0:
            matris[d] = {dd: gecis[d][dd] / toplam for dd in dongu_durumlari}
        else:
            matris[d] = {dd: 1 / 3 for dd in dongu_durumlari}

    # Ortalama kalma süresi
    ort_sure = {
        d: float(np.mean(kalma[d])) if kalma[d] else 1.0
        for d in dongu_durumlari
    }

    # En çok geçilen döngü
    freqs = {d: gecmis.count(d) for d in dongu_durumlari}
    en_cok = max(freqs, key=lambda d: freqs[d]) if any(freqs.values()) else DONGÜ_SAGIR
    toplam_valid = sum(1 for x in gecmis if x in dongu_durumlari)
    en_cok_oran = freqs[en_cok] / toplam_valid if toplam_valid > 0 else 0.0

    return {
        "matris":        matris,
        "ortalama_sure": ort_sure,
        "gecis_sayilari": gecis,
        "en_cok_gecilen": en_cok,
        "en_cok_oran":    en_cok_oran,
        "kalma_listesi":  kalma,
    }


def mevcut_dongü_bilgisi(
    gecmis: List[int],
    zaman_adimi_dk: float = 60.0,
) -> Dict[str, object]:
    """
    Şu an hangi döngüde kaç adırdır olduğunu döndürür.
    zaman_adimi_dk: Her adımın kaç dakikaya karşılık geldiği.
    Döndürür:
      mevcut_dongü:   int
      sur_adim:        int
      sure_saat:       float
      tahmin_kalan:    float (saat)
    """
    dongu_durumlari = [DONGÜ_GUCLU, DONGÜ_CURUK, DONGÜ_SAGIR]
    if not gecmis:
        return {"mevcut_dongü": 0, "sur_adim": 0, "sure_saat": 0.0, "tahmin_kalan": 0.0}

    son_durum = gecmis[-1]
    if son_durum not in dongu_durumlari:
        return {"mevcut_dongü": 0, "sur_adim": 0, "sure_saat": 0.0, "tahmin_kalan": 0.0}

    # Geriye doğru kaç adırdır aynı döngüde
    adim = 0
    for i in reversed(gecmis):
        if i == son_durum:
            adim += 1
        else:
            break

    sure_saat = adim * zaman_adimi_dk / 60.0

    # Tahmin: ortalama kalma süresi - şimdiye kadar geçen süre
    markov = markov_hesapla(gecmis)
    ort_sure_adim = markov["ortalama_sure"].get(son_durum, 1.0)
    kalan_adim = max(0.0, ort_sure_adim - adim)
    tahmin_kalan_saat = kalan_adim * zaman_adimi_dk / 60.0

    return {
        "mevcut_dongü":  son_durum,
        "sur_adim":       adim,
        "sure_saat":      sure_saat,
        "tahmin_kalan":   tahmin_kalan_saat,
        "markov":         markov,
    }


# ──────────────────────────────────────────────
# F. ISI HARİTASI İÇİN SINIFLANDIRMA
# ──────────────────────────────────────────────

def isi_haritasi_renkleri(
    btc_df: pd.DataFrame,
    alt_df: pd.DataFrame,
    adim_dk: int,
) -> pd.DataFrame:
    """
    Coin için ısı haritası verisi üretir.
    Her satır: zaman dilimi, renk kodu (1=baskın, -1=ezik, 0=nötr)
    """
    if btc_df is None or alt_df is None or btc_df.empty or alt_df.empty:
        return pd.DataFrame(columns=["datetime", "deger"])

    adim_str = f"{adim_dk}min"
    btc_r = btc_df.resample(adim_str).agg({"close": "last"}).dropna()
    alt_r = alt_df.resample(adim_str).agg({"close": "last"}).dropna()

    btc_ret = btc_r["close"].pct_change().fillna(0)
    alt_ret = alt_r["close"].pct_change().fillna(0)

    common = btc_ret.index.intersection(alt_ret.index)
    if len(common) == 0:
        return pd.DataFrame(columns=["datetime", "deger"])

    btc_c = btc_ret.loc[common]
    alt_c = alt_ret.loc[common]

    # Market avg = BTC (basitleştirilmiş, burada salt BTC)
    degerler = []
    for dt in common:
        bv = float(btc_c.loc[dt])
        av = float(alt_c.loc[dt])
        if av > bv + GUCLU_TOLERANS:
            degerler.append(1)
        elif av < bv + EZIK_TOLERANS:
            degerler.append(-1)
        else:
            degerler.append(0)

    result = pd.DataFrame({"datetime": list(common), "deger": degerler})
    result.set_index("datetime", inplace=True)
    return result


# ──────────────────────────────────────────────
# G. TÜM COİNLER İÇİN TOPLU PUANLAMA
# ──────────────────────────────────────────────

def tam_puanlama(
    btc_df: pd.DataFrame,
    alt_dfs: Dict[str, pd.DataFrame],
    weight_recent: float = DEFAULT_WEIGHT_RECENT,
    weight_mid:    float = DEFAULT_WEIGHT_MID,
    weight_old:    float = DEFAULT_WEIGHT_OLD,
) -> Dict[str, Dict]:
    """
    Tüm coinler için güç skoru + döngü + beta hesaplar.
    Döndürür:
    {
      symbol: {
        "skor":     float,
        "dongü":    int,
        "beta_n":   float|None,
        "beta_v":   float|None,
      }
    }
    """
    if btc_df is None or btc_df.empty:
        return {}

    skorlar  = hesapla_guc_skoru(btc_df, alt_dfs, weight_recent, weight_mid, weight_old)
    donguler = siniflandir_donguler(skorlar)
    betalar  = tum_betalar(btc_df, alt_dfs)

    result = {}
    for sym in skorlar:
        result[sym] = {
            "skor":   skorlar.get(sym, 50.0),
            "dongü":  donguler.get(sym, 0),
            "beta_n": betalar.get(sym, {}).get("normal"),
            "beta_v": betalar.get(sym, {}).get("volatilite"),
        }
    return result


# ──────────────────────────────────────────────
# H. MAKAS SKORU YARDIMCILARI
# ──────────────────────────────────────────────

def makas_skoru_hesapla(
    long_sym:   str,
    short_sym:  str,
    puanlama:   Dict[str, Dict],
    btc_panel_sinyal: bool,
    hafiza_penceresi: str,
    btc_df:     pd.DataFrame,
    alt_dfs:    Dict[str, pd.DataFrame],
    weight_recent: float = DEFAULT_WEIGHT_RECENT,
    weight_mid:    float = DEFAULT_WEIGHT_MID,
    weight_old:    float = DEFAULT_WEIGHT_OLD,
) -> Tuple[float, Dict]:
    """
    Makas skoru bileşenlerini hesaplar (0-100).
    Döndürür: (toplam_skor, detay_dict)
    """
    from config import MAKAS_AGIRLIK, TAKER_RATE

    detay = {}

    # 1. Volatilite Beta Farkı (%30)
    beta_v_long  = puanlama.get(long_sym,  {}).get("beta_v")
    beta_v_short = puanlama.get(short_sym, {}).get("beta_v")

    if beta_v_long is not None and beta_v_short is not None and beta_v_short != 0:
        beta_fark = abs(beta_v_long - beta_v_short)
        beta_n_l  = puanlama.get(long_sym,  {}).get("beta_n")
        beta_n_s  = puanlama.get(short_sym, {}).get("beta_n")
        beta_n_fark = abs(beta_n_l - beta_n_s) if (beta_n_l and beta_n_s) else 0.0
        puan_beta = min(100.0, beta_fark * 50)  # 2x fark = 100 puan
        detay["volatilite_beta_farki"] = {
            "normal_fark":      round(beta_n_fark, 3),
            "volatilite_fark":  round(beta_fark, 3),
            "puan":             round(puan_beta, 1),
            "gecti":            beta_fark >= 0.5,
        }
    else:
        puan_beta = 0.0
        detay["volatilite_beta_farki"] = {"gecti": False, "puan": 0}

    # 2. Karakter Güvenilirliği (%25)
    # Arşiv verisiyle hesaplama pahalı, burada puanlama dict'ten tahmin
    # Gerçek hesaplama main'de yapılır, buraya oran geçilir
    long_dongü  = puanlama.get(long_sym,  {}).get("dongü", 0)
    short_dongü = puanlama.get(short_sym, {}).get("dongü", 0)
    # Basit yaklaşım: skor oranı ile güvenilirlik tahmini
    long_skor  = puanlama.get(long_sym,  {}).get("skor", 50.0)
    short_skor = puanlama.get(short_sym, {}).get("skor", 50.0)
    guvn_long  = long_skor / 100.0
    guvn_short = (100 - short_skor) / 100.0
    puan_guvn  = (guvn_long + guvn_short) / 2 * 100
    detay["karakter_guvenilirlik"] = {
        f"{long_sym}_oran":  round(guvn_long, 3),
        f"{short_sym}_oran": round(guvn_short, 3),
        "puan":              round(puan_guvn, 1),
    }

    # 3. Döngüde Kalma (%20)
    gecmis_l = dongü_gecmis_hesapla(long_sym,  btc_df, alt_dfs.get(long_sym, pd.DataFrame()),
                                    hafiza_penceresi, weight_recent, weight_mid, weight_old,
                                    tum_alt_dfs=alt_dfs)
    gecmis_s = dongü_gecmis_hesapla(short_sym, btc_df, alt_dfs.get(short_sym, pd.DataFrame()),
                                    hafiza_penceresi, weight_recent, weight_mid, weight_old,
                                    tum_alt_dfs=alt_dfs)

    bilgi_l = mevcut_dongü_bilgisi(gecmis_l)
    bilgi_s = mevcut_dongü_bilgisi(gecmis_s)

    l_sure  = bilgi_l.get("sure_saat", 99.0)
    s_sure  = bilgi_s.get("sure_saat", 99.0)

    # Yeni giriş = düşük süre = yüksek puan
    puan_kalma = max(0.0, 100.0 - (l_sure + s_sure) * 5)
    detay["dongude_kalma"] = {
        f"{long_sym}_sure":      round(l_sure, 2),
        f"{long_sym}_tahmin":    round(bilgi_l.get("tahmin_kalan", 0), 2),
        f"{short_sym}_sure":     round(s_sure, 2),
        f"{short_sym}_tahmin":   round(bilgi_s.get("tahmin_kalan", 0), 2),
        "puan":                  round(puan_kalma, 1),
    }

    # 4. Volatilite Sinyali (%15)
    puan_vol = 100.0 if btc_panel_sinyal else 0.0
    detay["volatilite_sinyali"] = {
        "sinyal": btc_panel_sinyal,
        "puan":   puan_vol,
    }

    # 5. Komisyon Başa Baş (%10)
    beklenen_makas = abs(long_skor - short_skor) / 100.0 * 0.05  # Kaba tahmin
    komisyon_toplam = TAKER_RATE * 4  # 2 giriş + 2 çıkış
    basa_bas_esigi  = komisyon_toplam * 3
    gecti_komisyon  = beklenen_makas >= basa_bas_esigi
    puan_komisyon   = 100.0 if gecti_komisyon else (beklenen_makas / basa_bas_esigi * 100 if basa_bas_esigi > 0 else 0)
    detay["komisyon_basa_bas"] = {
        "gereken_min":   round(basa_bas_esigi * 100, 4),
        "beklenen_makas": round(beklenen_makas * 100, 4),
        "gecti":         gecti_komisyon,
        "puan":          round(puan_komisyon, 1),
    }

    # Toplam ağırlıklı skor
    agirliklar = MAKAS_AGIRLIK
    toplam = (
        puan_beta   * agirliklar["volatilite_beta_farki"]
        + puan_guvn * agirliklar["karakter_guvenilirlik"]
        + puan_kalma * agirliklar["dongude_kalma"]
        + puan_vol   * agirliklar["volatilite_sinyali"]
        + puan_komisyon * agirliklar["komisyon_basa_bas"]
    )

    return round(toplam, 1), detay


def en_iyi_makas_cifti(
    puanlama: Dict[str, Dict],
    btc_panel_sinyal: bool,
    hafiza_penceresi: str,
    btc_df: pd.DataFrame,
    alt_dfs: Dict[str, pd.DataFrame],
    weight_recent: float = DEFAULT_WEIGHT_RECENT,
    weight_mid:    float = DEFAULT_WEIGHT_MID,
    weight_old:    float = DEFAULT_WEIGHT_OLD,
) -> Optional[Tuple[str, str, float]]:
    """
    Tüm Döngü1 × Döngü2 kombinasyonlarını değerlendirip en yüksek skorlu çifti döndürür.
    Döndürür: (long_sym, short_sym, skor) veya None
    """
    dongü1_coinler = [s for s, v in puanlama.items() if v.get("dongü") == DONGÜ_GUCLU]
    dongü2_coinler = [s for s, v in puanlama.items() if v.get("dongü") == DONGÜ_CURUK]

    if not dongü1_coinler or not dongü2_coinler:
        return None

    en_iyi_skor = -1.0
    en_iyi_long = None
    en_iyi_short = None

    for l_sym in dongü1_coinler:
        for s_sym in dongü2_coinler:
            skor, _ = makas_skoru_hesapla(
                l_sym, s_sym, puanlama, btc_panel_sinyal,
                hafiza_penceresi, btc_df, alt_dfs,
                weight_recent, weight_mid, weight_old,
            )
            if skor > en_iyi_skor:
                en_iyi_skor  = skor
                en_iyi_long  = l_sym
                en_iyi_short = s_sym

    if en_iyi_long is None:
        return None
    return (en_iyi_long, en_iyi_short, en_iyi_skor)
