# btc_panel.py — Bitcoin Volatilite Paneli
# ─────────────────────────────────────────────────────────────────────────────

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from config import (
    BTC_PANEL_PERIYOTLAR, BTC_PANEL_DEFAULT,
    BOLL_WINDOW, BOLL_STD, BOLL_NARROW_PCT, ATR_WINDOW,
    PATLAMA_GECMIS_GUN, RENK_POZITIF, RENK_NEGATIF, RENK_NOTR,
)
from archive_manager import load_funding_rates, get_latest_funding_rate

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# TEKNİK İNDİKATÖR HESAPLAMA
# ──────────────────────────────────────────────

def hesapla_bollinger(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bantlarını hesaplar.
    Döndürür: (orta, ust, alt, genislik)
    """
    close = df["close"].astype(float)
    orta  = close.rolling(BOLL_WINDOW).mean()
    std   = close.rolling(BOLL_WINDOW).std()
    ust   = orta + BOLL_STD * std
    alt   = orta - BOLL_STD * std
    genislik = (ust - alt) / orta.replace(0, np.nan)
    return orta, ust, alt, genislik


def hesapla_atr(df: pd.DataFrame) -> pd.Series:
    """ATR (Average True Range) hesaplar."""
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(ATR_WINDOW).mean()


def hesapla_obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume hesaplar."""
    close  = df["close"].astype(float)
    volume = df["volume"].astype(float)
    delta  = close.diff()
    sinyal = np.where(delta > 0, 1, np.where(delta < 0, -1, 0))
    obv = (volume * sinyal).cumsum()
    return pd.Series(obv, index=df.index)


def volatilite_sinyali(df: pd.DataFrame) -> Dict:
    """
    Bollinger sıkışma + ATR durumunu değerlendirir.
    Döndürür: {"patlama_yakin": bool, "bb_dar": bool, "atr_dusuk": bool, "detay": str}
    """
    if df is None or len(df) < max(BOLL_WINDOW, ATR_WINDOW) + 5:
        return {"patlama_yakin": False, "bb_dar": False, "atr_dusuk": False, "detay": "Yetersiz veri"}

    _, _, _, genislik = hesapla_bollinger(df)
    atr     = hesapla_atr(df)
    atr_ort = atr.rolling(ATR_WINDOW * 2).mean()

    son_genislik = genislik.dropna()
    son_atr      = atr.dropna()
    son_atr_ort  = atr_ort.dropna()

    if son_genislik.empty or son_atr.empty:
        return {"patlama_yakin": False, "bb_dar": False, "atr_dusuk": False, "detay": "Yetersiz veri"}

    threshold = son_genislik.quantile(BOLL_NARROW_PCT / 100)
    bb_dar    = bool(son_genislik.iloc[-1] < threshold)
    atr_dusuk = bool(
        son_atr.iloc[-1] < son_atr_ort.iloc[-1]
        if not son_atr_ort.empty else False
    )
    patlama_yakin = bb_dar and atr_dusuk

    detay_parts = []
    detay_parts.append(
        f"Bollinger Bant Genişliği → Son {BOLL_NARROW_PCT} mumun en darına yakın: "
        f"{'VAR ✅' if bb_dar else 'YOK ❌'}"
    )
    detay_parts.append(
        f"ATR ({ATR_WINDOW}) → Ortalamanın altında: "
        f"{'EVET ✅' if atr_dusuk else 'HAYIR ❌'}"
    )
    if patlama_yakin:
        detay_parts.append("**Durum: Enerji birikiyor, patlama yakın** 🔥")
    else:
        detay_parts.append("Durum: Normal volatilite")

    return {
        "patlama_yakin": patlama_yakin,
        "bb_dar":        bb_dar,
        "atr_dusuk":     atr_dusuk,
        "detay":         "\n".join(detay_parts),
    }


def yon_sinyali(df: pd.DataFrame, symbol: str = "BTC") -> Dict:
    """
    OBV + Funding Rate ile yön sinyali değerlendirir.
    Döndürür: {"yon": "YUKARI_AGIRLIKLI"|"ASAGI_AGIRLIKLI"|"CAKISAN", "detay": str, "celisen": bool}
    """
    obv = hesapla_obv(df)
    obv_trend = None
    if len(obv) >= 10:
        son10 = obv.iloc[-10:]
        if son10.iloc[-1] > son10.iloc[0]:
            obv_trend = "pozitif"
        elif son10.iloc[-1] < son10.iloc[0]:
            obv_trend = "negatif"
        else:
            obv_trend = "yatay"

    # Funding Rate
    funding = get_latest_funding_rate(symbol)
    funding_str   = ""
    funding_yukari = None
    if funding is not None:
        if funding < 0:
            funding_str    = f"Negatif ({funding*100:.4f}%) → Short squeeze riski yüksek ✅"
            funding_yukari = True
        elif funding > 0.001:
            funding_str    = f"Pozitif ({funding*100:.4f}%) → Long tasfiye riski ⚠️"
            funding_yukari = False
        else:
            funding_str    = f"Nötr ({funding*100:.4f}%)"
            funding_yukari = None

    # Fiyat trendi (son 10 mum)
    close = df["close"].astype(float)
    fiyat_trend = None
    if len(close) >= 10:
        if close.iloc[-1] > close.iloc[-10]:
            fiyat_trend = "yukari"
        elif close.iloc[-1] < close.iloc[-10]:
            fiyat_trend = "asagi"
        else:
            fiyat_trend = "yatay"

    # OBV yorumu
    obv_yorum = ""
    if obv_trend == "pozitif":
        if fiyat_trend != "yukari":
            obv_yorum = "OBV → Fiyat yatay, OBV yükseliyor → Birikim sinyali ✅"
        else:
            obv_yorum = "OBV → Fiyat ve OBV birlikte yükseliyor ✅"
    elif obv_trend == "negatif":
        obv_yorum = "OBV → Düşüş baskısı devam ediyor ❌"
    else:
        obv_yorum = "OBV → Yatay seyrediyor ⚖️"

    # Çelişen sinyal tespiti
    celisen = False
    if obv_trend == "pozitif" and funding_yukari is False:
        celisen = True
    elif obv_trend == "negatif" and funding_yukari is True:
        celisen = True

    # Genel yön kararı
    pozitif_sayac = 0
    negatif_sayac = 0
    if obv_trend == "pozitif":      pozitif_sayac += 1
    elif obv_trend == "negatif":    negatif_sayac += 1
    if funding_yukari is True:      pozitif_sayac += 1
    elif funding_yukari is False:   negatif_sayac += 1

    if celisen:
        yon = "CAKISAN"
    elif pozitif_sayac > negatif_sayac:
        yon = "YUKARI_AGIRLIKLI"
    elif negatif_sayac > pozitif_sayac:
        yon = "ASAGI_AGIRLIKLI"
    else:
        yon = "NOTR"

    detay_parts = [obv_yorum]
    if funding_str:
        detay_parts.append(f"Funding Rate → {funding_str}")
    if celisen:
        detay_parts.append("⚠️ **Sinyaller çelişiyor, net yön yok**")

    return {
        "yon":       yon,
        "detay":     "\n".join(detay_parts),
        "celisen":   celisen,
        "obv_trend": obv_trend,
        "funding":   funding,
    }


def hacim_sinyali(df: pd.DataFrame) -> Dict:
    """
    Son 10 mum hacim trendi değerlendirir.
    Döndürür: {"durum": str, "detay": str, "birikim_var": bool}
    """
    if df is None or len(df) < 15:
        return {"durum": "YETERSİZ VERİ", "detay": "", "birikim_var": False}

    hacim = df["volume"].astype(float)
    son10 = hacim.iloc[-10:]
    ort   = hacim.iloc[-30:].mean() if len(hacim) >= 30 else hacim.mean()

    son10_ort = son10.mean()
    trend_str = ""
    birikim   = False

    if son10_ort > ort * 1.1:
        trend_str = "Son 10 mumda ortalamanın üzerinde 📈"
        birikim   = True
    elif son10_ort < ort * 0.9:
        trend_str = "Son 10 mumda ortalamanın altında 📉"
    else:
        trend_str = "Son 10 mumda ortalamayla uyumlu ➡️"

    obv = hesapla_obv(df)
    obv_son5 = obv.iloc[-5:]
    obv_egim = "Pozitif, alım baskısı devam ediyor ✅" if obv_son5.iloc[-1] > obv_son5.iloc[0] else "Negatif ❌"

    durum = "BİRİKİM VAR" if birikim else "NORMAL"

    detay = f"Hacim trendi → {trend_str}\nOBV eğimi → {obv_egim}"
    return {"durum": durum, "detay": detay, "birikim_var": birikim}


# ──────────────────────────────────────────────
# GEÇMİŞ PATLAMA KAYITLARI
# ──────────────────────────────────────────────

def gecmis_patlamalar(btc_daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Son PATLAMA_GECMIS_GUN günde tespit edilen patlama anlarını döndürür.
    Döndürür: DataFrame [tarih, sure_saat, buyukluk_pct, yon]
    """
    if btc_daily_df is None or btc_daily_df.empty:
        return pd.DataFrame(columns=["tarih", "sure_saat", "buyukluk_pct", "yon"])

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=PATLAMA_GECMIS_GUN)
    df = btc_daily_df[btc_daily_df.index >= cutoff].copy()

    if len(df) < max(BOLL_WINDOW, ATR_WINDOW) + 2:
        return pd.DataFrame(columns=["tarih", "sure_saat", "buyukluk_pct", "yon"])

    _, _, _, genislik = hesapla_bollinger(df)
    atr     = hesapla_atr(df)
    atr_ort = atr.rolling(ATR_WINDOW * 2).mean()

    threshold = genislik.quantile(BOLL_NARROW_PCT / 100)

    kayitlar = []
    for i in range(1, len(df)):
        onceki_gen = genislik.iloc[i - 1] if not pd.isna(genislik.iloc[i - 1]) else 1.0
        bugun_atr  = atr.iloc[i]      if not pd.isna(atr.iloc[i])      else 0.0
        bugun_ort  = atr_ort.iloc[i]  if not pd.isna(atr_ort.iloc[i])  else 0.0

        if onceki_gen < threshold and bugun_atr > bugun_ort:
            tarih  = df.index[i]
            kapani = float(df["close"].iloc[i])
            acilis = float(df["open"].iloc[i])
            buyukluk = (kapani - acilis) / acilis * 100 if acilis != 0 else 0.0
            kayitlar.append({
                "tarih":         tarih.strftime("%Y-%m-%d"),
                "sure_saat":     24,  # Günlük mum = 24 saat
                "buyukluk_pct":  round(buyukluk, 2),
                "yon":           "Yukarı" if buyukluk > 0 else "Aşağı",
            })

    result_df = pd.DataFrame(kayitlar)
    return result_df


# ──────────────────────────────────────────────
# STREAMLIT PANEL ÇİZİMİ
# ──────────────────────────────────────────────

def render_btc_panel(btc_df: Optional[pd.DataFrame], btc_daily_df: Optional[pd.DataFrame]) -> Dict:
    """
    Bitcoin Volatilite Paneli'ni çizer.
    Döndürür: {"patlama_yakin": bool} — Makas paneli için sinyal.
    """
    st.subheader("₿ Bitcoin Volatilite Paneli")

    # Periyot seçici
    if "btc_panel_periyot" not in st.session_state:
        st.session_state["btc_panel_periyot"] = BTC_PANEL_DEFAULT

    cols_per = st.columns(len(BTC_PANEL_PERIYOTLAR))
    for i, p in enumerate(BTC_PANEL_PERIYOTLAR):
        aktif = st.session_state["btc_panel_periyot"] == p
        label = f"**{p}**" if aktif else p
        if cols_per[i].button(label, key=f"btcper_{p}", use_container_width=True):
            st.session_state["btc_panel_periyot"] = p
            st.rerun()

    if btc_df is None or btc_df.empty:
        st.warning("⚠️ Bitcoin verisi henüz yüklenmedi.")
        return {"patlama_yakin": False}

    # Sinyalleri hesapla
    vol_sig  = volatilite_sinyali(btc_df)
    yon_sig  = yon_sinyali(btc_df, "BTC")
    hacim_sig = hacim_sinyali(btc_df)

    # ── Üst Yargı Kartları ──
    st.markdown("---")
    kart1, kart2, kart3 = st.columns(3)

    # Volatilite kartı
    with kart1:
        vol_label = "PATLAMA YAKLAŞIYOR 🔥" if vol_sig["patlama_yakin"] else "NORMAL"
        vol_renk  = RENK_POZITIF if vol_sig["patlama_yakin"] else RENK_NOTR
        st.markdown(
            f"""<div style="border:2px solid {vol_renk}; border-radius:8px; padding:12px; text-align:center;">
            <div style="font-size:11px; color:{RENK_NOTR};">⚡ VOLATİLİTE</div>
            <div style="font-size:14px; font-weight:bold; color:{vol_renk};">{vol_label}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        with st.expander("Detay ▾"):
            st.markdown(vol_sig["detay"])

    # Yön kartı
    with kart2:
        yon_map = {
            "YUKARI_AGIRLIKLI": ("YUKARI AĞIRLIKLI 📈", RENK_POZITIF),
            "ASAGI_AGIRLIKLI":  ("AŞAĞI AĞIRLIKLI 📉", RENK_NEGATIF),
            "CAKISAN":          ("ÇAKIŞAN SİNYAL ⚠️",  "#ffc107"),
            "NOTR":             ("NÖTR ⚖️",             RENK_NOTR),
        }
        yon_txt, yon_renk = yon_map.get(yon_sig["yon"], ("BİLİNMİYOR", RENK_NOTR))
        st.markdown(
            f"""<div style="border:2px solid {yon_renk}; border-radius:8px; padding:12px; text-align:center;">
            <div style="font-size:11px; color:{RENK_NOTR};">📈 YÖN</div>
            <div style="font-size:14px; font-weight:bold; color:{yon_renk};">{yon_txt}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        with st.expander("Detay ▾"):
            st.markdown(yon_sig["detay"])

    # Hacim kartı
    with kart3:
        hac_txt  = hacim_sig["durum"]
        hac_renk = RENK_POZITIF if hacim_sig["birikim_var"] else RENK_NOTR
        st.markdown(
            f"""<div style="border:2px solid {hac_renk}; border-radius:8px; padding:12px; text-align:center;">
            <div style="font-size:11px; color:{RENK_NOTR};">📊 HACİM</div>
            <div style="font-size:14px; font-weight:bold; color:{hac_renk};">{hac_txt}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        with st.expander("Detay ▾"):
            st.markdown(hacim_sig["detay"])

    # ── Bollinger + Fiyat Grafiği ──
    st.markdown("---")
    st.markdown("##### Bitcoin Fiyat + Bollinger Bantları")
    _ciz_btc_grafik(btc_df)

    # ── Geçmiş Patlama Kayıtları ──
    st.markdown("---")
    st.markdown(f"##### Son {PATLAMA_GECMIS_GUN} Günlük Patlama Kayıtları")
    if btc_daily_df is not None and not btc_daily_df.empty:
        patlama_df = gecmis_patlamalar(btc_daily_df)
        if patlama_df.empty:
            st.info("Bu dönemde patlama kaydı bulunamadı.")
        else:
            # Ortalama patlama sıklığı
            ortalama_siklik = PATLAMA_GECMIS_GUN / max(len(patlama_df), 1)
            st.caption(f"Ortalama patlama sıklığı: her **{ortalama_siklik:.1f}** günde bir ({len(patlama_df)} patlama)")
            st.dataframe(
                patlama_df.rename(columns={
                    "tarih":        "Tarih",
                    "sure_saat":    "Süre (saat)",
                    "buyukluk_pct": "Büyüklük (%)",
                    "yon":          "Yön",
                }),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("Arşiv verisi yüklenmeden geçmiş patlama kayıtları gösterilemiyor.")

    return {"patlama_yakin": vol_sig["patlama_yakin"]}


def _ciz_btc_grafik(df: pd.DataFrame) -> None:
    """Bitcoin fiyat grafiğini Bollinger bantlarıyla çizer."""
    orta, ust, alt, _ = hesapla_bollinger(df)
    obv = hesapla_obv(df)

    fig = go.Figure()

    # Bollinger bantları
    fig.add_trace(go.Scatter(
        x=df.index, y=ust, name="BB Üst",
        line=dict(color="#37474f", dash="dot"), showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=alt, name="BB Alt",
        line=dict(color="#37474f", dash="dot"),
        fill="tonexty", fillcolor="rgba(55,71,79,0.08)",
        showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=orta, name="BB Orta",
        line=dict(color="#78909c", dash="dash"), showlegend=True,
    ))

    # Fiyat mumu
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"],
        low=df["low"],   close=df["close"],
        name="BTC", increasing_line_color="#00e676",
        decreasing_line_color="#ff1744",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=350,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", x=0, y=1.08),
    )
    st.plotly_chart(fig, use_container_width=True)
