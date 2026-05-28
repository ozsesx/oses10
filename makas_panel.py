# makas_panel.py — Makas Paneli (Pair Trading Karar Destek)
# ─────────────────────────────────────────────────────────────────────────────

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    DONGÜ_GUCLU, DONGÜ_CURUK, MAKAS_MIN_SKOR,
    TAKER_RATE, TAKER_RATE_BNB, MAKER_RATE, MAKER_RATE_BNB,
    RENK_POZITIF, RENK_NEGATIF, RENK_NOTR, RENK_KART,
)
from archive_manager import get_latest_funding_rate
from scoring import (
    makas_skoru_hesapla,
    en_iyi_makas_cifti,
    dongü_gecmis_hesapla,
    mevcut_dongü_bilgisi,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# MAKAS FIRSAT BANNERI
# ──────────────────────────────────────────────

def render_makas_banner(
    puanlama:         Dict[str, Dict],
    btc_panel_sinyal: bool,
    hafiza_penceresi: str,
    btc_df:           pd.DataFrame,
    alt_dfs:          Dict[str, pd.DataFrame],
    weight_recent:    float,
    weight_mid:       float,
    weight_old:       float,
) -> Optional[Tuple[str, str, float]]:
    """
    Ekranın en üstünde makas fırsat banner'ını gösterir.
    Döndürür: (long_sym, short_sym, skor) veya None
    """
    sonuc = en_iyi_makas_cifti(
        puanlama, btc_panel_sinyal, hafiza_penceresi,
        btc_df, alt_dfs, weight_recent, weight_mid, weight_old,
    )
    if sonuc is None:
        return None

    long_sym, short_sym, skor = sonuc
    if skor < MAKAS_MIN_SKOR:
        return None

    st.markdown(
        f"""<div style="
            background: linear-gradient(90deg, #1a2e1a 0%, #0e1117 100%);
            border: 2px solid {RENK_POZITIF};
            border-radius: 10px;
            padding: 14px 20px;
            margin-bottom: 16px;
            font-size: 16px;
            font-weight: bold;
            color: {RENK_POZITIF};
        ">
        ⚡ Güçlü makas fırsatı:&nbsp;
        <span style="color:#ffffff;">{long_sym}USDT</span>
        &nbsp;(Long) /&nbsp;
        <span style="color:{RENK_NEGATIF};">{short_sym}USDT</span>
        &nbsp;(Short) → Skor:&nbsp;
        <span style="color:#ffc107;">{skor:.0f}/100</span>
        </div>""",
        unsafe_allow_html=True,
    )
    return sonuc


# ──────────────────────────────────────────────
# MAKAS SKORU DETAY AÇILIMI
# ──────────────────────────────────────────────

def render_makas_detay(detay: Dict, long_sym: str, short_sym: str) -> None:
    """Makas skoru bileşenlerini expander içinde ayrıntılı gösterir."""

    def _cizgi():
        st.markdown("━" * 40)

    _cizgi()

    # 1. Volatilite Beta Farkı
    d = detay.get("volatilite_beta_farki", {})
    st.markdown("**Volatilite Beta Farkı**")
    st.markdown(
        f"Normal Beta Farkı:      `{d.get('normal_fark', '—')}x`\n\n"
        f"Volatilite Beta Farkı:  `{d.get('volatilite_fark', '—')}x` "
        f"{'✅' if d.get('gecti') else '❌'}   — Puan: {d.get('puan', 0):.0f}"
    )
    _cizgi()

    # 2. Karakter Güvenilirliği
    d = detay.get("karakter_guvenilirlik", {})
    st.markdown("**Karakter Güvenilirliği**")
    st.markdown(
        f"`{long_sym}USDT` güvenilirlik tahmini:  "
        f"`{d.get(f'{long_sym}_oran', 0)*100:.0f}%` "
        f"{'✅' if d.get(f'{long_sym}_oran', 0) > 0.75 else '⚠️'}\n\n"
        f"`{short_sym}USDT` güvenilirlik tahmini: "
        f"`{d.get(f'{short_sym}_oran', 0)*100:.0f}%` "
        f"{'✅' if d.get(f'{short_sym}_oran', 0) > 0.75 else '⚠️'}\n\n"
        f"Puan: {d.get('puan', 0):.0f}"
    )
    _cizgi()

    # 3. Döngüde Kalma
    d = detay.get("dongude_kalma", {})
    l_sure    = d.get(f"{long_sym}_sure",    0)
    l_tahmin  = d.get(f"{long_sym}_tahmin",  0)
    s_sure    = d.get(f"{short_sym}_sure",   0)
    s_tahmin  = d.get(f"{short_sym}_tahmin", 0)
    st.markdown("**Döngüde Kalma**")
    st.markdown(
        f"`{long_sym}USDT`: {l_sure:.1f} saattir Döngü 1'de | "
        f"Tahmini: {l_tahmin:.1f} saat daha "
        f"{'✅' if l_sure < 2 else '⚠️'}\n\n"
        f"`{short_sym}USDT`: {s_sure:.1f} saattir Döngü 2'de | "
        f"Tahmini: {s_tahmin:.1f} saat daha "
        f"{'✅' if s_sure < 2 else '⚠️'}\n\n"
        f"Puan: {d.get('puan', 0):.0f}"
    )
    _cizgi()

    # 4. Volatilite Sinyali
    d = detay.get("volatilite_sinyali", {})
    st.markdown("**Volatilite Sinyali**")
    st.markdown(
        f"Bollinger Sıkışma: {'VAR ✅' if d.get('sinyal') else 'YOK ❌'}\n\n"
        f"Puan: {d.get('puan', 0):.0f}"
    )
    _cizgi()

    # 5. Komisyon Başa Baş
    d = detay.get("komisyon_basa_bas", {})
    st.markdown("**Komisyon Başa Baş**")
    st.markdown(
        f"Gereken minimum makas: `%{d.get('gereken_min', 0):.4f}`\n\n"
        f"Beklenen makas:        `%{d.get('beklenen_makas', 0):.4f}` "
        f"{'✅' if d.get('gecti') else '❌'}\n\n"
        f"Puan: {d.get('puan', 0):.0f}"
    )
    _cizgi()


# ──────────────────────────────────────────────
# KASA VE RİSK HESAPLAYICI
# ──────────────────────────────────────────────

def render_kasa_hesaplayici(long_sym: str, short_sym: str) -> None:
    """Kasa + kaldıraç + komisyon hesaplayıcı arayüzü."""
    st.markdown("### 💼 Kasa ve Risk Hesaplayıcı")

    col1, col2, col3 = st.columns(3)
    with col1:
        kasa_tl = st.number_input("Kasa (TL)", min_value=0.0, value=30000.0, step=1000.0, key="kasa_tl")
    with col2:
        risk_tl = st.number_input("Risk (TL)", min_value=0.0, value=2000.0, step=100.0, key="risk_tl")
    with col3:
        usd_kur = st.number_input("USD/TL Kur", min_value=1.0, value=33.0, step=0.5, key="usd_kur")

    col4, col5 = st.columns([3, 1])
    with col4:
        kaldıraç = st.slider("Kaldıraç", min_value=1, max_value=20, value=5, step=1, key="kaldirac")
    with col5:
        bnb_ile = st.selectbox("BNB öde", ["Evet", "Hayır"], key="bnb_ile") == "Evet"

    # USD cinsine çevir
    risk_usd    = risk_tl / usd_kur if usd_kur > 0 else 0.0
    kontrat_usd = risk_usd * kaldıraç

    # Komisyon oranı seç
    taker = TAKER_RATE_BNB if bnb_ile else TAKER_RATE

    # 2 coin × 2 işlem (giriş + çıkış)
    giris_komisyon  = kontrat_usd * taker * 2  # Her iki coin için giriş
    cikis_komisyon  = kontrat_usd * taker * 2  # Her iki coin için çıkış
    toplam_komisyon = giris_komisyon + cikis_komisyon

    # Funding fee (tahmini 8 saat)
    fr_long  = get_latest_funding_rate(long_sym)  or 0.0
    fr_short = get_latest_funding_rate(short_sym) or 0.0
    ort_fr   = (abs(fr_long) + abs(fr_short)) / 2
    funding_fee = kontrat_usd * ort_fr * 2  # 2 pozisyon

    toplam_maliyet = toplam_komisyon + funding_fee

    # Beklenen makas tahmin (placeholder — gerçekte scoring'ten gelir)
    beklenen_makas_pct = st.number_input(
        "Beklenen Makas (%)",
        min_value=0.0, max_value=50.0, value=2.1, step=0.1, key="beklenen_makas_pct",
    )
    brut_kar  = kontrat_usd * beklenen_makas_pct / 100
    net_kar   = brut_kar - toplam_maliyet
    basa_bas  = toplam_komisyon / kontrat_usd * 100 if kontrat_usd > 0 else 0.0

    # Tablo göster
    st.markdown("---")
    sol, sag = st.columns(2)
    with sol:
        st.markdown(
            f"""<div style="background:{RENK_KART}; border-radius:8px; padding:14px; font-family:monospace; font-size:13px; line-height:2;">
            <b>Kontrat büyüklüğü:</b><br>
            <span style="color:{RENK_POZITIF};">{kontrat_usd:,.2f} USDT</span><br>
            ━━━━━━━━━━━━━━━━━━━━━━<br>
            Giriş komisyonu (x2 coin): {giris_komisyon:.4f} USDT<br>
            Çıkış komisyonu (x2 coin): {cikis_komisyon:.4f} USDT<br>
            Funding Fee (tahmini 8s):  {funding_fee:.4f} USDT<br>
            <b>Toplam maliyet:</b>           <span style="color:{RENK_NEGATIF};">{toplam_maliyet:.4f} USDT</span>
            </div>""",
            unsafe_allow_html=True,
        )
    with sag:
        net_renk = RENK_POZITIF if net_kar > 0 else RENK_NEGATIF
        st.markdown(
            f"""<div style="background:{RENK_KART}; border-radius:8px; padding:14px; font-family:monospace; font-size:13px; line-height:2;">
            ━━━━━━━━━━━━━━━━━━━━━━<br>
            Beklenen Makas: <b>%{beklenen_makas_pct:.2f}</b><br>
            Brüt Kar: {brut_kar:.4f} USDT<br>
            <b>Net Kar:</b> <span style="color:{net_renk};">{net_kar:.4f} USDT</span><br>
            ━━━━━━━━━━━━━━━━━━━━━━<br>
            Komisyon başa baş: <b>%{basa_bas:.4f}</b>
            </div>""",
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────
# ANA MAKAS PANELİ
# ──────────────────────────────────────────────

def render_makas_panel(
    puanlama:         Dict[str, Dict],
    btc_panel_sinyal: bool,
    hafiza_penceresi: str,
    btc_df:           Optional[pd.DataFrame],
    alt_dfs:          Dict[str, pd.DataFrame],
    weight_recent:    float,
    weight_mid:       float,
    weight_old:       float,
) -> None:
    """Makas Paneli ana render fonksiyonu."""

    st.subheader("✂️ Makas Paneli")

    if not puanlama:
        st.warning("⚠️ Tarama verisi bekleniyor...")
        return

    dongü1_coinler = sorted(
        [s for s, v in puanlama.items() if v.get("dongü") == DONGÜ_GUCLU],
        key=lambda s: puanlama[s].get("skor", 0), reverse=True,
    )
    dongü2_coinler = sorted(
        [s for s, v in puanlama.items() if v.get("dongü") == DONGÜ_CURUK],
        key=lambda s: puanlama[s].get("skor", 100),
    )

    if not dongü1_coinler or not dongü2_coinler:
        st.info("Döngü 1 veya Döngü 2 coinleri henüz yok.")
        return

    # ── Fırsat Bannerı ──
    en_iyi = render_makas_banner(
        puanlama, btc_panel_sinyal, hafiza_penceresi,
        btc_df, alt_dfs, weight_recent, weight_mid, weight_old,
    )

    # Ön seçim: banner'dan gelen veya varsayılan
    default_long  = en_iyi[0] if en_iyi else dongü1_coinler[0]
    default_short = en_iyi[1] if en_iyi else dongü2_coinler[0]

    if "makas_long"  not in st.session_state: st.session_state["makas_long"]  = default_long
    if "makas_short" not in st.session_state: st.session_state["makas_short"] = default_short

    # ── Coin Seçimi ──
    col_l, col_s = st.columns(2)
    with col_l:
        st.markdown("**Long (Döngü 1 — Güçlü)**")
        long_idx = dongü1_coinler.index(st.session_state["makas_long"]) \
            if st.session_state["makas_long"] in dongü1_coinler else 0
        secilen_long = st.selectbox(
            "Long Coin", dongü1_coinler, index=long_idx,
            format_func=lambda s: f"{s}USDT  (Skor: {puanlama[s]['skor']:.0f})",
            key="sel_long",
        )
        st.session_state["makas_long"] = secilen_long

    with col_s:
        st.markdown("**Short (Döngü 2 — Çürük)**")
        short_idx = dongü2_coinler.index(st.session_state["makas_short"]) \
            if st.session_state["makas_short"] in dongü2_coinler else 0
        secilen_short = st.selectbox(
            "Short Coin", dongü2_coinler, index=short_idx,
            format_func=lambda s: f"{s}USDT  (Skor: {puanlama[s]['skor']:.0f})",
            key="sel_short",
        )
        st.session_state["makas_short"] = secilen_short

    long_sym  = st.session_state["makas_long"]
    short_sym = st.session_state["makas_short"]

    st.markdown("---")

    # ── Makas Skoru ──
    if btc_df is not None and long_sym in alt_dfs and short_sym in alt_dfs:
        skor, detay = makas_skoru_hesapla(
            long_sym, short_sym, puanlama, btc_panel_sinyal,
            hafiza_penceresi, btc_df, alt_dfs,
            weight_recent, weight_mid, weight_old,
        )

        skor_renk = RENK_POZITIF if skor >= 75 else ("#ffc107" if skor >= MAKAS_MIN_SKOR else RENK_NEGATIF)
        st.markdown(
            f"""<div style="text-align:center; margin:8px 0;">
            <span style="font-size:28px; font-weight:bold; color:{skor_renk};">
            Makas Skoru: {skor:.0f} / 100
            </span>
            </div>""",
            unsafe_allow_html=True,
        )

        with st.expander("📋 Makas Skoru Detayı"):
            render_makas_detay(detay, long_sym, short_sym)

        # ── Makas Daralma Uyarısı ──
        _kontrol_makas_daralma(long_sym, short_sym, puanlama)

    else:
        st.info("Makas skoru hesaplanamadı — veri bekleniyor.")

    # ── Kasa Hesaplayıcı ──
    st.markdown("---")
    render_kasa_hesaplayici(long_sym, short_sym)


def _kontrol_makas_daralma(
    long_sym: str,
    short_sym: str,
    puanlama:  Dict[str, Dict],
) -> None:
    """
    Bir önceki taramaya göre volatilite beta farkı daraldıysa uyarı gösterir.
    """
    anahtar = f"onceki_beta_fark_{long_sym}_{short_sym}"
    mevcut_l = puanlama.get(long_sym,  {}).get("beta_v")
    mevcut_s = puanlama.get(short_sym, {}).get("beta_v")

    if mevcut_l is not None and mevcut_s is not None:
        mevcut_fark = abs(mevcut_l - mevcut_s)
        onceki_fark = st.session_state.get(anahtar)
        if onceki_fark is not None and mevcut_fark < onceki_fark * 0.85:
            st.warning(
                f"⚠️ **MAKAS DARALIYOR!** Beta farkı: "
                f"`{onceki_fark:.3f}x` → `{mevcut_fark:.3f}x`. "
                f"Pozisyonu gözden geçir."
            )
        st.session_state[anahtar] = mevcut_fark
