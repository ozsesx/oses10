# main.py — Streamlit Ana Uygulama
# ─────────────────────────────────────────────────────────────────────────────
# Render.com start:
#   streamlit run main.py --server.port $PORT --server.address 0.0.0.0

import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    SABIT_COINLER, MAX_MANUEL_EK, MAX_COIN,
    PERIYOT_LISTESI, PERIYOT_KONFIG,
    DEFAULT_WEIGHT_RECENT, DEFAULT_WEIGHT_MID, DEFAULT_WEIGHT_OLD,
    HAFIZA_LISTESI, HAFIZA_GUNLER,
    DONGÜ_GUCLU, DONGÜ_CURUK, DONGÜ_SAGIR, DONGÜ_RENKLER, DONGÜ_ETIKET, DONGÜ_BOYUT,
    AUTO_REFRESH_DAKIKA, STALE_DAKIKA,
    RENK_ARKA_PLAN, RENK_KART, RENK_YENILE_BTN, RENK_POZITIF, RENK_NEGATIF, RENK_NOTR,
    ISI_ADIM_KISIT, ISI_ADIM_DAKIKA,
)
from data_manager import (
    get_exchange,
    toplu_tarama,
    filtrele_gecerli_semboller,
    tespit_hacim_patlamasi,
    fetch_btc_panel_data,
    fetch_funding_rates,
    fetch_latest_funding_rate,
)
from archive_manager import (
    init_db,
    arsiv_hazir_mi,
    arsiv_ozet,
    ilk_kurulum_arsiv,
    gunluk_guncelleme,
    start_scheduler,
    load_daily_ohlcv,
    load_ayrisma_gucu_all,
    hesapla_volatilite_anlari,
)
from scoring import (
    tam_puanlama,
    dongü_gecmis_hesapla,
    mevcut_dongü_bilgisi,
    markov_hesapla,
    isi_haritasi_renkleri,
    hesapla_karakter_guvenilirlik,
)
from btc_panel import render_btc_panel
from makas_panel import render_makas_panel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# SAYFA AYARLARI
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Kripto Makas Bot",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Mobil Uyumlu Custom CSS ──
st.markdown(
    f"""
    <style>
    /* Genel arka plan */
    .stApp {{ background-color: {RENK_ARKA_PLAN}; }}

    /* Coin kart stili */
    .coin-card {{
        background-color: {RENK_KART};
        border-radius: 10px;
        padding: 10px 12px;
        margin-bottom: 8px;
        font-size: 12px;
        line-height: 1.6;
        word-break: break-word;
    }}
    .coin-card-d1 {{ border: 2px solid {RENK_POZITIF}; }}
    .coin-card-d2 {{ border: 2px solid {RENK_NEGATIF}; }}
    .coin-card-d3 {{ border: 2px solid {RENK_NOTR}; }}
    .coin-card-d0 {{ border: 1px solid #2a2a3e; }}

    /* Durum matrisi kareleri */
    .mat-kare {{
        display: inline-block;
        width: 38px;
        height: 38px;
        border-radius: 6px;
        margin: 2px;
        cursor: pointer;
        font-size: 9px;
        text-align: center;
        line-height: 38px;
        font-weight: bold;
        overflow: hidden;
        white-space: nowrap;
    }}

    /* Yenile butonu turuncu */
    button[data-testid="baseButton-secondary"]:has(span:contains("🔄")) {{
        background-color: {RENK_YENILE_BTN} !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
    }}

    /* Mobil responsive */
    @media (max-width: 768px) {{
        .coin-card {{ font-size: 11px; padding: 8px; }}
        .mat-kare  {{ width: 30px; height: 30px; line-height: 30px; font-size: 8px; }}
        h1, h2, h3 {{ font-size: 16px !important; }}
    }}

    /* Footer */
    .footer-uyari {{
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: #111;
        color: #666;
        text-align: center;
        font-size: 11px;
        padding: 4px;
        z-index: 9999;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# ŞİFRE KORUMASI
# ──────────────────────────────────────────────

def sifre_kontrolu() -> bool:
    """Kullanıcı şifre ekranını gösterir. Doğruysa True döner."""
    if st.session_state.get("giris_yapildi", False):
        return True

    st.markdown("## 🔐 Kripto Makas Bot")
    st.markdown("Devam etmek için şifre girin.")
    with st.form("login_form"):
        sifre = st.text_input("Şifre", type="password", placeholder="●●●●●●●●")
        giris = st.form_submit_button("Giriş Yap", use_container_width=True)

    if giris:
        import os
        try:
            dogru_sifre = (
                os.environ.get("APP_PASSWORD")
                or st.secrets.get("passwords", {}).get("app_password")
                or "admin"
            )
        except Exception:
            dogru_sifre = os.environ.get("APP_PASSWORD", "admin")

        if sifre == dogru_sifre:
            st.session_state["giris_yapildi"] = True
            st.rerun()
        else:
            st.error("❌ Hatalı şifre.")
    return False


# ──────────────────────────────────────────────
# SESSION STATE BAŞLATMA
# ──────────────────────────────────────────────

def init_session_state() -> None:
    defaults = {
        "giris_yapildi":        False,
        "son_tarama_zaman":     None,
        "global_periyot":       "1 Gün",
        "global_weight_recent": DEFAULT_WEIGHT_RECENT,
        "global_weight_mid":    DEFAULT_WEIGHT_MID,
        "global_weight_old":    DEFAULT_WEIGHT_OLD,
        "hafiza_penceresi":     "1 Gün",
        "manuel_coinler":       [],
        "tum_data":             {},
        "puanlama":             {},
        "btc_df":               None,
        "btc_daily_df":         None,
        "btc_panel_sinyal":     False,
        "secilen_coin":         None,
        "makas_long":           None,
        "makas_short":          None,
        "exchange":             None,
        "arsiv_baslatildi":     False,
        "dongu_gecmis_cache":   {},
        "coin_detay_acik":      None,
        "onceki_beta_farklari": {},
        "xray_periyot":         "1 Gün",
        "xray_weight_recent":   DEFAULT_WEIGHT_RECENT,
        "xray_weight_mid":      DEFAULT_WEIGHT_MID,
        "xray_weight_old":      DEFAULT_WEIGHT_OLD,
        "xray_hafiza":          "1 Gün",
        "xray_isi_adim":        "15 dakika",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ──────────────────────────────────────────────
# EXCHANGE VE ARŞİV BAŞLATMA
# ──────────────────────────────────────────────

@st.cache_resource
def get_cached_exchange():
    return get_exchange()


def baslat_arsiv_ve_scheduler(exchange) -> None:
    """İlk açılışta DB oluştur, arşiv yoksa kur, scheduler'ı başlat."""
    if st.session_state.get("arsiv_baslatildi"):
        return

    init_db()
    semboller = _aktif_semboller()
    start_scheduler(exchange, semboller)

    if not arsiv_hazir_mi():
        with st.spinner("📦 Arşiv ilk kez oluşturuluyor (~90 saniye)..."):
            prog = st.empty()
            def cb(i, total, sym):
                prog.text(f"📥 {sym} arşivleniyor... ({i+1}/{total})")
            ilk_kurulum_arsiv(exchange, semboller, progress_cb=cb)
            prog.text("✅ Arşiv hazır!")

    st.session_state["arsiv_baslatildi"] = True


# ──────────────────────────────────────────────
# AKTİF SEMBOL LİSTESİ
# ──────────────────────────────────────────────

def _aktif_semboller() -> List[str]:
    manuel = st.session_state.get("manuel_coinler", [])
    return list(dict.fromkeys(SABIT_COINLER + manuel))[:MAX_COIN]


# ──────────────────────────────────────────────
# VERİ TARAMA
# ──────────────────────────────────────────────

def _test_binance_baglanti() -> Tuple[bool, str]:
    """Binance API'ye test isteği atar, sonucu döndürür."""
    import requests as req
    try:
        r = req.get("https://fapi.binance.com/fapi/v1/ping", timeout=10)
        if r.status_code == 200:
            return True, "OK"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)


def veri_tara(zorunlu: bool = False) -> None:
    """
    Canlı veriyi tarar ve session_state'e yazar.
    zorunlu=True ise zaman kontrolü yapılmaz.
    """
    son = st.session_state.get("son_tarama_zaman")
    if not zorunlu and son is not None:
        gecen = (datetime.now(timezone.utc) - son).total_seconds() / 60
        if gecen < STALE_DAKIKA:
            return

    # Önce bağlantı testi
    bagli, hata_mesaj = _test_binance_baglanti()
    if not bagli:
        st.error(f"❌ Binance API'ye ulaşılamıyor: `{hata_mesaj}`")
        st.info("💡 Render free plan bazen Binance IP'lerini engeller. Sayfayı yenile veya birkaç dakika bekle.")
        return

    exchange = get_cached_exchange()
    semboller = filtrele_gecerli_semboller(exchange, _aktif_semboller())
    periyot   = st.session_state["global_periyot"]

    prog = st.empty()
    tum_data = toplu_tarama(exchange, semboller, periyot, progress_placeholder=prog)
    prog.empty()

    if not tum_data:
        st.error("❌ Veri çekilemedi — Binance'e bağlanıldı ama veri gelmedi.")
        st.info("Periyot olarak **1 Gün** seç ve tekrar dene.")
        return

    btc_df = tum_data.pop("BTC", None)
    if btc_df is None:
        # BTC ayrıca çek
        from data_manager import fetch_ohlcv_with_resample
        btc_df = fetch_ohlcv_with_resample(exchange, "BTC/USDT:USDT", periyot)

    st.session_state["tum_data"]  = tum_data
    st.session_state["btc_df"]    = btc_df

    # BTC günlük (panel için arşivden)
    btc_daily = load_daily_ohlcv("BTC", 365)
    st.session_state["btc_daily_df"] = btc_daily if not btc_daily.empty else None

    # Puanlama
    puanlama = tam_puanlama(
        btc_df, tum_data,
        st.session_state["global_weight_recent"],
        st.session_state["global_weight_mid"],
        st.session_state["global_weight_old"],
    )
    # Ayrışma gücü ekle
    ayrisma = load_ayrisma_gucu_all()
    for sym in puanlama:
        puanlama[sym]["ayrisma"] = ayrisma.get(sym, 0.0)

    st.session_state["puanlama"] = puanlama
    st.session_state["son_tarama_zaman"] = datetime.now(timezone.utc)

    # Hacim patlaması tespiti
    patlamalar = tespit_hacim_patlamasi(_aktif_semboller(), [])
    st.session_state["hacim_patlamalar"] = patlamalar


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

def render_sidebar(exchange) -> None:
    """Yan menüyü çizer: manuel coin ekleme, arşiv durumu vb."""
    with st.sidebar:
        st.markdown("### ✂️ Kripto Makas Bot")

        # Son tarama zamanı
        son = st.session_state.get("son_tarama_zaman")
        if son:
            gecen = int((datetime.now(timezone.utc) - son).total_seconds() / 60)
            st.caption(f"Son tarama: **{gecen} dk önce**")
        else:
            st.caption("Henüz tarama yapılmadı.")

        # Manuel coin ekleme
        st.markdown("---")
        st.markdown("**Manuel Coin Ekle**")
        manuel_coinler = st.session_state.get("manuel_coinler", [])
        if len(manuel_coinler) < MAX_MANUEL_EK:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                yeni_coin = st.text_input(
                    "Sembol (ör: WLD)", key="manuel_coin_input",
                    placeholder="SEMBOL", label_visibility="collapsed",
                ).strip().upper()
            with col_b:
                if st.button("Ekle", key="ekle_btn"):
                    if yeni_coin and yeni_coin not in SABIT_COINLER and yeni_coin not in manuel_coinler:
                        manuel_coinler.append(yeni_coin)
                        st.session_state["manuel_coinler"] = manuel_coinler
                        st.success(f"✅ {yeni_coin} eklendi")
                        st.rerun()
        else:
            st.caption(f"Maks. {MAX_MANUEL_EK} manuel coin eklendi.")

        if manuel_coinler:
            for mc in manuel_coinler:
                col_c, col_d = st.columns([3, 1])
                col_c.caption(mc)
                if col_d.button("❌", key=f"sil_{mc}"):
                    manuel_coinler.remove(mc)
                    st.session_state["manuel_coinler"] = manuel_coinler
                    st.rerun()

        # Hacim patlama uyarıları
        patlamalar = st.session_state.get("hacim_patlamalar", [])
        if patlamalar:
            st.markdown("---")
            st.markdown("**⚡ Hacim Patlaması**")
            for sym in patlamalar[:5]:
                col_e, col_f = st.columns([3, 1])
                col_e.caption(f"{sym} — Patlama!")
                if col_f.button("Ekle", key=f"hekle_{sym}"):
                    if sym not in manuel_coinler and len(manuel_coinler) < MAX_MANUEL_EK:
                        manuel_coinler.append(sym)
                        st.session_state["manuel_coinler"] = manuel_coinler
                        st.rerun()

        # Arşiv bilgisi
        st.markdown("---")
        st.markdown("**📦 Arşiv Durumu**")
        ozet = arsiv_ozet()
        st.caption(
            f"Sembol: {ozet['sembol_sayisi']} | "
            f"Satır: {ozet['satir_sayisi']:,}\n"
            f"Son güncelleme: {ozet['son_guncelleme']}"
        )
        if st.button("🔄 Arşivi Güncelle", key="arsiv_guncelle"):
            with st.spinner("Arşiv güncelleniyor..."):
                gunluk_guncelleme(exchange, _aktif_semboller())
            st.success("✅ Arşiv güncellendi")

        # Çıkış
        st.markdown("---")
        if st.button("🚪 Çıkış", key="cikis"):
            st.session_state["giris_yapildi"] = False
            st.rerun()


# ──────────────────────────────────────────────
# SEKME 1: CANLI İZLEME ODASI
# ──────────────────────────────────────────────

def render_canli_izleme() -> None:
    """Canlı İzleme Odası sekmesi."""

    # ── Üst Kontrol Paneli ──
    with st.container():
        st.markdown("#### ⏱ Zaman Penceresi")
        cols_per = st.columns(len(PERIYOT_LISTESI))
        for i, p in enumerate(PERIYOT_LISTESI):
            aktif = st.session_state["global_periyot"] == p
            label = f"**{p}**" if aktif else p
            if cols_per[i].button(label, key=f"gper_{p}", use_container_width=True):
                st.session_state["global_periyot"] = p
                veri_tara(zorunlu=True)
                st.rerun()

        st.markdown("#### ⚖️ Dönem Ağırlıkları")
        w1, w2, w3, w4 = st.columns([2, 2, 2, 2])
        with w1:
            st.session_state["global_weight_recent"] = st.number_input(
                "Yakın", value=float(st.session_state["global_weight_recent"]),
                min_value=0.0, max_value=20.0, step=0.1, key="wt_recent",
            )
        with w2:
            st.session_state["global_weight_mid"] = st.number_input(
                "Orta", value=float(st.session_state["global_weight_mid"]),
                min_value=0.0, max_value=20.0, step=0.1, key="wt_mid",
            )
        with w3:
            st.session_state["global_weight_old"] = st.number_input(
                "Eski", value=float(st.session_state["global_weight_old"]),
                min_value=0.0, max_value=20.0, step=0.1, key="wt_old",
            )
        with w4:
            # Hafıza penceresi
            hafiza_idx = HAFIZA_LISTESI.index(st.session_state["hafiza_penceresi"]) \
                if st.session_state["hafiza_penceresi"] in HAFIZA_LISTESI else 0
            st.session_state["hafiza_penceresi"] = st.selectbox(
                "Hafıza", HAFIZA_LISTESI, index=hafiza_idx, key="hafiza_sel",
            )

        # Yenile butonu (turuncu, sağ üst)
        _, col_btn = st.columns([8, 2])
        with col_btn:
            if st.button("🔄 Yeniden Tara", key="yenile_btn", use_container_width=True,
                         type="secondary"):
                veri_tara(zorunlu=True)
                st.rerun()

    puanlama = st.session_state.get("puanlama", {})
    btc_df   = st.session_state.get("btc_df")
    tum_data = st.session_state.get("tum_data", {})

    if not puanlama:
        st.info("⌛ Tarama başlatılıyor...")
        veri_tara()
        st.rerun()
        return

    # ── Makas Fırsat Bannerı ──
    from makas_panel import render_makas_banner
    render_makas_banner(
        puanlama,
        st.session_state.get("btc_panel_sinyal", False),
        st.session_state["hafiza_penceresi"],
        btc_df, tum_data,
        st.session_state["global_weight_recent"],
        st.session_state["global_weight_mid"],
        st.session_state["global_weight_old"],
    )

    # ── Üç Döngü Başlıkları ──
    st.markdown("---")
    col_d1, col_d2, col_d3 = st.columns(3)
    col_d1.markdown(f"<h5 style='color:{DONGÜ_RENKLER[1]};'>🟢 Döngü 1 — En Güçlü {DONGÜ_BOYUT}</h5>", unsafe_allow_html=True)
    col_d2.markdown(f"<h5 style='color:{DONGÜ_RENKLER[2]};'>🔴 Döngü 2 — En Çürük {DONGÜ_BOYUT}</h5>", unsafe_allow_html=True)
    col_d3.markdown(f"<h5 style='color:{DONGÜ_RENKLER[3]};'>⚫ Döngü 3 — En Sağır {DONGÜ_BOYUT}</h5>", unsafe_allow_html=True)

    d1 = sorted([s for s, v in puanlama.items() if v["dongü"] == DONGÜ_GUCLU], key=lambda s: puanlama[s]["skor"], reverse=True)
    d2 = sorted([s for s, v in puanlama.items() if v["dongü"] == DONGÜ_CURUK], key=lambda s: puanlama[s]["skor"])
    d3 = [s for s, v in puanlama.items() if v["dongü"] == DONGÜ_SAGIR]

    max_len = max(len(d1), len(d2), len(d3), 1)
    for i in range(max_len):
        c1, c2, c3 = st.columns(3)
        for col, liste, dongü_no in [(c1, d1, DONGÜ_GUCLU), (c2, d2, DONGÜ_CURUK), (c3, d3, DONGÜ_SAGIR)]:
            with col:
                if i < len(liste):
                    _render_coin_karti(liste[i], puanlama, dongü_no, btc_df, tum_data)

    # ── Alt: Durum Matrisi ──
    st.markdown("---")
    st.markdown("#### 🔲 Genel Durum Matrisi")
    _render_durum_matrisi(puanlama)


def _render_coin_karti(
    sym: str,
    puanlama: Dict,
    dongü: int,
    btc_df: Optional[pd.DataFrame],
    tum_data: Dict,
) -> None:
    """Tek bir coin kartını çizer."""
    info  = puanlama.get(sym, {})
    skor  = info.get("skor", 0.0)
    beta_n = info.get("beta_n")
    beta_v = info.get("beta_v")
    ayrisma = info.get("ayrisma", 0.0)

    renk_cls = {DONGÜ_GUCLU: "d1", DONGÜ_CURUK: "d2", DONGÜ_SAGIR: "d3"}.get(dongü, "d0")
    border_renk = DONGÜ_RENKLER.get(dongü, "#2a2a3e")

    # Beta satırı
    beta_str = ""
    if beta_n is not None:
        beta_dir = ""
        if beta_v is not None:
            diff = beta_v - beta_n
            beta_dir = f" <span style='color:#ffc107;'>{'⬆️' if diff > 0 else '⬇️'}</span>" if abs(diff) > 0.05 else ""
        beta_str = (
            f"BTC+1% → <b>{beta_n:+.2f}%</b>(N)"
            f"{f' | <b>{beta_v:+.2f}%</b>(V){beta_dir}' if beta_v is not None else ''}"
        )

    # Döngü bilgisi (Markov)
    sure_str = ""
    gecmis_cache = st.session_state.get("dongu_gecmis_cache", {})
    cache_key = f"{sym}_{st.session_state['hafiza_penceresi']}"
    gecmis = gecmis_cache.get(cache_key)
    if gecmis:
        bilgi = mevcut_dongü_bilgisi(gecmis)
        sure_h    = bilgi.get("sure_saat", 0)
        tahmin_h  = bilgi.get("tahmin_kalan", 0)
        sure_str  = f"⏱ {sure_h:.1f}s | ~{tahmin_h:.1f}s daha"

    html = (
        f"""<div class="coin-card coin-card-{renk_cls}" style="border-color:{border_renk};">"""
        f"""<b style="font-size:14px;">{sym}USDT</b><br>"""
        f"""Güç: <b>{skor:.0f}/100</b> | 6A: <b>{ayrisma*100:.0f}%</b><br>"""
        f"""{beta_str}<br>"""
        f"""{sure_str}"""
        f"""</div>"""
    )
    if st.markdown(html, unsafe_allow_html=True):
        pass

    # Tıklanabilir detay butonu
    if st.button(f"🔍 {sym}", key=f"detay_{sym}", use_container_width=True):
        st.session_state["coin_detay_acik"] = sym
        st.session_state["_aktif_sekme"] = "Röntgen"
        st.rerun()


def _render_durum_matrisi(puanlama: Dict) -> None:
    """Tüm coinlerin küçük kare matrisini çizer."""
    semboller = list(puanlama.keys())
    satirlar  = []
    satir     = []
    for i, sym in enumerate(semboller):
        satir.append(sym)
        if len(satir) == 10:
            satirlar.append(satir)
            satir = []
    if satir:
        satirlar.append(satir)

    for satir in satirlar:
        cols = st.columns(len(satir))
        for j, sym in enumerate(satir):
            dongü = puanlama.get(sym, {}).get("dongü", 0)
            renk  = DONGÜ_RENKLER.get(dongü, DONGÜ_RENKLER[0])
            with cols[j]:
                if st.button(
                    sym[:4],
                    key=f"mat_{sym}",
                    help=f"{sym} — Döngü {dongü}",
                    use_container_width=True,
                ):
                    st.session_state["coin_detay_acik"] = sym
                    st.rerun()
                st.markdown(
                    f"<div style='height:4px; background:{renk}; border-radius:2px; margin-top:-8px;'></div>",
                    unsafe_allow_html=True,
                )


# ──────────────────────────────────────────────
# SEKME 4: RÖNTGEN ODASI
# ──────────────────────────────────────────────

def render_rontgen_odasi() -> None:
    """Röntgen Odası sekmesi — tek coin derinlemesine analiz."""
    st.subheader("🔬 Röntgen Odası")

    puanlama = st.session_state.get("puanlama", {})
    tum_data = st.session_state.get("tum_data", {})
    btc_df   = st.session_state.get("btc_df")

    tum_semboller = sorted(list(puanlama.keys()) if puanlama else SABIT_COINLER)

    # Önceden seçilmiş coin varsa ona git
    default_coin = st.session_state.get("coin_detay_acik") or (tum_semboller[0] if tum_semboller else "ETH")
    if default_coin not in tum_semboller:
        default_coin = tum_semboller[0]

    secilen = st.selectbox(
        "Coin Seç",
        tum_semboller,
        index=tum_semboller.index(default_coin),
        format_func=lambda s: f"{s}USDT",
        key="rontgen_coin_sel",
    )
    st.session_state["coin_detay_acik"] = secilen

    # Coin özel periyot ve çarpanlar (global'den bağımsız)
    st.markdown("##### Özel Parametreler")
    cx1, cx2, cx3, cx4 = st.columns(4)
    with cx1:
        xray_per_idx = PERIYOT_LISTESI.index(st.session_state["xray_periyot"]) \
            if st.session_state["xray_periyot"] in PERIYOT_LISTESI else 0
        st.session_state["xray_periyot"] = st.selectbox(
            "Periyot", PERIYOT_LISTESI, index=xray_per_idx, key="xray_per",
        )
    with cx2:
        st.session_state["xray_weight_recent"] = st.number_input(
            "Yakın", value=float(st.session_state["xray_weight_recent"]),
            min_value=0.0, max_value=20.0, step=0.1, key="xr_wr",
        )
    with cx3:
        st.session_state["xray_weight_mid"] = st.number_input(
            "Orta", value=float(st.session_state["xray_weight_mid"]),
            min_value=0.0, max_value=20.0, step=0.1, key="xr_wm",
        )
    with cx4:
        st.session_state["xray_weight_old"] = st.number_input(
            "Eski", value=float(st.session_state["xray_weight_old"]),
            min_value=0.0, max_value=20.0, step=0.1, key="xr_wo",
        )

    hafiza_idx = HAFIZA_LISTESI.index(st.session_state["xray_hafiza"]) \
        if st.session_state["xray_hafiza"] in HAFIZA_LISTESI else 0
    col_h, col_isi = st.columns(2)
    with col_h:
        st.session_state["xray_hafiza"] = st.selectbox(
            "Hafıza Penceresi", HAFIZA_LISTESI, index=hafiza_idx, key="xray_haf",
        )
    with col_isi:
        isi_secenekler = ISI_ADIM_KISIT.get(st.session_state["xray_periyot"], ["15 dakika"])
        isi_mevcut = st.session_state["xray_isi_adim"]
        isi_idx = isi_secenekler.index(isi_mevcut) if isi_mevcut in isi_secenekler else 0
        st.session_state["xray_isi_adim"] = st.selectbox(
            "Isı Haritası Adım", isi_secenekler, index=isi_idx, key="xray_isi",
        )

    sym = secilen
    info = puanlama.get(sym, {})
    dongü = info.get("dongü", 0)
    skor  = info.get("skor",  50.0)
    beta_n = info.get("beta_n")
    beta_v = info.get("beta_v")
    ayrisma = info.get("ayrisma", 0.0)

    st.markdown("---")
    # Üst bilgiler
    dongü_etiket = DONGÜ_ETIKET.get(dongü, "Bilinmiyor")
    st.markdown(
        f"### {sym}USDT &nbsp;|&nbsp; {dongü_etiket} &nbsp;|&nbsp; Güç Skoru: **{skor:.0f}/100**"
    )
    st.caption(f"6A Ayrışma Gücü: **{ayrisma*100:.0f}%**")

    # Beta
    if beta_n is not None:
        yön_str = ""
        if beta_v is not None:
            diff = beta_v - beta_n
            yön_str = f"⬆️ Volatilitede güçleniyor" if diff > 0.05 \
                      else ("⬇️ Volatilitede zayıflıyor" if diff < -0.05 else "")
        st.markdown(
            f"**BTC +1%** → Normal: **{beta_n:+.2f}%** "
            f"{'| Volatilite: **' + f'{beta_v:+.2f}%**' if beta_v is not None else ''} "
            f"{yön_str}"
        )

    # ── Karakter Güvenilirliği ──
    st.markdown("---")
    st.markdown("##### Karakter Güvenilirliği")

    alt_df = tum_data.get(sym)
    if btc_df is not None and alt_df is not None:
        with st.spinner("Güvenilirlik hesaplanıyor..."):
            guvn = hesapla_karakter_guvenilirlik(sym, dongü, btc_df, alt_df)

        col_n, col_v = st.columns(2)
        with col_n:
            n_pct = guvn["normal_oran"] * 100
            st.markdown(
                f"**Normal anlarda:** `{n_pct:.0f}%` "
                f"{'✅' if n_pct >= 80 else '⚠️'}"
            )
            with st.expander(f"Detay ({guvn['normal_toplam']} normal an)"):
                st.write(
                    f"{guvn['normal_toplam']} normal anda incelendi.\n"
                    f"{guvn['normal_korudu']} anda karakterini korudu ({n_pct:.0f}%) ✅\n"
                    f"{guvn['normal_toplam'] - guvn['normal_korudu']} anda karakter eridi "
                    f"({100-n_pct:.0f}%) ⚠️"
                )
        with col_v:
            v_pct = guvn["vol_oran"] * 100
            st.markdown(
                f"**Volatilite anlarında:** `{v_pct:.0f}%` "
                f"{'✅' if v_pct >= 70 else '⚠️'}"
            )
            with st.expander(f"Detay ({guvn['vol_toplam']} volatilite anı)"):
                for idx, detay in enumerate(guvn.get("vol_detay", [])[:20]):
                    ikon = "✅" if detay["korudu"] else "⚠️"
                    tarih = detay["tarih"].strftime("%Y-%m-%d") if hasattr(detay["tarih"], "strftime") else str(detay["tarih"])
                    beta_d = f"(Beta: {detay['beta']:.2f}x)" if detay.get("beta") else ""
                    st.caption(
                        f"#{idx+1}  {tarih} | BTC: {detay['btc_ret']*100:+.2f}% → {sym}: "
                        f"{detay['alt_ret']*100:+.2f}%  {beta_d}  {ikon}"
                    )

    # ── Döngü Geçiş Haritası ──
    st.markdown("---")
    st.markdown("##### Döngü Geçiş Haritası")
    if btc_df is not None and alt_df is not None:
        gecmis = dongü_gecmis_hesapla(
            sym, btc_df, alt_df,
            st.session_state["xray_hafiza"],
            st.session_state["xray_weight_recent"],
            st.session_state["xray_weight_mid"],
            st.session_state["xray_weight_old"],
            tum_alt_dfs=tum_data,
        )
        _render_gecis_seridi(gecmis)

        # ── Markov Tahmini ──
        if gecmis:
            bilgi  = mevcut_dongü_bilgisi(gecmis)
            markov = bilgi.get("markov", {})
            sure_h = bilgi.get("sure_saat", 0)
            tahmin = bilgi.get("tahmin_kalan", 0)
            mvk_matris = markov.get("matris", {})
            ort_sure   = markov.get("ortalama_sure", {})
            en_cok     = markov.get("en_cok_gecilen", 0)
            en_cok_ort = markov.get("en_cok_oran", 0)

            st.markdown("##### Markov Tahmini")
            st.markdown(f"Seçilen pencere: **{st.session_state['xray_hafiza']}**")
            durum_str = DONGÜ_ETIKET.get(bilgi.get("mevcut_dongü", 0), "?")

            gecis_str = ""
            if bilgi.get("mevcut_dongü") in mvk_matris:
                gecisler = mvk_matris[bilgi["mevcut_dongü"]]
                for hedef, oran in gecisler.items():
                    if hedef != bilgi["mevcut_dongü"] and oran > 0:
                        gecis_str += f"- {DONGÜ_ETIKET.get(hedef,'?')} geçiş ihtimali: **{oran*100:.0f}%**\n"

            st.markdown(
                f"- {durum_str}'de ortalama kalma: **{ort_sure.get(bilgi.get('mevcut_dongü',0),1):.1f} adım**\n"
                f"{gecis_str}"
                f"- Şu an {durum_str}'de: **{sure_h:.1f} saat**\n"
                f"- Tahmini çıkış: **~{tahmin:.1f} saat sonra**\n"
                f"- En çok geçilen döngü: {DONGÜ_ETIKET.get(en_cok,'?')} (**{en_cok_ort*100:.0f}%**)"
            )

    # ── Isı Haritası ──
    st.markdown("---")
    st.markdown("##### Isı Haritası")
    if btc_df is not None and alt_df is not None:
        adim_str = st.session_state["xray_isi_adim"]
        adim_dk  = ISI_ADIM_DAKIKA.get(adim_str, 15)
        isi_df   = isi_haritasi_renkleri(btc_df, alt_df, adim_dk)
        _render_isi_haritasi(isi_df, sym)


def _render_gecis_seridi(gecmis: List[int]) -> None:
    """Döngü geçiş şeridini Plotly bar chart olarak çizer."""
    if not gecmis:
        st.info("Yeterli geçmiş verisi yok.")
        return

    renk_map = {DONGÜ_GUCLU: RENK_POZITIF, DONGÜ_CURUK: RENK_NEGATIF, DONGÜ_SAGIR: RENK_NOTR, 0: "#1e1e2e"}
    renkler  = [renk_map.get(d, "#1e1e2e") for d in gecmis]
    x_vals   = list(range(len(gecmis)))

    fig = go.Figure(go.Bar(
        x=x_vals,
        y=[1] * len(gecmis),
        marker_color=renkler,
        showlegend=False,
        hovertemplate=[f"{DONGÜ_ETIKET.get(d,'Diğer')}" for d in gecmis],
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=80,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        bargap=0.05,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("◀ Geçmiş &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Şu An ▶")


def _render_isi_haritasi(isi_df: pd.DataFrame, sym: str) -> None:
    """Isı haritasını Plotly ile çizer."""
    if isi_df is None or isi_df.empty:
        st.info("Isı haritası için yeterli veri yok.")
        return

    degerler = isi_df["deger"].values
    renkler  = []
    for d in degerler:
        if d > 0:
            renkler.append(RENK_POZITIF)
        elif d < 0:
            renkler.append(RENK_NEGATIF)
        else:
            renkler.append(RENK_NOTR)

    x_vals = list(range(len(degerler)))
    fig = go.Figure(go.Bar(
        x=x_vals,
        y=[1] * len(degerler),
        marker_color=renkler,
        showlegend=False,
        hovertemplate=[f"{'Baskın' if d > 0 else ('Ezik' if d < 0 else 'Nötr')}" for d in degerler],
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=100,
        title=f"{sym}USDT — Baskın/Ezik/Nötr Şeridi",
        title_font_color="#e8eaf6",
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        bargap=0.02,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"🟢 Baskın &nbsp;|&nbsp; 🔴 Ezik &nbsp;|&nbsp; ⚫ Nötr"
    )


# ──────────────────────────────────────────────
# OTOMATIK YENİLEME
# ──────────────────────────────────────────────

def otomatik_yenileme_kontrol() -> None:
    """10 dakikada bir otomatik yenileme yapılır."""
    son = st.session_state.get("son_tarama_zaman")
    if son is None:
        return
    gecen_dk = (datetime.now(timezone.utc) - son).total_seconds() / 60
    if gecen_dk >= AUTO_REFRESH_DAKIKA:
        veri_tara(zorunlu=True)
        st.rerun()


# ──────────────────────────────────────────────
# ANA AKIŞ
# ──────────────────────────────────────────────

def main():
    init_session_state()

    if not sifre_kontrolu():
        return

    exchange = get_cached_exchange()
    baslat_arsiv_ve_scheduler(exchange)

    render_sidebar(exchange)

    # İlk tarama
    veri_tara()

    # BTC panel sinyali hesapla (Bitcoin sekmesinden bağımsız önbellekle)
    btc_df = st.session_state.get("btc_df")
    if btc_df is not None:
        from btc_panel import volatilite_sinyali
        vsig = volatilite_sinyali(btc_df)
        st.session_state["btc_panel_sinyal"] = vsig.get("patlama_yakin", False)

    # Sekmeler
    tab1, tab2, tab3, tab4 = st.tabs([
        "📡 Canlı İzleme",
        "₿ Bitcoin Paneli",
        "✂️ Makas Paneli",
        "🔬 Röntgen Odası",
    ])

    with tab1:
        render_canli_izleme()

    with tab2:
        btc_panel_sonuc = render_btc_panel(
            btc_df,
            st.session_state.get("btc_daily_df"),
        )
        st.session_state["btc_panel_sinyal"] = btc_panel_sonuc.get("patlama_yakin", False)

    with tab3:
        render_makas_panel(
            puanlama=st.session_state.get("puanlama", {}),
            btc_panel_sinyal=st.session_state.get("btc_panel_sinyal", False),
            hafiza_penceresi=st.session_state["hafiza_penceresi"],
            btc_df=btc_df,
            alt_dfs=st.session_state.get("tum_data", {}),
            weight_recent=st.session_state["global_weight_recent"],
            weight_mid=st.session_state["global_weight_mid"],
            weight_old=st.session_state["global_weight_old"],
        )

    with tab4:
        render_rontgen_odasi()

    # Otomatik yenileme kontrolü
    otomatik_yenileme_kontrol()

    # Footer uyarısı
    st.markdown(
        """<div class="footer-uyari">
        ⚠️ Bu bir yatırım tavsiyesi değildir. Tüm işlem kararları kullanıcıya aittir.
        Kripto para piyasaları yüksek risk içerir.
        </div>""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
