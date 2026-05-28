# config.py — Sabitler ve Coin Evreni
# ─────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────
# COİN LİSTESİ
# ──────────────────────────────────────────────
SABIT_COINLER = [
    "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "DOGE", "DOT", "MATIC", "LINK",
    "UNI", "ATOM", "LTC", "ETC", "FIL", "APT", "ARB", "OP", "INJ", "SUI",
    "TIA", "SEI", "WIF", "BONK", "PEPE", "SHIB", "FTM", "NEAR", "ALGO", "VET",
    "SAND", "MANA", "AXS", "GALA", "ENJ", "CHZ", "HBAR", "ICP", "EOS", "XLM",
    "TRX", "NEO", "IOTA", "DASH", "ZEC", "XMR", "CAKE", "AAVE", "SNX", "CRV",
    "MKR", "COMP", "LDO", "RUNE", "STX", "FLOKI", "BLUR", "GMX", "DYDX", "IMX",
]

MAX_COIN = 70                   # Sabit 60 + maks 10 manuel ek
MAX_MANUEL_EK = 10

# ──────────────────────────────────────────────
# BYBIT API (Binance ABD sunucularında bloke — Bybit açık)
# ──────────────────────────────────────────────
EXCHANGE_ID = "bybit"
REQUEST_DELAY = 0.3
ARCHIVE_DELAY = 1.5
RATE_LIMIT_WAIT = [5, 15, 45]
MAX_RETRY = 3

BYBIT_BASE_URL   = "https://api.bybit.com"
KLINES_URL       = f"{BYBIT_BASE_URL}/v5/market/kline"
TICKER_URL       = f"{BYBIT_BASE_URL}/v5/market/tickers"
FUNDING_RATE_URL = f"{BYBIT_BASE_URL}/v5/market/funding/history"
PING_URL         = f"{BYBIT_BASE_URL}/v5/market/time"

# Eski compat
BINANCE_BASE_URL = BYBIT_BASE_URL

# ──────────────────────────────────────────────
# PERIYOT → CCXT TIMEFRAME EŞLEŞMESİ
# ──────────────────────────────────────────────
# Her UI periyotu için (timeframe, gereken_mum_sayısı) tuple döner.
# 8h/12h/1G/2G/3G/4G → düşük timeframe'den yeniden örneklenir.

PERIYOT_KONFIG = {
    "1m":      {"tf": "1m",  "limit": 960,   "resample": None},
    "3m":      {"tf": "3m",  "limit": 320,   "resample": None},
    "5m":      {"tf": "5m",  "limit": 1008,  "resample": None},
    "8h":      {"tf": "1m",  "limit": 480,   "resample": "8h"},
    "12h":     {"tf": "1m",  "limit": 720,   "resample": "12h"},
    "1 Gün":   {"tf": "5m",  "limit": 288,   "resample": "1D"},
    "2 Gün":   {"tf": "5m",  "limit": 576,   "resample": "1D"},
    "3 Gün":   {"tf": "5m",  "limit": 864,   "resample": "1D"},
    "4 Gün":   {"tf": "5m",  "limit": 1152,  "resample": "1D"},
    "1 Hafta": {"tf": "15m", "limit": 672,   "resample": "1D"},
    "15 Gün":  {"tf": "15m", "limit": 1440,  "resample": "1D"},
}

PERIYOT_LISTESI = list(PERIYOT_KONFIG.keys())

# ──────────────────────────────────────────────
# AĞIRLIK VARSAYILANLARI
# ──────────────────────────────────────────────
DEFAULT_WEIGHT_RECENT = 4.5
DEFAULT_WEIGHT_MID    = 1.5
DEFAULT_WEIGHT_OLD    = 0.2

# ──────────────────────────────────────────────
# HAFIZA PENCERELERİ (Markov)
# ──────────────────────────────────────────────
HAFIZA_LISTESI = ["1 Gün", "2 Gün", "3 Gün", "4 Gün", "1 Hafta", "2 Hafta", "1 Ay", "3 Ay"]

HAFIZA_GUNLER = {
    "1 Gün":   1,
    "2 Gün":   2,
    "3 Gün":   3,
    "4 Gün":   4,
    "1 Hafta": 7,
    "2 Hafta": 14,
    "1 Ay":    30,
    "3 Ay":    90,
}

# ──────────────────────────────────────────────
# DÖNGÜ TANIMLARI
# ──────────────────────────────────────────────
DONGÜ_GUCLU  = 1   # Yeşil (En Güçlü 15)
DONGÜ_CURUK  = 2   # Kırmızı (En Çürük 15)
DONGÜ_SAGIR  = 3   # Gri (En Sağır 15)
DONGÜ_BOYUT  = 15  # Her döngüdeki coin sayısı

DONGÜ_RENKLER = {
    DONGÜ_GUCLU: "#00c853",   # Koyu yeşil
    DONGÜ_CURUK: "#d50000",   # Koyu kırmızı
    DONGÜ_SAGIR: "#616161",   # Gri
    0:           "#1e1e2e",   # Listedeki diğerleri
}

DONGÜ_ETIKET = {
    DONGÜ_GUCLU: "Döngü 1 🟢",
    DONGÜ_CURUK: "Döngü 2 🔴",
    DONGÜ_SAGIR: "Döngü 3 ⚫",
}

# ──────────────────────────────────────────────
# VERİTABANI
# ──────────────────────────────────────────────
DB_PATH        = "/tmp/archive.db"    # Render.com free plan (restart'ta sıfırlanır)
DB_PATH_LOCAL  = "./archive.db"       # Yerel geliştirme için fallback
ARCHIVE_DAYS   = 365                  # 1 yıllık arşiv

# ──────────────────────────────────────────────
# KOMİSYON ORANLARI (config değiştir = yeter)
# ──────────────────────────────────────────────
TAKER_RATE         = 0.00040   # %0.040
TAKER_RATE_BNB     = 0.00036   # %0.036 (BNB ile ödeme)
MAKER_RATE         = 0.00020   # %0.020
MAKER_RATE_BNB     = 0.00018   # %0.018 (BNB ile ödeme)

# ──────────────────────────────────────────────
# VOLATİLİTE TANIMLAMA PARAMETRELERİ
# ──────────────────────────────────────────────
BOLL_WINDOW          = 20     # Bollinger bandı penceresi
BOLL_STD             = 2.0    # Bollinger standart sapma çarpanı
BOLL_NARROW_PCT      = 20     # Bant genişliğinin en dar % kaçı
ATR_WINDOW           = 14     # ATR penceresi

# ──────────────────────────────────────────────
# HACİM PATLAMASı EŞIĞI
# ──────────────────────────────────────────────
HACIM_PATLAMA_KATI   = 3.0    # 24s hacmi ortalama X katına çıkarsa uyar

# ──────────────────────────────────────────────
# MAKAS SKORU AĞIRLIKLARI
# ──────────────────────────────────────────────
MAKAS_AGIRLIK = {
    "volatilite_beta_farki":   0.30,
    "karakter_guvenilirlik":   0.25,
    "dongude_kalma":           0.20,
    "volatilite_sinyali":      0.15,
    "komisyon_basa_bas":       0.10,
}
MAKAS_MIN_SKOR = 60   # Banner için minimum skor eşiği

# ──────────────────────────────────────────────
# ARAYÜZ RENKLERİ
# ──────────────────────────────────────────────
RENK_ARKA_PLAN   = "#0e1117"
RENK_KART        = "#1a1c26"
RENK_YENILE_BTN  = "#ff6b35"
RENK_BASLIK      = "#e8eaf6"
RENK_POZITIF     = "#00e676"
RENK_NEGATIF     = "#ff1744"
RENK_NOTR        = "#78909c"

# ──────────────────────────────────────────────
# OTOMATIK YENİLEME
# ──────────────────────────────────────────────
AUTO_REFRESH_DAKIKA  = 10    # Her X dakikada bir otomatik taze çekim teklifi
STALE_DAKIKA         = 60    # 60 dakika geçmişse otomatik yeniden çek

# ──────────────────────────────────────────────
# ISITMA HARİTASI ADIM KISITLARI
# ──────────────────────────────────────────────
ISI_ADIM_KISIT = {
    "1m":      ["1 dakika"],
    "3m":      ["1 dakika", "5 dakika"],
    "5m":      ["1 dakika", "5 dakika"],
    "8h":      ["5 dakika", "15 dakika"],
    "12h":     ["5 dakika", "15 dakika"],
    "1 Gün":   ["5 dakika", "15 dakika"],
    "2 Gün":   ["5 dakika", "15 dakika"],
    "3 Gün":   ["15 dakika", "1 saat"],
    "4 Gün":   ["15 dakika", "1 saat"],
    "1 Hafta": ["1 saat"],
    "15 Gün":  ["1 saat"],
}

ISI_ADIM_DAKIKA = {
    "1 dakika":  1,
    "5 dakika":  5,
    "15 dakika": 15,
    "1 saat":    60,
}

# ──────────────────────────────────────────────
# BITCOIN PANELİ
# ──────────────────────────────────────────────
BTC_PANEL_PERIYOTLAR = ["4 Saat", "1 Gün", "1 Hafta"]
BTC_PANEL_DEFAULT    = "4 Saat"
BTC_PANEL_KONFIG = {
    "4 Saat":  {"tf": "1m",  "limit": 240,  "resample": "4h"},
    "1 Gün":   {"tf": "5m",  "limit": 288,  "resample": "1D"},
    "1 Hafta": {"tf": "15m", "limit": 672,  "resample": "1D"},
}

PATLAMA_GECMIS_GUN = 90   # Son 90 günlük patlama kayıtları gösterilir

# ──────────────────────────────────────────────
# 6 AYLIK AYRIŞMA GÜCÜ
# ──────────────────────────────────────────────
BTC_DUSUS_ESIK    = -0.02  # %2 ve üzeri düşüş günleri
AYRISMA_AY        = 6      # Son 6 ay

# ──────────────────────────────────────────────
# GÜÇ SKORU TOLERANSI
# ──────────────────────────────────────────────
GUCLU_TOLERANS    = 0.005   # Piyasa ortalamasının +%0.5 üstü → baskın
EZIK_TOLERANS     = -0.005  # Piyasa ortalamasının -%0.5 altı → ezik
