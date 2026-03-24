"""
SMC Trading Bot v1.3
Trader: @kasun9125 | Pori, Finland
Fixes: Smart quotes, API retry, Unmitigated OB/FVG,
       OB-edge SL, Env validation, Sent cleanup,
       4H bias stickiness, Configurable threshold
"""

import os
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# ─── CONFIGURATION ────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Configurable parameters (no more magic numbers)
CHECK_INTERVAL       = int(os.environ.get("CHECK_INTERVAL", "900"))       # seconds
SIGNAL_THRESHOLD     = int(os.environ.get("SIGNAL_THRESHOLD", "65"))      # min confluence score
SIGNAL_COOLDOWN      = int(os.environ.get("SIGNAL_COOLDOWN", "7200"))     # 2 hours per asset
API_RETRIES          = int(os.environ.get("API_RETRIES", "3"))            # Binance retry count
API_RETRY_DELAY      = int(os.environ.get("API_RETRY_DELAY", "5"))        # seconds between retries
SENT_CLEANUP_HOURS   = int(os.environ.get("SENT_CLEANUP_HOURS", "24"))    # clear sent dict after 24h

ASSETS = {
    "BTC/USDT": {"round": 1000, "symbol": "BTCUSDT"},
    "ETH/USDT": {"round": 100,  "symbol": "ETHUSDT"},
    "SOL/USDT": {"round": 10,   "symbol": "SOLUSDT"},
    "XRP/USDT": {"round": 0.1,  "symbol": "XRPUSDT"},
    "BNB/USDT": {"round": 10,   "symbol": "BNBUSDT"},
}

FINLAND_TZ = pytz.timezone("Europe/Helsinki")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("smc_bot.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ─── ENVIRONMENT VALIDATION ───────────────────────────────────
def validate_env() -> bool:
    """Validate all required environment variables on startup."""
    errors = []
    if not TELEGRAM_TOKEN:
        errors.append("TELEGRAM_TOKEN is not set in Railway Variables!")
    elif ":" not in TELEGRAM_TOKEN:
        errors.append("TELEGRAM_TOKEN format looks wrong — should be 1234567890:ABCdef...")
    if not TELEGRAM_CHAT_ID:
        errors.append("TELEGRAM_CHAT_ID is not set in Railway Variables!")
    elif not TELEGRAM_CHAT_ID.lstrip("-").isdigit():
        errors.append("TELEGRAM_CHAT_ID should be a number — get it from @userinfobot")
    if errors:
        for e in errors:
            log.error(f"CONFIG ERROR: {e}")
        return False
    log.info("Environment variables validated OK")
    return True


# ─── TELEGRAM ─────────────────────────────────────────────────
def send_telegram(message: str) -> bool:
    """Send message to Telegram with retry."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for attempt in range(1, API_RETRIES + 1):
        try:
            r = requests.post(url, json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML"
            }, timeout=10)
            data = r.json()
            if data.get("ok"):
                log.info("Telegram sent!")
                return True
            log.error(f"Telegram error (attempt {attempt}): {data}")
            if data.get("error_code") == 404:
                log.error("Token is wrong — check TELEGRAM_TOKEN in Railway Variables!")
                return False
        except Exception as e:
            log.error(f"Telegram exception (attempt {attempt}): {e}")
        if attempt < API_RETRIES:
            time.sleep(API_RETRY_DELAY)
    return False


# ─── DATA FETCHING WITH RETRY ─────────────────────────────────
def get_klines(symbol: str, interval: str, limit: int = 200):
    """Fetch OHLCV from Binance with retry logic."""
    url = "https://api.binance.me/api/v3/klines"
    fallback_url = "https://api.binance.com/api/v3/klines"
    for attempt in range(1, API_RETRIES + 1):
        try:
            r = requests.get(url, params={
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }, timeout=15)
            r.raise_for_status()
            raw = r.json()
            if not raw or len(raw) < 30:
                log.warning(f"{symbol} {interval}: only {len(raw) if raw else 0} candles")
                return None
            df = pd.DataFrame(raw, columns=[
                "ts", "open", "high", "low", "close", "vol",
                "ct", "qv", "trades", "tbb", "tbq", "ignore"
            ])
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
            if len(df) < 30:
                return None
            log.info(f"{symbol} {interval}: {len(df)} candles OK")
            return df
        except requests.exceptions.HTTPError as e:
            if r.status_code == 429:
                log.warning(f"Binance rate limit hit — waiting 60s")
                time.sleep(60)
            else:
                log.error(f"Binance HTTP error (attempt {attempt}): {e}")
        except Exception as e:
            log.error(f"get_klines {symbol} {interval} (attempt {attempt}): {e}")
        if attempt < API_RETRIES:
            time.sleep(API_RETRY_DELAY)
    log.error(f"Failed to fetch {symbol} {interval} after {API_RETRIES} attempts")
    return None


def safe_get(series, idx=-1):
    """Safely get float from pandas series."""
    try:
        v = float(series.iloc[idx])
        return None if np.isnan(v) else v
    except Exception:
        return None


# ─── INDICATORS ───────────────────────────────────────────────
def calc_rsi(closes: pd.Series, period: int = 14) -> float:
    """Calculate RSI — used as divergence/exhaustion check only."""
    try:
        if len(closes) < period + 5:
            return 50.0
        d    = closes.diff()
        up   = d.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        down = (-d.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        rs   = up / down.replace(0, np.nan)
        rsi  = 100 - (100 / (1 + rs))
        v    = safe_get(rsi)
        return round(v, 1) if v else 50.0
    except Exception:
        return 50.0


def calc_ema(closes: pd.Series, period: int) -> float:
    """Calculate EMA and return last value."""
    try:
        v = safe_get(closes.ewm(span=period, adjust=False).mean())
        return v if v else 0.0
    except Exception:
        return 0.0


def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range."""
    try:
        tr = (df["high"] - df["low"]).tail(period)
        atr = float(tr.mean())
        return atr if not np.isnan(atr) and atr > 0 else 0.0
    except Exception:
        return 0.0


# ─── STRUCTURE DETECTION (4H BIAS — STICKY) ──────────────────
def detect_structure(df: pd.DataFrame) -> str:
    """
    Detect market structure using 4H swing highs/lows.
    Uses larger lookback to avoid flipping on minor pullbacks.
    Prioritizes clear HH/HL or LH/LL patterns only.
    """
    try:
        if len(df) < 50:
            return "UNKNOWN"
        h = df["high"].values.astype(float)
        l = df["low"].values.astype(float)
        n = len(h)

        # Use larger window (5) to avoid noise — more sticky
        sh, sl = [], []
        for i in range(5, n - 5):
            if h[i] == max(h[i-5:i+6]):
                sh.append((i, h[i]))
            if l[i] == min(l[i-5:i+6]):
                sl.append((i, l[i]))

        # Need at least 3 swings for reliable structure
        if len(sh) >= 3 and len(sl) >= 3:
            # Check last 2 swing highs and lows
            hh = sh[-1][1] > sh[-2][1]
            hl = sl[-1][1] > sl[-2][1]
            lh = sh[-1][1] < sh[-2][1]
            ll = sl[-1][1] < sl[-2][1]

            # Require BOTH conditions for clean bias
            if hh and hl:
                return "BULLISH"
            if lh and ll:
                return "BEARISH"
            # Only one condition = transitioning
            if lh and hl:
                return "TRANSITIONING"
            if hh and ll:
                return "TRANSITIONING"

        return "CHOPPY"
    except Exception as e:
        log.error(f"detect_structure: {e}")
        return "UNKNOWN"


# ─── OB DETECTION WITH MITIGATION CHECK ───────────────────────
def detect_ob(df: pd.DataFrame, bias: str, current_price: float) -> dict:
    """
    Detect Order Block with unmitigated check.
    An OB is only valid if price has NOT returned to fill it yet.
    """
    empty = {"found": False, "high": 0.0, "low": 0.0, "mid": 0.0, "sl": 0.0}
    try:
        n = len(df)
        if n < 15:
            return empty

        for i in range(n - 5, 3, -1):
            if i + 1 >= n:
                continue
            c   = df.iloc[i]
            c1  = df.iloc[i + 1]
            ch  = float(c["high"]);  cl  = float(c["low"])
            co  = float(c["open"]); cc  = float(c["close"])
            c1o = float(c1["open"]); c1c = float(c1["close"])

            if any(np.isnan([ch, cl, co, cc, c1o, c1c])):
                continue

            ob_found = False
            if bias == "BULLISH" and cc < co and c1c > c1o and c1c > ch:
                ob_found = True
            if bias == "BEARISH" and cc > co and c1c < c1o and c1c < cl:
                ob_found = True

            if ob_found:
                # MITIGATION CHECK
                # For bullish OB: price should not have closed below OB low
                # For bearish OB: price should not have closed above OB high
                mitigated = False
                future_closes = df["close"].iloc[i+2:].values

                if bias == "BULLISH":
                    if any(float(fc) < cl for fc in future_closes):
                        mitigated = True
                    # Also check current price is near OB (within 2%)
                    if current_price < cl * 0.98:
                        mitigated = True
                elif bias == "BEARISH":
                    if any(float(fc) > ch for fc in future_closes):
                        mitigated = True
                    # Also check current price is near OB (within 2%)
                    if current_price > ch * 1.02:
                        mitigated = True

                if not mitigated:
                    # SL placed at OB edge (not ATR-based)
                    sl_level = round(cl * 0.9995, 4) if bias == "BULLISH" else round(ch * 1.0005, 4)
                    return {
                        "found": True,
                        "high":  round(ch, 4),
                        "low":   round(cl, 4),
                        "mid":   round((ch + cl) / 2, 4),
                        "sl":    sl_level
                    }
    except Exception as e:
        log.error(f"detect_ob: {e}")
    return empty


# ─── FVG DETECTION WITH MITIGATION CHECK ──────────────────────
def detect_fvg(df: pd.DataFrame, bias: str, current_price: float) -> dict:
    """
    Detect Fair Value Gap with mitigation check.
    Only valid if price has not already filled the gap.
    """
    empty = {"found": False, "high": 0.0, "low": 0.0, "mid": 0.0}
    try:
        n = len(df)
        if n < 10:
            return empty

        for i in range(n - 4, 2, -1):
            if i - 1 < 0 or i + 1 >= n:
                continue
            c1h = float(df.iloc[i-1]["high"]); c1l = float(df.iloc[i-1]["low"])
            c3h = float(df.iloc[i+1]["high"]); c3l = float(df.iloc[i+1]["low"])

            if any(np.isnan([c1h, c1l, c3h, c3l])):
                continue

            fvg_found = False
            fvg_high = fvg_low = 0.0

            if bias == "BULLISH" and c3l > c1h:
                fvg_found = True
                fvg_high = c3l
                fvg_low  = c1h
            elif bias == "BEARISH" and c3h < c1l:
                fvg_found = True
                fvg_high = c1l
                fvg_low  = c3h

            if fvg_found:
                # MITIGATION CHECK
                # FVG is mitigated if price has already traded through it
                mitigated = False
                future_data = df.iloc[i+2:]

                if bias == "BULLISH":
                    if any(float(r["low"]) < fvg_low for _, r in future_data.iterrows()):
                        mitigated = True
                elif bias == "BEARISH":
                    if any(float(r["high"]) > fvg_high for _, r in future_data.iterrows()):
                        mitigated = True

                if not mitigated:
                    return {
                        "found": True,
                        "high":  round(fvg_high, 4),
                        "low":   round(fvg_low, 4),
                        "mid":   round((fvg_high + fvg_low) / 2, 4)
                    }
    except Exception as e:
        log.error(f"detect_fvg: {e}")
    return empty


# ─── SWEEP / CHOCH ────────────────────────────────────────────
def detect_sweep(df: pd.DataFrame, bias: str) -> bool:
    """Detect recent liquidity sweep."""
    try:
        if len(df) < 25:
            return False
        rh = float(df["high"].iloc[-22:-4].max())
        rl = float(df["low"].iloc[-22:-4].min())
        lh = float(df["high"].iloc[-1])
        ll = float(df["low"].iloc[-1])
        lc = float(df["close"].iloc[-1])
        if bias == "BEARISH":
            return lh >= rh * 0.999 and lc < rh * 0.997
        if bias == "BULLISH":
            return ll <= rl * 1.001 and lc > rl * 1.003
    except Exception as e:
        log.error(f"detect_sweep: {e}")
    return False


def detect_choch(df: pd.DataFrame, bias: str) -> bool:
    """Detect Change of Character or Break of Structure."""
    try:
        if len(df) < 20:
            return False
        t  = df.tail(15)
        lc = float(t["close"].iloc[-1])
        if bias == "BEARISH":
            return lc < float(t["low"].iloc[:-2].min())
        if bias == "BULLISH":
            return lc > float(t["high"].iloc[:-2].max())
    except Exception as e:
        log.error(f"detect_choch: {e}")
    return False


# ─── PDH / PDL ────────────────────────────────────────────────
def get_pdh_pdl(df) -> dict:
    """Get Previous Day High and Low."""
    try:
        if df is None or len(df) < 2:
            return {"pdh": 0.0, "pdl": 0.0}
        return {
            "pdh": round(float(df["high"].iloc[-2]), 4),
            "pdl": round(float(df["low"].iloc[-2]),  4)
        }
    except Exception:
        return {"pdh": 0.0, "pdl": 0.0}


def check_pdh_pdl(df, pdh: float, pdl: float) -> dict:
    """Check for PDH/PDL breakout and retest."""
    empty = {"signal": None, "level": 0, "type": ""}
    try:
        if pdh == 0 or len(df) < 3:
            return empty
        price = float(df["close"].iloc[-1])
        high  = float(df["high"].iloc[-1])
        low   = float(df["low"].iloc[-1])
        if high > pdh and abs(price - pdh) / pdh < 0.002:
            return {"signal": "BUY",  "level": pdh, "type": "PDH Retest"}
        if low < pdl and abs(price - pdl) / pdl < 0.002:
            return {"signal": "SELL", "level": pdl, "type": "PDL Retest"}
    except Exception:
        pass
    return empty


# ─── ROUND NUMBERS ────────────────────────────────────────────
def get_round_nums(price: float, step: float) -> dict:
    """Get nearest institutional round numbers."""
    try:
        lo = round(np.floor(price / step) * step, 4)
        return {"lower": lo, "upper": round(lo + step, 4)}
    except Exception:
        return {"lower": 0.0, "upper": 0.0}


# ─── SESSION DETECTION ────────────────────────────────────────
def get_session() -> dict:
    """Detect current trading session for Finland."""
    now    = datetime.now(FINLAND_TZ)
    h      = now.hour + now.minute / 60
    summer = 4 <= now.month <= 9

    if summer:
        if   3  <= h < 11: s, p, m = "Asian Session",      "LOW",   -15
        elif 11 <= h < 16: s, p, m = "London Session",      "HIGH",    0
        elif 16 <= h < 19: s, p, m = "London/NY Overlap",   "BEST",    5
        elif 19 <= h < 21: s, p, m = "NY Session",          "HIGH",    0
        else:               s, p, m = "Off Hours",           "AVOID", -15
        kz = "London KZ" if 11 <= h < 12 else "NY KZ" if 16 <= h < 17 else "None"
    else:
        if   2  <= h < 10: s, p, m = "Asian Session",      "LOW",   -15
        elif 10 <= h < 15: s, p, m = "London Session",      "HIGH",    0
        elif 15 <= h < 18: s, p, m = "London/NY Overlap",   "BEST",    5
        elif 18 <= h < 20: s, p, m = "NY Session",          "HIGH",    0
        else:               s, p, m = "Off Hours",           "AVOID", -15
        kz = "London KZ" if 10 <= h < 11 else "NY KZ" if 15 <= h < 16 else "None"

    if kz != "None":
        m += 5

    # Astro layer (kept as per trader preference)
    wd = now.weekday()
    if   wd == 2: ast, am = "Wednesday Mercury", 5
    elif wd == 3: ast, am = "Thursday Jupiter",  5
    elif wd == 5: ast, am = "Saturday Saturn",  -5
    else:         ast, am = now.strftime("%A"),   0

    return {
        "session":   s,
        "priority":  p,
        "killzone":  kz,
        "score_mod": m,
        "astro":     ast,
        "amod":      am,
        "time_fi":   now.strftime("%H:%M EET"),
        "date_fi":   now.strftime("%d %b %Y"),
    }


# ─── CONFLUENCE SCORING ───────────────────────────────────────
def score_confluence(bias, ob, fvg, sweep, choch,
                     rsi, sess, price, rn) -> dict:
    """
    Calculate SMC confluence score.
    RSI used as exhaustion check only — not trend confirmation.
    """
    sc = 0; facts = []; warns = []

    if bias in ("BULLISH", "BEARISH"):
        sc += 20
        facts.append(f"HTF {bias.capitalize()}")

    if ob["found"] or fvg["found"]:
        sc += 20
        z = "OB+FVG" if ob["found"] and fvg["found"] else \
            "OB" if ob["found"] else "FVG"
        facts.append(f"Unmitigated {z} zone")

    if sweep:
        sc += 20
        facts.append("Liquidity sweep")

    if choch:
        sc += 20
        facts.append("CHOCH/BOS confirmed")

    # RSI used as exhaustion check only (extreme levels)
    if bias == "BEARISH" and rsi > 75:
        sc += 5
        facts.append(f"RSI exhaustion ({rsi})")
    elif bias == "BULLISH" and rsi < 25:
        sc += 5
        facts.append(f"RSI exhaustion ({rsi})")

    if sess["priority"] in ("HIGH", "BEST"):
        sc += 5
        facts.append(sess["session"])

    if sess["killzone"] != "None":
        sc += 5
        facts.append("ICT Killzone")

    if price > 0 and rn["lower"] > 0:
        near = (abs(price - rn["lower"]) / price < 0.003 or
                abs(price - rn["upper"]) / price < 0.003)
        if near:
            sc += 5
            facts.append("Round number")

    # Session and astro modifiers
    sc += sess["score_mod"] + sess["amod"]
    sc  = max(0, min(100, sc))

    # Warnings
    if rsi < 35 and bias == "BEARISH":
        warns.append(f"RSI oversold ({rsi}) — bounce risk")
    if rsi > 65 and bias == "BULLISH":
        warns.append(f"RSI overbought ({rsi}) — pullback risk")
    if sess["priority"] in ("AVOID", "LOW"):
        warns.append(f"{sess['session']} — low probability")

    return {"score": sc, "factors": facts, "warns": warns}


# ─── SENT DICTIONARY CLEANUP ──────────────────────────────────
def cleanup_sent(sent: dict) -> dict:
    """Remove old entries from sent dictionary to prevent memory growth."""
    cutoff = time.time() - (SENT_CLEANUP_HOURS * 3600)
    cleaned = {k: v for k, v in sent.items() if v > cutoff}
    removed = len(sent) - len(cleaned)
    if removed > 0:
        log.info(f"Cleaned {removed} old entries from sent dictionary")
    return cleaned


# ─── SIGNAL GENERATION ────────────────────────────────────────
def generate_signal(asset: str, info: dict):
    """Generate SMC trading signal for an asset."""
    try:
        log.info(f"--- Analyzing {asset} ---")

        # Fetch data
        df4  = get_klines(info["symbol"], "4h",  200)
        df15 = get_klines(info["symbol"], "15m", 200)
        df1d = get_klines(info["symbol"], "1d",   15)

        if df4 is None or df15 is None:
            log.warning(f"{asset}: Missing data")
            return None

        log.info(f"{asset}: df4={len(df4)} df15={len(df15)} candles")

        # Current price
        price = safe_get(df15["close"])
        if not price:
            log.warning(f"{asset}: No price")
            return None

        # HTF bias (sticky — uses 4H with large window)
        bias = detect_structure(df4)
        log.info(f"{asset}: bias={bias} price={price}")

        if bias in ("CHOPPY", "TRANSITIONING", "UNKNOWN"):
            log.info(f"{asset}: Skip — {bias}")
            return None

        # MA filter
        e20 = calc_ema(df4["close"], 20)
        e50 = calc_ema(df4["close"], 50)
        if e20 == 0 or e50 == 0:
            return None
        ma = "BULLISH" if e20 > e50 else "BEARISH"
        if ma != bias:
            log.info(f"{asset}: MA conflict {ma} vs {bias}")
            return None

        # Technical analysis
        rsi   = calc_rsi(df15["close"])
        ob    = detect_ob(df15, bias, price)       # with mitigation check
        fvg   = detect_fvg(df15, bias, price)      # with mitigation check
        sweep = detect_sweep(df15, bias)
        choch = detect_choch(df15, bias)
        pdhl  = get_pdh_pdl(df1d)
        pdhs  = check_pdh_pdl(df15, pdhl["pdh"], pdhl["pdl"])
        rn    = get_round_nums(price, info["round"])
        sess  = get_session()
        conf  = score_confluence(bias, ob, fvg, sweep, choch,
                                 rsi, sess, price, rn)
        score = conf["score"]

        log.info(f"{asset}: score={score} ob={ob['found']} fvg={fvg['found']} "
                 f"sweep={sweep} choch={choch} rsi={rsi}")

        if score < SIGNAL_THRESHOLD:
            log.info(f"{asset}: Score {score} < threshold {SIGNAL_THRESHOLD}")
            return None

        # LEVELS — SL at OB edge (not ATR-based)
        atr = calc_atr(df15)
        if atr == 0:
            atr = price * 0.001

        if bias == "BULLISH":
            ea  = round(price, 4)
            ec  = round(ob["high"] if ob["found"] else price * 0.999, 4)
            eo  = round(ob["mid"]  if ob["found"] else price * 0.9995, 4)
            # SL below OB low if OB found, else ATR-based
            sl  = ob["sl"] if ob["found"] else round(price - atr * 1.5, 4)
            tp1 = round(price + atr * 2.0, 4)
            tp2 = round(price + atr * 4.0, 4)
            sig = "BUY"; emo = "📈"
        else:
            ea  = round(price, 4)
            ec  = round(ob["low"] if ob["found"] else price * 1.001, 4)
            eo  = round(ob["mid"] if ob["found"] else price * 1.0005, 4)
            # SL above OB high if OB found, else ATR-based
            sl  = ob["sl"] if ob["found"] else round(price + atr * 1.5, 4)
            tp1 = round(price - atr * 2.0, 4)
            tp2 = round(price - atr * 4.0, 4)
            sig = "SELL"; emo = "📉"

        risk   = abs(price - sl)
        reward = abs(tp1 - price)
        rr     = round(reward / risk, 1) if risk > 0 else 0

        facts = "\n".join([f"✅ {f}" for f in conf["factors"][:4]])
        warns = "\n".join([f"⚠️ {w}" for w in conf["warns"]]) if conf["warns"] else ""
        pnote = f"\n🎯 <b>{pdhs['type']}</b> @ {pdhs['level']}" if pdhs["signal"] else ""

        msg = (
            f"─────────────────────────────\n"
            f"🤖 <b>SMC BOT SIGNAL</b>\n"
            f"─────────────────────────────\n"
            f"📊 <b>{asset} | 4H+15M</b>\n"
            f"⏰ {sess['session']} | KZ: {sess['killzone']}\n"
            f"🪐 {sess['astro']} | {sess['time_fi']}\n"
            f"─────────────────────────────\n"
            f"{emo} <b>{sig}</b>\n"
            f"💯 Confidence: <b>{score}%</b>\n"
            f"─────────────────────────────\n"
            f"Aggressive:   <code>{ea}</code>\n"
            f"Conservative: <code>{ec}</code>\n"
            f"50% OB:       <code>{eo}</code>\n\n"
            f"SL:  <code>{sl}</code>\n"
            f"TP1: <code>{tp1}</code>\n"
            f"TP2: <code>{tp2}</code>\n"
            f"RR:  1:{rr}\n"
            f"─────────────────────────────\n"
            f"PDH: <code>{pdhl['pdh']}</code> | "
            f"PDL: <code>{pdhl['pdl']}</code>\n"
            f"─────────────────────────────\n"
            f"{facts}{pnote}\n"
            f"{warns}\n"
            f"⏳ TP1 est: 2-8 hours\n"
            f"─────────────────────────────\n"
            f"<i>Confirm with Claude Project first!</i>"
        )
        log.info(f"{asset}: Signal {sig} | Score {score}")
        return msg

    except Exception as e:
        log.error(f"generate_signal {asset}: {e}")
        import traceback
        log.error(traceback.format_exc())
        return None


# ─── MAIN ─────────────────────────────────────────────────────
def run():
    log.info("=" * 50)
    log.info("SMC Trading Bot v1.3 starting...")
    log.info(f"Threshold: {SIGNAL_THRESHOLD}% | Interval: {CHECK_INTERVAL}s")
    log.info(f"Assets: {', '.join(ASSETS.keys())}")
    log.info("=" * 50)

    # Validate environment
    if not validate_env():
        log.error("Fix environment variables and restart!")
        return

    # Startup message
    send_telegram(
        "🤖 <b>SMC Trading Bot v1.3 Started!</b>\n\n"
        "Monitoring:\n"
        "• BTC/USDT | ETH/USDT | SOL/USDT\n"
        "• XRP/USDT | BNB/USDT\n\n"
        f"📍 Finland (EET/EEST)\n"
        f"⏱ Every {CHECK_INTERVAL//60} minutes\n"
        f"🎯 Signal threshold: {SIGNAL_THRESHOLD}%\n"
        "✅ v1.3 — All fixes applied!\n\n"
        "<i>Always confirm with Claude Project!</i>"
    )

    sent     = {}   # tracks last signal time per asset
    cycle_no = 0

    while True:
        try:
            cycle_no += 1
            sess = get_session()
            log.info(f"Cycle #{cycle_no} | {sess['time_fi']} | {sess['session']}")

            # Skip off hours
            if sess["priority"] == "AVOID":
                log.info("Off Hours — skipping cycle")
                time.sleep(CHECK_INTERVAL)
                continue

            # Cleanup old sent entries every 10 cycles
            if cycle_no % 10 == 0:
                sent = cleanup_sent(sent)

            # Analyze each asset
            for asset, info in ASSETS.items():
                msg = generate_signal(asset, info)
                if msg:
                    last_sent = sent.get(asset, 0)
                    if time.time() - last_sent > SIGNAL_COOLDOWN:
                        success = send_telegram(msg)
                        if success:
                            sent[asset] = time.time()
                        time.sleep(3)
                    else:
                        remaining = int((SIGNAL_COOLDOWN - (time.time() - last_sent)) / 60)
                        log.info(f"{asset}: Cooldown active — {remaining} mins remaining")

            log.info(f"Cycle #{cycle_no} done. Next in {CHECK_INTERVAL//60} mins.")
            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            log.info("Bot stopped by user.")
            send_telegram("🛑 SMC Bot v1.3 stopped.")
            break
        except Exception as e:
            log.error(f"Main loop error: {e}")
            import traceback
            log.error(traceback.format_exc())
            time.sleep(60)


if __name__ == "__main__":
    run()
