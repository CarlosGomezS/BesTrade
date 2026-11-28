"""Radar Cocodrilo Pro v11.3 — Gestor Patrimonial EUR (Warren Buffett & Charlie Munger).

Novedades v9:
- Owner Earnings reales: descuenta Stock-Based Compensation (SBC) del FCF para evitar trampas en tech.
- Buscador inteligente por nombre en modo 'custom' (ej. 'Alphabet' -> GOOGL, 'Inditex' -> ITX.MC).
- Aviso de presentación de resultados próximos (<= 10 días) para no comprar a ciegas antes de earnings.
- Radar de compras de directivos (Insider Buying últimos 90 días) en candidatas top.
- Termómetro de mercado (VIX, Bono EE. UU. 10A y distancia S&P 500 a máximos) en Telegram y web.
- Diario automático de aciertos (data/history.json) con seguimiento de rentabilidad real de alertas.
- Mantiene Nivel 0 'ENTRADA_YA', sharding paralelo (6 bloques), rotación diaria y Gemini 3 escéptico.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import statistics
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from universe import detect_region, get_universe, parse_custom_tickers
from oracle_v10 import VALID_VERDICTS, classify_quality_valuation_quadrant, compute_oracle_checks, extract_gemini_json, is_oro_puro, run_tribunal_ia, verify_with_sec_edgar

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_LATEST_JSON = DATA_DIR / "latest-scan.json"
OUT_DAILY_JSON = DATA_DIR / "daily-scan.json"
OUT_WEEKLY_JSON = DATA_DIR / "weekly-scan.json"
OUT_CUSTOM_JSON = DATA_DIR / "custom-scan.json"
OUT_HISTORY_JSON = DATA_DIR / "history.json"
OUT_COOLDOWN_JSON = DATA_DIR / "alert-cooldown.json"

UNIVERSE_MODE = os.environ.get("UNIVERSE_MODE", "rotacion_diaria").strip().lower()
CUSTOM_TICKERS = os.environ.get("CUSTOM_TICKERS", "").strip()
TEST_MODE = os.environ.get("TEST_MODE", "false").strip().lower() == "true"
MIN_QUALITY_SCORE = int(os.environ.get("MIN_QUALITY_SCORE", "75"))
MARGIN_OF_SAFETY_PCT = float(os.environ.get("MARGIN_OF_SAFETY_PCT", "25"))
MIN_DRAWDOWN_ALERT = float(os.environ.get("MIN_DRAWDOWN_ALERT", "18"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "3"))
MIN_MARKET_CAP_USD = float(os.environ.get("MIN_MARKET_CAP_USD", "1000000000"))  # >= 1B USD
ROTATION_BATCH_SIZE = int(os.environ.get("ROTATION_BATCH_SIZE", "220"))

SHARD_INDEX = int(os.environ.get("SHARD_INDEX", "0"))
SHARD_TOTAL = int(os.environ.get("SHARD_TOTAL", "1"))
MERGE_DIR = os.environ.get("MERGE_DIR", "").strip()
ALLOW_PARTIAL_MERGE = os.environ.get("ALLOW_PARTIAL_MERGE", "false").strip().lower() == "true"
MAX_DISCARDED_IN_WEB_JSON = int(os.environ.get("MAX_DISCARDED_IN_WEB_JSON", "250"))
URGENT_ONLY = os.environ.get("URGENT_ONLY", "false").strip().lower() == "true"
URGENT_DOUBLE_PING = os.environ.get("URGENT_DOUBLE_PING", "true").strip().lower() == "true"
ALERT_COOLDOWN_DAYS = float(os.environ.get("ALERT_COOLDOWN_DAYS", "7"))
ALERT_RETRIGGER_DROP_PCT = float(os.environ.get("ALERT_RETRIGGER_DROP_PCT", "5.0"))
SCANNER_VERSION = "v11.3-oraculo"
_RAW_BUDGET_ENV = (os.environ.get("DEFAULT_TRADE_BUDGET_EUR") or os.environ.get("CASH_POT_EUR") or "").strip()
try:
    DEFAULT_TRADE_BUDGET_EUR: Optional[float] = float(_RAW_BUDGET_ENV) if (_RAW_BUDGET_ENV and float(_RAW_BUDGET_ENV) > 0) else None
except Exception:
    DEFAULT_TRADE_BUDGET_EUR = None
DEFAULT_STOP_LOSS_MODE = os.environ.get("DEFAULT_STOP_LOSS_MODE", "AUTO").strip().upper()

TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
FIREBASE_SERVICE_ACCOUNT = os.environ.get("FIREBASE_SERVICE_ACCOUNT", "").strip()
FIREBASE_DEVICE_TOKEN = os.environ.get("FIREBASE_DEVICE_TOKEN", "").strip()
FIREBASE_TOPIC = os.environ.get("FIREBASE_TOPIC", "entrar_ya").strip()
FIREBASE_SEND_BOTH = os.environ.get("FIREBASE_SEND_BOTH", "false").strip().lower() == "true"
APP_BASE_URL = os.environ.get("APP_BASE_URL", "https://carlosgomezs.github.io/BesTrade").strip().rstrip("/")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODELS = ("gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite")
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"

FALLBACK_FX_EUR: Dict[str, float] = {
    "EUR": 1.0, "USD": 0.94, "GBP": 1.19, "CHF": 1.07, "CAD": 0.67, "JPY": 0.0061,
    "SEK": 0.087, "NOK": 0.085, "DKK": 0.134, "HKD": 0.121, "AUD": 0.61,
}

SUBUNIT_CURRENCIES: Dict[str, Tuple[str, float]] = {
    "GBp": ("GBP", 100.0),
    "GBX": ("GBP", 100.0),
    "ZAc": ("ZAR", 100.0),
    "ILA": ("ILS", 100.0),
}

_print_lock = threading.Lock()
_fx_lock = threading.Lock()
_fx_cache: Dict[Tuple[str, str], Optional[float]] = {}
_fx_hist_cache: Dict[Tuple[str, str], Optional[pd.Series]] = {}


def safe_float(val: Any) -> Optional[float]:
    try:
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return None
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return None


def normalize_currency(raw_currency: Optional[str]) -> Tuple[str, float]:
    raw = (raw_currency or "USD").strip()
    if raw in SUBUNIT_CURRENCIES:
        return SUBUNIT_CURRENCIES[raw]
    return raw.upper(), 1.0


def get_fx_rate(src_cur: str, dst_cur: str) -> Optional[float]:
    src = (src_cur or "USD").upper()
    dst = (dst_cur or "USD").upper()
    if src == dst:
        return 1.0
    key = (src, dst)
    with _fx_lock:
        if key in _fx_cache:
            return _fx_cache[key]

    rate: Optional[float] = None
    for pair, invert in ((f"{src}{dst}=X", False), (f"{dst}{src}=X", True)):
        try:
            hist = yf.Ticker(pair).history(period="5d")
            if hist is not None and not hist.empty and "Close" in hist:
                val = safe_float(hist["Close"].dropna().iloc[-1])
                if val and val > 0:
                    rate = (1.0 / val) if invert else val
                    break
        except Exception:
            continue

    if rate is None and src != "USD" and dst != "USD":
        r1 = get_fx_rate(src, "USD")
        r2 = get_fx_rate("USD", dst)
        if r1 and r2:
            rate = r1 * r2

    with _fx_lock:
        _fx_cache[key] = rate
        if rate and rate > 0:
            _fx_cache[(dst, src)] = 1.0 / rate
    return rate


def resolve_custom_inputs(raw_str: str) -> List[str]:
    """Permite escribir tickers (GOOGL, ITX.MC) o nombres de empresa ('Alphabet, Inditex')."""
    if not raw_str or not raw_str.strip():
        return []
    parts = [p.strip() for p in re.split(r"[,;\n]+", raw_str) if p.strip()]
    resolved: List[str] = []

    for item in parts:
        upper_item = item.upper()
        if re.match(r"^[A-Z0-9\-\^]{1,5}$", upper_item) or re.match(r"^[A-Z0-9\-]{1,7}\.[A-Z]{1,3}$", upper_item):
            if len(upper_item) <= 5 or "." in upper_item:
                resolved.append(upper_item)
                continue

        q = urllib.parse.quote(item)
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={q}&quotesCount=5&newsCount=0"
        found_sym: Optional[str] = None
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            for quote in data.get("quotes") or []:
                sym = (quote.get("symbol") or "").strip().upper()
                qtype = (quote.get("quoteType") or "").upper()
                if sym and qtype in ("EQUITY", "ADR", ""):
                    found_sym = sym
                    break
        except Exception:
            pass

        if found_sym:
            print(f"🔎 Resuelto '{item}' -> {found_sym}")
            resolved.append(found_sym)
        else:
            resolved.extend(parse_custom_tickers(item))

    seen = set()
    out: List[str] = []
    for sym in resolved:
        if sym and sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


def get_market_context() -> Dict[str, Any]:
    """Termómetro de mercado: VIX (miedo), Bono EE. UU. 10A y distancia del S&P 500 a máximos."""
    ctx: Dict[str, Any] = {
        "vix": None,
        "us10y_pct": None,
        "sp500_drawdown_pct": None,
        "regime": "NORMAL",
        "summary": "Clima de mercado normal",
    }
    try:
        vix_h = yf.Ticker("^VIX").history(period="5d")
        if vix_h is not None and not vix_h.empty:
            v = safe_float(vix_h["Close"].dropna().iloc[-1])
            if v is not None:
                ctx["vix"] = round(v, 1)
    except Exception:
        pass

    try:
        tnx_h = yf.Ticker("^TNX").history(period="5d")
        if tnx_h is not None and not tnx_h.empty:
            y = safe_float(tnx_h["Close"].dropna().iloc[-1])
            if y is not None:
                ctx["us10y_pct"] = round(y / 10.0 if y > 20 else y, 2)
    except Exception:
        pass

    try:
        sp_h = yf.Ticker("^GSPC").history(period="1y")
        if sp_h is not None and not sp_h.empty:
            closes = sp_h["Close"].dropna()
            last_c = safe_float(closes.iloc[-1])
            max_c = safe_float(closes.max())
            if last_c and max_c and max_c > 0:
                ctx["sp500_drawdown_pct"] = round(((last_c / max_c) - 1.0) * 100.0, 1)
    except Exception:
        pass

    vix = ctx["vix"]
    dd = ctx["sp500_drawdown_pct"]
    if (vix is not None and vix >= 28) or (dd is not None and dd <= -15):
        ctx["regime"] = "PANICO"
        ctx["summary"] = "Miedo alto en mercado: entorno ideal para cazar gangas"
    elif (vix is not None and vix >= 21) or (dd is not None and dd <= -8):
        ctx["regime"] = "CAUTELA"
        ctx["summary"] = "Volatilidad elevada: empiezan a aparecer descuentos"
    elif (vix is not None and vix <= 14) and (dd is None or dd >= -3):
        ctx["regime"] = "EUFORIA"
        ctx["summary"] = "Mercado caro y confiado: máxima exigencia con el margen"
    else:
        ctx["regime"] = "NORMAL"
        ctx["summary"] = "Mercado tranquilo: ser selectivo y paciente"

    return ctx


def get_row_series(df: Optional[pd.DataFrame], candidates: List[str]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    idx_map = {str(k).strip().lower(): k for k in df.index}
    for name in candidates:
        key = idx_map.get(name.lower())
        if key is not None:
            s = df.loc[key]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[0]
            s = s.dropna()
            if not s.empty:
                try:
                    dt_idx = pd.to_datetime(s.index, errors="coerce")
                    if dt_idx.notna().all():
                        s = s.iloc[dt_idx.argsort()[::-1]]
                except Exception:
                    pass
                return s
    return None


def classify_sector_profile(sector: str, industry: str) -> str:
    s = (sector or "").lower()
    ind = (industry or "").lower()
    if "financial" in s or "bank" in ind or "insurance" in ind or "real estate" in s or "reit" in ind:
        return "financial"
    if any(w in ind for w in ("discount store", "grocery", "supermarket", "retail", "warehouse", "wholesale", "distribution")):
        return "retail"
    if "technology" in s or "communication" in s or "software" in ind:
        return "tech"
    return "general"


def extract_next_earnings(info: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    """Extrae fecha próxima de resultados y días restantes sin peticiones extra."""
    now = datetime.now(timezone.utc)
    for key in ("earningsTimestampStart", "earningsTimestamp"):
        ts = safe_float(info.get(key))
        if ts and ts > 1e8:
            try:
                dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
                days = (dt.date() - now.date()).days
                if -1 <= days <= 120:
                    return dt.strftime("%Y-%m-%d"), max(0, days)
            except Exception:
                continue
    return None, None


def compute_roic_and_statements(t: yf.Ticker, info: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "roic_pct": None,
        "fcf_years_pos": None,
        "fcf_years_total": None,
        "fcf_latest": None,
        "fcf_latest_date": None,
        "fcf_normalized": None,
        "fcf_spike_capped": False,
        "sbc_latest": None,
        "sbc_pct_of_fcf": None,
        "owner_earnings_adjusted": False,
        "fcf_cagr_pct": None,
        "eps_cagr_pct": None,
        "share_change_pct": None,
        "net_debt_ebitda": None,
        "ebitda_negative": False,
        "net_cash": None,
    }

    try:
        fin = t.financials
        bs = t.balance_sheet
        cf = t.cashflow
    except Exception:
        fin = bs = cf = None

    # 1) Free Cash Flow y Stock-Based Compensation (Owner Earnings reales)
    fcf_series = get_row_series(cf, ["Free Cash Flow"])
    if fcf_series is None:
        ocf = get_row_series(cf, ["Operating Cash Flow", "Total Cash From Operating Activities"])
        capex = get_row_series(cf, ["Capital Expenditure", "Capital Expenditures"])
        if ocf is not None and capex is not None:
            common = ocf.index.intersection(capex.index)
            if len(common) > 0:
                fcf_series = ocf.loc[common] + capex.loc[common].apply(lambda x: -abs(x))
                try:
                    dt_idx = pd.to_datetime(fcf_series.index, errors="coerce")
                    if dt_idx.notna().all():
                        fcf_series = fcf_series.iloc[dt_idx.argsort()[::-1]]
                except Exception:
                    pass

    sbc_series = get_row_series(cf, ["Stock Based Compensation", "Share Based Compensation"])
    sbc_latest = None
    if sbc_series is not None and len(sbc_series) >= 1:
        val_sbc = safe_float(sbc_series.values[0])
        if val_sbc is not None and val_sbc > 0:
            sbc_latest = val_sbc
            out["sbc_latest"] = sbc_latest

    if fcf_series is not None and len(fcf_series) >= 1:
        raw_vals: List[float] = []
        adj_vals: List[float] = []
        raw_dates: List[Optional[str]] = []
        for idx_col, v in fcf_series.items():
            fv = safe_float(v)
            if fv is None:
                continue
            raw_vals.append(fv)
            try:
                dt_c = pd.to_datetime(idx_col, errors="coerce")
                raw_dates.append(dt_c.strftime("%Y-%m-%d") if pd.notna(dt_c) else None)
            except Exception:
                raw_dates.append(None)
            sbc_col = 0.0
            if sbc_series is not None and idx_col in sbc_series.index:
                sv = safe_float(sbc_series.get(idx_col))
                if sv is not None and sv > 0:
                    sbc_col = sv
            adj_vals.append(fv - sbc_col)

        if raw_vals:
            out["fcf_years_total"] = len(raw_vals)
            out["fcf_years_pos"] = sum(1 for v in raw_vals if v > 0)
            out["fcf_latest_date"] = raw_dates[0] if raw_dates else None

            use_vals = raw_vals
            if sbc_latest and raw_vals[0] > 0:
                sbc_ratio = (sbc_latest / raw_vals[0]) * 100.0
                out["sbc_pct_of_fcf"] = round(sbc_ratio, 1)
                if sbc_ratio >= 5.0:
                    use_vals = adj_vals
                    out["owner_earnings_adjusted"] = True

            out["fcf_latest"] = use_vals[0]
            ni_check = get_row_series(fin, ["Net Income", "Net Income Common Stockholders"])
            ni_latest = safe_float(ni_check.values[0]) if (ni_check is not None and len(ni_check) >= 1) else None
            sustained_growth = (
                len(use_vals) >= 3
                and use_vals[0] > 0
                and use_vals[1] > 0
                and use_vals[2] > 0
                and use_vals[0] >= use_vals[1] * 0.95
                and use_vals[1] > use_vals[2] * 1.05
            )
            backed_by_ni = ni_latest is not None and ni_latest > 0 and use_vals[0] <= ni_latest * 1.35

            pos_vals = [v for v in use_vals[:4] if v > 0]
            if sustained_growth and backed_by_ni:
                # Crecimiento real y sostenido respaldado por beneficio neto: no recortar el FCF
                out["fcf_normalized"] = use_vals[0]
            elif len(pos_vals) >= 2 and use_vals[0] > 0:
                med_fcf = statistics.median(pos_vals)
                cap_fcf = med_fcf * 1.25
                if use_vals[0] > cap_fcf:
                    out["fcf_normalized"] = cap_fcf
                    out["fcf_spike_capped"] = True
                else:
                    out["fcf_normalized"] = use_vals[0]
            else:
                out["fcf_normalized"] = use_vals[0]

            if len(raw_vals) >= 3 and raw_vals[0] > 0 and raw_vals[-1] > 0:
                out["fcf_cagr_pct"] = ((raw_vals[0] / raw_vals[-1]) ** (1.0 / (len(raw_vals) - 1)) - 1.0) * 100.0

    if out["fcf_latest"] is None:
        fallback_fcf = safe_float(info.get("freeCashflow"))
        out["fcf_latest"] = fallback_fcf
        out["fcf_normalized"] = fallback_fcf

    # 2) Crecimiento de Beneficio Neto
    ni_series = get_row_series(fin, ["Net Income", "Net Income Common Stockholders"])
    if ni_series is not None and len(ni_series) >= 3:
        vals = [safe_float(v) for v in ni_series.values]
        vals = [v for v in vals if v is not None]
        if len(vals) >= 3 and vals[0] > 0 and vals[-1] > 0:
            out["eps_cagr_pct"] = ((vals[0] / vals[-1]) ** (1.0 / (len(vals) - 1)) - 1.0) * 100.0

    # 3) Recompras vs Dilución
    sh_series = get_row_series(fin, ["Diluted Average Shares", "Basic Average Shares"])
    if sh_series is not None and len(sh_series) >= 2:
        vals = [safe_float(v) for v in sh_series.values if safe_float(v) and safe_float(v) > 0]
        if len(vals) >= 2:
            out["share_change_pct"] = ((vals[0] / vals[-1]) - 1.0) * 100.0

    # 4) ROIC real = NOPAT / Capital Invertido (mediana)
    ebit_s = get_row_series(fin, ["EBIT", "Operating Income"])
    tax_s = get_row_series(fin, ["Tax Provision"])
    pretax_s = get_row_series(fin, ["Pretax Income"])
    debt_s = get_row_series(bs, ["Total Debt"])
    eq_s = get_row_series(bs, ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"])
    cash_s = get_row_series(bs, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"])

    roic_list: List[float] = []
    if ebit_s is not None and eq_s is not None:
        common_cols = ebit_s.index.intersection(eq_s.index)
        for col in common_cols[:4]:
            ebit = safe_float(ebit_s.get(col))
            eq = safe_float(eq_s.get(col))
            debt = safe_float(debt_s.get(col)) if (debt_s is not None and col in debt_s.index) else 0.0
            cash = safe_float(cash_s.get(col)) if (cash_s is not None and col in cash_s.index) else 0.0
            if ebit is None or eq is None:
                continue
            tax_rate = 0.21
            if tax_s is not None and pretax_s is not None and col in tax_s.index and col in pretax_s.index:
                tp, pt = safe_float(tax_s.get(col)), safe_float(pretax_s.get(col))
                if tp is not None and pt is not None and pt > 0:
                    tax_rate = min(max(tp / pt, 0.05), 0.35)
            nopat = ebit * (1.0 - tax_rate)
            invested_capital = (debt or 0.0) + eq - (cash or 0.0)
            if invested_capital > 0:
                roic_list.append((nopat / invested_capital) * 100.0)

    if roic_list:
        out["roic_pct"] = statistics.median(roic_list)
    else:
        roe = safe_float(info.get("returnOnEquity"))
        if roe is not None:
            out["roic_pct"] = roe * 100.0

    # 5) Caja Neta y Deuda Neta / EBITDA
    total_debt = safe_float(info.get("totalDebt"))
    total_cash = safe_float(info.get("totalCash"))
    ebitda = safe_float(info.get("ebitda"))

    if total_cash is not None and total_debt is not None:
        out["net_cash"] = total_cash - total_debt
        net_debt = total_debt - total_cash
        if ebitda is not None:
            if ebitda <= 0:
                out["ebitda_negative"] = True
            else:
                out["net_debt_ebitda"] = net_debt / ebitda

    return out


def compute_conservative_dcf(
    fcf_total: Optional[float],
    shares: Optional[float],
    net_cash: Optional[float],
    hist_growth_pct: Optional[float],
    analyst_growth_pct: Optional[float],
    discount_pct: float = 10.0,
    margin_safety_pct: float = 25.0,
) -> Optional[Dict[str, Any]]:
    if fcf_total is None or fcf_total <= 0 or shares is None or shares <= 0:
        return None

    fcf_per_share = fcf_total / shares
    net_cash_ps = (net_cash / shares) if net_cash is not None else 0.0

    growth_inputs = [g for g in (hist_growth_pct, analyst_growth_pct) if g is not None]
    if growth_inputs:
        raw_g = min(growth_inputs) if len(growth_inputs) > 1 else growth_inputs[0]
        g1_base = min(max(raw_g, 2.0), 13.0)
    else:
        g1_base = 5.0

    def single_dcf(g1_pct: float, r_pct: float) -> Optional[float]:
        r = r_pct / 100.0
        g1 = g1_pct / 100.0
        g2 = min(max(g1 * 0.6, 0.0), 0.07)
        g_term = min(max(g1 * 0.3, 0.0), 0.025)
        if r <= g_term + 0.005:
            return None
        pv = 0.0
        cf = fcf_per_share
        for y in range(1, 11):
            cf *= (1.0 + (g1 if y <= 5 else g2))
            pv += cf / ((1.0 + r) ** y)
        tv = (cf * (1.0 + g_term)) / (r - g_term)
        pv_tv = tv / ((1.0 + r) ** 10)
        return max(0.0, pv + pv_tv + net_cash_ps)

    intrinsic_base = single_dcf(g1_base, discount_pct)
    intrinsic_bear = single_dcf(max(0.0, g1_base * 0.65 - 1.5), discount_pct + 1.5)
    intrinsic_bull = single_dcf(min(18.0, g1_base * 1.15 + 1.5), max(7.5, discount_pct - 0.5))
    if intrinsic_base is None or intrinsic_base <= 0:
        return None

    target_base = intrinsic_base * (1.0 - margin_safety_pct / 100.0)
    return {
        "fcf_per_share": round(fcf_per_share, 2),
        "net_cash_per_share": round(net_cash_ps, 2),
        "assumed_growth_pct": round(g1_base, 1),
        "intrinsic_bear": round(intrinsic_bear, 2) if intrinsic_bear is not None else None,
        "intrinsic_base": round(intrinsic_base, 2),
        "intrinsic_bull": round(intrinsic_bull, 2) if intrinsic_bull is not None else None,
        "target_price": round(target_base, 2),
        "margin_safety_pct": margin_safety_pct,
    }


def score_company(data: Dict[str, Any]) -> Dict[str, Any]:
    score = 0
    known = 0
    hard_fails: List[str] = []
    strengths: List[str] = []
    breakdown: Dict[str, Optional[int]] = {
        "roic": None,
        "margin": None,
        "fcf": None,
        "debt": None,
        "growth": None,
        "shareholder": None,
    }
    profile = data["sector_profile"]

    def add(key: str, pts: int, max_pts: int) -> None:
        nonlocal score, known
        score += pts
        known += max_pts
        breakdown[key] = round((pts / max_pts) * 100)

    roic = data.get("roic_pct")
    if roic is not None:
        if profile == "financial":
            pts = 25 if roic >= 15 else 20 if roic >= 12 else 12 if roic >= 9 else 0
            add("roic", pts, 25)
            if roic < 8:
                hard_fails.append(f"Rentabilidad capital baja ({roic:.1f}%)")
            elif roic >= 14:
                strengths.append(f"Alta rentabilidad sobre capital ({roic:.1f}%)")
        else:
            pts = 25 if roic >= 18 else 20 if roic >= 14 else 14 if roic >= 10 else 6 if roic >= 6 else 0
            add("roic", pts, 25)
            if roic < 6:
                hard_fails.append(f"ROIC demasiado bajo ({roic:.1f}%)")
            elif roic >= 15:
                strengths.append(f"ROIC excelente ({roic:.1f}%)")

    opm = data.get("operating_margin_pct")
    if opm is not None:
        if profile == "retail":
            pts = 15 if opm >= 6 else 13 if opm >= 3.2 else 8 if opm >= 2 else 0
            add("margin", pts, 15)
            if opm < 1.5:
                hard_fails.append(f"Margen operativo muy débil ({opm:.1f}%)")
            elif opm >= 3.2:
                strengths.append(f"Margen sólido para retail de alto volumen ({opm:.1f}%)")
        elif profile == "tech":
            pts = 15 if opm >= 25 else 11 if opm >= 15 else 6 if opm >= 8 else 0
            add("margin", pts, 15)
            if opm < 5:
                hard_fails.append(f"Margen operativo débil en tecnología ({opm:.1f}%)")
            elif opm >= 25:
                strengths.append(f"Márgenes operativos muy altos ({opm:.1f}%)")
        else:
            pts = 15 if opm >= 20 else 11 if opm >= 12 else 6 if opm >= 7 else 0
            add("margin", pts, 15)
            if opm < 4 and profile != "financial":
                hard_fails.append(f"Margen operativo bajo ({opm:.1f}%)")
            elif opm >= 20:
                strengths.append(f"Margen operativo alto ({opm:.1f}%)")

    if profile != "financial":
        f_pos, f_tot = data.get("fcf_years_pos"), data.get("fcf_years_total")
        if f_pos is not None and f_tot and f_tot >= 2:
            ratio = f_pos / f_tot
            pts = 20 if ratio >= 0.99 else 14 if ratio >= 0.75 else 6 if ratio >= 0.5 else 0
            add("fcf", pts, 20)
            if ratio < 0.5:
                hard_fails.append(f"Flujo de caja irregular ({f_pos}/{f_tot} años positivos)")
            elif ratio >= 0.99:
                strengths.append(f"Genera caja real todos los años ({f_pos}/{f_tot})")
        elif data.get("fcf_latest") is not None:
            if data["fcf_latest"] > 0:
                add("fcf", 12, 20)
            else:
                add("fcf", 0, 20)
                hard_fails.append("Flujo de caja libre negativo")

    if data.get("ebitda_negative") and profile != "financial":
        add("debt", 0, 20)
        hard_fails.append("EBITDA negativo / pérdidas operativas")
    elif profile != "financial" and data.get("net_debt_ebitda") is not None:
        nde = data["net_debt_ebitda"]
        pts = 20 if nde <= 0.5 else 16 if nde <= 1.5 else 10 if nde <= 2.5 else 4 if nde <= 3.5 else 0
        add("debt", pts, 20)
        if nde > 3.5:
            hard_fails.append(f"Deuda elevada ({nde:.1f}x EBITDA)")
        elif nde <= 0:
            strengths.append("Más caja que deuda (Caja Neta)")
        elif nde <= 1.5:
            strengths.append(f"Deuda muy controlada ({nde:.1f}x EBITDA)")

    growth_ref = data.get("eps_cagr_pct") if data.get("eps_cagr_pct") is not None else data.get("fcf_cagr_pct")
    if growth_ref is None:
        growth_ref = data.get("revenue_growth_pct")
    if growth_ref is not None:
        pts = 10 if growth_ref >= 10 else 8 if growth_ref >= 5 else 5 if growth_ref >= 1 else 0
        add("growth", pts, 10)
        if growth_ref < -5:
            hard_fails.append(f"Contracción fuerte del negocio ({growth_ref:.1f}% anual)")
        elif growth_ref >= 10:
            strengths.append(f"Crecimiento compuesto sólido ({growth_ref:.1f}% anual)")

    sh_chg = data.get("share_change_pct")
    div_y = data.get("dividend_yield_pct") or 0.0
    if sh_chg is not None:
        if sh_chg <= -2.0:
            add("shareholder", 10, 10)
            strengths.append(f"Recompra acciones netas ({sh_chg:.1f}%)")
        elif sh_chg <= 1.0:
            add("shareholder", 8 if div_y >= 1.5 else 7, 10)
        elif sh_chg <= 4.0:
            add("shareholder", 4, 10)
        else:
            add("shareholder", 0, 10)
            hard_fails.append(f"Dilución alta (+{sh_chg:.1f}% acciones)")

    min_known = 50 if profile == "financial" else 70
    normalized = round((score / known) * 100) if known > 0 else 0
    passes = known >= min_known and len(hard_fails) == 0 and normalized >= MIN_QUALITY_SCORE
    return {
        "quality_score": normalized,
        "data_coverage_pct": known,
        "score_breakdown": breakdown,
        "hard_fails": hard_fails,
        "strengths": strengths[:4],
        "passes_quality": passes,
    }


def _warmup_yf_session() -> None:
    """Inicializa cookie/crumb de yfinance en un solo hilo antes de lanzar el pool paralelo."""
    for sym in ("AAPL", "MSFT"):
        try:
            _ = yf.Ticker(sym).history(period="5d")
            time.sleep(0.5)
            return
        except Exception:
            time.sleep(1.5)


def analyze_ticker_with_retry(ticker: str, retries: int = 3) -> Optional[Dict[str, Any]]:
    for attempt in range(retries + 1):
        try:
            res = analyze_ticker(ticker)
            if res is not None:
                return res
            return None
        except Exception as exc:
            msg = str(exc).lower()
            is_auth = any(k in msg for k in ("401", "crumb", "unauthorized"))
            transient = is_auth or any(
                k in msg
                for k in ("429", "too many", "timeout", "timed out", "rate", "502", "503", "504", "connection", "json")
            )
            if attempt < retries and transient:
                base_wait = 4.0 if is_auth else 2.5
                time.sleep(base_wait * (attempt + 1) + random.uniform(0.5, 1.8))
                continue
            return None
    return None


def analyze_ticker(ticker: str) -> Optional[Dict[str, Any]]:
    t = yf.Ticker(ticker)
    info = t.info or {}

    currency, price_div = normalize_currency(info.get("currency"))
    fin_currency, _ = normalize_currency(info.get("financialCurrency") or currency)

    raw_price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"))
    if raw_price is None or raw_price <= 0:
        return None
    price = raw_price / price_div

    mkt_cap = safe_float(info.get("marketCap"))
    if mkt_cap is not None:
        fx_to_usd = get_fx_rate(currency, "USD")
        mkt_cap_usd = (mkt_cap * fx_to_usd) if fx_to_usd else mkt_cap
        if UNIVERSE_MODE != "custom" and mkt_cap_usd < MIN_MARKET_CAP_USD:
            return None
    else:
        mkt_cap_usd = None

    name = info.get("shortName") or info.get("longName") or ticker
    sector = info.get("sector") or "Otros"
    industry = info.get("industry") or ""
    profile = classify_sector_profile(sector, industry)
    region = detect_region(ticker)

    raw_high52 = safe_float(info.get("fiftyTwoWeekHigh"))
    raw_low52 = safe_float(info.get("fiftyTwoWeekLow"))
    high52 = (raw_high52 / price_div) if raw_high52 else None
    low52 = (raw_low52 / price_div) if raw_low52 else None
    drawdown_52w = ((price / high52) - 1.0) * 100.0 if (high52 and high52 > 0) else None

    shares = (mkt_cap / price) if (mkt_cap and price > 0) else safe_float(info.get("sharesOutstanding"))
    if shares is None or shares <= 0:
        shares = safe_float(info.get("sharesOutstanding"))

    opm = safe_float(info.get("operatingMargins"))
    rev_g = safe_float(info.get("revenueGrowth"))
    earn_g = safe_float(info.get("earningsGrowth"))

    div_y = safe_float(info.get("trailingAnnualDividendYield"))
    if div_y is not None:
        div_y = div_y * 100.0
    else:
        raw_dy = safe_float(info.get("dividendYield"))
        if raw_dy is not None:
            div_y = raw_dy * 100.0 if raw_dy < 0.20 else raw_dy

    earnings_date, earnings_days = extract_next_earnings(info)

    stm = compute_roic_and_statements(t, info)
    data_warnings: List[str] = []

    fx_fin_to_trade = get_fx_rate(fin_currency, currency)
    fx_ok = fx_fin_to_trade is not None and fx_fin_to_trade > 0

    fcf_latest_conv = (stm["fcf_latest"] * fx_fin_to_trade) if (fx_ok and stm["fcf_latest"] is not None) else None
    fcf_norm_conv = (stm["fcf_normalized"] * fx_fin_to_trade) if (fx_ok and stm["fcf_normalized"] is not None) else None
    net_cash_conv = (stm["net_cash"] * fx_fin_to_trade) if (fx_ok and stm["net_cash"] is not None) else None

    if not fx_ok and fin_currency != currency:
        data_warnings.append(f"Sin tipo de cambio {fin_currency}->{currency}: DCF y P/FCF omitidos")

    if stm["fcf_spike_capped"]:
        data_warnings.append("FCF del último año anormalmente alto; DCF normalizado con mediana histórica")

    if stm["sbc_pct_of_fcf"] is not None and stm["sbc_pct_of_fcf"] >= 20.0:
        data_warnings.append(f"Pago en acciones a empleados alto ({stm['sbc_pct_of_fcf']}% del FCF): DCF descontado a caja real")

    if stm["roic_pct"] is not None and stm["roic_pct"] > 100.0:
        data_warnings.append("ROIC >100%: posible patrimonio neto muy bajo por recompras o efecto contable")

    p_fcf = (mkt_cap / fcf_latest_conv) if (mkt_cap and fcf_latest_conv and fcf_latest_conv > 0) else None

    hist_g = stm["eps_cagr_pct"] if stm["eps_cagr_pct"] is not None else stm["fcf_cagr_pct"]
    analyst_g = (earn_g * 100.0) if earn_g is not None else ((rev_g * 100.0) if rev_g is not None else None)

    if profile == "financial":
        dcf = None
        data_warnings.append("Sector financiero/REIT: DCF estándar por FCF desactivado (valorar por P/B, ROE o AFFO)")
    else:
        dcf = compute_conservative_dcf(
            fcf_total=fcf_norm_conv,
            shares=shares,
            net_cash=net_cash_conv,
            hist_growth_pct=hist_g,
            analyst_growth_pct=analyst_g,
            discount_pct=10.0,
            margin_safety_pct=MARGIN_OF_SAFETY_PCT,
        )

    record: Dict[str, Any] = {
        "ticker": ticker,
        "name": name,
        "region": region,
        "currency": currency,
        "financial_currency": fin_currency,
        "fx_rate_applied": round(fx_fin_to_trade, 4) if fx_ok else None,
        "sector": sector,
        "industry": industry,
        "sector_profile": profile,
        "market_cap_b": round((mkt_cap_usd or mkt_cap) / 1e9, 1) if (mkt_cap_usd or mkt_cap) else None,
        "price": round(price, 2),
        "high_52w": round(high52, 2) if high52 else None,
        "low_52w": round(low52, 2) if low52 else None,
        "drawdown_52w_pct": round(drawdown_52w, 1) if drawdown_52w is not None else None,
        "trailing_pe": round(safe_float(info.get("trailingPE")), 1) if safe_float(info.get("trailingPE")) else None,
        "forward_pe": round(safe_float(info.get("forwardPE")), 1) if safe_float(info.get("forwardPE")) else None,
        "p_fcf": round(p_fcf, 1) if p_fcf is not None else None,
        "roic_pct": round(stm["roic_pct"], 1) if stm["roic_pct"] is not None else None,
        "operating_margin_pct": round(opm * 100.0, 1) if opm is not None else None,
        "revenue_growth_pct": round(rev_g * 100.0, 1) if rev_g is not None else None,
        "eps_cagr_pct": round(stm["eps_cagr_pct"], 1) if stm["eps_cagr_pct"] is not None else None,
        "fcf_cagr_pct": round(stm["fcf_cagr_pct"], 1) if stm["fcf_cagr_pct"] is not None else None,
        "fcf_years_pos": stm["fcf_years_pos"],
        "fcf_years_total": stm["fcf_years_total"],
        "fcf_latest": fcf_latest_conv,
        "fcf_latest_date": stm["fcf_latest_date"],
        "fcf_latest_trade": round(fcf_latest_conv, 2) if fcf_latest_conv is not None else None,
        "sbc_pct_of_fcf": stm["sbc_pct_of_fcf"],
        "owner_earnings_adjusted": stm["owner_earnings_adjusted"],
        "net_debt_ebitda": round(stm["net_debt_ebitda"], 2) if stm["net_debt_ebitda"] is not None else None,
        "ebitda_negative": stm["ebitda_negative"],
        "share_change_pct": round(stm["share_change_pct"], 1) if stm["share_change_pct"] is not None else None,
        "dividend_yield_pct": round(div_y, 2) if div_y is not None else None,
        "earnings_date": earnings_date,
        "earnings_days": earnings_days,
        "insider_buys_90d": None,
        "insider_status": "NOT_CHECKED",
        "dcf": dcf,
        "data_warnings": data_warnings,
        "recent_news": [],
        "news_status": "NOT_CHECKED",
    }

    q = score_company(record)
    record.update(q)

    discount_pct = None
    upside_base = None
    in_dcf_zone = False
    below_bear = False
    if dcf and dcf["intrinsic_base"] > 0:
        discount_pct = round((1.0 - (price / dcf["intrinsic_base"])) * 100.0, 1)
        upside_base = round(((dcf["intrinsic_base"] / price) - 1.0) * 100.0, 1)
        in_dcf_zone = price <= dcf["target_price"]
        below_bear = bool(dcf["intrinsic_bear"] and price <= dcf["intrinsic_bear"])
        if discount_pct > 75.0 or upside_base > 250.0:
            record["data_warnings"].append("Descuento DCF extremo (>75%): verificar FCF y cuentas oficiales antes de actuar")

    deep_panic_cheap = (
        drawdown_52w is not None
        and drawdown_52w <= -MIN_DRAWDOWN_ALERT
        and ((p_fcf is not None and p_fcf <= 20) or (record["forward_pe"] is not None and record["forward_pe"] <= 18))
        and (dcf is None or price <= dcf["intrinsic_base"] * 1.05)
    )

    if record["passes_quality"] and (in_dcf_zone or deep_panic_cheap):
        status = "VIGILAR" if profile == "financial" else "OPORTUNIDAD"
    elif record["passes_quality"] and (
        (drawdown_52w is not None and drawdown_52w <= -14.0)
        or (dcf is not None and price <= dcf["intrinsic_base"] * 1.05)
    ):
        status = "VIGILAR"
    elif record["passes_quality"]:
        status = "CALIDAD_CARA"
    else:
        status = "DESCARTADA"

    record.pop("fcf_latest", None)
    record["discount_vs_base_pct"] = discount_pct
    record["upside_base_pct"] = upside_base
    record["in_dcf_zone"] = in_dcf_zone
    record["below_bear"] = below_bear
    record["status"] = status
    record["ai_check"] = None
    record["ai_verdict"] = None
    record["trap_risk_score"] = None
    record["oro_puro"] = False
    record.update(compute_oracle_checks(t, info, record))
    record["quality_valuation"] = classify_quality_valuation_quadrant(record)
    return record
def check_insider_buying(ticker: str) -> Tuple[Optional[int], str]:
    """Cuenta compras reales de directivos en 90d distinguiendo 0 real vs ERROR de Yahoo."""
    try:
        df = yf.Ticker(ticker).insider_transactions
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return 0, "OK_EMPTY"
        now_naive = datetime.now(timezone.utc).replace(tzinfo=None)
        buys = 0
        cols_lower = {str(c).strip().lower(): c for c in df.columns}
        txt_col = cols_lower.get("text") or cols_lower.get("transaction")
        date_col = cols_lower.get("start date") or cols_lower.get("date")
        if not txt_col:
            return None, "UNAVAILABLE"
        for _, row in df.head(25).iterrows():
            txt = str(row.get(txt_col) or "").lower()
            if "purchase" not in txt and "buy" not in txt:
                continue
            if "option" in txt or "award" in txt or "grant" in txt:
                continue
            if date_col and row.get(date_col) is not None:
                try:
                    dt = pd.to_datetime(row.get(date_col))
                    if pd.notna(dt):
                        dt_naive = dt.tz_localize(None) if getattr(dt, "tzinfo", None) else dt
                        if (now_naive - dt_naive.to_pydatetime()).days > 90:
                            continue
                except Exception:
                    pass
            buys += 1
        return buys, "OK"
    except Exception:
        return None, "ERROR"


def extract_recent_news(ticker: str, limit: int = 5) -> Tuple[List[str], str]:
    """Extrae titulares recientes distinguiendo sin noticias vs ERROR de Yahoo."""
    headlines: List[str] = []
    try:
        raw_news = yf.Ticker(ticker).news
        if raw_news is None:
            return [], "OK_EMPTY"
    except Exception:
        return [], "ERROR"
    for item in raw_news:
        if not isinstance(item, dict):
            continue
        content = item.get("content") if isinstance(item.get("content"), dict) else item
        title = (content.get("title") or "").strip()
        if not title:
            continue
        pub = ""
        provider = content.get("provider")
        if isinstance(provider, dict):
            pub = (provider.get("displayName") or "").strip()
        elif isinstance(item.get("publisher"), str):
            pub = item.get("publisher", "").strip()
        date_str = ""
        pub_date = content.get("pubDate") or content.get("displayTime")
        if isinstance(pub_date, str) and len(pub_date) >= 10:
            date_str = pub_date[:10]
        elif item.get("providerPublishTime"):
            try:
                date_str = datetime.fromtimestamp(int(item["providerPublishTime"]), tz=timezone.utc).strftime("%Y-%m-%d")
            except Exception:
                pass
        prefix = f"[{date_str}] " if date_str else ""
        suffix = f" ({pub})" if pub else ""
        entry = f"{prefix}{title}{suffix}"
        if entry not in headlines:
            headlines.append(entry)
        if len(headlines) >= limit:
            break
    return headlines, ("OK" if headlines else "OK_EMPTY")


def clean_ai_text(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:800]


def optional_gemini_buffett_check(rec: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Filtro IA escéptico con salida JSON estricta. Devuelve (texto, veredicto_enum, error)."""
    if not GEMINI_API_KEY:
        return None, None, "GEMINI_API_KEY no configurada"

    news = rec.get("recent_news") or []
    news_status = rec.get("news_status") or ("OK" if news else "OK_EMPTY")
    if news:
        news_block = "\n".join(f"- {h}" for h in news)
    elif news_status == "ERROR":
        news_block = "- ERROR obteniendo titulares en Yahoo Finance (no asumas ausencia de problemas)."
    else:
        news_block = "- Sin titulares recientes disponibles en Yahoo Finance."

    warnings_block = "; ".join(rec.get("data_warnings") or []) or "Ninguna"
    dcf = rec.get("dcf") or {}
    dcf_info = f"{dcf.get('intrinsic_base')} {rec['currency']} (obj -25%: {dcf.get('target_price')})" if dcf else "N/A"
    ins_val = rec.get("insider_buys_90d")
    ins_st = rec.get("insider_status")
    insider_txt = f"Sí ({ins_val} compras)" if (ins_val or 0) > 0 else ("Error proveedor" if ins_st == "ERROR" else "Sin compras recientes")

    prompt = (
        "Eres un analista ESCÉPTICO estilo Charlie Munger buscando razones para NO comprar (evitar trampas de valor).\n"
        "No regales el veredicto TEMPORAL: si faltan noticias claras que expliquen la caída usa EVIDENCIA_INSUFICIENTE; "
        "si hay riesgo estructural usa DUDOSO o PERMANENTE.\n"
        f"Empresa: {rec['name']} ({rec['ticker']}), Sector: {rec['sector']} / {rec['industry']}.\n"
        f"Datos: Precio {rec['price']} {rec['currency']} (caída 52s: {rec['drawdown_52w_pct']}%), "
        f"Calidad: {rec['quality_score']}/100, ROIC: {rec['roic_pct']}%, Margen op: {rec['operating_margin_pct']}%, "
        f"Deuda/EBITDA: {rec['net_debt_ebitda']}x, P/FCF: {rec['p_fcf']}x, PER futuro: {rec['forward_pe']}x, "
        f"DCF base (caja real): {dcf_info}, Compras de directivos 90d: {insider_txt}.\n"
        f"Alertas de datos: {warnings_block}.\n"
        f"Titulares recientes (ignora noticias genéricas que no hablen de esta empresa):\n{news_block}\n\n"
        "Devuelve EXCLUSIVAMENTE un objeto JSON válido en español con estas 4 claves exactas:\n"
        '{"motivo_caida": "...", "foso": "...", "mayor_riesgo": "...", "veredicto": "TEMPORAL | DUDOSO | PERMANENTE | EVIDENCIA_INSUFICIENTE"}'
    )

    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 900,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingLevel": "LOW"},
        },
    }).encode("utf-8")

    last_err = None
    for model in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
        for attempt in range(2):
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=25) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                parsed, err = extract_gemini_json(body)
                if err or not parsed:
                    last_err = f"{model}: {err}"
                    continue
                verdict = str(parsed.get("veredicto") or "").strip().upper().replace(" ", "_")
                if verdict not in VALID_VERDICTS:
                    last_err = f"{model}: veredicto inválido ({verdict})"
                    continue
                motivo = str(parsed.get("motivo_caida") or "").strip()
                foso = str(parsed.get("foso") or "").strip()
                riesgo = str(parsed.get("mayor_riesgo") or "").strip()
                if len(motivo) < 8 or len(foso) < 8 or len(riesgo) < 8:
                    last_err = f"{model}: respuesta JSON incompleta"
                    continue
                txt = (
                    f"1) Por qué cae: {motivo}\n"
                    f"2) Foso: {foso}\n"
                    f"3) Mayor riesgo: {riesgo}\n"
                    f"4) Veredicto: {verdict}"
                )
                return clean_ai_text(txt), verdict, None
            except urllib.error.HTTPError as exc:
                err_body = ""
                try:
                    err_body = exc.read().decode("utf-8", errors="ignore")[:160]
                except Exception:
                    pass
                last_err = f"{model} HTTP {exc.code}: {err_body}"
                if exc.code in (429, 500, 503) and attempt == 0:
                    time.sleep(3.5)
                    continue
                break
            except Exception as exc:
                last_err = f"{model}: {exc}"
                if attempt == 0:
                    time.sleep(2.0)
                    continue
                break

    print(f"⚠️ Gemini falló en {rec['ticker']}: {last_err}")
    return None, None, last_err


def esc_html(s: Any) -> str:
    return str(s if s is not None else "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def send_telegram(html_text: str) -> None:
    if not TG_TOKEN or not TG_CHAT_ID:
        print("⚠️ Sin TELEGRAM_BOT_TOKEN/CHAT_ID. Mensaje:\n", html_text)
        return
    chunks: List[str] = []
    rest = html_text
    while len(rest) > 3800:
        cut = rest.rfind("\n", 0, 3800)
        idx = cut if cut > 500 else 3800
        chunks.append(rest[:idx])
        rest = rest[idx:]
    if rest.strip():
        chunks.append(rest)

    for chunk in chunks:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        payload = json.dumps({
            "chat_id": TG_CHAT_ID,
            "text": chunk,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }).encode("utf-8")
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    if resp.status >= 300:
                        raise RuntimeError(f"Telegram HTTP {resp.status}")
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                time.sleep(2.0 * (attempt + 1))
        if last_exc is not None:
            raise last_exc


def _get_fcm_access_token(sa_info: Dict[str, Any]) -> str:
    """Obtiene token OAuth2 para Firebase Cloud Messaging HTTP v1 usando google-auth."""
    from google.auth.transport.requests import Request as GoogleAuthRequest
    from google.oauth2 import service_account

    creds = service_account.Credentials.from_service_account_info(
        sa_info,
        scopes=["https://www.googleapis.com/auth/firebase.messaging"],
    )
    creds.refresh(GoogleAuthRequest())
    if not creds.token:
        raise RuntimeError("No se pudo obtener access_token OAuth2 de Firebase")
    return str(creds.token)


def send_fcm_push(
    title: str,
    body: str,
    ticker: Optional[str] = None,
    tier: str = "ENTRADA_YA",
) -> int:
    """Envía notificación push nativa vía Firebase Cloud Messaging (HTTP v1) a tokens o al topic de la APK."""
    if not FIREBASE_SERVICE_ACCOUNT:
        return 0

    tokens = [
        tok.strip()
        for tok in re.split(r"[,;\s]+", FIREBASE_DEVICE_TOKEN or "")
        if tok.strip()
    ]
    topic = (FIREBASE_TOPIC or "").strip()
    if not tokens and not topic:
        return 0

    try:
        sa_info = json.loads(FIREBASE_SERVICE_ACCOUNT)
        project_id = str(sa_info.get("project_id") or "").strip()
        if not project_id:
            raise ValueError("FIREBASE_SERVICE_ACCOUNT sin project_id")
        access_token = _get_fcm_access_token(sa_info)
    except Exception as exc:
        print(f"⚠️ Error autenticando Firebase FCM: {exc}")
        return 0

    url = f"https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"
    rel_link = f"./?ticker={urllib.parse.quote(ticker)}" if ticker else "./?filter=ENTRADA_YA"
    abs_link = f"{APP_BASE_URL}/?ticker={urllib.parse.quote(ticker)}" if ticker else f"{APP_BASE_URL}/?filter=ENTRADA_YA"

    targets: List[Tuple[str, str]] = [("token", tok) for tok in tokens[:10]]
    if topic and (not targets or FIREBASE_SEND_BOTH):
        targets.append(("topic", topic))

    sent_ok = 0
    for target_kind, target_val in targets:
        base_msg: Dict[str, Any] = {
                "notification": {
                    "title": title[:120],
                    "body": body[:240],
                },
                "data": {
                    "ticker": str(ticker or ""),
                    "tier": str(tier or "ENTRADA_YA"),
                    "url": rel_link,
                    "title": title[:120],
                    "body": body[:240],
                },
                "webpush": {
                    "headers": {
                        "Urgency": "high",
                        "TTL": "3600",
                    },
                    "notification": {
                        "title": title[:120],
                        "body": body[:240],
                        "tag": f"cocodrilo-entrar-ya-{ticker or 'alert'}",
                        "renotify": True,
                        "requireInteraction": True,
                    },
                    "fcm_options": {
                        "link": abs_link,
                    },
                },
            }
        base_msg[target_kind] = target_val
        message_payload = {"message": base_msg}
        req = urllib.request.Request(
            url,
            data=json.dumps(message_payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                if 200 <= resp.status < 300:
                    sent_ok += 1
        except Exception as exc:
            print(f"⚠️ Error enviando push FCM ({target_kind}={target_val[:14]}...): {exc}")

    if sent_ok > 0:
        print(f"🔔 Push Firebase FCM enviado ({sent_ok}/{len(targets)} destino(s)).")
    return sent_ok


def to_eur(amount: Any, currency: Optional[str]) -> Optional[float]:
    """Convierte a EUR de forma estricta; devuelve None si no existe tipo de cambio válido (nunca falsea EUR)."""
    val = safe_float(amount)
    if val is None:
        return None
    cur = (currency or "EUR").strip().upper()
    if cur == "EUR":
        return val
    fx = get_fx_rate(cur, "EUR")
    if fx is None or fx <= 0:
        fx = FALLBACK_FX_EUR.get(cur)
    return (val * fx) if (fx and fx > 0) else None


def fmt_eur(amount: Any, currency: Optional[str]) -> str:
    raw = safe_float(amount)
    if raw is None:
        return "—"
    val = to_eur(raw, currency)
    if val is not None:
        return f"{val:.2f} €"
    cur = (currency or "").strip().upper()
    return f"{raw:.2f} {cur} (sin FX)".strip()


def compute_adaptive_stop_profile(r: Dict[str, Any], ref_entry_eur: float) -> Tuple[float, Optional[float], str]:
    """Calcula Stop Loss adaptativo según sector, volatilidad 52s y soporte técnico (seguro con datos incompletos)."""
    sec = str(r.get("sector") or "").lower()
    prof = str(r.get("sector_profile") or "").lower()
    if prof == "retail" or any(k in sec for k in ("health", "defensive", "utilities", "staple")):
        base_pct, tag = 14.0, "Defensivo"
    elif prof == "tech" or any(k in sec for k in ("tech", "communication", "semiconductor")):
        base_pct, tag = 20.0, "Tech/Volátil"
    else:
        base_pct, tag = 17.0, "Estándar"

    vol_adj = 0.0
    h52 = safe_float(r.get("high_52w"))
    l52 = safe_float(r.get("low_52w"))
    has_range52 = bool(h52 is not None and l52 is not None and h52 > l52 > 0)
    if has_range52:
        range52 = (h52 - l52) / h52
        if range52 >= 0.45:
            vol_adj = 3.0
        elif range52 >= 0.35:
            vol_adj = 1.5
        elif range52 <= 0.22:
            vol_adj = -2.0
    pct = max(11.0, min(25.0, round(base_pct + vol_adj)))

    cur = (r.get("currency") or "USD").upper()
    low52_eur = to_eur(l52, cur) if has_range52 else None
    if low52_eur and ref_entry_eur > 0 and (ref_entry_eur * 0.76 <= low52_eur < ref_entry_eur * 0.94):
        support_stop = low52_eur * 0.97
        sup_pct = abs(((support_stop / ref_entry_eur) - 1.0) * 100.0)
        if 10.0 <= sup_pct <= 26.0:
            return round(sup_pct, 1), support_stop, f"Auto · {tag} + Soporte 52s"

    stop_eur = (ref_entry_eur * (1.0 - pct / 100.0)) if ref_entry_eur > 0 else None
    suffix = "" if has_range52 else " (sin rango 52s)"
    return pct, stop_eur, f"Auto · {tag}{suffix}"


def compute_trade_plan_eur(
    r: Dict[str, Any],
    budget_eur: Optional[float] = None,
    stop_loss_pct: Optional[float] = None,
    stop_loss_mode: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Calcula plan operativo en EUR (3 tramos de entrada, Salida 50%/100% y Stop Loss adaptativo o Solo Tesis)."""
    cur = (r.get("currency") or "USD").upper()
    p_now = to_eur(r.get("price"), cur)
    if p_now is None or p_now <= 0:
        return None

    dcf = r.get("dcf") or {}
    base_eur = to_eur(dcf.get("intrinsic_base"), cur) if dcf.get("intrinsic_base") else None
    bear_eur = to_eur(dcf.get("intrinsic_bear"), cur) if dcf.get("intrinsic_bear") else None
    buy_obj_eur = to_eur(dcf.get("target_price"), cur) if dcf.get("target_price") else None

    in_buy_zone = (r.get("status") == "OPORTUNIDAD") or (buy_obj_eur is not None and p_now <= buy_obj_eur)
    p1 = p_now if in_buy_zone else (buy_obj_eur if (buy_obj_eur and buy_obj_eur < p_now) else p_now)
    p2 = bear_eur if (bear_eur and p1 * 0.55 < bear_eur < p1 * 0.96) else (p1 * 0.88)
    p3 = p2 * 0.88

    eff_budget = budget_eur if (budget_eur is not None and float(budget_eur) > 0) else DEFAULT_TRADE_BUDGET_EUR
    is_example_budget = eff_budget is None
    budget = max(10.0, float(eff_budget)) if not is_example_budget else 80.0
    m1, m2, m3 = budget * 0.35, budget * 0.35, budget * 0.30
    total_shares = (m1 / p1) + (m2 / p2) + (m3 / p3)
    avg_entry = (budget / total_shares) if total_shares > 0 else p1

    exit1 = base_eur if (base_eur and base_eur > avg_entry * 1.15) else (avg_entry * 1.25)
    exit2 = max(base_eur * 1.25, exit1 * 1.12) if base_eur else (avg_entry * 1.45)
    exit1_pct = ((exit1 / avg_entry) - 1.0) * 100.0
    exit2_pct = ((exit2 / avg_entry) - 1.0) * 100.0

    mode = (stop_loss_mode or DEFAULT_STOP_LOSS_MODE or "AUTO").strip().upper()
    is_thesis_only = (mode == "THESIS") and (stop_loss_pct is None)

    if stop_loss_pct is not None:
        sl_mult = 1.0 - (max(5.0, min(45.0, float(stop_loss_pct))) / 100.0)
        stop_loss = min(avg_entry * sl_mult, p3 * 0.94)
        sl_reason = f"Manual -{float(stop_loss_pct):.0f}%"
    elif is_thesis_only:
        _, adap_stop, _ = compute_adaptive_stop_profile(r, avg_entry)
        base_sl = adap_stop if adap_stop is not None else (avg_entry * 0.83)
        stop_loss = min(base_sl, p3 * 0.94)
        sl_reason = "Solo Tesis (Ref. técnica)"
    elif mode != "AUTO" and mode.replace(".", "", 1).isdigit():
        sl_val = float(mode)
        sl_mult = 1.0 - (max(5.0, min(45.0, sl_val)) / 100.0)
        stop_loss = min(avg_entry * sl_mult, p3 * 0.94)
        sl_reason = f"Manual -{sl_val:.0f}%"
    else:
        _, adap_stop, sl_reason = compute_adaptive_stop_profile(r, avg_entry)
        base_sl = adap_stop if adap_stop is not None else (avg_entry * 0.83)
        stop_loss = min(base_sl, p3 * 0.94)

    stop_loss_pct_calc = ((stop_loss / avg_entry) - 1.0) * 100.0

    return {
        "p_now": round(p_now, 2),
        "budget": round(budget, 0),
        "is_example_budget": is_example_budget,
        "m1": round(m1, 0),
        "m2": round(m2, 0),
        "m3": round(m3, 0),
        "p1": round(p1, 2),
        "p2": round(p2, 2),
        "p3": round(p3, 2),
        "avg_entry": round(avg_entry, 2),
        "exit1": round(exit1, 2),
        "exit2": round(exit2, 2),
        "exit1_pct": round(exit1_pct, 1),
        "exit2_pct": round(exit2_pct, 1),
        "stop_loss": round(stop_loss, 2),
        "stop_loss_pct": round(stop_loss_pct_calc, 1),
        "sl_reason": sl_reason,
        "stop_loss_mode": "THESIS" if is_thesis_only else "TACTICAL",
        "in_buy_zone": 1.0 if in_buy_zone else 0.0,
    }


def send_fcm_urgent_alerts(entradas_ya: List[Dict[str, Any]]) -> int:
    """Envía un push prioritario de Firebase por cada oportunidad ENTRAR YA detectada (máx. 3)."""
    if not entradas_ya or not FIREBASE_SERVICE_ACCOUNT or (not FIREBASE_DEVICE_TOKEN and not FIREBASE_TOPIC):
        return 0
    total_sent = 0
    for r in entradas_ya[:3]:
        tkr = r.get("ticker") or ""
        name = (r.get("name") or tkr)[:28]
        price_txt = fmt_eur(r.get("price"), r.get("currency"))
        dd_txt = f"{r['drawdown_52w_pct']}%" if r.get("drawdown_52w_pct") is not None else "—"
        oro_prefix = "👑 ORO PURO · " if r.get("oro_puro") else ""
        plan = compute_trade_plan_eur(r)
        if plan:
            t1_str = "35% bote" if plan.get("is_example_budget") else f"{plan['m1']:.0f}€"
            sl_str = "Solo Tesis" if plan.get("stop_loss_mode") == "THESIS" else f"SL {plan['stop_loss']:.2f}€"
            plan_txt = f" · T1: {t1_str} | Salida: {plan['exit1']:.2f}€ | {sl_str}"
        else:
            plan_txt = ""
        title = f"🚨 ENTRAR YA: {tkr} ({name})"
        body = (
            f"{oro_prefix}{price_txt} (52s: {dd_txt}) · Score {r.get('quality_score', '—')}/100{plan_txt}"
        )
        total_sent += send_fcm_push(title=title, body=body, ticker=tkr, tier="ENTRADA_YA")
    return total_sent


def load_alert_cooldown(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Carga el registro anti-spam de alertas ENTRAR YA desde data/alert-cooldown.json."""
    target = path or OUT_COOLDOWN_JSON
    if not target.exists():
        return {}
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
        entries = raw.get("tickers") if isinstance(raw, dict) else None
        if isinstance(entries, dict):
            return {str(k).upper(): v for k, v in entries.items() if isinstance(v, dict)}
    except Exception as exc:
        print(f"⚠️ No se pudo leer {target.name}: {exc}")
    return {}


def filter_new_urgent_alerts(
    entradas_ya: List[Dict[str, Any]],
    cooldown_map: Dict[str, Dict[str, Any]],
    now_dt: Optional[datetime] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Filtra alertas repetidas: solo avisa si es nueva, pasaron ALERT_COOLDOWN_DAYS o cayó >= ALERT_RETRIGGER_DROP_PCT%."""
    now = now_dt or datetime.now(timezone.utc)
    to_notify: List[Dict[str, Any]] = []
    suppressed: List[str] = []

    for r in entradas_ya:
        tkr = str(r.get("ticker") or "").upper()
        if not tkr:
            continue
        prev = cooldown_map.get(tkr)
        if not prev:
            to_notify.append(r)
            continue

        elapsed_days: Optional[float] = None
        last_iso = str(prev.get("last_alert_at") or "").strip()
        if last_iso:
            try:
                prev_dt = datetime.fromisoformat(last_iso.replace("Z", "+00:00"))
                if prev_dt.tzinfo is None:
                    prev_dt = prev_dt.replace(tzinfo=timezone.utc)
                elapsed_days = (now - prev_dt).total_seconds() / 86400.0
            except Exception:
                elapsed_days = None

        last_price = safe_float(prev.get("last_alert_price"))
        cur_price = safe_float(r.get("price"))
        extra_drop_pct = None
        if last_price and cur_price and last_price > 0:
            extra_drop_pct = ((cur_price / last_price) - 1.0) * 100.0

        expired = elapsed_days is None or elapsed_days >= ALERT_COOLDOWN_DAYS
        dropped_more = extra_drop_pct is not None and extra_drop_pct <= -ALERT_RETRIGGER_DROP_PCT

        if expired or dropped_more:
            to_notify.append(r)
        else:
            suppressed.append(tkr)
            days_txt = f"{elapsed_days:.1f}d" if elapsed_days is not None else "?"
            chg_txt = f"{extra_drop_pct:+.1f}%" if extra_drop_pct is not None else "N/D"
            print(
                f"🔇 Cooldown anti-spam activo para {tkr} (avisada hace {days_txt}, var. {chg_txt}; "
                f"se silencia hasta {ALERT_COOLDOWN_DAYS:g}d o caída <= -{ALERT_RETRIGGER_DROP_PCT:g}%)."
            )

    return to_notify, suppressed


def save_alert_cooldown(
    notified: List[Dict[str, Any]],
    cooldown_map: Dict[str, Dict[str, Any]],
    now_dt: Optional[datetime] = None,
    path: Optional[Path] = None,
) -> None:
    """Guarda fecha y precio del último aviso ENTRAR YA para evitar notificaciones repetidas."""
    if not notified:
        return
    now = now_dt or datetime.now(timezone.utc)
    now_iso = now.isoformat()
    updated = dict(cooldown_map)

    for r in notified:
        tkr = str(r.get("ticker") or "").upper()
        if not tkr:
            continue
        prev = updated.get(tkr) or {}
        updated[tkr] = {
            "last_alert_at": now_iso,
            "last_alert_price": safe_float(r.get("price")),
            "currency": r.get("currency") or "",
            "tier": r.get("opportunity_tier") or "ENTRADA_YA",
            "count": int(prev.get("count") or 0) + 1,
        }

    # Limpieza de registros con más de 90 días para mantener ligero el JSON
    pruned: Dict[str, Dict[str, Any]] = {}
    for tkr, info in updated.items():
        keep = True
        ts = str(info.get("last_alert_at") or "")
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if (now - dt).days > 90:
                    keep = False
            except Exception:
                pass
        if keep:
            pruned[tkr] = info

    target = path or OUT_COOLDOWN_JSON
    target.parent.mkdir(parents=True, exist_ok=True)
    payload_cd = {
        "updated_at": now_iso,
        "cooldown_days": ALERT_COOLDOWN_DAYS,
        "retrigger_drop_pct": ALERT_RETRIGGER_DROP_PCT,
        "tickers": pruned,
    }
    target.write_text(json.dumps(payload_cd, indent=2, ensure_ascii=False), encoding="utf-8")


def extract_ai_verdict(ai_text: Optional[str]) -> Optional[str]:
    """Fallback estricto: solo acepta un veredicto exacto al final de la línea VEREDICTO."""
    if not ai_text:
        return None
    m = re.search(r"(?im)^\s*(?:\d\)\s*)?veredicto\s*:\s*(TEMPORAL|DUDOSO|PERMANENTE|EVIDENCIA_INSUFICIENTE)\b", ai_text)
    return m.group(1).upper() if m else None


def has_severe_warning(rec: Dict[str, Any]) -> bool:
    warns = rec.get("data_warnings") or []
    return any(
        k in w.lower()
        for w in warns
        for k in ("extremo", "sin tipo de cambio", "financiero")
    )


def compute_conviction_score(rec: Dict[str, Any]) -> float:
    score = float(rec.get("quality_score") or 0)
    disc = rec.get("discount_vs_base_pct")
    if disc is not None:
        score += min(max(disc, -20.0), 60.0) * 0.75
    if rec.get("below_bear"):
        score += 14.0
    elif rec.get("in_dcf_zone"):
        score += 6.0

    fpe = rec.get("forward_pe")
    pfcf = rec.get("p_fcf")
    if fpe is not None:
        if fpe <= 14:
            score += 10.0
        elif fpe <= 19:
            score += 5.0
        elif fpe > 26:
            score -= 8.0
    elif pfcf is not None and pfcf <= 14:
        score += 6.0

    nde = rec.get("net_debt_ebitda")
    if nde is not None:
        if nde <= 0:
            score += 6.0
        elif nde <= 1.2:
            score += 3.0

    if (rec.get("insider_buys_90d") or 0) > 0:
        score += 6.0

    if has_severe_warning(rec):
        score -= 18.0
    elif rec.get("data_warnings"):
        score -= 4.0

    verdict = rec.get("ai_verdict") or extract_ai_verdict(rec.get("ai_check"))
    if verdict == "TEMPORAL":
        score += 8.0
    elif verdict == "EVIDENCIA_INSUFICIENTE":
        score -= 6.0
    elif verdict == "DUDOSO":
        score -= 10.0
    elif verdict == "PERMANENTE":
        score -= 35.0

    return round(score, 1)


def classify_opportunity_tier(rec: Dict[str, Any]) -> Optional[str]:
    if rec.get("status") != "OPORTUNIDAD":
        return None

    q = rec.get("quality_score") or 0
    disc = rec.get("discount_vs_base_pct") or -999.0
    dd = rec.get("drawdown_52w_pct") if rec.get("drawdown_52w_pct") is not None else 0.0
    fpe = rec.get("forward_pe")
    pfcf = rec.get("p_fcf")
    roic = rec.get("roic_pct") or 0.0
    nde = rec.get("net_debt_ebitda")
    f_pos = rec.get("fcf_years_pos")
    f_tot = rec.get("fcf_years_total")
    in_dcf = bool(rec.get("in_dcf_zone"))
    below_bear = bool(rec.get("below_bear"))
    severe_warn = has_severe_warning(rec)
    verdict = rec.get("ai_verdict")

    fcf_perfect = f_pos is not None and f_tot is not None and f_tot >= 3 and f_pos == f_tot
    debt_fortress = nde is not None and nde <= 1.2
    ultra_cheap = fpe is not None and fpe <= 15.5 and (pfcf is None or pfcf <= 15.0)
    multiple_sane = (fpe is not None and fpe <= 20.0) or (fpe is None and pfcf is not None and pfcf <= 16.0)

    if (
        q >= 92
        and roic >= 18.0
        and debt_fortress
        and fcf_perfect
        and in_dcf
        and below_bear
        and 40.0 <= disc <= 75.0
        and dd <= -25.0
        and ultra_cheap
        and not severe_warn
        and verdict == "TEMPORAL"
    ):
        return "ENTRADA_YA"

    if (
        q >= 90
        and in_dcf
        and (below_bear or disc >= 35.0)
        and dd <= -20.0
        and multiple_sane
        and not severe_warn
        and verdict not in ("DUDOSO", "PERMANENTE")
    ):
        return "GANGA"

    if (
        q >= 85
        and (in_dcf or (disc >= 20.0 and dd <= -18.0))
        and not severe_warn
        and verdict != "PERMANENTE"
    ):
        return "MUY_BUENA"

    return "BUENA"


def format_detailed_opp(r: Dict[str, Any], header_line: str) -> str:
    dcf = r.get("dcf") or {}
    cur = r["currency"]
    dd = f"{r['drawdown_52w_pct']}%" if r["drawdown_52w_pct"] is not None else "—"
    pfcf = f"{r['p_fcf']}x" if r["p_fcf"] is not None else "—"
    fpe = f"{r['forward_pe']}x" if r["forward_pe"] is not None else "—"
    nde = f"{r['net_debt_ebitda']}x" if r["net_debt_ebitda"] is not None else "—"
    disc = f" (+{r['discount_vs_base_pct']}% desc.)" if (r.get("discount_vs_base_pct") is not None and r["discount_vs_base_pct"] > 0) else ""
    bear_tag = " 🔥 Bajo conservador" if r.get("below_bear") else ""
    ins_tag = f" · 👔 {r['insider_buys_90d']} compras directivos" if (r.get("insider_buys_90d") or 0) > 0 else ""

    price_eur_txt = fmt_eur(r.get("price"), cur)
    block = [
        header_line,
        f"• Precio: <b>{price_eur_txt}</b> (52s: <b>{dd}</b>){bear_tag}{ins_tag}",
        f"• Calidad: <b>{r['quality_score']}/100</b> | ROIC: {r['roic_pct']}% | Deuda: {nde}",
        f"• Múltiplos: P/FCF <b>{pfcf}</b> | PER Futuro: <b>{fpe}</b>",
    ]
    if dcf:
        oe_note = " (caja real -SBC)" if r.get("owner_earnings_adjusted") else ""
        block.append(
            f"• DCF Base{oe_note}: <b>{fmt_eur(dcf.get('intrinsic_base'), cur)}</b> | Compra (-{int(MARGIN_OF_SAFETY_PCT)}%): <b>{fmt_eur(dcf.get('target_price'), cur)}</b>{disc}"
        )
    plan = compute_trade_plan_eur(r)
    if plan:
        t1_lbl = "Hoy" if plan["in_buy_zone"] else "Al tocar obj."
        if plan.get("is_example_budget"):
            block.append(
                f"• 💶 <b>Plan entrada (35% / 35% / 30% · ajusta a tu bote real):</b> "
                f"T1 ({t1_lbl}) a <b>{plan['p1']:.2f} €</b> | "
                f"T2 a <b>{plan['p2']:.2f} €</b> | "
                f"T3 a <b>{plan['p3']:.2f} €</b>"
            )
        else:
            block.append(
                f"• 💶 <b>Entrada EUR ({plan['budget']:.0f} €):</b> "
                f"T1 ({t1_lbl}) <b>{plan['m1']:.0f} €</b> a {plan['p1']:.2f} € | "
                f"T2 <b>{plan['m2']:.0f} €</b> a {plan['p2']:.2f} € | "
                f"T3 <b>{plan['m3']:.0f} €</b> a {plan['p3']:.2f} €"
            )
        block.append(
            f"• 🎯 <b>Salida (TP):</b> 50% en <b>{plan['exit1']:.2f} €</b> (+{plan['exit1_pct']:.0f}%) | "
            f"Total en <b>{plan['exit2']:.2f} €</b> (+{plan['exit2_pct']:.0f}%) | "
            f"🛑 <b>Stop Loss:</b> <b>{plan['stop_loss']:.2f} €</b> ({plan['stop_loss_pct']:.0f}% · {esc_html(plan.get('sl_reason', 'Auto'))})"
        )
    if r.get("earnings_days") is not None and r["earnings_days"] <= 10:
        block.append(f"• 📅 <b>OJO RESULTADOS:</b> presenta en <b>{r['earnings_days']} días</b> ({r['earnings_date']}) — prudencia antes de entrar fuerte.")
    if r.get("strengths"):
        block.append(f"• Claves: {esc_html(' · '.join(r['strengths'][:3]))}")
    if r.get("hard_fails"):
        block.append(f"• ❌ <i>Fallos críticos:</i> {esc_html(' · '.join(r['hard_fails']))}")
    if r.get("piotroski_score") is not None or r.get("piotroski_classic_score") is not None:
        cov_txt = f" | Cob.Int: {r['interest_coverage']}x" if r.get("interest_coverage") is not None else ""
        cls_txt = (
            f" | Clásico: <b>{r['piotroski_classic_score']}/{r['piotroski_classic_max']}</b>"
            if r.get("piotroski_classic_score") is not None
            else ""
        )
        adp_txt = (
            f"Adaptado <b>{r['piotroski_score']}/{r['piotroski_max']}</b>"
            if r.get("piotroski_score") is not None
            else "Adaptado N/D"
        )
        block.append(f"• 🛡️ Oráculo: Piotroski {adp_txt}{cls_txt}{cov_txt}")
    if r.get("quality_valuation") and r["quality_valuation"].get("quadrant_label"):
        block.append(f"• 🧭 Cuadrante: <b>{esc_html(r['quality_valuation']['quadrant_label'])}</b>")
    if r.get("sec_note") and r.get("sec_status") not in (None, "NO_US"):
        sec_ic = "✅" if r.get("sec_verified") is True else "🚨" if r.get("sec_verified") is False else "ℹ️"
        block.append(f"• {sec_ic} <i>SEC EDGAR:</i> {esc_html(r['sec_note'])}")
    if r.get("oro_puro"):
        block.append("• 👑 <b>SEÑAL ORO PURO (1-2/AÑO):</b> doble verificación superada + riesgo trampa bajo.")
    if r.get("oracle_flags"):
        block.append(f"• 🚩 <i>Anti-Trampas:</i> {esc_html(' | '.join(r['oracle_flags']))}")
    if r.get("data_warnings"):
        block.append(f"• ⚠️ <i>Aviso:</i> {esc_html(' | '.join(r['data_warnings']))}")
    if r.get("ai_check"):
        v_tag = f" ({r['ai_verdict']})" if r.get("ai_verdict") else ""
        block.append(f"• 🤖 <i>IA + Noticias{v_tag}:</i>\n{esc_html(r['ai_check'])}")
    return "\n".join(block)


def build_urgent_telegram_alert(entradas_ya: List[Dict[str, Any]], gap_pts: Optional[float] = None) -> str:
    now_str = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    lines = [
        "🚨🔥🚨 <b>ALERTA MÁXIMA: ENTRAR YA</b> 🚨🔥🚨",
        f"<i>{now_str} · Oportunidad extrema detectada por Radar Cocodrilo v11.3</i>",
    ]
    if gap_pts is not None and gap_pts >= 12:
        lines.append(f"⚡ <b>MUY DESPLAZADA DEL RESTO:</b> saca <b>+{gap_pts:.0f} pts</b> de ventaja al siguiente valor.")

    for idx, r in enumerate(entradas_ya[:3], start=1):
        hdr = f"\n🚨👑 <b>#{idx} ENTRAR YA — {esc_html(r['name'])} ({esc_html(r['ticker'])})</b> · <i>{esc_html(r['region'])}</i>"
        lines.append(format_detailed_opp(r, hdr))

    lines.append(
        "\n⚡ <b>ACCIÓN INMEDIATA:</b> dedica 3 minutos a confirmar las noticias de hoy y el último trimestre; "
        "si no hay fraude ni deterioro estructural, es zona de entrada escalonada."
    )
    return "\n".join(lines)


def build_telegram_report(
    results: List[Dict[str, Any]],
    failed_count: int,
    ai_status: Dict[str, Any],
    total_scanned: int,
    market_ctx: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    date_str = datetime.now(timezone.utc).strftime("%d/%m/%Y")

    if UNIVERSE_MODE == "custom":
        lines = [
            f"🔎 <b>ANÁLISIS EXPRÉS COCODRILO v11.3 — {date_str}</b>",
            f"<i>Empresas analizadas: {len(results)}/{total_scanned}</i>",
        ]
        for r in results[:10]:
            st = r.get("opportunity_tier") or r["status"]
            hdr = f"\n📌 <b>{esc_html(r['name'])} ({esc_html(r['ticker'])})</b> · <i>{esc_html(st)} · {esc_html(r['region'])}</i>"
            lines.append(format_detailed_opp(r, hdr))
        return "\n".join(lines)

    opps = [r for r in results if r["status"] == "OPORTUNIDAD"]
    watch = [r for r in results if r["status"] == "VIGILAR"]

    entradas_ya = [r for r in opps if r.get("opportunity_tier") == "ENTRADA_YA"]
    gangas = [r for r in opps if r.get("opportunity_tier") == "GANGA"]
    muy_buenas = [r for r in opps if r.get("opportunity_tier") == "MUY_BUENA"]
    buenas = [r for r in opps if r.get("opportunity_tier") == "BUENA"]

    for group in (entradas_ya, gangas, muy_buenas, buenas):
        group.sort(key=lambda r: -(r.get("conviction_score") or 0))
    watch.sort(key=lambda r: (r["drawdown_52w_pct"] if r["drawdown_52w_pct"] is not None else 0, -r["quality_score"]))

    if not opps and not watch and not TEST_MODE:
        return None

    ya_txt = f"🚨 {len(entradas_ya)} ENTRAR YA · " if entradas_ya else ""
    tier_summary = f"{ya_txt}🏆 {len(gangas)} Ganga · 🥇 {len(muy_buenas)} Muy buenas · 🥈 {len(buenas)} Buenas · 🟡 {len(watch)} Vigilar"
    lines = [
        f"🐊 <b>RADAR COCODRILO PRO v11.3 — {date_str}</b>",
        f"<i>{esc_html(UNIVERSE_MODE.upper())} · {len(results)} analizadas (de {total_scanned}) · {tier_summary}</i>",
    ]
    if market_ctx and (market_ctx.get("vix") is not None or market_ctx.get("us10y_pct") is not None):
        v_str = f"VIX {market_ctx['vix']}" if market_ctx.get("vix") is not None else ""
        y_str = f"Bono 10A {market_ctx['us10y_pct']}%" if market_ctx.get("us10y_pct") is not None else ""
        s_str = f"S&amp;P {market_ctx['sp500_drawdown_pct']}%" if market_ctx.get("sp500_drawdown_pct") is not None else ""
        parts_m = " · ".join(x for x in (v_str, y_str, s_str) if x)
        lines.append(f"🌡️ <b>Clima {esc_html(market_ctx['regime'])}:</b> {parts_m}")

    if entradas_ya:
        lines.append("\n🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥\n🚨🔥 <b>ENTRAR YA — OPORTUNIDAD HISTÓRICA</b> 🔥🚨\n🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥")
        for idx, r in enumerate(entradas_ya[:3], start=1):
            hdr = f"\n🚨👑 <b>#{idx} ENTRAR YA — {esc_html(r['name'])} ({esc_html(r['ticker'])})</b> · <i>{esc_html(r['region'])}</i>"
            lines.append(format_detailed_opp(r, hdr))

    if gangas:
        lines.append("\n━━━━━━━━━━━━━━━━━━━\n🚨🏆 <b>GANGA EXCEPCIONAL (MÁXIMA CONVICCIÓN)</b> 🏆🚨\n━━━━━━━━━━━━━━━━━━━")
        for idx, r in enumerate(gangas[:3], start=1):
            crown = "👑 <b>#1 TOP GANGA DEL DÍA — " if idx == 1 else f"🏆 <b>#{idx} GANGA — "
            hdr = f"\n{crown}{esc_html(r['name'])} ({esc_html(r['ticker'])})</b> · <i>{esc_html(r['region'])}</i>"
            lines.append(format_detailed_opp(r, hdr))

    if muy_buenas:
        lines.append("\n━━━━━━━━━━━━━━━━━━━\n🥇 <b>MUY BUENAS (Alta calidad + buen margen)</b>\n━━━━━━━━━━━━━━━━━━━")
        for r in muy_buenas[:5]:
            hdr = f"\n🥇 <b>{esc_html(r['name'])} ({esc_html(r['ticker'])})</b> · <i>{esc_html(r['region'])}</i>"
            lines.append(format_detailed_opp(r, hdr))
        if len(muy_buenas) > 5:
            lines.append(f"\n<i>…y {len(muy_buenas) - 5} muy buenas más en el panel web.</i>")

    if buenas:
        lines.append("\n━━━━━━━━━━━━━━━━━━━\n🥈 <b>BUENAS (Interesantes / con matices)</b>\n━━━━━━━━━━━━━━━━━━━")
        for r in buenas[:6]:
            dd = f"{r['drawdown_52w_pct']}%" if r["drawdown_52w_pct"] is not None else "—"
            pfcf = f"{r['p_fcf']}x" if r["p_fcf"] is not None else "—"
            fpe = f"{r['forward_pe']}x" if r["forward_pe"] is not None else "—"
            obj = fmt_eur(r['dcf']['target_price'], r['currency']) if r.get("dcf") else "—"
            plan_b = compute_trade_plan_eur(r)
            sl_tp = f" | TP {plan_b['exit1']:.2f} € · SL {plan_b['stop_loss']:.2f} €" if plan_b else ""
            ai_tag = ""
            if r.get("ai_verdict") == "PERMANENTE":
                ai_tag = " · ⚠️ IA: POSIBLE TRAMPA"
            elif r.get("ai_verdict"):
                ai_tag = f" · 🤖 IA: {r['ai_verdict']}"
            earn_tag = f" · 📅 Res.{r['earnings_days']}d" if (r.get("earnings_days") is not None and r["earnings_days"] <= 10) else ""
            lines.append(
                f"• <b>{esc_html(r['ticker'])}</b> ({esc_html(r['name'][:22])}): <b>{fmt_eur(r['price'], r['currency'])}</b> "
                f"(52s: {dd}) | Score <b>{r['quality_score']}</b> | Obj: {obj}{sl_tp}{ai_tag}{earn_tag}"
            )
            if r.get("ai_check") and not entradas_ya and not gangas and not muy_buenas:
                lines.append(f"  <i>{esc_html(r['ai_check'])}</i>")
        if len(buenas) > 6:
            lines.append(f"<i>…y {len(buenas) - 6} buenas más en la app.</i>")

    if not opps:
        lines.append("\n😴 <b>Hoy no hay ninguna empresa de calidad en zona clara de compra.</b> El cocodrilo espera.")

    if watch:
        lines.append("\n━━━━━━━━━━━━━━━━━━━\n🟡 <b>EN VIGILANCIA (Acercándose)</b>\n━━━━━━━━━━━━━━━━━━━")
        for r in watch[:6]:
            dd = f"{r['drawdown_52w_pct']}%" if r["drawdown_52w_pct"] is not None else "—"
            obj = fmt_eur(r['dcf']['target_price'], r['currency']) if r.get("dcf") else "—"
            lines.append(
                f"• <b>{esc_html(r['ticker'])}</b> ({esc_html(r['name'][:20])}): {fmt_eur(r['price'], r['currency'])} "
                f"| 52s: {dd} | Score {r['quality_score']} | Compra: {obj}"
            )
            if not opps and r.get("ai_check"):
                lines.append(f"  🤖 <i>IA ({esc_html(r['ticker'])}):</i> {esc_html(r['ai_check'])}")

    if ai_status["enabled"]:
        if ai_status["errors"] == 0 and ai_status["attempted"] > 0:
            lines.append(f"\n🤖 <i>IA Gemini OK ({ai_status['ok']}/{ai_status['attempted']} con titulares recientes).</i>")
        elif ai_status["errors"] > 0:
            lines.append(
                f"\n⚠️ <i>IA Gemini con incidencias ({ai_status['ok']}/{ai_status['attempted']} OK · {esc_html(ai_status['last_error'])}).</i>"
            )
    else:
        lines.append("\nℹ️ <i>IA Gemini desactivada (sin GEMINI_API_KEY).</i>")

    if failed_count > 0:
        lines.append(f"<i>ℹ️ {failed_count} tickers omitidos por falta de datos o capitalización &lt; 1B USD.</i>")

    lines.append(
        "\n⚠️ <b>Regla de oro:</b> ni siquiera una 🚨 ENTRAR YA o 🏆 Ganga se compra sin 3 minutos de contraste de cuentas y noticias."
    )
    return "\n".join(lines)


def _extract_datetime_series(df: Optional[pd.DataFrame], col: str = "Close") -> Optional[pd.Series]:
    """Devuelve serie ordenada con DatetimeIndex normalizado (sin tz); None si el índice no es temporal real."""
    if df is None or df.empty or col not in df:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            parsed_idx = pd.to_datetime(df.index, errors="coerce")
            if parsed_idx.isna().any() or parsed_idx.min().year < 1990:
                return None
            s = df[col].copy()
            s.index = parsed_idx
        except Exception:
            return None
    else:
        s = df[col].copy()
    idx_naive = s.index.tz_localize(None) if s.index.tz is not None else s.index
    s.index = idx_naive.normalize()
    s = pd.to_numeric(s, errors="coerce").dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s if not s.empty else None


def _get_fx_history_to_eur(src_cur: str, start_date_str: str) -> Optional[pd.Series]:
    """Obtiene serie histórica diaria de cambio src_cur -> EUR desde 7 días antes de start_date_str."""
    src = (src_cur or "EUR").strip().upper()
    if src == "EUR":
        return None
    key = (src, start_date_str[:10])
    with _fx_lock:
        if key in _fx_hist_cache:
            return _fx_hist_cache[key]

    try:
        d0 = datetime.strptime(start_date_str[:10], "%Y-%m-%d").date()
        fetch_start = (d0 - timedelta(days=7)).strftime("%Y-%m-%d")
    except Exception:
        return None

    series_out: Optional[pd.Series] = None
    for pair, invert in ((f"{src}EUR=X", False), (f"EUR{src}=X", True)):
        try:
            h = yf.Ticker(pair).history(start=fetch_start, auto_adjust=False)
            s = _extract_datetime_series(h, "Close")
            if s is None or s.empty:
                continue
            s = s[s > 0]
            if s.empty:
                continue
            series_out = (1.0 / s) if invert else s
            break
        except Exception:
            continue

    with _fx_lock:
        _fx_hist_cache[key] = series_out
    return series_out


def _fx_rate_on_or_before(fx_series: Optional[pd.Series], date_str: str) -> Optional[float]:
    """Devuelve el tipo de cambio en la última sesión <= date_str (o primera en <=7 días si es festivo inicial)."""
    if fx_series is None or fx_series.empty:
        return None
    try:
        target_ts = pd.Timestamp(datetime.strptime(date_str[:10], "%Y-%m-%d").date())
        prior = fx_series[fx_series.index <= target_ts]
        if not prior.empty:
            val = safe_float(prior.iloc[-1])
            return val if (val and val > 0) else None
        first_ts = fx_series.index[0]
        if (first_ts - target_ts).days <= 7:
            val = safe_float(fx_series.iloc[0])
            return val if (val and val > 0) else None
    except Exception:
        return None
    return None


def _cumulative_split_factor(
    ticker: str,
    alert_date_str: str,
    end_date_str: Optional[str] = None,
    splits_series: Optional[pd.Series] = None,
) -> float:
    """Factor acumulado de splits estrictamente posteriores a alert_date_str (ej. 2.0 para split 2:1)."""
    try:
        d0 = pd.Timestamp(datetime.strptime(alert_date_str[:10], "%Y-%m-%d").date())
        d1 = pd.Timestamp(datetime.strptime(end_date_str[:10], "%Y-%m-%d").date()) if end_date_str else None
        sp_s = splits_series if splits_series is not None else getattr(yf.Ticker(ticker), "splits", None)
        if sp_s is None or not isinstance(sp_s, pd.Series) or sp_s.empty:
            return 1.0
        idx_naive = pd.to_datetime(sp_s.index, errors="coerce")
        if idx_naive.isna().any():
            return 1.0
        if getattr(idx_naive, "tz", None) is not None:
            idx_naive = idx_naive.tz_localize(None)
        idx_norm = idx_naive.normalize()
        mask = idx_norm > d0
        if d1 is not None:
            mask = mask & (idx_norm <= d1)
        factor = 1.0
        for raw_v in sp_s[mask].values:
            v = safe_float(raw_v)
            if v is not None and v > 0:
                factor *= v
        return factor if (math.isfinite(factor) and factor > 0) else 1.0
    except Exception:
        return 1.0


def _period_return_pct(symbol: str, start_date_str: str) -> Optional[float]:
    """Calcula rentabilidad % en divisa local desde start_date_str usando historial ajustado (incluye dividendos)."""
    try:
        d0 = datetime.strptime(start_date_str[:10], "%Y-%m-%d")
        h = yf.Ticker(symbol).history(start=d0.strftime("%Y-%m-%d"), auto_adjust=True)
        if h is None or h.empty or "Close" not in h:
            return None
        closes = h["Close"].dropna()
        if len(closes) < 2:
            return None
        p_start = safe_float(closes.iloc[0])
        p_end = safe_float(closes.iloc[-1])
        if p_start and p_end and p_start > 0:
            return round(((p_end / p_start) - 1.0) * 100.0, 1)
    except Exception:
        return None
    return None


def _period_return_eur_pct(
    symbol: str,
    symbol_currency: str,
    start_date_str: str,
    end_date_str: Optional[str] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Devuelve (retorno_local_pct, retorno_eur_pct) usando FX histórico para homogeneizar comparaciones en EUR."""
    try:
        d0 = datetime.strptime(start_date_str[:10], "%Y-%m-%d")
        h = yf.Ticker(symbol).history(start=d0.strftime("%Y-%m-%d"), auto_adjust=True)
        if h is None or h.empty or "Close" not in h:
            return None, None
        closes_raw = h["Close"].dropna()
        if len(closes_raw) < 2:
            return None, None
        p_start = safe_float(closes_raw.iloc[0])
        p_end = safe_float(closes_raw.iloc[-1])
        if not (p_start and p_end and p_start > 0):
            return None, None
        local_ret = round(((p_end / p_start) - 1.0) * 100.0, 1)

        cur = (symbol_currency or "USD").strip().upper()
        if cur == "EUR":
            return local_ret, local_ret

        dt_series = _extract_datetime_series(h, "Close")
        d_start_eff = dt_series.index[0].strftime("%Y-%m-%d") if dt_series is not None else start_date_str[:10]
        d_end_eff = dt_series.index[-1].strftime("%Y-%m-%d") if dt_series is not None else (end_date_str or start_date_str)[:10]

        fx_s = _get_fx_history_to_eur(cur, d_start_eff)
        fx_start = _fx_rate_on_or_before(fx_s, d_start_eff)
        fx_end = _fx_rate_on_or_before(fx_s, d_end_eff)
        if fx_start and fx_end and fx_start > 0 and fx_end > 0:
            eur_ret = round((((p_end * fx_end) / (p_start * fx_start)) - 1.0) * 100.0, 1)
            return local_ret, eur_ret
        return local_ret, None
    except Exception:
        return None, None


def _max_drawdown_after_alert_pct(
    ticker: str,
    init_price: float,
    alert_date_str: str,
    hist_df: Optional[pd.DataFrame] = None,
) -> Optional[float]:
    """Calcula la máxima caída porcentual sufrida desde la fecha de alerta respecto al precio de entrada (split-ajustado)."""
    if init_price <= 0:
        return None
    try:
        d0 = datetime.strptime(alert_date_str[:10], "%Y-%m-%d")
        h = hist_df if hist_df is not None else yf.Ticker(ticker).history(start=d0.strftime("%Y-%m-%d"), auto_adjust=False)
        if h is None or h.empty:
            return None
        col = "Low" if "Low" in h else ("Close" if "Close" in h else None)
        if not col:
            return None
        vals = h[col].dropna()
        if vals.empty:
            return None
        min_p = safe_float(vals.min())
        if min_p is not None and min_p > 0:
            dd = ((min_p / init_price) - 1.0) * 100.0
            return round(min(0.0, dd), 1)
    except Exception:
        return None
    return None


def _max_gain_after_alert_pct(
    ticker: str,
    init_price: float,
    alert_date_str: str,
    hist_df: Optional[pd.DataFrame] = None,
) -> Optional[float]:
    """Calcula la máxima subida porcentual (MFE) alcanzada desde la fecha de alerta respecto al precio de entrada (split-ajustado)."""
    if init_price <= 0:
        return None
    try:
        d0 = datetime.strptime(alert_date_str[:10], "%Y-%m-%d")
        h = hist_df if hist_df is not None else yf.Ticker(ticker).history(start=d0.strftime("%Y-%m-%d"), auto_adjust=False)
        if h is None or h.empty:
            return None
        col = "High" if "High" in h else ("Close" if "Close" in h else None)
        if not col:
            return None
        vals = h[col].dropna()
        if vals.empty:
            return None
        max_p = safe_float(vals.max())
        if max_p is not None and max_p > 0:
            gain = ((max_p / init_price) - 1.0) * 100.0
            return round(max(0.0, gain), 1)
    except Exception:
        return None
    return None


def _cash_dividends_in_window(
    ticker: str,
    start_date_str: str,
    end_date_str: Optional[str] = None,
    div_series: Optional[pd.Series] = None,
) -> float:
    """Suma dividendos en efectivo (sin reinversión) pagados entre start_date_str y end_date_str (inclusive)."""
    try:
        d0 = pd.Timestamp(datetime.strptime(start_date_str[:10], "%Y-%m-%d").date())
        d1 = pd.Timestamp(datetime.strptime(end_date_str[:10], "%Y-%m-%d").date()) if end_date_str else None
        div_s = div_series if div_series is not None else yf.Ticker(ticker).dividends
        if div_s is None or div_s.empty:
            return 0.0
        idx_naive = pd.to_datetime(div_s.index, errors="coerce")
        if getattr(idx_naive, "tz", None) is not None:
            idx_naive = idx_naive.tz_localize(None)
        idx_norm = idx_naive.normalize()
        mask = idx_norm >= d0
        if d1 is not None:
            mask = mask & (idx_norm <= d1)
        vals = [safe_float(x) for x in div_s[mask].values]
        return sum(v for v in vals if v is not None and v > 0)
    except Exception:
        return 0.0


def _cash_dividends_in_window_eur(
    ticker: str,
    currency: str,
    start_date_str: str,
    end_date_str: Optional[str] = None,
    div_series: Optional[pd.Series] = None,
    fx_series: Optional[pd.Series] = None,
    fallback_fx_end: Optional[float] = None,
) -> Optional[float]:
    """Suma dividendos en efectivo convertidos a EUR usando el FX de cada fecha de pago (o fallback_fx_end)."""
    cur = (currency or "EUR").strip().upper()
    if cur == "EUR":
        return _cash_dividends_in_window(ticker, start_date_str, end_date_str, div_series)
    try:
        d0 = pd.Timestamp(datetime.strptime(start_date_str[:10], "%Y-%m-%d").date())
        d1 = pd.Timestamp(datetime.strptime(end_date_str[:10], "%Y-%m-%d").date()) if end_date_str else None
        div_s = div_series if div_series is not None else yf.Ticker(ticker).dividends
        if div_s is None or div_s.empty:
            return 0.0
        idx_naive = pd.to_datetime(div_s.index, errors="coerce")
        if getattr(idx_naive, "tz", None) is not None:
            idx_naive = idx_naive.tz_localize(None)
        idx_norm = idx_naive.normalize()
        mask = idx_norm >= d0
        if d1 is not None:
            mask = mask & (idx_norm <= d1)
        total_eur = 0.0
        for dt_val, raw_div in zip(idx_norm[mask], div_s[mask].values):
            dv = safe_float(raw_div)
            if dv is None or dv <= 0:
                continue
            dt_str = pd.Timestamp(dt_val).strftime("%Y-%m-%d")
            fx_d = _fx_rate_on_or_before(fx_series, dt_str) or fallback_fx_end
            if fx_d is None or fx_d <= 0:
                return None
            total_eur += dv * fx_d
        return total_eur
    except Exception:
        return None


def _total_return_with_dividends_pct(
    ticker: str,
    init_price: float,
    cur_price: float,
    alert_date_str: str,
    div_series: Optional[pd.Series] = None,
) -> float:
    """Calcula rentabilidad % (precio + dividendos en efectivo, sin reinversión) desde alert_date_str."""
    base_ret = ((cur_price / init_price) - 1.0) * 100.0
    divs_sum = _cash_dividends_in_window(ticker, alert_date_str, end_date_str=None, div_series=div_series)
    total_ret = (((cur_price + divs_sum) / init_price) - 1.0) * 100.0 if divs_sum > 0 else base_ret
    return round(total_ret, 1)


def _milestone_return_at_horizon(
    ticker: str,
    init_price: float,
    alert_date_str: str,
    horizon_days: int,
    hist_df: Optional[pd.DataFrame] = None,
    div_series: Optional[pd.Series] = None,
    currency: str = "USD",
    fx_series: Optional[pd.Series] = None,
    fx_alert: Optional[float] = None,
) -> Tuple[Optional[float], Optional[str], Optional[float]]:
    """Calcula (retorno_local_pct, fecha_cierre_exacta, retorno_eur_pct) en la primera sesión >= alert_date + horizon_days."""
    if init_price <= 0:
        return None, None, None
    try:
        d0 = datetime.strptime(alert_date_str[:10], "%Y-%m-%d").date()
        target_d = d0 + timedelta(days=int(horizon_days))
        h = hist_df if hist_df is not None else yf.Ticker(ticker).history(start=d0.strftime("%Y-%m-%d"), auto_adjust=False)
        closes = _extract_datetime_series(h, "Close")
        if closes is None or closes.empty:
            return None, None, None
        target_ts = pd.Timestamp(target_d)
        eligible = closes[closes.index >= target_ts]
        if eligible.empty:
            return None, None, None
        actual_ts = eligible.index[0]
        close_at_h = safe_float(eligible.iloc[0])
        if close_at_h is None or close_at_h <= 0:
            return None, None, None
        actual_date_str = actual_ts.strftime("%Y-%m-%d")
        divs_sum = _cash_dividends_in_window(ticker, alert_date_str, end_date_str=actual_date_str, div_series=div_series)
        ret_local = round((((close_at_h + divs_sum) / init_price) - 1.0) * 100.0, 1)

        cur = (currency or "USD").strip().upper()
        ret_eur: Optional[float] = None
        if cur == "EUR":
            ret_eur = ret_local
        elif fx_alert and fx_alert > 0 and fx_series is not None:
            fx_h = _fx_rate_on_or_before(fx_series, actual_date_str)
            if fx_h and fx_h > 0:
                divs_eur = _cash_dividends_in_window_eur(
                    ticker, cur, alert_date_str, end_date_str=actual_date_str,
                    div_series=div_series, fx_series=fx_series, fallback_fx_end=fx_h
                )
                if divs_eur is not None:
                    ret_eur = round(((((close_at_h * fx_h) + divs_eur) / (init_price * fx_alert)) - 1.0) * 100.0, 1)
        return ret_local, actual_date_str, ret_eur
    except Exception:
        return None, None, None


def update_alert_history(results: List[Dict[str, Any]], now_dt: Optional[datetime] = None) -> None:
    """Registra alertas y actualiza rentabilidad (divisa local + EUR homogeneizada), splits, hitos exactos y benchmarks SPY/URTH."""
    now_dt = now_dt or datetime.now(timezone.utc)
    today = now_dt.strftime("%Y-%m-%d")
    now_iso = now_dt.isoformat()
    history: List[Dict[str, Any]] = []
    if OUT_HISTORY_JSON.exists():
        try:
            parsed = json.loads(OUT_HISTORY_JSON.read_text(encoding="utf-8"))
            if isinstance(parsed, dict) and isinstance(parsed.get("entries"), list):
                history = parsed["entries"]
            elif isinstance(parsed, list):
                history = parsed
        except Exception:
            history = []

    price_map = {r["ticker"]: r for r in results}

    for entry in history:
        tkr = entry.get("ticker")
        init_p = safe_float(entry.get("alert_price"))
        alert_d = str(entry.get("alert_date") or today)[:10]
        cur = (entry.get("currency") or "USD").strip().upper()
        if not tkr or not init_p or init_p <= 0:
            continue

        t_obj = None
        try:
            t_obj = yf.Ticker(tkr)
        except Exception:
            t_obj = None

        rec = price_map.get(tkr)
        cur_p = safe_float(rec.get("price")) if rec else None
        if cur_p is None and t_obj is not None:
            try:
                hist_t = t_obj.history(period="5d")
                if hist_t is not None and not hist_t.empty and "Close" in hist_t:
                    cur_p = safe_float(hist_t["Close"].dropna().iloc[-1])
            except Exception:
                cur_p = None

        if cur_p is not None and cur_p > 0:
            div_s = None
            splits_s = None
            hist_full = None
            if t_obj is not None:
                try:
                    div_s = t_obj.dividends
                except Exception:
                    div_s = None
                try:
                    splits_s = getattr(t_obj, "splits", None)
                except Exception:
                    splits_s = None
                try:
                    hist_full = t_obj.history(start=alert_d, auto_adjust=False)
                except Exception:
                    hist_full = None

            split_factor = _cumulative_split_factor(tkr, alert_d, splits_series=splits_s)
            eff_init_p = init_p / split_factor if split_factor > 0 else init_p

            price_ret_pct = round(((cur_p / eff_init_p) - 1.0) * 100.0, 1)
            tot_ret_pct = _total_return_with_dividends_pct(tkr, eff_init_p, cur_p, alert_d, div_series=div_s)

            entry["last_price"] = round(cur_p, 2)
            entry["split_factor_since_alert"] = round(split_factor, 4)
            entry["adjusted_alert_price"] = round(eff_init_p, 4)
            entry["price_return_pct"] = price_ret_pct
            entry["return_pct"] = tot_ret_pct
            entry["return_methodology"] = "price_plus_cash_dividends_split_adjusted"

            # Homogeneización en EUR con tipo de cambio histórico
            if cur == "EUR":
                fx_s = None
                fx_alert: Optional[float] = 1.0
                fx_now: Optional[float] = 1.0
            else:
                fx_s = _get_fx_history_to_eur(cur, alert_d)
                fx_alert = _fx_rate_on_or_before(fx_s, alert_d)
                fx_now = _fx_rate_on_or_before(fx_s, today)

            if fx_alert and fx_now and fx_alert > 0 and fx_now > 0:
                divs_eur = _cash_dividends_in_window_eur(
                    tkr, cur, alert_d, end_date_str=None,
                    div_series=div_s, fx_series=fx_s, fallback_fx_end=fx_now
                )
                entry["alert_price_eur"] = round(init_p * fx_alert, 2)
                entry["adjusted_alert_price_eur"] = round(eff_init_p * fx_alert, 4)
                entry["last_price_eur"] = round(cur_p * fx_now, 2)
                entry["fx_alert_to_eur"] = round(fx_alert, 6)
                entry["fx_last_to_eur"] = round(fx_now, 6)
                entry["fx_impact_pct"] = round(((fx_now / fx_alert) - 1.0) * 100.0, 1)
                entry["price_return_eur_pct"] = round((((cur_p * fx_now) / (eff_init_p * fx_alert)) - 1.0) * 100.0, 1)
                if divs_eur is not None:
                    entry["return_eur_pct"] = round(((((cur_p * fx_now) + divs_eur) / (eff_init_p * fx_alert)) - 1.0) * 100.0, 1)
                    entry["fx_status"] = "OK"
                else:
                    entry["return_eur_pct"] = None
                    entry["fx_status"] = "UNAVAILABLE"
            else:
                entry["return_eur_pct"] = None
                entry["price_return_eur_pct"] = None
                entry["fx_status"] = "UNAVAILABLE"

            mdd_pct = _max_drawdown_after_alert_pct(tkr, eff_init_p, alert_d, hist_df=hist_full)
            if mdd_pct is not None:
                prev_mdd = safe_float(entry.get("max_drawdown_after_alert_pct"))
                entry["max_drawdown_after_alert_pct"] = min(mdd_pct, prev_mdd) if prev_mdd is not None else mdd_pct
            mfe_pct = _max_gain_after_alert_pct(tkr, eff_init_p, alert_d, hist_df=hist_full)
            if mfe_pct is not None:
                prev_mfe = safe_float(entry.get("max_gain_after_alert_pct"))
                entry["max_gain_after_alert_pct"] = max(mfe_pct, prev_mfe) if prev_mfe is not None else mfe_pct

            entry["last_checked"] = today
            entry["last_checked_at"] = now_iso
            if rec:
                entry["current_status"] = rec.get("opportunity_tier") or rec.get("status")
            entry.setdefault("milestones_pct", {"d30": None, "d90": None, "d180": None, "d365": None})
            entry.setdefault("milestones_eur_pct", {"d30": None, "d90": None, "d180": None, "d365": None})
            entry.setdefault("milestone_dates", {"d30": None, "d90": None, "d180": None, "d365": None})
            entry.setdefault("status", "ACTIVA")

            try:
                d0_date = datetime.strptime(alert_d, "%Y-%m-%d").date()
                elapsed_days = (now_dt.date() - d0_date).days
                needed_horizons: List[Tuple[int, str]] = []
                for horizon, key in ((30, "d30"), (90, "d90"), (180, "d180"), (365, "d365")):
                    if elapsed_days < horizon:
                        continue
                    cur_ms = entry["milestones_pct"].get(key)
                    cur_ms_eur = entry["milestones_eur_pct"].get(key)
                    cur_ms_date = entry["milestone_dates"].get(key)
                    drifted = False
                    if cur_ms is not None:
                        if not cur_ms_date:
                            drifted = True
                        else:
                            try:
                                rec_d = datetime.strptime(str(cur_ms_date)[:10], "%Y-%m-%d").date()
                                target_d = d0_date + timedelta(days=horizon)
                                if (rec_d - target_d).days > 7 or rec_d < target_d:
                                    drifted = True
                            except Exception:
                                drifted = True
                    missing_eur = (cur_ms_eur is None) and (fx_alert is not None)
                    if cur_ms is None or drifted or missing_eur or split_factor != 1.0:
                        needed_horizons.append((horizon, key))

                for horizon, key in needed_horizons:
                    ms_ret, ms_date, ms_ret_eur = _milestone_return_at_horizon(
                        tkr, eff_init_p, alert_d, horizon,
                        hist_df=hist_full, div_series=div_s,
                        currency=cur, fx_series=fx_s, fx_alert=fx_alert,
                    )
                    if ms_ret is not None and ms_date is not None:
                        entry["milestones_pct"][key] = ms_ret
                        entry["milestone_dates"][key] = ms_date
                        if ms_ret_eur is not None:
                            entry["milestones_eur_pct"][key] = ms_ret_eur
            except Exception:
                pass

        # Benchmarks homogéneos: ETFs ajustados con dividendos en USD y homogeneizados a EUR
        sp_pct, sp_eur_pct = _period_return_eur_pct("SPY", "USD", alert_d, today)
        msci_pct, msci_eur_pct = _period_return_eur_pct("URTH", "USD", alert_d, today)
        ret_eur_val = safe_float(entry.get("return_eur_pct"))
        alpha_msci_eur = (
            round(ret_eur_val - msci_eur_pct, 1)
            if (ret_eur_val is not None and msci_eur_pct is not None)
            else None
        )
        entry["benchmarks"] = {
            "sp500_pct": sp_pct,
            "sp500_eur_pct": sp_eur_pct,
            "sp500_symbol": "SPY",
            "msci_world_pct": msci_pct,
            "msci_world_eur_pct": msci_eur_pct,
            "msci_world_symbol": "URTH",
            "alpha_msci_eur_pp": alpha_msci_eur,
            "currency_base": "EUR" if (msci_eur_pct is not None and sp_eur_pct is not None) else "USD",
            "methodology": "total_return_etf_adjusted_eur_homogenized",
            "benchmark_as_of": today,
        }

    for r in results:
        tier = r.get("opportunity_tier")
        is_top = tier in ("ENTRADA_YA", "GANGA") or bool(r.get("oro_puro"))
        if not is_top:
            continue
        recent_dup = False
        for e in history:
            if e.get("ticker") == r["ticker"]:
                try:
                    d0 = datetime.strptime(str(e.get("alert_date", "2000-01-01"))[:10], "%Y-%m-%d")
                    d1 = datetime.strptime(today, "%Y-%m-%d")
                    if (d1 - d0).days < 30:
                        recent_dup = True
                        break
                except Exception:
                    pass
        if not recent_dup:
            cur_r = (r.get("currency") or "USD").strip().upper()
            p_eur_init = to_eur(r["price"], cur_r)
            history.insert(0, {
                "ticker": r["ticker"],
                "name": r["name"],
                "currency": cur_r,
                "tier": "ORO_PURO" if r.get("oro_puro") else tier,
                "alert_date": today,
                "alert_timestamp": now_iso,
                "alert_price": r["price"],
                "adjusted_alert_price": r["price"],
                "split_factor_since_alert": 1.0,
                "alert_price_eur": round(p_eur_init, 2) if p_eur_init is not None else None,
                "adjusted_alert_price_eur": round(p_eur_init, 4) if p_eur_init is not None else None,
                "quality_score": r["quality_score"],
                "ai_verdict": r.get("ai_verdict"),
                "fundamentals_snapshot": {
                    "piotroski_classic_score": r.get("piotroski_classic_score"),
                    "piotroski_classic_max": r.get("piotroski_classic_max"),
                    "piotroski_adapted_score": r.get("piotroski_score"),
                    "piotroski_adapted_max": r.get("piotroski_max"),
                    "sec_status": r.get("sec_status"),
                    "net_debt_ebitda": r.get("net_debt_ebitda"),
                    "fcf_latest_date": r.get("fcf_latest_date"),
                    "quadrant": (r.get("quality_valuation") or {}).get("quadrant_code"),
                },
                "return_methodology": "price_plus_cash_dividends_split_adjusted",
                "milestones_pct": {"d30": None, "d90": None, "d180": None, "d365": None},
                "milestones_eur_pct": {"d30": None, "d90": None, "d180": None, "d365": None},
                "milestone_dates": {"d30": None, "d90": None, "d180": None, "d365": None},
                "benchmarks": {
                    "msci_world_pct": None,
                    "msci_world_eur_pct": None,
                    "msci_world_symbol": "URTH",
                    "sp500_pct": None,
                    "sp500_eur_pct": None,
                    "sp500_symbol": "SPY",
                    "alpha_msci_eur_pp": None,
                    "currency_base": "EUR",
                    "methodology": "total_return_etf_adjusted_eur_homogenized",
                },
                "status": "ACTIVA",
                "last_price": r["price"],
                "last_price_eur": round(p_eur_init, 2) if p_eur_init is not None else None,
                "price_return_pct": 0.0,
                "return_pct": 0.0,
                "price_return_eur_pct": 0.0 if p_eur_init is not None else None,
                "return_eur_pct": 0.0 if p_eur_init is not None else None,
                "fx_status": "OK" if p_eur_init is not None else "UNAVAILABLE",
                "max_drawdown_after_alert_pct": 0.0,
                "max_gain_after_alert_pct": 0.0,
                "last_checked": today,
                "current_status": tier,
            })

    history = history[:150]
    payload = {
        "updated_at": now_iso,
        "entries": history,
    }
    OUT_HISTORY_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_scan_batch(tickers: List[str]) -> Tuple[List[Dict[str, Any]], int]:
    t0 = time.time()
    if len(tickers) > 5:
        _warmup_yf_session()
    results: List[Dict[str, Any]] = []
    failed = 0
    done = 0

    def worker(sym: str) -> Optional[Dict[str, Any]]:
        time.sleep(random.uniform(0.35, 0.85))
        return analyze_ticker_with_retry(sym, retries=3)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {pool.submit(worker, sym): sym for sym in tickers}
        for fut in as_completed(future_map):
            done += 1
            rec = fut.result()
            if rec:
                results.append(rec)
            else:
                failed += 1
            if done % 40 == 0 or done == len(tickers):
                elapsed = time.time() - t0
                with _print_lock:
                    print(f"  Progreso: {done}/{len(tickers)} | OK: {len(results)} | Omitidas: {failed} | {elapsed:.0f}s")
    return results, failed


def load_merged_shards(merge_dir: Path) -> Tuple[List[Dict[str, Any]], int, int, int]:
    """Fusiona shards exigiendo integridad completa (rechaza publicaciones parciales salvo override)."""
    shard_files = sorted(merge_dir.rglob("shard-*.json"))
    if not shard_files:
        raise RuntimeError(f"No se encontraron archivos shard-*.json en {merge_dir}")

    by_ticker: Dict[str, Dict[str, Any]] = {}
    total_scanned = 0
    total_failed = 0
    max_elapsed = 0
    seen_indices = set()
    expected_total: Optional[int] = SHARD_TOTAL if SHARD_TOTAL > 1 else None
    seen_modes = set()
    corrupt_files: List[str] = []

    for sf in shard_files:
        try:
            data = json.loads(sf.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("companies"), list):
                raise ValueError("Estructura de shard inválida")
        except Exception as exc:
            corrupt_files.append(f"{sf.name}: {exc}")
            continue

        s_idx = data.get("shard_index")
        s_tot = data.get("shard_total")
        s_mode = data.get("universe_mode")
        if s_tot and int(s_tot) > 1:
            if expected_total is None:
                expected_total = int(s_tot)
            elif expected_total != int(s_tot):
                corrupt_files.append(f"{sf.name}: shard_total inconsistente ({s_tot} != {expected_total})")
                continue
        if s_idx is not None:
            seen_indices.add(int(s_idx))
        if s_mode:
            seen_modes.add(str(s_mode))

        total_scanned += int(data.get("total_scanned") or 0)
        total_failed += int(data.get("total_failed") or 0)
        max_elapsed = max(max_elapsed, int(data.get("elapsed_sec") or 0))
        for rec in data.get("companies") or []:
            tkr = rec.get("ticker")
            if tkr and tkr not in by_ticker:
                by_ticker[tkr] = rec

    if corrupt_files and not ALLOW_PARTIAL_MERGE:
        raise RuntimeError(f"Fusión abortada por shards corruptos: {'; '.join(corrupt_files)}")
    if len(seen_modes) > 1 and not ALLOW_PARTIAL_MERGE:
        raise RuntimeError(f"Fusión abortada: universos mezclados en shards ({seen_modes})")
    if expected_total is not None:
        missing = sorted(set(range(expected_total)) - seen_indices)
        if missing and not ALLOW_PARTIAL_MERGE:
            raise RuntimeError(
                f"Fusión abortada: faltan shards {missing} (recibidos {len(seen_indices)}/{expected_total}). "
                "No se sobrescribe weekly-scan.json con un escaneo incompleto."
            )

    print(f"🧩 Fusionados {len(seen_indices)}/{expected_total or len(shard_files)} bloques íntegros: {len(by_ticker)} empresas válidas (de {total_scanned} escaneadas).")
    return list(by_ticker.values()), total_scanned, total_failed, max_elapsed


def build_fx_rates_to_eur(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Mapa de tipos de cambio hacia EUR para que la web calcule Mi Cartera sin mezclar divisas."""
    currencies = {"EUR", "USD", "GBP", "CHF", "CAD", "JPY"}
    for r in results:
        c = (r.get("currency") or "").upper()
        if c:
            currencies.add(c)
    rates: Dict[str, float] = {"EUR": 1.0}
    for cur in sorted(currencies):
        if cur == "EUR":
            continue
        fx = get_fx_rate(cur, "EUR")
        if fx and fx > 0:
            rates[cur] = round(fx, 6)
    return rates


def trim_for_mobile_web(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(results) <= 600:
        return results
    keep_good = [r for r in results if r["status"] != "DESCARTADA"]
    discarded = [r for r in results if r["status"] == "DESCARTADA"]
    discarded.sort(key=lambda r: -(r.get("quality_score") or 0))
    return keep_good + discarded[:MAX_DISCARDED_IN_WEB_JSON]


def main() -> None:
    t0 = time.time()

    if MERGE_DIR:
        print(f"🧩 Cocodrilo Pro v11.3 — Fusionando bloques de {MERGE_DIR} (modo {UNIVERSE_MODE})")
        results, total_scanned, failed, shard_elapsed = load_merged_shards(Path(MERGE_DIR))
        elapsed_total = max(shard_elapsed, round(time.time() - t0))
    else:
        if UNIVERSE_MODE == "custom":
            all_tickers = resolve_custom_inputs(CUSTOM_TICKERS)
            if not all_tickers:
                raise RuntimeError("No se pudo resolver ningún ticker válido desde CUSTOM_TICKERS.")
        else:
            all_tickers = get_universe(UNIVERSE_MODE, CUSTOM_TICKERS, rotation_batch_size=ROTATION_BATCH_SIZE)

        if SHARD_TOTAL > 1:
            tickers = [sym for idx, sym in enumerate(all_tickers) if idx % SHARD_TOTAL == SHARD_INDEX]
            print(
                f"🐊 Cocodrilo Pro v11.3 | Modo: {UNIVERSE_MODE} | Shard {SHARD_INDEX + 1}/{SHARD_TOTAL} "
                f"| Empresas: {len(tickers)} (total universo: {len(all_tickers)})"
            )
            results, failed = run_scan_batch(tickers)
            shard_out_dir = DATA_DIR / "shards"
            shard_out_dir.mkdir(parents=True, exist_ok=True)
            shard_file = shard_out_dir / f"shard-{SHARD_INDEX}.json"
            shard_payload = {
                "shard_index": SHARD_INDEX,
                "shard_total": SHARD_TOTAL,
                "universe_mode": UNIVERSE_MODE,
                "total_scanned": len(tickers),
                "total_ok": len(results),
                "total_failed": failed,
                "elapsed_sec": round(time.time() - t0),
                "companies": results,
            }
            shard_file.write_text(json.dumps(shard_payload, ensure_ascii=False), encoding="utf-8")
            print(f"💾 Shard guardado en {shard_file} ({len(results)} OK en {shard_payload['elapsed_sec']}s)")
            if len(results) == 0:
                raise RuntimeError(f"Shard {SHARD_INDEX} devolvió 0 empresas válidas.")
            return

        print(f"🐊 Escáner Cocodrilo Pro v11.3 | Modo: {UNIVERSE_MODE} | Empresas: {len(all_tickers)} | Hilos: {MAX_WORKERS}")
        total_scanned = len(all_tickers)
        results, failed = run_scan_batch(all_tickers)
        elapsed_total = round(time.time() - t0)

    if len(results) == 0:
        raise RuntimeError("0 empresas válidas analizadas. Abortando para no sobrescribir datos buenos en el panel.")

    market_ctx = get_market_context() if UNIVERSE_MODE != "custom" else None

    status_order = {"OPORTUNIDAD": 0, "VIGILAR": 1, "CALIDAD_CARA": 2, "DESCARTADA": 3}
    for r in results:
        r["ai_verdict"] = None
        r["conviction_score"] = compute_conviction_score(r)
    results.sort(key=lambda r: (status_order.get(r["status"], 9), -(r["conviction_score"] or 0)))

    # Compras de directivos (Insider Buying) + Auditoría oficial SEC EDGAR en candidatas top
    insider_targets = results[:12] if UNIVERSE_MODE == "custom" else [r for r in results if r["status"] == "OPORTUNIDAD"][:15]
    for r in insider_targets:
        buys, ins_status = check_insider_buying(r["ticker"])
        r["insider_buys_90d"] = buys
        r["insider_status"] = ins_status
        if buys and buys > 0:
            r["strengths"] = ([f"Directivos comprando acciones ({buys} op. 90d)"] + (r.get("strengths") or []))[:4]
        sec_info = verify_with_sec_edgar(r)
        r.update(sec_info)
        if sec_info.get("sec_verified") is False:
            r["data_warnings"] = (r.get("data_warnings") or []) + [sec_info["sec_note"]]

    ai_status: Dict[str, Any] = {
        "enabled": bool(GEMINI_API_KEY),
        "attempted": 0,
        "ok": 0,
        "errors": 0,
        "last_error": None,
    }

    if GEMINI_API_KEY:
        if UNIVERSE_MODE == "custom":
            targets = results[:8]
        else:
            max_ai = 8 if (UNIVERSE_MODE in ("global_mega", "global_1000") or URGENT_ONLY) else 6
            targets = [r for r in results if r["status"] == "OPORTUNIDAD"][:max_ai]
            if not targets and not URGENT_ONLY:
                targets = [r for r in results if r["status"] == "VIGILAR"][:2]

        for idx_t, r in enumerate(targets):
            ai_status["attempted"] += 1
            news_list, news_st = extract_recent_news(r["ticker"], limit=5)
            r["recent_news"] = news_list
            r["news_status"] = news_st
            use_tribunal = (idx_t < 6) or (UNIVERSE_MODE == "custom")
            if use_tribunal:
                print(f"⚖️ Tribunal IA JSON (Fiscal vs Abogado) para {r['ticker']}...")
                ai_text, ai_verdict, trap_risk, ai_err = run_tribunal_ia(r, GEMINI_API_KEY, GEMINI_MODELS)
                r["trap_risk_score"] = trap_risk
            else:
                print(f"🤖 Consultando Gemini JSON + titulares recientes para {r['ticker']}...")
                ai_text, ai_verdict, ai_err = optional_gemini_buffett_check(r)
            if ai_text and ai_verdict:
                r["ai_check"] = ai_text
                r["ai_verdict"] = ai_verdict
                ai_status["ok"] += 1
            else:
                ai_status["errors"] += 1
                ai_status["last_error"] = ai_err
            time.sleep(2.5)

    tier_order = {"ENTRADA_YA": 0, "GANGA": 1, "MUY_BUENA": 2, "BUENA": 3, None: 9}
    for r in results:
        r["conviction_score"] = compute_conviction_score(r)
        tier = classify_opportunity_tier(r)
        sec_fail_us = (r.get("region") == "EE. UU." and r.get("sec_verified") is not True)
        if tier == "ENTRADA_YA" and (r.get("sec_verified") is False or sec_fail_us or r.get("is_cyclical_risk") or r.get("forward_ok") is not True):
            tier = "MUY_BUENA"
        r["opportunity_tier"] = tier
        r["oro_puro"] = is_oro_puro(r)
        r["quality_valuation"] = classify_quality_valuation_quadrant(r)

    results.sort(
        key=lambda r: (
            status_order.get(r["status"], 9),
            tier_order.get(r["opportunity_tier"], 9),
            -(r["conviction_score"] or 0),
        )
    )

    entradas_ya = [r for r in results if r.get("opportunity_tier") == "ENTRADA_YA"]
    opps_all = [r for r in results if r["status"] == "OPORTUNIDAD"]
    gap_pts = None
    if len(entradas_ya) >= 1 and len(opps_all) >= 2:
        gap_pts = round((opps_all[0].get("conviction_score") or 0) - (opps_all[1].get("conviction_score") or 0), 1)

    if URGENT_ONLY:
        cooldown_map = load_alert_cooldown()
        new_entradas, suppressed_tkrs = filter_new_urgent_alerts(entradas_ya, cooldown_map)
        if new_entradas:
            urgent_msg = build_urgent_telegram_alert(new_entradas, gap_pts)
            send_telegram(urgent_msg)
            send_fcm_urgent_alerts(new_entradas)
            save_alert_cooldown(new_entradas, cooldown_map)
            print(f"🚨 ALERTA MÁXIMA enviada ({len(new_entradas)} nuevas ENTRAR YA).")
            if URGENT_DOUBLE_PING:
                time.sleep(25)
                tickers_str = ", ".join(r["ticker"] for r in new_entradas[:3])
                send_telegram(f"🔔🚨 <b>RECORDATORIO URGENTE COCODRILO:</b> revisa <b>{esc_html(tickers_str)}</b> ahora mismo (zona ENTRAR YA).")
        elif TEST_MODE:
            sil_txt = f" (silenciadas por cooldown: {', '.join(suppressed_tkrs)})" if suppressed_tkrs else ""
            send_telegram(
                f"🛡️ <b>Vigilante Intradía OK</b> ({len(results)} analizadas): sin novedades en 🚨 ENTRAR YA{esc_html(sil_txt)}."
            )
            send_fcm_push(
                title="🛡️ Cocodrilo Intradía OK (Prueba)",
                body=f"{len(results)} revisadas: sin novedades en ENTRAR YA{sil_txt}.",
                ticker=None,
                tier="TEST",
            )
            print(f"✅ Chequeo intradía OK (sin novedades{sil_txt}).")
        else:
            print(f"😴 Intradía sin novedades ENTRAR YA (silenciadas: {suppressed_tkrs or 'ninguna'}). Silencio total.")
        return

    counts = {
        "entrada_ya": len(entradas_ya),
        "gangas": sum(1 for r in results if r["opportunity_tier"] == "GANGA"),
        "muy_buenas": sum(1 for r in results if r["opportunity_tier"] == "MUY_BUENA"),
        "buenas": sum(1 for r in results if r["opportunity_tier"] == "BUENA"),
        "oportunidades": len(opps_all),
        "vigilar": sum(1 for r in results if r["status"] == "VIGILAR"),
        "calidad_cara": sum(1 for r in results if r["status"] == "CALIDAD_CARA"),
        "descartadas": sum(1 for r in results if r["status"] == "DESCARTADA"),
    }

    web_companies = trim_for_mobile_web(results)

    fx_rates_eur = build_fx_rates_to_eur(results)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "scanner_version": SCANNER_VERSION,
        "fx_rates_eur": fx_rates_eur,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe_mode": UNIVERSE_MODE,
        "total_scanned": total_scanned,
        "total_ok": len(results),
        "web_stored": len(web_companies),
        "total_failed": failed,
        "elapsed_sec": elapsed_total,
        "market_context": market_ctx,
        "ai_status": ai_status,
        "counts": counts,
        "companies": web_companies,
    }
    json_payload = json.dumps(summary, indent=2, ensure_ascii=False)

    if UNIVERSE_MODE == "custom":
        OUT_CUSTOM_JSON.write_text(json_payload, encoding="utf-8")
        print(f"💾 Análisis exprés guardado en {OUT_CUSTOM_JSON.name} (sin tocar daily/weekly).")
    else:
        OUT_LATEST_JSON.write_text(json_payload, encoding="utf-8")
        is_weekly = UNIVERSE_MODE in ("global_1000", "global_mega")
        target_split = OUT_WEEKLY_JSON if is_weekly else OUT_DAILY_JSON
        target_split.write_text(json_payload, encoding="utf-8")
        update_alert_history(results)
        print(
            f"💾 Guardado en {OUT_LATEST_JSON.name}, {target_split.name} y history.json "
            f"({len(results)} analizadas, {len(web_companies)} en web, {elapsed_total}s)"
        )

    if entradas_ya and UNIVERSE_MODE != "custom":
        cooldown_map = load_alert_cooldown()
        new_entradas, _ = filter_new_urgent_alerts(entradas_ya, cooldown_map)
        if new_entradas:
            send_telegram(build_urgent_telegram_alert(new_entradas, gap_pts))
            send_fcm_urgent_alerts(new_entradas)
            save_alert_cooldown(new_entradas, cooldown_map)
            time.sleep(2)

    msg = build_telegram_report(results, failed, ai_status, total_scanned, market_ctx)
    if msg:
        send_telegram(msg)
        print("✅ Alerta enviada a Telegram.")
    else:
        print("😴 Sin oportunidades hoy y TEST_MODE=false.")


if __name__ == "__main__":
    main()
