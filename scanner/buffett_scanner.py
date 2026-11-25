"""Radar Cocodrilo Pro v9 — Escáner Global Experto (Warren Buffett & Charlie Munger).

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from universe import detect_region, get_universe, parse_custom_tickers

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_LATEST_JSON = DATA_DIR / "latest-scan.json"
OUT_DAILY_JSON = DATA_DIR / "daily-scan.json"
OUT_WEEKLY_JSON = DATA_DIR / "weekly-scan.json"
OUT_CUSTOM_JSON = DATA_DIR / "custom-scan.json"
OUT_HISTORY_JSON = DATA_DIR / "history.json"

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
MAX_DISCARDED_IN_WEB_JSON = int(os.environ.get("MAX_DISCARDED_IN_WEB_JSON", "250"))
URGENT_ONLY = os.environ.get("URGENT_ONLY", "false").strip().lower() == "true"
URGENT_DOUBLE_PING = os.environ.get("URGENT_DOUBLE_PING", "true").strip().lower() == "true"

TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODELS = ("gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite")
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"

SUBUNIT_CURRENCIES: Dict[str, Tuple[str, float]] = {
    "GBp": ("GBP", 100.0),
    "GBX": ("GBP", 100.0),
    "ZAc": ("ZAR", 100.0),
    "ILA": ("ILS", 100.0),
}

_print_lock = threading.Lock()
_fx_lock = threading.Lock()
_fx_cache: Dict[Tuple[str, str], Optional[float]] = {}


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
        for idx_col, v in fcf_series.items():
            fv = safe_float(v)
            if fv is None:
                continue
            raw_vals.append(fv)
            sbc_col = 0.0
            if sbc_series is not None and idx_col in sbc_series.index:
                sv = safe_float(sbc_series.get(idx_col))
                if sv is not None and sv > 0:
                    sbc_col = sv
            adj_vals.append(fv - sbc_col)

        if raw_vals:
            out["fcf_years_total"] = len(raw_vals)
            out["fcf_years_pos"] = sum(1 for v in raw_vals if v > 0)

            use_vals = raw_vals
            if sbc_latest and raw_vals[0] > 0:
                sbc_ratio = (sbc_latest / raw_vals[0]) * 100.0
                out["sbc_pct_of_fcf"] = round(sbc_ratio, 1)
                if sbc_ratio >= 5.0:
                    use_vals = adj_vals
                    out["owner_earnings_adjusted"] = True

            out["fcf_latest"] = use_vals[0]
            pos_vals = [v for v in use_vals[:4] if v > 0]
            if len(pos_vals) >= 2 and use_vals[0] > 0:
                med_fcf = statistics.median(pos_vals)
                cap_fcf = med_fcf * 1.15
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


def analyze_ticker_with_retry(ticker: str, retries: int = 3) -> Optional[Dict[str, Any]]:
    for attempt in range(retries + 1):
        try:
            res = analyze_ticker(ticker)
            if res is not None:
                return res
            return None
        except Exception as exc:
            msg = str(exc).lower()
            transient = any(
                k in msg
                for k in ("429", "too many", "timeout", "timed out", "rate", "502", "503", "504", "connection", "json")
            )
            if attempt < retries and transient:
                time.sleep(2.5 * (attempt + 1) + random.uniform(0.4, 1.5))
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
        "sbc_pct_of_fcf": stm["sbc_pct_of_fcf"],
        "owner_earnings_adjusted": stm["owner_earnings_adjusted"],
        "net_debt_ebitda": round(stm["net_debt_ebitda"], 2) if stm["net_debt_ebitda"] is not None else None,
        "ebitda_negative": stm["ebitda_negative"],
        "share_change_pct": round(stm["share_change_pct"], 1) if stm["share_change_pct"] is not None else None,
        "dividend_yield_pct": round(div_y, 2) if div_y is not None else None,
        "earnings_date": earnings_date,
        "earnings_days": earnings_days,
        "insider_buys_90d": 0,
        "dcf": dcf,
        "data_warnings": data_warnings,
        "recent_news": [],
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
    return record
