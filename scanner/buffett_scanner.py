"""Radar Cocodrilo Pro v7 Blindado — Escáner Automático (Warren Buffett & Charlie Munger).

Mejoras de blindaje v7:
- Normalización de subunidades (GBp/GBX/ZAc/ILA) y conversión FX automática (financialCurrency -> currency -> USD).
- Ordenación cronológica explícita de estados financieros y FCF normalizado (anti-picos puntuales).
- Uso de acciones implícitas (MarketCap / Precio) para evitar errores de ratio en ADRs.
- Protección en sector financiero/REIT (sin DCF estándar por FCF y tope en estado VIGILAR).
- Alertas de cordura (data_warnings) para ROIC o descuentos DCF extremos.
- Gemini con titulares reales recientes de Yahoo Finance, reintentos, modelo de respaldo y estado visible en Telegram.
- Doble salida JSON (daily-scan.json / weekly-scan.json + latest-scan.json) y fallo explícito si 0 empresas válidas.
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
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from universe import detect_region, get_universe

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_LATEST_JSON = DATA_DIR / "latest-scan.json"
OUT_DAILY_JSON = DATA_DIR / "daily-scan.json"
OUT_WEEKLY_JSON = DATA_DIR / "weekly-scan.json"

UNIVERSE_MODE = os.environ.get("UNIVERSE_MODE", "elite200").strip().lower()
CUSTOM_TICKERS = os.environ.get("CUSTOM_TICKERS", "").strip()
TEST_MODE = os.environ.get("TEST_MODE", "false").strip().lower() == "true"
MIN_QUALITY_SCORE = int(os.environ.get("MIN_QUALITY_SCORE", "75"))
MARGIN_OF_SAFETY_PCT = float(os.environ.get("MARGIN_OF_SAFETY_PCT", "25"))
MIN_DRAWDOWN_ALERT = float(os.environ.get("MIN_DRAWDOWN_ALERT", "18"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "3"))
MIN_MARKET_CAP_USD = float(os.environ.get("MIN_MARKET_CAP_USD", "1000000000"))  # >= 1B USD

TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODELS = ("gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite")


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
    """Devuelve (código ISO en mayúsculas, divisor de precio si cotiza en subunidad)."""
    raw = (raw_currency or "USD").strip()
    if raw in SUBUNIT_CURRENCIES:
        return SUBUNIT_CURRENCIES[raw]
    return raw.upper(), 1.0


def get_fx_rate(src_cur: str, dst_cur: str) -> Optional[float]:
    """Obtiene el tipo de cambio src -> dst con caché segura entre hilos y pivote USD."""
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


def get_row_series(df: Optional[pd.DataFrame], candidates: List[str]) -> Optional[pd.Series]:
    """Extrae una fila financiera y la ordena explícitamente de más reciente a más antigua."""
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


def compute_roic_and_statements(t: yf.Ticker, info: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "roic_pct": None,
        "fcf_years_pos": None,
        "fcf_years_total": None,
        "fcf_latest": None,
        "fcf_normalized": None,
        "fcf_spike_capped": False,
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

    # 1) Free Cash Flow (ordenado del más reciente al más antiguo)
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

    if fcf_series is not None and len(fcf_series) >= 1:
        vals = [safe_float(v) for v in fcf_series.values]
        vals = [v for v in vals if v is not None]
        if vals:
            out["fcf_years_total"] = len(vals)
            out["fcf_years_pos"] = sum(1 for v in vals if v > 0)
            out["fcf_latest"] = vals[0]
            pos_vals = [v for v in vals[:4] if v > 0]
            if len(pos_vals) >= 2 and vals[0] > 0:
                med_fcf = statistics.median(pos_vals)
                cap_fcf = med_fcf * 1.15
                if vals[0] > cap_fcf:
                    out["fcf_normalized"] = cap_fcf
                    out["fcf_spike_capped"] = True
                else:
                    out["fcf_normalized"] = vals[0]
            else:
                out["fcf_normalized"] = vals[0]

            if len(vals) >= 3 and vals[0] > 0 and vals[-1] > 0:
                out["fcf_cagr_pct"] = ((vals[0] / vals[-1]) ** (1.0 / (len(vals) - 1)) - 1.0) * 100.0

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

    # 4) ROIC real = NOPAT / Capital Invertido (con mediana para evitar años atípicos)
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

    # 5) Caja Neta y Deuda Neta / EBITDA (en divisa financiera)
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

    # 1) ROIC / ROE (25 pts)
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

    # 2) Margen Operativo adaptado al sector (15 pts)
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

    # 3) Consistencia FCF (20 pts — omitido en bancos/REITs donde FCF no es estándar)
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

    # 4) Solidez de Balance — Deuda Neta / EBITDA (20 pts — omitido en financieros)
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

    # 5) Crecimiento compuesto (10 pts)
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

    # 6) Trato al accionista — Recompras / Dividendo (10 pts)
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
            # Si devolvió None sin excepción, no reintentar indefinidamente
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
        if mkt_cap_usd < MIN_MARKET_CAP_USD:
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

    # Acciones implícitas (mkt_cap / price) evitan descuadres en ADRs (ej. 1 ADR = 5 acciones ordinarias)
    shares = (mkt_cap / price) if (mkt_cap and price > 0) else safe_float(info.get("sharesOutstanding"))
    if shares is None or shares <= 0:
        shares = safe_float(info.get("sharesOutstanding"))

    opm = safe_float(info.get("operatingMargins"))
    rev_g = safe_float(info.get("revenueGrowth"))
    earn_g = safe_float(info.get("earningsGrowth"))

    # Rentabilidad por dividendo sin ambigüedad de escala
    div_y = safe_float(info.get("trailingAnnualDividendYield"))
    if div_y is not None:
        div_y = div_y * 100.0
    else:
        raw_dy = safe_float(info.get("dividendYield"))
        if raw_dy is not None:
            div_y = raw_dy * 100.0 if raw_dy < 0.20 else raw_dy

    stm = compute_roic_and_statements(t, info)
    data_warnings: List[str] = []

    # Conversión de FCF y Caja Neta desde divisa financiera a divisa de cotización
    fx_fin_to_trade = get_fx_rate(fin_currency, currency)
    fx_ok = fx_fin_to_trade is not None and fx_fin_to_trade > 0

    fcf_latest_conv = (stm["fcf_latest"] * fx_fin_to_trade) if (fx_ok and stm["fcf_latest"] is not None) else None
    fcf_norm_conv = (stm["fcf_normalized"] * fx_fin_to_trade) if (fx_ok and stm["fcf_normalized"] is not None) else None
    net_cash_conv = (stm["net_cash"] * fx_fin_to_trade) if (fx_ok and stm["net_cash"] is not None) else None

    if not fx_ok and fin_currency != currency:
        data_warnings.append(f"Sin tipo de cambio {fin_currency}->{currency}: DCF y P/FCF omitidos")

    if stm["fcf_spike_capped"]:
        data_warnings.append("FCF del último año anormalmente alto; DCF normalizado con mediana histórica")

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
        "net_debt_ebitda": round(stm["net_debt_ebitda"], 2) if stm["net_debt_ebitda"] is not None else None,
        "ebitda_negative": stm["ebitda_negative"],
        "share_change_pct": round(stm["share_change_pct"], 1) if stm["share_change_pct"] is not None else None,
        "dividend_yield_pct": round(div_y, 2) if div_y is not None else None,
        "dcf": dcf,
        "data_warnings": data_warnings,
        "recent_news": [],
    }

    q = score_company(record)
    record.update(q)

    # Descuento vs Valor Intrínseco Base (positivo = barata)
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

    # Eliminar campo auxiliar crudo para mantener JSON ligero
    record.pop("fcf_latest", None)
    record["discount_vs_base_pct"] = discount_pct
    record["upside_base_pct"] = upside_base
    record["in_dcf_zone"] = in_dcf_zone
    record["below_bear"] = below_bear
    record["status"] = status
    record["ai_check"] = None
    return record


def extract_recent_news(ticker: str, limit: int = 5) -> List[str]:
    """Descarga los últimos titulares reales de Yahoo Finance solo para candidatas a IA."""
    headlines: List[str] = []
    try:
        raw_news = yf.Ticker(ticker).news or []
    except Exception:
        return headlines
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
    return headlines


def clean_ai_text(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:800]


def optional_gemini_buffett_check(rec: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Consulta a Gemini inyectando métricas + titulares recientes. Devuelve (texto, error)."""
    if not GEMINI_API_KEY:
        return None, "GEMINI_API_KEY no configurada"

    news = extract_recent_news(rec["ticker"], limit=5)
    rec["recent_news"] = news
    news_block = "\n".join(f"- {h}" for h in news) if news else "- Sin titulares recientes disponibles en Yahoo Finance."
    warnings_block = "; ".join(rec.get("data_warnings") or []) or "Ninguna"
    dcf = rec.get("dcf") or {}
    dcf_info = f"{dcf.get('intrinsic_base')} {rec['currency']} (obj -25%: {dcf.get('target_price')})" if dcf else "N/A"

    prompt = (
        f"Eres un analista estricto estilo Warren Buffett y Charlie Munger. Sin rodeos ni texto de relleno.\n"
        f"Empresa: {rec['name']} ({rec['ticker']}), Sector: {rec['sector']} / {rec['industry']}.\n"
        f"Datos: Precio {rec['price']} {rec['currency']} (caída 52s: {rec['drawdown_52w_pct']}%), "
        f"Calidad: {rec['quality_score']}/100, ROIC: {rec['roic_pct']}%, Margen op: {rec['operating_margin_pct']}%, "
        f"Deuda/EBITDA: {rec['net_debt_ebitda']}x, P/FCF: {rec['p_fcf']}x, PER futuro: {rec['forward_pe']}x, DCF base: {dcf_info}.\n"
        f"Alertas de datos: {warnings_block}.\n"
        f"Titulares recientes:\n{news_block}\n\n"
        "Responde en español directo en máximo 4 líneas cortas (sin usar asteriscos ni negritas):\n"
        "1) Por qué cae: resume el motivo según los titulares (o indica si no hay catalizador claro en noticias).\n"
        "2) Foso competitivo: ¿sigue intacto o se erosiona?\n"
        "3) Mayor riesgo real para no comprarla.\n"
        "4) Veredicto Cocodrilo: TEMPORAL (oportunidad), DUDOSO (esperar resultados) o PERMANENTE (posible trampa de valor)."
    )

    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 320},
    }).encode("utf-8")

    last_err = None
    for model in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
        for attempt in range(2):
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=25) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                parts = body["candidates"][0]["content"]["parts"]
                text = "".join(p.get("text", "") for p in parts)
                return clean_ai_text(text), None
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
    return None, last_err


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


def build_telegram_report(
    results: List[Dict[str, Any]],
    failed_count: int,
    ai_status: Dict[str, Any],
) -> Optional[str]:
    opps = [r for r in results if r["status"] == "OPORTUNIDAD"]
    watch = [r for r in results if r["status"] == "VIGILAR"]

    opps.sort(key=lambda r: (-(r["upside_base_pct"] or -999), -r["quality_score"]))
    watch.sort(key=lambda r: (r["drawdown_52w_pct"] if r["drawdown_52w_pct"] is not None else 0, -r["quality_score"]))

    if not opps and not watch and not TEST_MODE:
        return None

    date_str = datetime.now(timezone.utc).strftime("%d/%m/%Y")
    lines = [
        f"🐊 <b>RADAR COCODRILO PRO v7 — {date_str}</b>",
        f"<i>Universo: {esc_html(UNIVERSE_MODE.upper())} · Analizadas: {len(results)} · 🟢 Oportunidades: {len(opps)} · 🟡 Vigilancia: {len(watch)}</i>",
    ]

    if opps:
        lines.append("\n<b>🟢 TOP OPORTUNIDADES (Calidad Alta + Precio Castigado)</b>")
        for r in opps[:10]:
            dcf = r.get("dcf") or {}
            cur = r["currency"]
            dd = f"{r['drawdown_52w_pct']}%" if r["drawdown_52w_pct"] is not None else "—"
            pfcf = f"{r['p_fcf']}x" if r["p_fcf"] is not None else "—"
            fpe = f"{r['forward_pe']}x" if r["forward_pe"] is not None else "—"
            nde = f"{r['net_debt_ebitda']}x" if r["net_debt_ebitda"] is not None else "—"
            block = [
                f"\n🎯 <b>{esc_html(r['name'])} ({esc_html(r['ticker'])})</b> · <i>{esc_html(r['region'])}</i>",
                f"• Precio: <b>{r['price']} {cur}</b> (desde máx. 52s: <b>{dd}</b>)",
                f"• Calidad Buffett: <b>{r['quality_score']}/100</b> | ROIC: {r['roic_pct']}% | Deuda/EBITDA: {nde}",
                f"• Múltiplos: P/FCF <b>{pfcf}</b> | PER Futuro: <b>{fpe}</b>",
            ]
            if dcf:
                tag = " 🔥 (bajo escenario conservador)" if r["below_bear"] else ""
                block.append(
                    f"• Valor DCF Base: <b>{dcf['intrinsic_base']} {cur}</b> | Objetivo (-{int(MARGIN_OF_SAFETY_PCT)}%): <b>{dcf['target_price']} {cur}</b>{tag}"
                )
            if r["strengths"]:
                block.append(f"• Puntos fuertes: {esc_html(' · '.join(r['strengths'][:3]))}")
            if r.get("data_warnings"):
                block.append(f"• ⚠️ <i>Aviso datos:</i> {esc_html(' | '.join(r['data_warnings']))}")
            if r.get("ai_check"):
                block.append(f"• 🤖 <i>Filtro IA + Noticias:</i>\n{esc_html(r['ai_check'])}")
            lines.append("\n".join(block))
        if len(opps) > 10:
            lines.append(f"\n<i>…y {len(opps) - 10} oportunidades más disponibles en el panel web visual.</i>")
    else:
        lines.append("\n😴 <b>Hoy no hay ninguna empresa de calidad en zona clara de compra.</b> El cocodrilo espera.")

    if watch:
        lines.append("\n<b>🟡 EN VIGILANCIA (Acercándose a zona de compra)</b>")
        for r in watch[:8]:
            dd = f"{r['drawdown_52w_pct']}%" if r["drawdown_52w_pct"] is not None else "—"
            obj = f"{r['dcf']['target_price']} {r['currency']}" if r.get("dcf") else "—"
            line = (
                f"• <b>{esc_html(r['ticker'])}</b> ({esc_html(r['name'][:20])}): {r['price']} {r['currency']} "
                f"| 52s: {dd} | Score {r['quality_score']} | Obj: {obj}"
            )
            lines.append(line)
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
        "\n⚠️ <b>Antes de invertir:</b> contrasta las cuentas oficiales y confirma si la caída es <i>TEMPORAL o PERMANENTE</i>."
    )
    return "\n".join(lines)


def main() -> None:
    tickers = get_universe(UNIVERSE_MODE, CUSTOM_TICKERS)
    t0 = time.time()
    print(f"🐊 Escáner Cocodrilo Pro v7 Blindado | Modo: {UNIVERSE_MODE} | Empresas: {len(tickers)} | Hilos: {MAX_WORKERS}")

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

    # Protección anti-sobrescritura: si Yahoo Finance bloqueó toda la tanda, abortar y disparar alerta de fallo
    if len(results) == 0:
        raise RuntimeError("0 empresas válidas analizadas. Abortando para no sobrescribir datos buenos en el panel.")

    order = {"OPORTUNIDAD": 0, "VIGILAR": 1, "CALIDAD_CARA": 2, "DESCARTADA": 3}
    results.sort(key=lambda r: (order.get(r["status"], 9), -r["quality_score"], -(r["upside_base_pct"] or -999)))

    ai_status: Dict[str, Any] = {
        "enabled": bool(GEMINI_API_KEY),
        "attempted": 0,
        "ok": 0,
        "errors": 0,
        "last_error": None,
    }

    if GEMINI_API_KEY:
        targets = [r for r in results if r["status"] == "OPORTUNIDAD"][:6]
        if not targets:
            targets = [r for r in results if r["status"] == "VIGILAR"][:2]
        for r in targets:
            ai_status["attempted"] += 1
            print(f"🤖 Consultando Gemini + titulares recientes para {r['ticker']}...")
            ai_text, ai_err = optional_gemini_buffett_check(r)
            if ai_text:
                r["ai_check"] = ai_text
                ai_status["ok"] += 1
            else:
                ai_status["errors"] += 1
                ai_status["last_error"] = ai_err
            time.sleep(2.5)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "scanner_version": "v7-blindado",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe_mode": UNIVERSE_MODE,
        "total_scanned": len(tickers),
        "total_ok": len(results),
        "total_failed": failed,
        "elapsed_sec": round(time.time() - t0),
        "ai_status": ai_status,
        "counts": {
            "oportunidades": sum(1 for r in results if r["status"] == "OPORTUNIDAD"),
            "vigilar": sum(1 for r in results if r["status"] == "VIGILAR"),
            "calidad_cara": sum(1 for r in results if r["status"] == "CALIDAD_CARA"),
            "descartadas": sum(1 for r in results if r["status"] == "DESCARTADA"),
        },
        "companies": results,
    }
    json_payload = json.dumps(summary, indent=2, ensure_ascii=False)

    # Mantiene latest-scan.json por compatibilidad y guarda además su archivo separado (diario vs semanal)
    OUT_LATEST_JSON.write_text(json_payload, encoding="utf-8")
    target_split = OUT_WEEKLY_JSON if UNIVERSE_MODE == "global_1000" else OUT_DAILY_JSON
    target_split.write_text(json_payload, encoding="utf-8")
    print(f"💾 Guardado en {OUT_LATEST_JSON.name} y {target_split.name} ({len(results)} empresas en {summary['elapsed_sec']}s)")

    msg = build_telegram_report(results, failed, ai_status)
    if msg:
        send_telegram(msg)
        print("✅ Alerta enviada a Telegram.")
    else:
        print("😴 Sin oportunidades hoy y TEST_MODE=false.")


if __name__ == "__main__":
    main()
