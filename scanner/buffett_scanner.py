"""Radar Cocodrilo Pro v6 — Escáner Masivo Automático (Warren Buffett & Charlie Munger).

Escanea desde ~200 empresas de élite mundial (diario) hasta ~1.000+ empresas globales
(S&P 500 + S&P 400 MidCap + Nasdaq 100 + Campeones de Europa) sin que tengas que meter
ningún dato manual.

- Descarga estados financieros reales (4 años) y cotizaciones vía Yahoo Finance (gratis).
- Concurrencia suave (hilos limitados + reintentos) para analizar ~1.000 empresas en ~15 min.
- Calcula puntuación objetiva Buffett (0-100) con desglose visual de 6 pilares + bloqueos anti-trampa.
- Calcula valoración DCF a 10 años en 3 escenarios (Conservador, Base y Optimista) ajustando caja/deuda neta.
- Envía alertas claras a Telegram y actualiza data/latest-scan.json para el panel web visual.
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

from universe import detect_region, get_universe

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "data" / "latest-scan.json"

UNIVERSE_MODE = os.environ.get("UNIVERSE_MODE", "elite200").strip().lower()
CUSTOM_TICKERS = os.environ.get("CUSTOM_TICKERS", "").strip()
TEST_MODE = os.environ.get("TEST_MODE", "false").strip().lower() == "true"
MIN_QUALITY_SCORE = int(os.environ.get("MIN_QUALITY_SCORE", "75"))
MARGIN_OF_SAFETY_PCT = float(os.environ.get("MARGIN_OF_SAFETY_PCT", "25"))
MIN_DRAWDOWN_ALERT = float(os.environ.get("MIN_DRAWDOWN_ALERT", "18"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "3"))
MIN_MARKET_CAP_USD = float(os.environ.get("MIN_MARKET_CAP_USD", "1000000000"))  # >= 1B para evitar chicharros

TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

_print_lock = threading.Lock()


def safe_float(val: Any) -> Optional[float]:
    try:
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return None
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return None


def get_row_series(df: Optional[pd.DataFrame], candidates: List[str]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    idx_map = {str(k).strip().lower(): k for k in df.index}
    for name in candidates:
        key = idx_map.get(name.lower())
        if key is not None:
            s = df.loc[key].dropna()
            if not s.empty:
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

    # 1) Free Cash Flow (últimos años disponibles)
    fcf_series = get_row_series(cf, ["Free Cash Flow"])
    if fcf_series is None:
        ocf = get_row_series(cf, ["Operating Cash Flow", "Total Cash From Operating Activities"])
        capex = get_row_series(cf, ["Capital Expenditure", "Capital Expenditures"])
        if ocf is not None and capex is not None:
            common = ocf.index.intersection(capex.index)
            if len(common) > 0:
                fcf_series = ocf.loc[common] + capex.loc[common].apply(lambda x: -abs(x))

    if fcf_series is not None and len(fcf_series) >= 1:
        vals = [safe_float(v) for v in fcf_series.values]
        vals = [v for v in vals if v is not None]
        if vals:
            out["fcf_years_total"] = len(vals)
            out["fcf_years_pos"] = sum(1 for v in vals if v > 0)
            out["fcf_latest"] = vals[0]
            if len(vals) >= 3 and vals[0] > 0 and vals[-1] > 0:
                out["fcf_cagr_pct"] = ((vals[0] / vals[-1]) ** (1.0 / (len(vals) - 1)) - 1.0) * 100.0

    if out["fcf_latest"] is None:
        out["fcf_latest"] = safe_float(info.get("freeCashflow"))

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

    # 4) ROIC real = NOPAT / Capital Invertido
    ebit_s = get_row_series(fin, ["EBIT", "Operating Income"])
    tax_s = get_row_series(fin, ["Tax Provision"])
    pretax_s = get_row_series(fin, ["Pretax Income"])
    debt_s = get_row_series(bs, ["Total Debt"])
    eq_s = get_row_series(bs, ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"])
    cash_s = get_row_series(bs, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"])

    roic_list = []
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
        out["roic_pct"] = sum(roic_list) / len(roic_list)
    else:
        roe = safe_float(info.get("returnOnEquity"))
        if roe is not None:
            out["roic_pct"] = roe * 100.0

    # 5) Caja Neta y Deuda Neta / EBITDA (con detector de EBITDA negativo)
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
    if intrinsic_base is None:
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

    def add(key: str, pts: int, max_pts: int):
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

    # 3) Consistencia FCF (20 pts)
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

    # 4) Solidez de Balance — Deuda Neta / EBITDA (20 pts)
    if data.get("ebitda_negative"):
        add("debt", 0, 20)
        hard_fails.append("EBITDA negativo / pérdidas operativas")
    elif profile == "financial":
        add("debt", 14, 20)
    elif data.get("net_debt_ebitda") is not None:
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

    normalized = round((score / known) * 100) if known > 0 else 0
    passes = known >= 70 and len(hard_fails) == 0 and normalized >= MIN_QUALITY_SCORE
    return {
        "quality_score": normalized,
        "data_coverage_pct": known,
        "score_breakdown": breakdown,
        "hard_fails": hard_fails,
        "strengths": strengths[:4],
        "passes_quality": passes,
    }


def analyze_ticker_with_retry(ticker: str, retries: int = 2) -> Optional[Dict[str, Any]]:
    for attempt in range(retries + 1):
        try:
            return analyze_ticker(ticker)
        except Exception as exc:
            msg = str(exc).lower()
            if attempt < retries and ("429" in msg or "too many" in msg or "timeout" in msg or "rate" in msg):
                time.sleep(2.5 * (attempt + 1) + random.uniform(0.3, 1.2))
                continue
            return None
    return None


def analyze_ticker(ticker: str) -> Optional[Dict[str, Any]]:
    t = yf.Ticker(ticker)
    info = t.info or {}
    price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"))
    if price is None or price <= 0:
        return None

    mkt_cap = safe_float(info.get("marketCap"))
    if mkt_cap is not None and mkt_cap < MIN_MARKET_CAP_USD:
        return None

    name = info.get("shortName") or info.get("longName") or ticker
    currency = (info.get("currency") or "USD").upper()
    sector = info.get("sector") or "Otros"
    industry = info.get("industry") or ""
    profile = classify_sector_profile(sector, industry)
    region = detect_region(ticker)

    high52 = safe_float(info.get("fiftyTwoWeekHigh"))
    low52 = safe_float(info.get("fiftyTwoWeekLow"))
    drawdown_52w = ((price / high52) - 1.0) * 100.0 if (high52 and high52 > 0) else None

    shares = safe_float(info.get("sharesOutstanding"))
    if (shares is None or shares <= 0) and mkt_cap and price > 0:
        shares = mkt_cap / price

    opm = safe_float(info.get("operatingMargins"))
    rev_g = safe_float(info.get("revenueGrowth"))
    earn_g = safe_float(info.get("earningsGrowth"))
    div_y = safe_float(info.get("dividendYield"))
    if div_y is not None and div_y < 0.25:
        div_y = div_y * 100.0

    stm = compute_roic_and_statements(t, info)
    fcf_latest = stm["fcf_latest"]
    p_fcf = (mkt_cap / fcf_latest) if (mkt_cap and fcf_latest and fcf_latest > 0) else None

    hist_g = stm["eps_cagr_pct"] if stm["eps_cagr_pct"] is not None else stm["fcf_cagr_pct"]
    analyst_g = (earn_g * 100.0) if earn_g is not None else ((rev_g * 100.0) if rev_g is not None else None)

    dcf = compute_conservative_dcf(
        fcf_total=fcf_latest,
        shares=shares,
        net_cash=stm["net_cash"],
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
        "sector": sector,
        "industry": industry,
        "sector_profile": profile,
        "market_cap_b": round(mkt_cap / 1e9, 1) if mkt_cap else None,
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
        "fcf_years_total": stm["fcf_years_tota
