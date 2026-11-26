"""Radar Cocodrilo v10.3 — Módulo Francotirador (Oráculo Anti-Trampas Tolerancia Cero).

Correcciones v10.3 tras tercera auditoría técnica:
1) Forward Check real: separa crecimiento histórico de ventas (info.revenueGrowth) de las previsiones
   futuras de ingresos (t.revenue_estimate 0y/+1y). Si faltan previsiones futuras de ingresos o BPA,
   forward_ok es None (MISSING) y BLOQUEA ORO PURO y ENTRADA_YA.
2) Validación estricta de entero en riesgo_trampa: rechaza flotantes/decimales (ej. 3.7) y booleanos.
3) SEC EDGAR depurado:
   - Elimina CapitalExpendituresIncurredButNotYetPaid (CapEx devengado no pagado en efectivo).
   - Soporta enmiendas 10-K/A (prioriza la cifra corregida más reciente para el mismo cierre fiscal).
   - Combina múltiples etiquetas de CapEx entre distintos ejercicios fiscales sin quedarse atrapado en la primera.
   - Soporta ejercicios fiscales de 53 semanas (rango 330-385 días).
"""

from __future__ import annotations

import gzip
import json
import math
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ORACLE_VERSION = "v10.3-oraculo"
SEC_USER_AGENT = "RadarCocodriloPro/10.3 (contact: carlosgomezs.radar@gmail.com)"
VALID_VERDICTS = ("TEMPORAL", "DUDOSO", "PERMANENTE", "EVIDENCIA_INSUFICIENTE")

_sec_cik_map: Optional[Dict[str, str]] = None
_sec_cik_error: Optional[str] = None


def _sf(val: Any) -> Optional[float]:
    try:
        if val is None or isinstance(val, bool) or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return None
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return None


def _parse_iso_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str or not isinstance(date_str, str):
        return None
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _read_http_json(url: str, headers: Dict[str, str], timeout: int = 15) -> Any:
    req_headers = dict(headers)
    req_headers.setdefault("Accept-Encoding", "gzip")
    req = urllib.request.Request(url, headers=req_headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        if resp.headers.get("Content-Encoding", "").lower() == "gzip" or raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        return json.loads(raw.decode("utf-8"))


def _row(df: Optional[pd.DataFrame], names: List[str]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    idx_map = {str(k).strip().lower(): k for k in df.index}
    for n in names:
        k = idx_map.get(n.lower())
        if k is not None:
            s = df.loc[k]
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


def extract_forward_revenue_growth(t: Any, info: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """Extrae crecimiento previsto de ingresos futuros (0y / +1y) desde t.revenue_estimate.

    Devuelve (porcentaje_crecimiento_previsto_o_None, estado):
    - OK: previsión obtenida (si existen 0y y +1y, toma el más conservador para detectar frenazos).
    - MISSING: Yahoo no devuelve estimaciones anuales de ingresos.
    - ERROR: fallo consultando t.revenue_estimate.
    """
    direct = _sf(info.get("forwardRevenueGrowth"))
    if direct is not None:
        return round(direct * 100.0, 1), "OK"

    try:
        rev_est = getattr(t, "revenue_estimate", None)
    except Exception:
        return None, "ERROR"

    if rev_est is None or not isinstance(rev_est, pd.DataFrame) or rev_est.empty:
        return None, "MISSING"

    idx_map = {str(k).strip().lower(): k for k in rev_est.index}
    cols_map = {str(c).strip().lower(): c for c in rev_est.columns}
    growth_col = cols_map.get("growth")
    avg_col = cols_map.get("avg")
    ago_col = cols_map.get("yearagorevenue")

    found_pcts: List[float] = []
    for period in ("0y", "+1y"):
        row_key = idx_map.get(period)
        if row_key is None:
            continue
        row = rev_est.loc[row_key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        g_val = _sf(row.get(growth_col)) if growth_col is not None else None
        if g_val is None and avg_col is not None and ago_col is not None:
            avg_v = _sf(row.get(avg_col))
            ago_v = _sf(row.get(ago_col))
            if avg_v is not None and ago_v is not None and ago_v > 0:
                g_val = (avg_v / ago_v) - 1.0
        if g_val is not None:
            found_pcts.append(round(g_val * 100.0, 1))

    if not found_pcts:
        return None, "MISSING"
    return min(found_pcts), "OK"


def compute_oracle_checks(t: Any, info: Dict[str, Any], rec: Dict[str, Any]) -> Dict[str, Any]:
    """Calcula Piotroski (sin reescalar), cobertura de intereses, estabilidad de márgenes y Forward Check real."""
    out: Dict[str, Any] = {
        "oracle_version": ORACLE_VERSION,
        "piotroski_score": None,
        "piotroski_max": 0,
        "interest_coverage": None,
        "margin_spread_pct": None,
        "is_cyclical_risk": False,
        "hist_rev_growth_pct": None,
        "forward_eps_growth_pct": None,
        "forward_rev_growth_pct": None,
        "forward_rev_status": "MISSING",
        "forward_ok": None,
        "forward_status": "MISSING",
        "oracle_flags": [],
    }
    flags: List[str] = []

    try:
        fin = t.financials
        bs = t.balance_sheet
        cf = t.cashflow
    except Exception:
        fin = bs = cf = None

    # --- 1) PIOTROSKI F-SCORE (puntuación bruta real, SIN reescalar) ---
    ni = _row(fin, ["Net Income", "Net Income Common Stockholders"])
    rev = _row(fin, ["Total Revenue", "Operating Revenue"])
    gp = _row(fin, ["Gross Profit"])
    ebit = _row(fin, ["EBIT", "Operating Income"])
    int_exp = _row(fin, ["Interest Expense", "Interest Expense Non Operating"])
    ocf = _row(cf, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    assets = _row(bs, ["Total Assets"])
    lt_debt = _row(bs, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"])
    cur_a = _row(bs, ["Current Assets", "Total Current Assets"])
    cur_l = _row(bs, ["Current Liabilities", "Total Current Liabilities"])
    shares = _row(fin, ["Diluted Average Shares", "Basic Average Shares"])

    f_pts = 0
    f_max = 0

    def v(series: Optional[pd.Series], idx: int = 0) -> Optional[float]:
        if series is None or len(series) <= idx:
            return None
        return _sf(series.values[idx])

    ni0, ni1 = v(ni, 0), v(ni, 1)
    a0, a1 = v(assets, 0), v(assets, 1)
    ocf0 = v(ocf, 0)

    roa0 = (ni0 / a0) if (ni0 is not None and a0 and a0 > 0) else None
    roa1 = (ni1 / a1) if (ni1 is not None and a1 and a1 > 0) else None
    if roa0 is not None:
        f_max += 1
        if roa0 > 0:
            f_pts += 1
    if ocf0 is not None:
        f_max += 1
        if ocf0 > 0:
            f_pts += 1
    if roa0 is not None and roa1 is not None:
        f_max += 1
        if roa0 > roa1:
            f_pts += 1
    if ocf0 is not None and ni0 is not None:
        f_max += 1
        if ocf0 > ni0:
            f_pts += 1
    d0, d1 = v(lt_debt, 0), v(lt_debt, 1)
    if d0 is not None and d1 is not None and a0 and a1 and a0 > 0 and a1 > 0:
        f_max += 1
        if (d0 / a0) <= (d1 / a1) + 0.005:
            f_pts += 1
    ca0, ca1, cl0, cl1 = v(cur_a, 0), v(cur_a, 1), v(cur_l, 0), v(cur_l, 1)
    if all(x is not None for x in (ca0, ca1, cl0, cl1)) and cl0 > 0 and cl1 > 0:
        f_max += 1
        if (ca0 / cl0) >= (ca1 / cl1) - 0.02:
            f_pts += 1
    sh0, sh1 = v(shares, 0), v(shares, 1)
    if sh0 is not None and sh1 is not None and sh1 > 0:
        f_max += 1
        if sh0 <= sh1 * 1.005:
            f_pts += 1
    gp0, gp1, r0, r1 = v(gp, 0), v(gp, 1), v(rev, 0), v(rev, 1)
    if all(x is not None for x in (gp0, gp1, r0, r1)) and r0 > 0 and r1 > 0:
        f_max += 1
        if (gp0 / r0) >= (gp1 / r1) - 0.002:
            f_pts += 1
    if all(x is not None for x in (r0, r1, a0, a1)) and a0 > 0 and a1 > 0:
        f_max += 1
        if (r0 / a0) >= (r1 / a1) - 0.01:
            f_pts += 1

    out["piotroski_max"] = f_max
    if f_max >= 5:
        out["piotroski_score"] = f_pts
        if f_pts <= 4 and (f_max - f_pts) >= 3:
            flags.append(f"Piotroski débil ({f_pts}/{f_max} señales): tendencia financiera deteriorándose")
        if f_max < 7:
            flags.append(f"Piotroski parcial ({f_pts}/{f_max}): cobertura insuficiente para ORO PURO (mín. 7/9)")
    elif f_max > 0:
        flags.append(f"Piotroski incompleto ({f_max}/9 señales disponibles)")

    # --- 2) COBERTURA DE INTERESES (EBIT / |Intereses|) ---
    ebit0, ie0 = v(ebit, 0), v(int_exp, 0)
    if ebit0 is not None and ie0 is not None and abs(ie0) > 0:
        cov = ebit0 / abs(ie0)
        out["interest_coverage"] = round(cov, 1)
        if cov < 5.0 and rec.get("sector_profile") != "financial":
            flags.append(f"Cobertura de intereses justa ({cov:.1f}x)")

    # --- 3) ANTI-CÍCLICAS (Estabilidad de margen operativo 3-4 años) ---
    if ebit is not None and rev is not None:
        common = ebit.index.intersection(rev.index)
        margins: List[float] = []
        for col in list(common)[:4]:
            ev, rv = _sf(ebit.get(col)), _sf(rev.get(col))
            if ev is not None and rv is not None and rv > 0:
                margins.append((ev / rv) * 100.0)
        if len(margins) >= 3:
            spread = max(margins) - min(margins)
            out["margin_spread_pct"] = round(spread, 1)
            avg_m = sum(margins) / len(margins)
            if spread >= 12.0 and (avg_m <= 0 or spread / max(abs(avg_m), 1.0) >= 0.45):
                out["is_cyclical_risk"] = True
                flags.append(f"Márgenes cíclicos/volátiles (oscilación {spread:.1f} pp en {len(margins)}a)")

    # --- 4) FORWARD CHECK REAL (separa crecimiento histórico de previsiones futuras) ---
    hist_rev_g = _sf(info.get("revenueGrowth"))
    if hist_rev_g is not None:
        out["hist_rev_growth_pct"] = round(hist_rev_g * 100.0, 1)
        if hist_rev_g < -0.05:
            flags.append(f"Ingresos históricos cayendo ({hist_rev_g * 100:.1f}%)")

    tr_eps = _sf(info.get("trailingEps"))
    fw_eps = _sf(info.get("forwardEps"))

    eps_checked = False
    fwd_rev_checked = False
    forward_failed = False

    if tr_eps is not None and fw_eps is not None and tr_eps > 0:
        eps_checked = True
        eps_chg = round(((fw_eps / tr_eps) - 1.0) * 100.0, 1)
        out["forward_eps_growth_pct"] = eps_chg
        if eps_chg < -10.0:
            forward_failed = True
            flags.append(f"BPA futuro previsto a la baja ({eps_chg:.1f}%)")

    # Solo consultar t.revenue_estimate en candidatas relevantes (o tests sin status) para evitar 429 masivos
    status_val = rec.get("status")
    if status_val in (None, "OPORTUNIDAD", "VIGILAR"):
        fwd_rev_pct, fwd_rev_st = extract_forward_revenue_growth(t, info)
        out["forward_rev_growth_pct"] = fwd_rev_pct
        out["forward_rev_status"] = fwd_rev_st
        if fwd_rev_pct is not None:
            fwd_rev_checked = True
            if fwd_rev_pct < -5.0:
                forward_failed = True
                flags.append(f"Ventas futuras previstas a la baja ({fwd_rev_pct:.1f}%)")
    else:
        out["forward_rev_status"] = "SKIPPED"

    if forward_failed:
        out["forward_ok"] = False
        out["forward_status"] = "FAIL"
    elif eps_checked and fwd_rev_checked:
        out["forward_ok"] = True
        out["forward_status"] = "OK"
    else:
        out["forward_ok"] = None
        out["forward_status"] = "MISSING"
        if status_val in (None, "OPORTUNIDAD", "VIGILAR"):
            missing_parts = []
            if not eps_checked:
                missing_parts.append("BPA futuro")
            if not fwd_rev_checked:
                missing_parts.append("ingresos futuros previstos")
            flags.append(f"Previsiones futuras incompletas (falta {' y '.join(missing_parts)})")

    out["oracle_flags"] = flags
    return out


def _load_sec_cik_map() -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """Descarga índice oficial de tickers -> CIK desde sec.gov distinguiendo error de red de ausencia real."""
    global _sec_cik_map, _sec_cik_error
    if _sec_cik_map is not None:
        return _sec_cik_map, None
    if _sec_cik_error is not None:
        return None, _sec_cik_error

    headers = {
        "User-Agent": SEC_USER_AGENT,
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
    }
    last_err = None
    for attempt in range(2):
        try:
            data = _read_http_json("https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=15)
            mapping: Dict[str, str] = {}
            if isinstance(data, dict):
                for entry in data.values():
                    if not isinstance(entry, dict):
                        continue
                    sym = str(entry.get("ticker") or "").upper().strip()
                    cik_raw = entry.get("cik_str")
                    if sym and cik_raw is not None:
                        mapping[sym] = str(cik_raw).zfill(10)
            if len(mapping) < 500:
                raise ValueError(f"Índice CIK sospechosamente corto ({len(mapping)} entradas)")
            _sec_cik_map = mapping
            return _sec_cik_map, None
        except Exception as exc:
            last_err = str(exc)
            if attempt == 0:
                time.sleep(1.5)

    _sec_cik_error = f"Error descargando índice CIK SEC: {last_err[:80]}"
    print(f"⚠️ {_sec_cik_error}")
    return None, _sec_cik_error


def _extract_annual_10k_by_end(
    facts_us_gaap: Dict[str, Any],
    tags: List[str],
    unit: str = "USD",
) -> Dict[str, float]:
    """Extrae valores de ejercicio fiscal completo (10-K y 10-K/A, 330-385 días incl. años de 53 semanas).

    Combina todas las etiquetas válidas entre ejercicios distintos y, para un mismo cierre fiscal,
    prioriza la presentación más reciente (10-K/A enmendado por delante del 10-K original) y la etiqueta primaria.
    """
    by_end: Dict[str, Tuple[str, int, int, float]] = {}
    for tag_idx, tag in enumerate(tags):
        node = facts_us_gaap.get(tag)
        if not isinstance(node, dict):
            continue
        arr = (node.get("units") or {}).get(unit) or []
        for x in arr:
            if not isinstance(x, dict):
                continue
            form = str(x.get("form") or "").upper().strip()
            if form not in ("10-K", "10-K/A"):
                continue
            frame = str(x.get("frame") or "")
            if frame and "Q" in frame.upper():
                continue
            val = _sf(x.get("val"))
            end_s = str(x.get("end") or "")[:10]
            start_s = str(x.get("start") or "")[:10]
            if val is None or len(end_s) != 10:
                continue
            if len(start_s) == 10:
                d_start, d_end = _parse_iso_date(start_s), _parse_iso_date(end_s)
                if d_start and d_end:
                    days = (d_end - d_start).days
                    if days < 330 or days > 385:
                        continue
            filed_s = str(x.get("filed") or "")
            is_amend = 1 if form == "10-K/A" else 0
            cand = (filed_s, is_amend, -tag_idx, val)
            prev = by_end.get(end_s)
            if prev is None or cand[:3] > prev[:3]:
                by_end[end_s] = cand
    return {end_s: item[3] for end_s, item in by_end.items()}


def verify_with_sec_edgar(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Cruza FCF oficial 10-K/10-K/A (USD) en SEC EDGAR contra Yahoo con alineación estricta de ejercicio fiscal."""
    ticker = (rec.get("ticker") or "").upper()
    if "." in ticker or rec.get("region") != "EE. UU.":
        return {"sec_status": "NO_US", "sec_verified": None, "sec_fiscal_end": None, "sec_note": "Empresa no cotizada en SEC (EE. UU.)"}

    fin_cur = (rec.get("financial_currency") or rec.get("currency") or "USD").upper()
    if fin_cur != "USD":
        return {"sec_status": "NON_USD", "sec_verified": None, "sec_fiscal_end": None, "sec_note": f"Reporta en {fin_cur} (no comparable en USD US-GAAP)"}

    cik_map, cik_err = _load_sec_cik_map()
    if cik_err or cik_map is None:
        return {"sec_status": "ERROR", "sec_verified": None, "sec_fiscal_end": None, "sec_note": cik_err or "Índice SEC no disponible"}

    cik = cik_map.get(ticker) or cik_map.get(ticker.replace("-", ".")) or cik_map.get(ticker.replace(".", "-"))
    if not cik:
        return {"sec_status": "NO_CIK", "sec_verified": None, "sec_fiscal_end": None, "sec_note": f"Ticker {ticker} sin CIK en índice SEC"}

    try:
        time.sleep(0.2)
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        headers = {"User-Agent": SEC_USER_AGENT, "Accept": "application/json", "Accept-Encoding": "gzip"}
        facts = _read_http_json(url, headers=headers, timeout=15)
        us_gaap = (facts.get("facts") or {}).get("us-gaap") or {}
        if not us_gaap:
            return {"sec_status": "NO_GAAP", "sec_verified": None, "sec_fiscal_end": None, "sec_note": "Sin etiquetas US-GAAP 10-K en SEC"}

        ocf_map = _extract_annual_10k_by_end(
            us_gaap,
            [
                "NetCashProvidedByUsedInOperatingActivities",
                "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
            ],
        )
        # Solo pagos reales de CapEx en efectivo (se excluye CapitalExpendituresIncurredButNotYetPaid)
        capex_map = _extract_annual_10k_by_end(
            us_gaap,
            [
                "PaymentsToAcquirePropertyPlantAndEquipment",
                "PaymentsToAcquireProductiveAssets",
                "PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets",
                "PaymentsForCapitalImprovements",
            ],
        )
        sbc_map = _extract_annual_10k_by_end(
            us_gaap,
            ["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"],
        )

        if not ocf_map:
            return {"sec_status": "INCOMPLETE", "sec_verified": None, "sec_fiscal_end": None, "sec_note": "SEC 10-K sin OCF anual estándar"}
        if not capex_map:
            return {"sec_status": "INCOMPLETE", "sec_verified": None, "sec_fiscal_end": None, "sec_note": "SEC 10-K sin pagos de CapEx en efectivo (no se asume 0 ni devengado)"}

        common_ends = sorted(set(ocf_map.keys()).intersection(capex_map.keys()), reverse=True)
        if not common_ends:
            return {"sec_status": "INCOMPLETE", "sec_verified": None, "sec_fiscal_end": None, "sec_note": "SEC 10-K sin cierre fiscal común entre OCF y CapEx"}

        yf_end_str = rec.get("fcf_latest_date")
        yf_dt = _parse_iso_date(yf_end_str)
        chosen_end = common_ends[0]
        if yf_dt is not None:
            best_end = None
            best_diff = 99999
            for end_s in common_ends:
                s_dt = _parse_iso_date(end_s)
                if s_dt is not None:
                    diff_d = abs((s_dt - yf_dt).days)
                    if diff_d < best_diff:
                        best_diff = diff_d
                        best_end = end_s
            if best_end is None or best_diff > 45:
                return {
                    "sec_status": "PERIOD_MISMATCH",
                    "sec_verified": None,
                    "sec_fiscal_end": common_ends[0],
                    "sec_note": f"Desfase de ejercicio fiscal: SEC ({common_ends[0]}) vs Yahoo ({yf_end_str})",
                }
            chosen_end = best_end
        else:
            return {
                "sec_status": "INCOMPLETE",
                "sec_verified": None,
                "sec_fiscal_end": chosen_end,
                "sec_note": f"Falta fecha fiscal de FCF en Yahoo para alinear contra SEC ({chosen_end})",
            }

        sec_dt = _parse_iso_date(chosen_end)
        if sec_dt is not None and (datetime.now(timezone.utc) - sec_dt).days > 550:
            return {
                "sec_status": "STALE",
                "sec_verified": None,
                "sec_fiscal_end": chosen_end,
                "sec_note": f"10-K SEC obsoleto (cierre {chosen_end})",
            }

        sec_ocf = ocf_map[chosen_end]
        sec_capex = capex_map[chosen_end]
        sec_sbc = sbc_map.get(chosen_end)

        sec_fcf = sec_ocf - abs(sec_capex)
        if rec.get("owner_earnings_adjusted"):
            if sec_sbc is None:
                return {
                    "sec_status": "INCOMPLETE",
                    "sec_verified": None,
                    "sec_fiscal_end": chosen_end,
                    "sec_note": f"SEC ({chosen_end}) sin etiqueta SBC para cruzar Owner Earnings",
                }
            ref_sec = sec_fcf - abs(sec_sbc)
        else:
            ref_sec = sec_fcf

        if ref_sec <= 0:
            return {
                "sec_status": "MISMATCH",
                "sec_verified": False,
                "sec_fiscal_end": chosen_end,
                "sec_note": f"ALERTA SEC {chosen_end}: caja oficial negativa ({ref_sec / 1e9:.2f}B USD) frente a Yahoo",
            }

        yf_fcf_ref = _sf(rec.get("fcf_latest_trade"))
        if yf_fcf_ref is None:
            mkt_b = rec.get("market_cap_b")
            pfcf = rec.get("p_fcf")
            if mkt_b and pfcf and pfcf > 0:
                yf_fcf_ref = (mkt_b * 1e9) / pfcf

        if yf_fcf_ref is None or yf_fcf_ref <= 0:
            return {
                "sec_status": "INCOMPLETE",
                "sec_verified": None,
                "sec_fiscal_end": chosen_end,
                "sec_note": f"SEC ({chosen_end}) FCF {ref_sec / 1e9:.2f}B USD, pero falta FCF válido en Yahoo para cruzar",
            }

        diff_pct = abs(yf_fcf_ref - ref_sec) / max(abs(ref_sec), 1.0) * 100.0
        if diff_pct > 30.0:
            return {
                "sec_status": "MISMATCH",
                "sec_verified": False,
                "sec_fiscal_end": chosen_end,
                "sec_note": f"Discrepancia SEC ({chosen_end}) vs Yahoo del {diff_pct:.0f}% (SEC {ref_sec / 1e9:.2f}B vs Yahoo {yf_fcf_ref / 1e9:.2f}B)",
            }

        return {
            "sec_status": "VERIFIED",
            "sec_verified": True,
            "sec_fiscal_end": chosen_end,
            "sec_note": f"Verificado 10-K SEC {chosen_end} (desviación {diff_pct:.0f}% en caja)",
        }
    except Exception as exc:
        return {"sec_status": "ERROR", "sec_verified": None, "sec_fiscal_end": None, "sec_note": f"SEC no disponible ({str(exc)[:60]})"}


def extract_gemini_json(body: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Extrae y valida de forma segura una respuesta JSON de Gemini sin asumir estructura fija."""
    if not isinstance(body, dict):
        return None, "Respuesta Gemini no es JSON objeto"
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        fb = body.get("promptFeedback") or {}
        reason = fb.get("blockReason") or "sin candidatos"
        return None, f"Gemini bloqueado/vacío ({reason})"

    cand0 = candidates[0] if isinstance(candidates[0], dict) else {}
    finish = cand0.get("finishReason")
    content = cand0.get("content") if isinstance(cand0.get("content"), dict) else {}
    parts = content.get("parts") if isinstance(content.get("parts"), list) else []
    raw_text = "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict) and not p.get("thought")).strip()
    if not raw_text:
        return None, f"Respuesta vacía (finishReason={finish})"

    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed, None
        return None, "JSON devuelto no es un objeto"
    except Exception:
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, dict):
                    return parsed, None
            except Exception:
                pass
        return None, f"JSON inválido de Gemini (finishReason={finish})"


def _normalize_verdict(val: Any) -> Optional[str]:
    if not isinstance(val, str):
        return None
    up = val.strip().upper().replace(" ", "_")
    return up if up in VALID_VERDICTS else None


def validate_tribunal_payload(parsed: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Valida que el JSON del Tribunal IA esté 100% completo y riesgo_trampa sea un entero estricto 0-10."""
    verdict = _normalize_verdict(parsed.get("veredicto"))
    if not verdict:
        return None, f"veredicto fuera de enum ({parsed.get('veredicto')})"

    raw_risk = parsed.get("riesgo_trampa")
    if type(raw_risk) is not int:
        return None, f"riesgo_trampa debe ser un entero exacto 0-10 (recibido {type(raw_risk).__name__}: {raw_risk})"
    if raw_risk < 0 or raw_risk > 10:
        return None, f"riesgo_trampa fuera de rango 0-10 ({raw_risk})"

    req_texts = {}
    for key in ("fiscal", "abogado", "comprobar_5min"):
        val = parsed.get(key)
        if not isinstance(val, str) or len(val.strip()) < 10:
            return None, f"campo '{key}' ausente o demasiado corto"
        req_texts[key] = val.strip()

    return {
        "veredicto": verdict,
        "riesgo_trampa": raw_risk,
        **req_texts,
    }, None


def run_tribunal_ia(
    rec: Dict[str, Any],
    gemini_api_key: str,
    gemini_models: Tuple[str, ...],
) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[str]]:
    """Tribunal Fiscal vs. Abogado con salida JSON estricta y validación completa de los 5 campos."""
    if not gemini_api_key:
        return None, None, None, "Sin GEMINI_API_KEY"

    news = rec.get("recent_news") or []
    news_status = rec.get("news_status") or ("OK" if news else "OK_EMPTY")
    if news:
        news_block = "\n".join(f"- {h}" for h in news)
    elif news_status == "ERROR":
        news_block = "- ERROR al obtener titulares de Yahoo Finance (no asumas ausencia de malas noticias)."
    else:
        news_block = "- Sin titulares recientes disponibles en Yahoo Finance."

    flags_block = "; ".join((rec.get("data_warnings") or []) + (rec.get("oracle_flags") or [])) or "Ninguna"
    dcf = rec.get("dcf") or {}
    sec_note = rec.get("sec_note") or "N/D"

    prompt = (
        "Actúa como un TRIBUNAL DE INVERSIÓN escéptico estilo Warren Buffett y Charlie Munger para decidir si una empresa "
        "es una OPORTUNIDAD DE ORO (solo 1-2 al año) o una TRAMPA DE VALOR.\n"
        "REGLA ESTRICTA: Si no hay titulares claros sobre esta empresa o faltan pruebas para saber por qué cae, "
        "NO inventes una explicación ni regales TEMPORAL; usa el veredicto EVIDENCIA_INSUFICIENTE.\n"
        f"Empresa: {rec['name']} ({rec['ticker']}), {rec['sector']} / {rec['industry']} ({rec['region']}).\n"
        f"Números: Precio {rec['price']} {rec['currency']} (52s: {rec['drawdown_52w_pct']}%), Score {rec['quality_score']}/100, "
        f"ROIC {rec['roic_pct']}%, Piotroski {rec.get('piotroski_score')}/{rec.get('piotroski_max')}, "
        f"Cobertura intereses {rec.get('interest_coverage')}x, P/FCF {rec['p_fcf']}x, PER futuro {rec['forward_pe']}x, "
        f"Previsión ingresos: {rec.get('forward_rev_growth_pct')}% ({rec.get('forward_rev_status')}), "
        f"DCF base {dcf.get('intrinsic_base')} {rec['currency']}, Compras directivos 90d: {rec.get('insider_buys_90d')}.\n"
        f"Auditoría SEC: {sec_note}.\n"
        f"Avisos automáticos: {flags_block}.\n"
        f"Titulares recientes:\n{news_block}\n\n"
        "Devuelve EXCLUSIVAMENTE un objeto JSON válido con estas 5 claves OBLIGATORIAS y no vacías en español:\n"
        '{\n'
        '  "fiscal": "argumento más duro para NO comprar (1-2 frases)",\n'
        '  "abogado": "por qué la caída podría ser temporal a 10 años (1-2 frases)",\n'
        '  "comprobar_5min": "las 2 verificaciones concretas que el inversor debe hacer hoy antes de comprar",\n'
        '  "riesgo_trampa": 5,\n'
        '  "veredicto": "TEMPORAL | DUDOSO | PERMANENTE | EVIDENCIA_INSUFICIENTE"\n'
        '}\n'
        "riesgo_trampa debe ser un número ENTERO exacto (sin decimales) del 0 (segurísima) al 10 (trampa mortal)."
    )

    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 1000,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingLevel": "LOW"},
        },
    }).encode("utf-8")

    last_err = None
    for model in gemini_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_api_key}"
        for attempt in range(2):
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=28) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                parsed, err = extract_gemini_json(body)
                if err or not parsed:
                    last_err = f"{model}: {err}"
                    continue

                valid, v_err = validate_tribunal_payload(parsed)
                if v_err or not valid:
                    last_err = f"{model}: respuesta incompleta ({v_err})"
                    continue

                formatted = (
                    f"FISCAL: {valid['fiscal']}\n"
                    f"ABOGADO: {valid['abogado']}\n"
                    f"COMPROBAR (5 min): {valid['comprobar_5min']}\n"
                    f"RIESGO_TRAMPA: {valid['riesgo_trampa']}/10\n"
                    f"VEREDICTO: {valid['veredicto']}"
                )[:980]
                return formatted, valid["veredicto"], valid["riesgo_trampa"], None
            except urllib.error.HTTPError as exc:
                last_err = f"{model} HTTP {exc.code}"
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
    return None, None, None, last_err


def is_oro_puro(rec: Dict[str, Any]) -> bool:
    """Filtro francotirador final v10.3 (tolerancia cero real): NINGUNA ausencia de datos aprueba."""
    if rec.get("status") != "OPORTUNIDAD":
        return False
    if rec.get("opportunity_tier") not in ("ENTRADA_YA", "GANGA"):
        return False

    # 1) Doble verificación oficial SEC EDGAR obligatoria con alineación fiscal
    if rec.get("sec_verified") is not True or rec.get("sec_status") != "VERIFIED":
        return False

    # 2) Anti-cíclicas y Forward Check real verificado (BPA futuro + ingresos futuros previstos)
    if rec.get("is_cyclical_risk") or rec.get("forward_ok") is not True:
        return False

    # 3) Piotroski BRUTO (sin reescalar): al menos 7 señales verificadas y 7 puntos reales obtenidos
    p_score = rec.get("piotroski_score")
    p_max = rec.get("piotroski_max") or 0
    if p_score is None or p_max < 7 or p_score < 7:
        return False

    # 4) Cobertura de intereses: si tiene deuda neta > 0.2x EBITDA, debe superar 8x obligatoriamente
    cov = rec.get("interest_coverage")
    nde = rec.get("net_debt_ebitda")
    if nde is None or nde > 0.2:
        if cov is None or cov < 8.0:
            return False

    # 5) Tribunal IA completo obligatorio: veredicto TEMPORAL + nota de riesgo entero estricto <= 4/10
    if rec.get("ai_verdict") != "TEMPORAL":
        return False
    risk = rec.get("trap_risk_score")
    if type(risk) is not int or risk < 0 or risk > 4:
        return False

    return True
