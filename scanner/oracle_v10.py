"""Radar Cocodrilo v10 — Módulo Francotirador (Oráculo Anti-Trampas).

Capas de blindaje:
1) Auditoría oficial SEC EDGAR (data.sec.gov) para empresas de EE. UU. (detecta datos sucios de Yahoo).
2) Piotroski F-Score (0-9) + Cobertura de intereses + Anti-Cíclicas (estabilidad de márgenes).
3) Forward Check (bloquea alertas si el beneficio o ventas futuras ya están cayendo).
4) Tribunal IA (Fiscal vs. Abogado + nota de riesgo de trampa 0-10) en las finalistas.
"""

from __future__ import annotations

import json
import math
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

SEC_USER_AGENT = "RadarCocodriloPro/10.0 (personal-value-scanner@github.com)"
_sec_cik_map: Optional[Dict[str, str]] = None


def _sf(val: Any) -> Optional[float]:
    try:
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return None
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return None


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


def compute_oracle_checks(t: Any, info: Dict[str, Any], rec: Dict[str, Any]) -> Dict[str, Any]:
    """Calcula Piotroski F-Score, cobertura de intereses, estabilidad de márgenes y Forward Check."""
    out: Dict[str, Any] = {
        "piotroski_score": None,
        "piotroski_max": 0,
        "interest_coverage": None,
        "margin_spread_pct": None,
        "is_cyclical_risk": False,
        "forward_ok": True,
        "oracle_flags": [],
    }
    flags: List[str] = []

    try:
        fin = t.financials
        bs = t.balance_sheet
        cf = t.cashflow
    except Exception:
        fin = bs = cf = None

    # --- 1) PIOTROSKI F-SCORE (0 a 9) ---
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

    if f_max >= 5:
        out["piotroski_score"] = f_pts
        out["piotroski_max"] = f_max
        scaled_9 = round((f_pts / f_max) * 9)
        if scaled_9 <= 4:
            flags.append(f"Piotroski débil ({f_pts}/{f_max}): tendencia financiera deteriorándose")

    # --- 2) COBERTURA DE INTERESES (EBIT / Intereses) ---
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

    # --- 4) FORWARD CHECK (Estimaciones futuras) ---
    tr_eps = _sf(info.get("trailingEps"))
    fw_eps = _sf(info.get("forwardEps"))
    rev_g = _sf(info.get("revenueGrowth"))

    if tr_eps is not None and fw_eps is not None and tr_eps > 0:
        eps_chg = ((fw_eps / tr_eps) - 1.0) * 100.0
        if eps_chg < -10.0:
            out["forward_ok"] = False
            flags.append(f"BPA futuro previsto a la baja ({eps_chg:.1f}%)")
    if rev_g is not None and rev_g < -0.05:
        out["forward_ok"] = False
        flags.append(f"Ingresos cayendo ({rev_g * 100:.1f}%)")

    out["oracle_flags"] = flags
    return out


def _load_sec_cik_map() -> Dict[str, str]:
    global _sec_cik_map
    if _sec_cik_map is not None:
        return _sec_cik_map
    _sec_cik_map = {}
    try:
        req = urllib.request.Request(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": SEC_USER_AGENT, "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for entry in data.values():
            sym = str(entry.get("ticker") or "").upper().strip()
            cik = str(entry.get("cik_str") or "").zfill(10)
            if sym and cik:
                _sec_cik_map[sym] = cik
    except Exception as exc:
        print(f"⚠️ No se pudo descargar índice CIK de SEC EDGAR: {exc}")
    return _sec_cik_map


def _latest_10k_value(facts_us_gaap: Dict[str, Any], tags: List[str], unit: str = "USD") -> Optional[float]:
    for tag in tags:
        node = facts_us_gaap.get(tag)
        if not isinstance(node, dict):
            continue
        arr = (node.get("units") or {}).get(unit) or []
        annual = [
            x for x in arr
            if x.get("form") in ("10-K", "20-F") and x.get("val") is not None
            and (not x.get("frame") or "Q" not in str(x.get("frame")))
        ]
        if not annual:
            continue
        annual.sort(key=lambda x: str(x.get("end") or ""), reverse=True)
        val = _sf(annual[0].get("val"))
        if val is not None:
            return val
    return None


def verify_with_sec_edgar(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Cruza FCF oficial 10-K en SEC EDGAR contra los datos de Yahoo para empresas de EE. UU."""
    ticker = (rec.get("ticker") or "").upper()
    if "." in ticker or rec.get("region") != "EE. UU.":
        return {"sec_status": "NO_US", "sec_verified": None, "sec_note": "Empresa no cotizada en SEC (EE. UU.)"}

    cik_map = _load_sec_cik_map()
    cik = cik_map.get(ticker) or cik_map.get(ticker.replace("-", "."))
    if not cik:
        return {"sec_status": "NO_CIK", "sec_verified": None, "sec_note": "Ticker sin CIK en SEC"}

    try:
        time.sleep(0.2)
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        req = urllib.request.Request(url, headers={"User-Agent": SEC_USER_AGENT, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            facts = json.loads(resp.read().decode("utf-8"))
        us_gaap = (facts.get("facts") or {}).get("us-gaap") or {}
        if not us_gaap:
            return {"sec_status": "NO_GAAP", "sec_verified": None, "sec_note": "Sin etiquetas US-GAAP en SEC"}

        sec_ocf = _latest_10k_value(us_gaap, ["NetCashProvidedByUsedInOperatingActivities"])
        sec_capex = _latest_10k_value(
            us_gaap,
            [
                "PaymentsToAcquirePropertyPlantAndEquipment",
                "PaymentsToAcquireProductiveAssets",
                "CapitalExpendituresIncurredButNotYetPaid",
            ],
        )
        sec_sbc = _latest_10k_value(us_gaap, ["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"]) or 0.0

        if sec_ocf is None:
            return {"sec_status": "INCOMPLETE", "sec_verified": None, "sec_note": "SEC sin OCF anual estándar"}

        sec_fcf = sec_ocf - abs(sec_capex or 0.0)
        sec_owner = sec_fcf - (sec_sbc if rec.get("owner_earnings_adjusted") else 0.0)

        if sec_fcf <= 0:
            return {
                "sec_status": "MISMATCH",
                "sec_verified": False,
                "sec_note": f"ALERTA SEC 10-K: FCF oficial negativo ({sec_fcf / 1e9:.2f}B USD) frente a datos positivos de Yahoo",
            }

        mkt_b = rec.get("market_cap_b")
        pfcf = rec.get("p_fcf")
        if mkt_b and pfcf and pfcf > 0:
            yf_fcf_est = (mkt_b * 1e9) / pfcf
            ref_sec = sec_owner if (rec.get("owner_earnings_adjusted") and sec_owner > 0) else sec_fcf
            diff_pct = abs(yf_fcf_est - ref_sec) / max(abs(ref_sec), 1.0) * 100.0
            if diff_pct > 30.0:
                return {
                    "sec_status": "MISMATCH",
                    "sec_verified": False,
                    "sec_note": f"Discrepancia SEC vs Yahoo del {diff_pct:.0f}% en FCF (SEC {ref_sec / 1e9:.2f}B vs Yahoo {yf_fcf_est / 1e9:.2f}B)",
                }
            return {
                "sec_status": "VERIFIED",
                "sec_verified": True,
                "sec_note": f"Verificado 10-K SEC EDGAR (desviación {diff_pct:.0f}% en caja)",
            }

        return {"sec_status": "VERIFIED", "sec_verified": True, "sec_note": f"FCF oficial SEC positivo ({sec_fcf / 1e9:.2f}B USD)"}
    except Exception as exc:
        return {"sec_status": "ERROR", "sec_verified": None, "sec_note": f"SEC no disponible ({str(exc)[:60]})"}


def run_tribunal_ia(
    rec: Dict[str, Any],
    gemini_api_key: str,
    gemini_models: Tuple[str, ...],
) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """Tribunal Fiscal vs. Abogado para las candidatas top. Devuelve (texto, trap_risk_0_10, error)."""
    if not gemini_api_key:
        return None, None, "Sin GEMINI_API_KEY"

    news = rec.get("recent_news") or []
    news_block = "\n".join(f"- {h}" for h in news) if news else "- Sin titulares recientes en Yahoo."
    flags_block = "; ".join((rec.get("data_warnings") or []) + (rec.get("oracle_flags") or [])) or "Ninguna"
    dcf = rec.get("dcf") or {}
    sec_note = rec.get("sec_note") or "N/D"

    prompt = (
        "Actúa como un TRIBUNAL DE INVERSIÓN estilo Warren Buffett y Charlie Munger para decidir si una empresa es una "
        "OPORTUNIDAD DE ORO (solo 1-2 al año) o una TRAMPA DE VALOR.\n"
        f"Empresa: {rec['name']} ({rec['ticker']}), {rec['sector']} / {rec['industry']} ({rec['region']}).\n"
        f"Números: Precio {rec['price']} {rec['currency']} (52s: {rec['drawdown_52w_pct']}%), Score {rec['quality_score']}/100, "
        f"ROIC {rec['roic_pct']}%, Piotroski {rec.get('piotroski_score')}/{rec.get('piotroski_max')}, "
        f"Cobertura intereses {rec.get('interest_coverage')}x, P/FCF {rec['p_fcf']}x, PER futuro {rec['forward_pe']}x, "
        f"DCF base {dcf.get('intrinsic_base')} {rec['currency']}, Compras directivos 90d: {rec.get('insider_buys_90d', 0)}.\n"
        f"Auditoría SEC: {sec_note}.\n"
        f"Avisos automáticos: {flags_block}.\n"
        f"Titulares recientes:\n{news_block}\n\n"
        "Responde en español directo, sin asteriscos ni negritas, en exactamente 5 líneas:\n"
        "FISCAL: el argumento más duro para NO comprar (disrupción, ciclo en pico, pérdida de moat o contabilidad).\n"
        "ABOGADO: por qué la caída podría ser temporal y el negocio seguir fuerte a 10 años.\n"
        "COMPROBAR (5 min): las 2 cosas exactas que el inversor debe verificar hoy antes de meter dinero.\n"
        "RIESGO_TRAMPA: X/10 (donde 0 es segurísima y 10 es trampa mortal).\n"
        "VEREDICTO: TEMPORAL, DUDOSO o PERMANENTE."
    )

    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 1200,
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
                parts = body["candidates"][0]["content"]["parts"]
                text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                text = re.sub(r"\*\*(.*?)\*\*", r"\1", text).strip()[:950]
                if not text:
                    last_err = f"{model}: respuesta vacía"
                    continue
                m_risk = re.search(r"RIESGO[_\s]*TRAMPA[^\d]{0,12}(\d{1,2})\s*/\s*10", text, flags=re.IGNORECASE)
                risk_num = min(10, max(0, int(m_risk.group(1)))) if m_risk else None
                return text, risk_num, None
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
    return None, None, last_err


def is_oro_puro(rec: Dict[str, Any]) -> bool:
    """Filtro francotirador final (1-2 disparos/año): exige doble verificación y cero banderas rojas."""
    if rec.get("status") != "OPORTUNIDAD":
        return False
    if rec.get("opportunity_tier") not in ("ENTRADA_YA", "GANGA"):
        return False
    if rec.get("sec_verified") is False:
        return False
    if rec.get("is_cyclical_risk"):
        return False
    if not rec.get("forward_ok", True):
        return False

    p_score = rec.get("piotroski_score")
    p_max = rec.get("piotroski_max") or 0
    if p_score is not None and p_max >= 5:
        if round((p_score / p_max) * 9) < 7:
            return False

    cov = rec.get("interest_coverage")
    nde = rec.get("net_debt_ebitda")
    if cov is not None and cov < 8.0 and (nde is None or nde > 0.2):
        return False

    if rec.get("ai_verdict") != "TEMPORAL":
        return False

    risk = rec.get("trap_risk_score")
    if risk is not None and risk > 4:
        return False

    return True
