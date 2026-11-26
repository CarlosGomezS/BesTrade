"""Radar Cocodrilo v10.1 — Módulo Francotirador (Oráculo Anti-Trampas Blindado).

Mejoras v10.1 tras auditoría:
1) SEC EDGAR estricto: nunca marca VERIFIED sin cruce real contra Yahoo y solo valida emisores USD.
2) Tribunal IA con salida JSON estricta (sin regex frágil) + estado EVIDENCIA_INSUFICIENTE.
3) Validación segura de respuesta Gemini (candidates / promptFeedback / finishReason).
4) Filtro ORO PURO tolerancia cero: la ausencia de verificación SEC, nota de riesgo o Piotroski BLOQUEA.
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

SEC_USER_AGENT = "RadarCocodriloPro/10.1 (personal-value-scanner@github.com)"
VALID_VERDICTS = ("TEMPORAL", "DUDOSO", "PERMANENTE", "EVIDENCIA_INSUFICIENTE")
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
    elif f_max > 0:
        out["piotroski_max"] = f_max
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
            if x.get("form") == "10-K" and x.get("val") is not None
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
    """Cruza FCF oficial 10-K (USD) en SEC EDGAR contra los datos de Yahoo.

    Regla estricta v10.1: solo devuelve sec_verified=True si logra comparar numéricamente
    el FCF oficial del 10-K contra el FCF de Yahoo con desviación <= 30%.
    """
    ticker = (rec.get("ticker") or "").upper()
    if "." in ticker or rec.get("region") != "EE. UU.":
        return {"sec_status": "NO_US", "sec_verified": None, "sec_note": "Empresa no cotizada en SEC (EE. UU.)"}

    fin_cur = (rec.get("financial_currency") or rec.get("currency") or "USD").upper()
    if fin_cur != "USD":
        return {"sec_status": "NON_USD", "sec_verified": None, "sec_note": f"Reporta en {fin_cur} (no comparable en USD US-GAAP)"}

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
            return {"sec_status": "NO_GAAP", "sec_verified": None, "sec_note": "Sin etiquetas US-GAAP 10-K en SEC"}

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
            return {"sec_status": "INCOMPLETE", "sec_verified": None, "sec_note": "SEC sin OCF anual 10-K estándar"}

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
        if not mkt_b or not pfcf or pfcf <= 0:
            return {
                "sec_status": "INCOMPLETE",
                "sec_verified": None,
                "sec_note": f"SEC FCF {sec_fcf / 1e9:.2f}B USD, pero falta MarketCap o P/FCF en Yahoo para cruzar",
            }

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
    except Exception as exc:
        return {"sec_status": "ERROR", "sec_verified": None, "sec_note": f"SEC no disponible ({str(exc)[:60]})"}


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


def run_tribunal_ia(
    rec: Dict[str, Any],
    gemini_api_key: str,
    gemini_models: Tuple[str, ...],
) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[str]]:
    """Tribunal Fiscal vs. Abogado con salida JSON estricta.

    Devuelve: (texto_formateado, veredicto_enum, trap_risk_0_10, error)
    """
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
        f"DCF base {dcf.get('intrinsic_base')} {rec['currency']}, Compras directivos 90d: {rec.get('insider_buys_90d')}.\n"
        f"Auditoría SEC: {sec_note}.\n"
        f"Avisos automáticos: {flags_block}.\n"
        f"Titulares recientes:\n{news_block}\n\n"
        "Devuelve EXCLUSIVAMENTE un objeto JSON válido con estas 5 claves exactas en español (sin markdown ni texto fuera del JSON):\n"
        '{\n'
        '  "fiscal": "argumento más duro para NO comprar (1-2 frases)",\n'
        '  "abogado": "por qué la caída podría ser temporal a 10 años (1-2 frases)",\n'
        '  "comprobar_5min": "las 2 verificaciones concretas que el inversor debe hacer hoy antes de comprar",\n'
        '  "riesgo_trampa": 5,\n'
        '  "veredicto": "TEMPORAL | DUDOSO | PERMANENTE | EVIDENCIA_INSUFICIENTE"\n'
        '}\n'
        "riesgo_trampa debe ser un número entero del 0 (segurísima) al 10 (trampa mortal)."
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

                verdict = _normalize_verdict(parsed.get("veredicto"))
                if not verdict:
                    last_err = f"{model}: veredicto fuera de enum ({parsed.get('veredicto')})"
                    continue

                raw_risk = _sf(parsed.get("riesgo_trampa"))
                risk_num = min(10, max(0, int(round(raw_risk)))) if raw_risk is not None else None

                fiscal = str(parsed.get("fiscal") or "—").strip()
                abogado = str(parsed.get("abogado") or "—").strip()
                check5 = str(parsed.get("comprobar_5min") or "—").strip()
                risk_txt = f"{risk_num}/10" if risk_num is not None else "N/D"

                formatted = (
                    f"FISCAL: {fiscal}\n"
                    f"ABOGADO: {abogado}\n"
                    f"COMPROBAR (5 min): {check5}\n"
                    f"RIESGO_TRAMPA: {risk_txt}\n"
                    f"VEREDICTO: {verdict}"
                )[:980]
                return formatted, verdict, risk_num, None
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
    """Filtro francotirador final v10.1 (tolerancia cero): la ausencia de pruebas SIEMPRE bloquea."""
    if rec.get("status") != "OPORTUNIDAD":
        return False
    if rec.get("opportunity_tier") not in ("ENTRADA_YA", "GANGA"):
        return False

    # 1) Doble verificación oficial SEC EDGAR obligatoria (None o False bloquean)
    if rec.get("sec_verified") is not True:
        return False

    # 2) Anti-cíclicas y estimaciones futuras obligatorias
    if rec.get("is_cyclical_risk") or rec.get("forward_ok") is not True:
        return False

    # 3) Piotroski obligatorio con cobertura >= 6/9 y nota escalada >= 7/9
    p_score = rec.get("piotroski_score")
    p_max = rec.get("piotroski_max") or 0
    if p_score is None or p_max < 6:
        return False
    if round((p_score / p_max) * 9) < 7:
        return False

    # 4) Cobertura de intereses: si tiene deuda neta > 0.2x EBITDA, debe superar 8x obligatoriamente
    cov = rec.get("interest_coverage")
    nde = rec.get("net_debt_ebitda")
    if nde is None or nde > 0.2:
        if cov is None or cov < 8.0:
            return False

    # 5) Tribunal IA obligatorio: veredicto TEMPORAL + nota de riesgo explícita <= 4/10
    if rec.get("ai_verdict") != "TEMPORAL":
        return False
    risk = rec.get("trap_risk_score")
    if risk is None or risk > 4:
        return False

    return True
