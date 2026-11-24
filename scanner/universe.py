"""Universos de empresas de calidad para el Radar Cocodrilo v6.

Modos disponibles:
- 'elite200'    : ~185 grandes negocios mundiales (EE. UU. + Europa + Asia ADR). Ideal diario L-V (~4 min).
- 'sp500'       : Todo el S&P 500 + campeones europeos (~580 empresas). (~9 min).
- 'global_1000' : S&P 500 + S&P 400 MidCap + Nasdaq 100 + Europa (~1.000+ empresas). Ideal sábados (~15 min).
- 'custom'      : Solo los tickers que indiques manualmente.
"""

from __future__ import annotations

import urllib.request
from typing import List
import pandas as pd

# ~75 Campeones de Europa y Mundo (Yahoo Finance tickers)
EUROPA_Y_MUNDO: List[str] = [
    # Países Bajos / Bélgica / Alemania / Francia
    "ASML.AS", "ADYEN.AS", "WKL.AS", "HEIA.AS", "PHIA.AS", "ASM.AS",
    "MC.PA", "RMS.PA", "OR.PA", "AI.PA", "SU.PA", "SAF.PA", "EL.PA", "DSY.PA", "DG.PA", "TTE.PA", "BNP.PA",
    "SAP.DE", "SIE.DE", "ALV.DE", "MUV2.DE", "ADS.DE", "IFX.DE", "DHL.DE", "MBG.DE", "DTE.DE", "HEN3.DE",
    # España (IBEX / Calidad)
    "ITX.MC", "AMS.MC", "VIS.MC", "IBE.MC", "FER.MC", "AENA.MC", "SAN.MC", "BBVA.MC", "LOG.MC", "ROVI.MC", "VID.MC",
    # Suiza / Nórdicas / Italia
    "NESN.SW", "ROG.SW", "NOVN.SW", "ABBN.SW", "CFR.SW", "GIVN.SW", "SIKA.SW", "PGHN.SW", "LONN.SW",
    "NVO", "NOVO-B.CO", "DSV.CO", "ATCO-A.ST", "INVE-B.ST", "ASSA-B.ST", "EVO.ST", "KNEBV.HE",
    "RACE", "ISP.MI", "UCG.MI", "MONC.MI", "PRY.MI",
    # Reino Unido
    "REL.L", "DGE.L", "ULVR.L", "AZN.L", "LSEG.L", "EXPN.L", "CPG.L", "HLMA.L", "SPX.L", "AUTO.L", "RIO.L", "SHEL.L",
    # Global ADRs de calidad
    "TSM", "MELI", "SHOP", "CSU.TO", "CNI", "CP", "TM", "SONY", "SAP", "UL"
]

# ~130 Compounders y Líderes de Calidad de EE. UU.
US_ELITE: List[str] = [
    # Consumo, Marcas y Retail de Calidad
    "KO", "PEP", "COST", "WMT", "PG", "MCD", "HD", "LOW", "NKE", "SBUX",
    "TJX", "ORLY", "AZO", "HSY", "MNST", "CL", "KMB", "MDLZ", "YUM", "CMG",
    "DPZ", "KR", "TGT", "ULTA", "LULU", "DECK", "ROL", "BF-B", "STZ",
    # Tecnología, Software, Cloud y Semiconductores
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "ADBE",
    "INTU", "ORCL", "CRM", "TXN", "QCOM", "KLAC", "LRCX", "AMAT", "SNPS",
    "CDNS", "NFLX", "NOW", "ANET", "PANW", "CRWD", "FTNT", "ADP", "PAYX",
    "ROP", "MSI", "APH", "MPWR", "ADI", "MCHP", "NXPI", "ACN", "IBM", "CSCO",
    "FDS", "VRSK", "Gartner" if False else "IT", "TYL", "CDW", "MANH",
    # Finanzas, Pagos, Datos y Seguros (Efecto red / Peajes)
    "BRK-B", "V", "MA", "AXP", "MCO", "SPGI", "MSCI", "FICO", "BLK",
    "CB", "PGR", "MMC", "AJG", "AON", "JPM", "CME", "ICE", "NDAQ", "SCHW",
    "BRO", "WRB", "TRV", "AFL", "MKL", "CBOE", "FI", "PYPL",
    # Salud, MedTech y Ciencias de la Vida
    "UNH", "JNJ", "LLY", "ABBV", "TMO", "DHR", "ISRG", "SYK", "ZTS", "IDXX",
    "ABT", "MDT", "BSX", "ELV", "CI", "MCK", "COR", "REGN", "VRTX", "RMD",
    "WST", "EW", "A", "MTD", "WAT", "IQV", "HCA",
    # Industriales, Defensa, Infraestructura y Energía de Calidad
    "XOM", "CVX", "COP", "SLB", "LIN", "SHW", "CTAS", "WM", "RSG", "UNP",
    "ODFL", "CPRT", "FAST", "GWW", "POOL", "WSO", "HEI", "TDG", "LMT", "GD",
    "NOC", "GE", "ETN", "PH", "ITW", "EMR", "HON", "CAT", "DE", "URI",
    "CSX", "NSC", "VRSN", "NVR", "LEN", "ECL", "APD", "PPG"
]

ELITE_200: List[str] = list(dict.fromkeys(US_ELITE + EUROPA_Y_MUNDO))


def _fetch_wiki_table_tickers(url: str, table_idx: int = 0, col_candidates=("Symbol", "Ticker", "Ticker symbol")) -> List[str]:
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "CocodriloBuffettBot/6.0 (Educational Stock Screener)"}
        )
        with urllib.request.urlopen(req, timeout=18) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        tables = pd.read_html(html)
        if not tables:
            return []
        # Buscar la tabla que contenga alguna de las columnas candidatas
        target_df = None
        for idx_try in ([table_idx] + list(range(len(tables)))):
            if idx_try >= len(tables):
                continue
            df = tables[idx_try]
            cols = [str(c).strip() for c in df.columns]
            if any(c in cols for c in col_candidates) and len(df) >= 50:
                target_df = df
                break
        if target_df is None:
            return []
        col = next((c for c in col_candidates if c in [str(x).strip() for x in target_df.columns]), target_df.columns[0])
        out = []
        for raw in target_df[col].dropna().astype(str).tolist():
            clean = raw.strip().upper().replace(".", "-")
            if clean and len(clean) <= 10 and " " not in clean:
                out.append(clean)
        return list(dict.fromkeys(out))
    except Exception as exc:
        print(f"⚠️ No se pudo descargar {url}: {exc}")
        return []


def fetch_sp500() -> List[str]:
    return _fetch_wiki_table_tickers("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", 0)


def fetch_sp400_midcap() -> List[str]:
    return _fetch_wiki_table_tickers("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", 0)


def fetch_nasdaq100() -> List[str]:
    return _fetch_wiki_table_tickers("https://en.wikipedia.org/wiki/Nasdaq-100", 4)


def detect_region(ticker: str) -> str:
    t = ticker.upper()
    if any(t.endswith(s) for s in (".MC", ".PA", ".AS", ".DE", ".SW", ".L", ".MI", ".CO", ".ST", ".HE", ".BR")):
        return "Europa"
    if t.endswith(".TO"):
        return "Canadá"
    if t in ("TSM", "MELI", "NVO", "RACE", "SAP", "UL", "TM", "SONY"):
        return "Global"
    return "EE. UU."


def get_universe(mode: str = "elite200", custom_tickers: str = "") -> List[str]:
    mode = (mode or "elite200").strip().lower()
    extra = [t.strip().upper() for t in custom_tickers.replace(";", ",").split(",") if t.strip()]

    if mode == "custom" and extra:
        return list(dict.fromkeys(extra))

    if mode == "sp500":
        sp500 = fetch_sp500()
        merged = (sp500 if len(sp500) >= 400 else US_ELITE) + EUROPA_Y_MUNDO + extra
        return list(dict.fromkeys(merged))

    if mode in ("global_1000", "global", "all"):
        sp500 = fetch_sp500()
        sp400 = fetch_sp400_midcap()
        ndx = fetch_nasdaq100()
        print(f"📡 Listas descargadas -> S&P500: {len(sp500)} | S&P400 MidCap: {len(sp400)} | Nasdaq100: {len(ndx)} | Europa/Élite: {len(EUROPA_Y_MUNDO)}")
        merged = US_ELITE + sp500 + sp400 + ndx + EUROPA_Y_MUNDO + extra
        return list(dict.fromkeys(merged))

    # Por defecto: elite200 (~200 grandes empresas mundiales)
    return list(dict.fromkeys(ELITE_200 + extra))
