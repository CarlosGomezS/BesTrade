"""Universo Global v8 — Radar Cocodrilo Pro.

Combina fuentes vivas (S&P 500, S&P 400 MidCap, S&P 600 SmallCap, Nasdaq 100)
con listas ampliadas de Europa, Canadá, Japón, Australia y ADRs globales.
- Sin dependencias extra (usa urllib + html.parser de la librería estándar).
- Soporta rotación diaria inteligente (~330 empresas/día, barriendo >2.000 distintas cada 2 semanas).
- Soporta modo custom para analizar al vuelo los tickers que elijas desde la app.
"""

from __future__ import annotations

import csv
import io
import random
import re
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"

# Núcleo de élite vigilado a diario (~115 empresas top EE. UU., Europa y Global)
ELITE_CORE: List[str] = [
    # EE. UU. Calidad / Tech / Salud / Consumo / Industria / Finanzas
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "TSM", "ORCL", "CRM",
    "ADBE", "INTU", "NOW", "SNPS", "CDNS", "KLAC", "LRCX", "AMAT", "TXN", "ADI",
    "ANET", "PANW", "CRWD", "FTNT", "ACN", "IT", "FICO", "SPGI", "MCO", "MSCI",
    "V", "MA", "PYPL", "FI", "INTC", "QCOM", "CSCO", "IBM", "NFLX", "BKNG",
    "ABNB", "UBER", "MELI", "COST", "WMT", "HD", "LOW", "TJX", "ROST", "DECK",
    "LULU", "NKE", "SBUX", "MCD", "CMG", "KO", "PEP", "PG", "CL", "MNST",
    "HSY", "MDLZ", "PM", "UNH", "ELV", "JNJ", "LLY", "ABBV", "MRK", "TMO",
    "DHR", "ISRG", "SYK", "BSX", "VRTX", "REGN", "ZTS", "IDXX", "A", "WAT",
    "LIN", "SHW", "ECL", "CAT", "DE", "HON", "ETN", "PH", "TT", "CARR",
    "CTAS", "WM", "RSG", "UNP", "ODFL", "CPRT", "FAST", "GWW", "URI", "ROL",
    "BRK-B", "JPM", "BLK", "CB", "AJG", "MMC", "PGR", "SPG",
    # Europa Top
    "ASML.AS", "ADYEN.AS", "WKL.AS", "REL.L", "AZN.L", "ULVR.L", "DGE.L", "EXPN.L",
    "LSEG.L", "CPG.L", "MC.PA", "RMS.PA", "OR.PA", "SU.PA", "AI.PA", "SAF.PA",
    "DSY.PA", "EL.PA", "SAP.DE", "SIE.DE", "DTE.DE", "MUV2.DE", "ALV.DE", "IFX.DE",
    "NOVO-B.CO", "DSV.CO", "COLO-B.CO", "ITX.MC", "AMS.MC", "FER.MC", "IBE.MC",
    "NESN.SW", "NOVN.SW", "ABBN.SW", "SIKA.SW", "GIVN.SW", "LONN.SW", "CFR.SW",
    "RACE.MI", "ATCO-A.ST", "ASSA-B.ST", "EVO.ST", "KNEBV.HE",
]

# Europa ampliada (~215 empresas líquidas de calidad y mid-caps europeas)
EUROPE_EXPANDED: List[str] = [
    # España (.MC)
    "ITX.MC", "AMS.MC", "FER.MC", "IBE.MC", "AENA.MC", "CLNX.MC", "LOG.MC", "VID.MC",
    "VIS.MC", "CIE.MC", "ROVI.MC", "ACS.MC", "NTGY.MC", "ELE.MC", "RED.MC", "ENG.MC",
    "MAP.MC", "SAN.MC", "BBVA.MC", "CABK.MC", "BKT.MC", "SAB.MC", "IAG.MC", "TEF.MC",
    "GRF.MC", "PUIG.MC", "EBRO.MC", "FLUI.MC", "ACX.MC", "MTS.MC",
    # Francia (.PA)
    "MC.PA", "RMS.PA", "OR.PA", "SU.PA", "AI.PA", "SAF.PA", "DSY.PA", "EL.PA",
    "AIR.PA", "SAN.PA", "TTE.PA", "CS.PA", "BNP.PA", "ACA.PA", "SGO.PA", "DG.PA",
    "LR.PA", "CAP.PA", "PUB.PA", "EDEN.PA", "BVI.PA", "RI.PA", "BN.PA", "KER.PA",
    "ML.PA", "HO.PA", "VIE.PA", "EN.PA", "ENGI.PA", "ORA.PA", "DIM.PA", "ERF.PA",
    "STMPA.PA", "TEP.PA", "WLN.PA", "SW.PA", "GTT.PA", "SOI.PA", "SK.PA", "RNO.PA",
    # Alemania (.DE)
    "SAP.DE", "SIE.DE", "DTE.DE", "ALV.DE", "MUV2.DE", "IFX.DE", "ADS.DE", "MBG.DE",
    "BMW.DE", "VOW3.DE", "P911.DE", "DHL.DE", "BAS.DE", "BAYN.DE", "MRK.DE", "SHL.DE",
    "MTX.DE", "RHM.DE", "BEI.DE", "HEN3.DE", "SY1.DE", "FRE.DE", "FME.DE", "EOAN.DE",
    "RWE.DE", "HEI.DE", "DB1.DE", "DBK.DE", "CBK.DE", "HNR1.DE", "NEM.DE", "AFX.DE",
    "SRT3.DE", "KNIN.DE", "G1A.DE", "BOSS.DE", "PUM.DE", "ZAL.DE", "DHER.DE", "HOT.DE",
    # Países Bajos (.AS) y Bélgica (.BR)
    "ASML.AS", "ADYEN.AS", "WKL.AS", "ASM.AS", "BESI.AS", "UNA.AS", "AD.AS", "HEIA.AS",
    "PRX.AS", "INGA.AS", "ABN.AS", "NN.AS", "ASRNL.AS", "KPN.AS", "IMCD.AS", "DSFIR.AS",
    "AKZA.AS", "RAND.AS", "LIGHT.AS", "EXO.AS", "ABI.BR", "ARGX.BR", "UCB.BR", "KBC.BR",
    "LOTB.BR", "ACKB.BR", "GBLB.BR", "SOF.BR", "SYENS.BR", "SOLB.BR", "UMI.BR",
    # Reino Unido (.L)
    "AZN.L", "SHEL.L", "HSBA.L", "ULVR.L", "REL.L", "LSEG.L", "DGE.L", "RIO.L",
    "GSK.L", "BP.L", "BATS.L", "EXPN.L", "CPG.L", "AHT.L", "HLMA.L", "SGE.L",
    "SPX.L", "AUTO.L", "RMV.L", "III.L", "BA.L", "RR.L", "RKT.L", "IHG.L",
    "NXT.L", "WTB.L", "TSCO.L", "SBRY.L", "NG.L", "SSE.L", "SVT.L", "UU.L",
    "PRU.L", "LGEN.L", "AV.L", "BARC.L", "LLOY.L", "NWG.L", "STAN.L", "IMB.L",
    "CRDA.L", "DPLM.L", "BNZL.L", "ITRK.L", "SMIN.L", "IMI.L", "MRO.L", "PSN.L",
    "BDEV.L", "INF.L", "PSON.L", "WPP.L", "JD.L", "BKG.L", "GAW.L", "HLN.L",
    # Suiza (.SW)
    "NESN.SW", "NOVN.SW", "ABBN.SW", "UBSG.SW", "ZURN.SW", "CFR.SW", "SIKA.SW", "GIVN.SW",
    "LONN.SW", "ALCON.SW", "ALC.SW", "HOLN.SW", "SREN.SW", "SLHN.SW", "PGHN.SW", "GEBN.SW",
    "KNIN.SW", "LOGN.SW", "SOON.SW", "STMN.SW", "VACN.SW", "SCHP.SW", "SCMN.SW", "SGSN.SW",
    "LISP.SW", "BARN.SW", "EMSN.SW", "TEMN.SW", "VAT.SW",
    # Nórdicos (.CO, .ST, .OL, .HE)
    "NOVO-B.CO", "DSV.CO", "COLO-B.CO", "CARL-B.CO", "MAERSK-B.CO", "ORSTED.CO", "VWS.CO",
    "GMAB.CO", "DEMANT.CO", "PNDORA.CO", "TRYG.CO", "DANSKE.CO", "NSIS-B.CO", "RBREW.CO",
    "ATCO-A.ST", "INVE-B.ST", "VOLV-B.ST", "ASSA-B.ST", "SAND.ST", "HEXA-B.ST", "EVO.ST",
    "HM-B.ST", "EPI-A.ST", "EQT.ST", "NIBE-B.ST", "ALFA.ST", "LIFCO-B.ST", "INDT.ST",
    "LAGR-B.ST", "ADDT-B.ST", "AAK.ST", "ESSITY-B.ST", "SEB-A.ST", "SHB-A.ST", "SWED-A.ST",
    "ERIC-B.ST", "SKF-B.ST", "SCA-B.ST", "SAAB-B.ST", "THULE.ST", "FORTNOX.ST",
    "EQNR.OL", "DNB.OL", "TEL.OL", "MOWI.OL", "ORK.OL", "NHY.OL", "YAR.OL", "KOG.OL",
    "TOM.OL", "SALM.OL", "GJF.OL", "STB.OL",
    "KNEBV.HE", "NDA-FI.HE", "SAMPO.HE", "NESTE.HE", "UPM.HE", "FORTUM.HE", "NOKIA.HE",
    "ELISA.HE", "WRT1V.HE", "METSO.HE", "ORNBV.HE", "KCR.HE", "HUH1V.HE", "VALMT.HE",
    # Italia (.MI), Irlanda (.IR), Austria (.VI), Portugal (.LS)
    "RACE.MI", "ENEL.MI", "ISP.MI", "UCG.MI", "ENI.MI", "G.MI", "MONC.MI", "PRY.MI",
    "AMP.MI", "REC.MI", "CPR.MI", "INW.MI", "TRN.MI", "SRG.MI", "PST.MI", "LDO.MI",
    "BZU.MI", "DIA.MI", "IP.MI", "SOL.MI", "RYA.IR", "KRZ.IR", "GLB.IR", "EBS.VI",
    "OMV.VI", "VER.VI", "VOE.VI", "ANDR.VI", "EDP.LS", "GALP.LS", "JMT.LS",
]

# Internacional ampliada: Canadá (.TO), Japón (.T), Australia (.AX), Hong Kong (.HK) y ADRs globales
INTL_EXPANDED: List[str] = [
    # Canadá (.TO)
    "CSU.TO", "SHOP.TO", "RY.TO", "TD.TO", "BN.TO", "BAM.TO", "CNR.TO", "CP.TO",
    "ATD.TO", "DOL.TO", "WCN.TO", "TRI.TO", "TFII.TO", "WSP.TO", "GIB-A.TO", "OTEX.TO",
    "L.TO", "MRU.TO", "EMP-A.TO", "IFC.TO", "FFH.TO", "MFC.TO", "SLF.TO", "ENB.TO",
    "TRP.TO", "CNQ.TO", "SU.TO", "AEM.TO", "WPM.TO", "FNV.TO", "CCJ", "KXS.TO",
    "DSG.TO", "TOI.V", "STN.TO", "CAE.TO", "BNS.TO", "BMO.TO", "CM.TO", "NA.TO",
    # Japón (.T) — Empresas industriales, robótica, semiconductores y consumo de alta calidad
    "6861.T", "8035.T", "6857.T", "4063.T", "6367.T", "6273.T", "6594.T", "6954.T",
    "6501.T", "6758.T", "7974.T", "9983.T", "7203.T", "4519.T", "4568.T", "4502.T",
    "4543.T", "7741.T", "6098.T", "4661.T", "9432.T", "9433.T", "8058.T", "8001.T",
    "8031.T", "8002.T", "8053.T", "8766.T", "8306.T", "8316.T", "6981.T", "6920.T",
    "6146.T", "6723.T", "4578.T", "2914.T", "3382.T", "4452.T", "7309.T", "6301.T",
    # Australia (.AX)
    "CSL.AX", "CBA.AX", "BHP.AX", "WES.AX", "WOW.AX", "COL.AX", "MQG.AX", "REA.AX",
    "CAR.AX", "WTC.AX", "XRO.AX", "PME.AX", "COH.AX", "RMD.AX", "SHL.AX", "BXB.AX",
    "ALL.AX", "TCL.AX", "GMG.AX", "QBE.AX", "SUN.AX", "NAB.AX", "ANZ.AX", "WBC.AX",
    "TLS.AX", "TWE.AX", "ARB.AX", "JBH.AX", "NWL.AX", "HUB.AX",
    # Asia / LatAm / ADRs líquidos en EE. UU. y HK
    "TSM", "BABA", "JD", "PDD", "NTES", "TCEHY", "0700.HK", "9988.HK", "1299.HK", "0388.HK",
    "SE", "GRAB", "CPNG", "INFY", "HDB", "IBN", "SONY", "TM", "MUFG", "NVO",
    "ASML", "SAP", "SPOT", "SHOP", "NU", "STNE", "PAGS", "GLOB", "DESP", "VIST",
    "BAP", "FMX", "KOF", "AMX", "VALE", "ITUB", "ERJ", "CYBR", "CHKP", "MNDY",
    "WIX", "NICE", "ONON", "BIRK", "ARM", "CRH", "LIN", "ACN", "ETN", "TT",
]

# Respaldo amplio de EE. UU. si Wikipedia fallase puntualmente
US_FALLBACK_EXTRA: List[str] = [
    "ABT", "ACGL", "ADM", "ADP", "AEE", "AEP", "AFL", "AIG", "AIZ", "AJG",
    "AKAM", "ALB", "ALGN", "ALL", "ALLE", "AMCR", "AMD", "AME", "AMGN", "AMP",
    "AMT", "ANSS", "AON", "AOS", "APA", "APD", "APH", "APTV", "ARE", "ATO",
    "AVB", "AVY", "AWK", "AXON", "AXP", "AZO", "BA", "BAC", "BALL", "BAX",
    "BBY", "BDX", "BEN", "BF-B", "BG", "BIIB", "BIO", "BK", "BKR", "BLDR",
    "BMY", "BR", "BRO", "BWA", "BX", "BXP", "C", "CAG", "CAH", "CBOE",
    "CCI", "CCL", "CDW", "CE", "CF", "CFG", "CHD", "CHRW", "CHTR", "CI",
    "CINF", "CLX", "CMA", "CMCSA", "CME", "CMI", "CMS", "CNC", "CNP", "COF",
    "COO", "COP", "COR", "CPB", "CPT", "CRL", "CSGP", "CSX", "CTLT", "CTRA",
    "CTSH", "CVS", "CVX", "D", "DAL", "DD", "DFS", "DG", "DGX", "DHI",
    "DIS", "DLR", "DLTR", "DOC", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK",
    "DVA", "DVN", "DXCM", "EA", "EBAY", "ED", "EFX", "EG", "EIX", "EL",
    "EMN", "EMR", "ENPH", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ES", "ESS",
    "ETR", "EVRG", "EW", "EXC", "EXPD", "EXPE", "EXR", "F", "FANG", "FCX",
    "FDS", "FDX", "FE", "FFIV", "FIS", "FITB", "FMC", "FOX", "FOXA", "FRT",
    "FSLR", "FTV", "GD", "GE", "GEHC", "GEN", "GEV", "GILD", "GIS", "GL",
    "GLW", "GM", "GNRC", "GPC", "GPN", "GRMN", "GS", "HAL", "HAS", "HBAN",
    "HCA", "HES", "HIG", "HII", "HLT", "HOLX", "HPE", "HPQ", "HRL", "HSIC",
    "HST", "HUBB", "HUM", "HWM", "ICE", "IEX", "IFF", "INCY", "INVH", "IP",
    "IPG", "IQV", "IR", "IRM", "ITW", "IVZ", "J", "JBHT", "JBL", "JCI",
    "JKHY", "JNPR", "K", "KDP", "KEY", "KEYS", "KHC", "KIM", "KKR", "KMB",
    "KMI", "KMX", "KR", "KVUE", "L", "LDOS", "LEN", "LH", "LHX", "LKQ",
    "LMT", "LNT", "LUV", "LVS", "LW", "LYB", "LYV", "MAA", "MAR", "MAS",
    "MCHP", "MCK", "MDT", "MET", "MGM", "MHK", "MKC", "MKTX", "MLM", "MMM",
    "MO", "MOH", "MOS", "MPC", "MPWR", "MRNA", "MS", "MSI", "MTB", "MTCH",
    "MTD", "MU", "NCLH", "NDAQ", "NDSN", "NEE", "NEM", "NI", "NOC", "NRG",
    "NSC", "NTAP", "NTRS", "NUE", "NVR", "NWS", "NWSA", "NXPI", "O", "OKE",
    "OMC", "ON", "OTIS", "OXY", "PANW", "PAYC", "PAYX", "PCAR", "PCG", "PEG",
    "PFE", "PFG", "PHM", "PKG", "PLD", "PNC", "PNR", "PNW", "PODD", "POOL",
    "PPG", "PPL", "PRU", "PSA", "PSX", "PTC", "PWR", "QRVO", "RCL", "RF",
    "RJF", "RL", "RMD", "ROP", "RVTY", "SBAC", "SCHW", "SHW", "SJM", "SLB",
    "SMCI", "SNA", "SO", "SOLV", "SPG", "SRE", "STE", "STLD", "STT", "STX",
    "STZ", "SWK", "SWKS", "SYF", "SYY", "T", "TAP", "TDG", "TDY", "TECH",
    "TEL", "TER", "TFC", "TFX", "TGT", "TMUS", "TPR", "TRGP", "TRMB", "TROW",
    "TRV", "TSCO", "TSLA", "TSN", "TTWO", "TXT", "TYL", "UAL", "UDR", "UHS",
    "ULTA", "UPS", "USB", "VICI", "VLO", "VLTO", "VMC", "VRSK", "VRSN", "VST",
    "VTR", "VTRS", "VZ", "WAB", "WBA", "WBD", "WDC", "WEC", "WELL", "WFC",
    "WM", "WMB", "WRB", "WST", "WTW", "WY", "WYNN", "XEL", "XOM", "XYL",
    "YUM", "ZBH", "ZBRA",
]

REGION_BY_SUFFIX: Dict[str, str] = {
    ".MC": "Europa", ".PA": "Europa", ".DE": "Europa", ".F": "Europa",
    ".AS": "Europa", ".BR": "Europa", ".L": "Europa", ".SW": "Europa",
    ".MI": "Europa", ".ST": "Europa", ".CO": "Europa", ".OL": "Europa",
    ".HE": "Europa", ".IR": "Europa", ".VI": "Europa", ".LS": "Europa",
    ".TO": "Canadá", ".V": "Canadá",
    ".T": "Asia-Pacífico", ".HK": "Asia-Pacífico", ".AX": "Asia-Pacífico",
    ".KS": "Asia-Pacífico", ".SI": "Asia-Pacífico",
}

KNOWN_GLOBAL_ADRS = {
    "TSM", "MELI", "BABA", "JD", "PDD", "NTES", "TCEHY", "SE", "GRAB", "CPNG",
    "INFY", "HDB", "IBN", "SONY", "TM", "MUFG", "NVO", "ASML", "SAP", "SPOT",
    "SHOP", "NU", "STNE", "PAGS", "GLOB", "DESP", "VIST", "BAP", "FMX", "KOF",
    "AMX", "VALE", "ITUB", "ERJ", "CYBR", "CHKP", "MNDY", "WIX", "NICE", "ONON",
    "BIRK", "ARM", "CCJ",
}


class _WikiTableParser(HTMLParser):
    """Extractor ligero de tablas HTML (sin lxml/bs4)."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: List[List[List[str]]] = []
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._cur_table: List[List[str]] = []
        self._cur_row: List[str] = []
        self._cur_cell: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        t = tag.lower()
        if t == "table":
            self._in_table = True
            self._cur_table = []
        elif t == "tr" and self._in_table:
            self._in_row = True
            self._cur_row = []
        elif t in ("td", "th") and self._in_row:
            self._in_cell = True
            self._cur_cell = []

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in ("td", "th") and self._in_cell:
            self._in_cell = False
            txt = " ".join("".join(self._cur_cell).split())
            self._cur_row.append(txt)
        elif t == "tr" and self._in_row:
            self._in_row = False
            if self._cur_row:
                self._cur_table.append(self._cur_row)
        elif t == "table" and self._in_table:
            self._in_table = False
            if self._cur_table:
                self.tables.append(self._cur_table)

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cur_cell.append(data)


def _fetch_text(url: str, timeout: int = 15) -> Optional[str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status >= 300:
                return None
            return resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return None


def _clean_us_ticker(sym: str) -> Optional[str]:
    s = re.sub(r"\[.*?\]", "", (sym or "")).strip().upper()
    if not s or " " in s or len(s) > 8:
        return None
    s = s.replace(".", "-")
    if not re.match(r"^[A-Z0-9\-]{1,7}$", s):
        return None
    return s


def _fetch_wikipedia_tickers(url: str, header_names: Tuple[str, ...] = ("symbol", "ticker", "ticker symbol")) -> List[str]:
    html = _fetch_text(url)
    if not html:
        return []
    parser = _WikiTableParser()
    try:
        parser.feed(html)
    except Exception:
        return []

    found: List[str] = []
    for table in parser.tables:
        if len(table) < 15:
            continue
        headers = [h.strip().lower() for h in table[0]]
        col_idx: Optional[int] = None
        for idx, h in enumerate(headers):
            if any(target == h or target in h for target in header_names):
                col_idx = idx
                break
        if col_idx is None:
            continue
        for row in table[1:]:
            if col_idx < len(row):
                sym = _clean_us_ticker(row[col_idx])
                if sym:
                    found.append(sym)
        if len(found) >= 30:
            break
    return found


def _fetch_sp500_csv_fallback() -> List[str]:
    txt = _fetch_text("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv")
    if not txt:
        return []
    out: List[str] = []
    try:
        reader = csv.DictReader(io.StringIO(txt))
        for row in reader:
            sym = _clean_us_ticker(row.get("Symbol", ""))
            if sym:
                out.append(sym)
    except Exception:
        return []
    return out


def _dedup(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for raw in items:
        t = (raw or "").strip().upper()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def parse_custom_tickers(custom_tickers: str) -> List[str]:
    if not custom_tickers:
        return []
    raw_items = re.split(r"[,;\s]+", custom_tickers.strip())
    out: List[str] = []
    for item in raw_items:
        t = item.strip().upper()
        if t and re.match(r"^[A-Z0-9\.\-\^]{1,12}$", t):
            out.append(t)
    return _dedup(out)


def detect_region(ticker: str) -> str:
    t = (ticker or "").strip().upper()
    for suf, reg in REGION_BY_SUFFIX.items():
        if t.endswith(suf):
            return reg
    if t in KNOWN_GLOBAL_ADRS:
        return "Global"
    return "EE. UU."


def _load_us_indices(include_smallcap: bool = True) -> List[str]:
    sp500 = _fetch_wikipedia_tickers("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    if len(sp500) < 300:
        sp500 = _fetch_sp500_csv_fallback() or sp500

    ndx100 = _fetch_wikipedia_tickers("https://en.wikipedia.org/wiki/Nasdaq-100")
    sp400 = _fetch_wikipedia_tickers("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies")
    sp600 = _fetch_wikipedia_tickers("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies") if include_smallcap else []

    combined = _dedup(sp500 + ndx100 + sp400 + sp600)
    if len(combined) < 400:
        combined = _dedup(combined + ELITE_CORE + US_FALLBACK_EXTRA)
    return combined


def get_universe(
    mode: str = "rotacion_diaria",
    custom_tickers: str = "",
    rotation_batch_size: int = 220,
) -> List[str]:
    """Devuelve la lista de tickers a escanear según el modo elegido."""
    m = (mode or "rotacion_diaria").strip().lower()
    extras = parse_custom_tickers(custom_tickers)

    if m == "custom":
        return extras if extras else _dedup(ELITE_CORE[:20])

    if m == "elite200":
        return _dedup(ELITE_CORE + EUROPE_EXPANDED[:80] + extras)

    if m == "sp500":
        sp500 = _fetch_wikipedia_tickers("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        if len(sp500) < 300:
            sp500 = _fetch_sp500_csv_fallback() or US_FALLBACK_EXTRA
        return _dedup(sp500 + extras)

    if m == "global_1000":
        us_large_mid = _load_us_indices(include_smallcap=False)
        return _dedup(ELITE_CORE + us_large_mid + EUROPE_EXPANDED + INTL_EXPANDED[:60] + extras)

    if m == "global_mega":
        us_all = _load_us_indices(include_smallcap=True)
        return _dedup(ELITE_CORE + us_all + EUROPE_EXPANDED + INTL_EXPANDED + extras)

    # Default: rotacion_diaria (núcleo fijo + lote diario rotativo del universo global ~2.000+)
    us_all = _load_us_indices(include_smallcap=True)
    full_global = _dedup(us_all + EUROPE_EXPANDED + INTL_EXPANDED)
    core_set = set(ELITE_CORE)
    rot_pool = [t for t in full_global if t not in core_set]

    rng = random.Random(20261125)
    rng.shuffle(rot_pool)

    day_ordinal = datetime.now(timezone.utc).toordinal()
    batch = max(50, int(rotation_batch_size))
    if rot_pool:
        start = (day_ordinal * batch) % len(rot_pool)
        if start + batch <= len(rot_pool):
            today_slice = rot_pool[start : start + batch]
        else:
            today_slice = rot_pool[start:] + rot_pool[: (start + batch) - len(rot_pool)]
    else:
        today_slice = []

    return _dedup(ELITE_CORE + today_slice + extras)
