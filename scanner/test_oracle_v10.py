"""Batería de pruebas automatizadas (unitarias + integración) — Radar Cocodrilo v10.5.

Ejecutar desde la raíz del repo:
    python -m unittest scanner/test_oracle_v10.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

SCANNER_DIR = Path(__file__).resolve().parent
if str(SCANNER_DIR) not in sys.path:
    sys.path.insert(0, str(SCANNER_DIR))

import oracle_v10 as o10


def _make_rev_estimate(growth_0y: float = 0.08, growth_1y: float = 0.09) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "avg": [540e6, 588e6],
            "yearAgoRevenue": [500e6, 540e6],
            "growth": [growth_0y, growth_1y],
        },
        index=["0y", "+1y"],
    )


class DummyTickerFull:
    def __init__(self, rev_estimate: pd.DataFrame | None = None, has_rev_est: bool = True):
        self.financials = pd.DataFrame({
            "2025-12-31": [120, 500, 300, 150, 10, 100],
            "2024-12-31": [100, 450, 265, 130, 10, 102],
            "2023-12-31": [90, 410, 240, 118, 10, 105],
        }, index=["Net Income", "Total Revenue", "Gross Profit", "EBIT", "Interest Expense", "Diluted Average Shares"])
        self.balance_sheet = pd.DataFrame({
            "2025-12-31": [1000, 100, 400, 150],
            "2024-12-31": [980, 120, 360, 160],
        }, index=["Total Assets", "Long Term Debt", "Current Assets", "Current Liabilities"])
        self.cashflow = pd.DataFrame({
            "2025-12-31": [140],
            "2024-12-31": [115],
        }, index=["Operating Cash Flow"])
        self.revenue_estimate = rev_estimate if rev_estimate is not None else (_make_rev_estimate() if has_rev_est else None)


class DummyTickerPartialPiotroski:
    """Solo 6 señales disponibles (de 9) y las 6 positivas -> NO debe reescalarse a 9/9."""
    def __init__(self):
        self.financials = pd.DataFrame({
            "2025-12-31": [120, 500, 150],
            "2024-12-31": [100, 450, 130],
        }, index=["Net Income", "Total Revenue", "EBIT"])
        self.balance_sheet = pd.DataFrame({
            "2025-12-31": [1000, 100],
            "2024-12-31": [980, 120],
        }, index=["Total Assets", "Long Term Debt"])
        self.cashflow = pd.DataFrame({
            "2025-12-31": [140],
            "2024-12-31": [115],
        }, index=["Operating Cash Flow"])
        self.revenue_estimate = _make_rev_estimate()


def _base_oro_record() -> dict:
    return {
        "ticker": "ACME",
        "region": "EE. UU.",
        "currency": "USD",
        "financial_currency": "USD",
        "status": "OPORTUNIDAD",
        "opportunity_tier": "GANGA",
        "sec_status": "MATCH",
        "sec_verified": True,
        "is_cyclical_risk": False,
        "forward_ok": True,
        "piotroski_score": 8,
        "piotroski_max": 9,
        "interest_coverage": 12.0,
        "net_debt_ebitda": 0.5,
        "ai_verdict": "TEMPORAL",
        "trap_risk_score": 3,
    }


class TestOracleV103(unittest.TestCase):
    def setUp(self):
        o10._sec_cik_map = None
        o10._sec_cik_error = None

    # --- 1) PIOTROSKI SIN REESCALADO ---
    def test_piotroski_no_rescaling_blocks_oro_puro(self):
        chk = o10.compute_oracle_checks(
            DummyTickerPartialPiotroski(),
            {"trailingEps": 5.0, "forwardEps": 5.5, "revenueGrowth": 0.10},
            {"sector_profile": "general", "status": "OPORTUNIDAD"},
        )
        self.assertEqual(chk["piotroski_score"], 6)
        self.assertEqual(chk["piotroski_max"], 6)
        rec = dict(_base_oro_record(), piotroski_score=chk["piotroski_score"], piotroski_max=chk["piotroski_max"])
        self.assertFalse(o10.is_oro_puro(rec))

    def test_piotroski_full_passes(self):
        chk = o10.compute_oracle_checks(
            DummyTickerFull(),
            {"trailingEps": 5.0, "forwardEps": 5.5, "revenueGrowth": 0.10},
            {"sector_profile": "general", "status": "OPORTUNIDAD"},
        )
        self.assertEqual(chk["piotroski_score"], 9)
        self.assertEqual(chk["piotroski_max"], 9)
        self.assertIs(chk["forward_ok"], True)
        self.assertEqual(chk["forward_rev_growth_pct"], 8.0)

    # --- 2) FORWARD CHECK REAL (HISTÓRICO VS PREVISIÓN FUTURA) ---
    def test_forward_check_missing_estimates_is_none_and_blocks(self):
        chk = o10.compute_oracle_checks(
            DummyTickerFull(has_rev_est=False),
            {"trailingEps": 5.0, "forwardEps": 5.6, "revenueGrowth": 0.18},  # Histórico +18% pero SIN previsión de ventas
            {"sector_profile": "general", "status": "OPORTUNIDAD"},
        )
        self.assertEqual(chk["hist_rev_growth_pct"], 18.0)
        self.assertIsNone(chk["forward_rev_growth_pct"])
        self.assertEqual(chk["forward_rev_status"], "MISSING")
        self.assertIsNone(chk["forward_ok"])
        self.assertEqual(chk["forward_status"], "MISSING")
        self.assertFalse(o10.is_oro_puro(dict(_base_oro_record(), forward_ok=chk["forward_ok"])))

    def test_forward_check_hist_positive_but_forward_revenue_negative_fails(self):
        chk = o10.compute_oracle_checks(
            DummyTickerFull(rev_estimate=_make_rev_estimate(growth_0y=0.02, growth_1y=-0.085)),
            {"trailingEps": 5.0, "forwardEps": 5.3, "revenueGrowth": 0.15},  # Histórico +15%, pero futuro +1y -8.5%
            {"sector_profile": "general", "status": "OPORTUNIDAD"},
        )
        self.assertEqual(chk["hist_rev_growth_pct"], 15.0)
        self.assertEqual(chk["forward_rev_growth_pct"], -8.5)
        self.assertIs(chk["forward_ok"], False)
        self.assertEqual(chk["forward_status"], "FAIL")
        self.assertFalse(o10.is_oro_puro(dict(_base_oro_record(), forward_ok=chk["forward_ok"])))

    def test_forward_check_negative_eps_fails(self):
        chk = o10.compute_oracle_checks(
            DummyTickerFull(),
            {"trailingEps": 5.0, "forwardEps": 4.0, "revenueGrowth": 0.05},
            {"sector_profile": "general", "status": "OPORTUNIDAD"},
        )
        self.assertIs(chk["forward_ok"], False)
        self.assertEqual(chk["forward_status"], "FAIL")

    # --- 3) SEC EDGAR: CAPEX DEVENGADO, MULTI-ETIQUETA, 53 SEMANAS Y 10-K/A ---
    @patch.object(o10, "_load_sec_cik_map", return_value=({"ACME": "0000001234"}, None))
    @patch.object(o10, "_read_http_json")
    def test_sec_missing_capex_returns_incomplete(self, mock_http, _mock_cik):
        mock_http.return_value = {
            "facts": {
                "us-gaap": {
                    "NetCashProvidedByUsedInOperatingActivities": {
                        "units": {"USD": [{"form": "10-K", "start": "2025-01-01", "end": "2025-12-31", "filed": "2026-02-20", "val": 10e9}]}
                    }
                }
            }
        }
        rec = {
            "ticker": "ACME", "region": "EE. UU.", "currency": "USD", "financial_currency": "USD",
            "fcf_latest_date": "2025-12-31", "fcf_latest_trade": 8e9, "owner_earnings_adjusted": False,
        }
        res = o10.verify_with_sec_edgar(rec)
        self.assertEqual(res["sec_status"], "INCOMPLETE")
        self.assertIsNone(res["sec_verified"])

    @patch.object(o10, "_load_sec_cik_map", return_value=({"ACME": "0000001234"}, None))
    @patch.object(o10, "_read_http_json")
    def test_sec_unpaid_accrued_capex_is_ignored_and_incomplete(self, mock_http, _mock_cik):
        """CapitalExpendituresIncurredButNotYetPaid NO debe usarse como CapEx pagado en efectivo."""
        mock_http.return_value = {
            "facts": {
                "us-gaap": {
                    "NetCashProvidedByUsedInOperatingActivities": {
                        "units": {"USD": [{"form": "10-K", "start": "2025-01-01", "end": "2025-12-31", "filed": "2026-02-20", "val": 10e9}]}
                    },
                    "CapitalExpendituresIncurredButNotYetPaid": {
                        "units": {"USD": [{"form": "10-K", "start": "2025-01-01", "end": "2025-12-31", "filed": "2026-02-20", "val": 1.5e9}]}
                    },
                }
            }
        }
        rec = {
            "ticker": "ACME", "region": "EE. UU.", "currency": "USD", "financial_currency": "USD",
            "fcf_latest_date": "2025-12-31", "fcf_latest_trade": 8.5e9, "owner_earnings_adjusted": False,
        }
        res = o10.verify_with_sec_edgar(rec)
        self.assertEqual(res["sec_status"], "INCOMPLETE")
        self.assertIsNone(res["sec_verified"])

    @patch.object(o10, "_load_sec_cik_map", return_value=({"ACME": "0000001234"}, None))
    @patch.object(o10, "_read_http_json")
    def test_sec_multiple_capex_tags_across_years(self, mock_http, _mock_cik):
        """Empresa que usó tag 1 en 2024 y cambió a tag 2 en 2025 debe alinear correctamente 2025."""
        mock_http.return_value = {
            "facts": {
                "us-gaap": {
                    "NetCashProvidedByUsedInOperatingActivities": {
                        "units": {"USD": [
                            {"form": "10-K", "start": "2024-01-01", "end": "2024-12-31", "filed": "2025-02-20", "val": 9e9},
                            {"form": "10-K", "start": "2025-01-01", "end": "2025-12-31", "filed": "2026-02-20", "val": 11e9},
                        ]}
                    },
                    "PaymentsToAcquirePropertyPlantAndEquipment": {
                        "units": {"USD": [
                            {"form": "10-K", "start": "2024-01-01", "end": "2024-12-31", "filed": "2025-02-20", "val": 2e9},
                        ]}
                    },
                    "PaymentsToAcquireProductiveAssets": {
                        "units": {"USD": [
                            {"form": "10-K", "start": "2025-01-01", "end": "2025-12-31", "filed": "2026-02-20", "val": 3e9},
                        ]}
                    },
                }
            }
        }
        rec = {
            "ticker": "ACME", "region": "EE. UU.", "currency": "USD", "financial_currency": "USD",
            "fcf_latest_date": "2025-12-31", "fcf_latest_trade": 8.0e9, "owner_earnings_adjusted": False,
        }
        res = o10.verify_with_sec_edgar(rec)
        self.assertEqual(res["sec_status"], "MATCH")
        self.assertIs(res["sec_verified"], True)
        self.assertEqual(res["sec_fiscal_end"], "2025-12-31")

    @patch.object(o10, "_load_sec_cik_map", return_value=({"ACME": "0000001234"}, None))
    @patch.object(o10, "_read_http_json")
    def test_sec_53_week_fiscal_year_accepted(self, mock_http, _mock_cik):
        """Ejercicio fiscal de 53 semanas (370-371 días) debe ser aceptado como anual válido."""
        mock_http.return_value = {
            "facts": {
                "us-gaap": {
                    "NetCashProvidedByUsedInOperatingActivities": {
                        "units": {"USD": [
                            {"form": "10-K", "start": "2024-12-29", "end": "2026-01-03", "filed": "2026-02-25", "val": 10e9},
                        ]}
                    },
                    "PaymentsToAcquirePropertyPlantAndEquipment": {
                        "units": {"USD": [
                            {"form": "10-K", "start": "2024-12-29", "end": "2026-01-03", "filed": "2026-02-25", "val": 2e9},
                        ]}
                    },
                }
            }
        }
        rec = {
            "ticker": "ACME", "region": "EE. UU.", "currency": "USD", "financial_currency": "USD",
            "fcf_latest_date": "2026-01-03", "fcf_latest_trade": 8.1e9, "owner_earnings_adjusted": False,
        }
        res = o10.verify_with_sec_edgar(rec)
        self.assertEqual(res["sec_status"], "MATCH")
        self.assertIs(res["sec_verified"], True)
        self.assertEqual(res["sec_fiscal_end"], "2026-01-03")

    @patch.object(o10, "_load_sec_cik_map", return_value=({"ACME": "0000001234"}, None))
    @patch.object(o10, "_read_http_json")
    def test_sec_10ka_restatement_overrides_original_10k(self, mock_http, _mock_cik):
        """Una enmienda 10-K/A posterior que rebaja la caja debe prevalecer sobre el 10-K original."""
        mock_http.return_value = {
            "facts": {
                "us-gaap": {
                    "NetCashProvidedByUsedInOperatingActivities": {
                        "units": {"USD": [
                            {"form": "10-K", "start": "2025-01-01", "end": "2025-12-31", "filed": "2026-02-20", "val": 10e9},
                            {"form": "10-K/A", "start": "2025-01-01", "end": "2025-12-31", "filed": "2026-05-10", "val": 5e9},
                        ]}
                    },
                    "PaymentsToAcquirePropertyPlantAndEquipment": {
                        "units": {"USD": [
                            {"form": "10-K", "start": "2025-01-01", "end": "2025-12-31", "filed": "2026-02-20", "val": 2e9},
                        ]}
                    },
                }
            }
        }
        # Yahoo aún muestra el FCF antiguo (8B), pero el 10-K/A corrigió FCF real a 3B -> MISMATCH
        rec = {
            "ticker": "ACME", "region": "EE. UU.", "currency": "USD", "financial_currency": "USD",
            "fcf_latest_date": "2025-12-31", "fcf_latest_trade": 8.0e9, "owner_earnings_adjusted": False,
        }
        res = o10.verify_with_sec_edgar(rec)
        self.assertEqual(res["sec_status"], "MISMATCH")
        self.assertIs(res["sec_verified"], False)

    @patch.object(o10, "_load_sec_cik_map", return_value=({"ACME": "0000001234"}, None))
    @patch.object(o10, "_read_http_json")
    def test_sec_acceptable_tolerance_blocks_oro_puro(self, mock_http, _mock_cik):
        """Desviación entre 15% y 30% se clasifica como ACCEPTABLE y NO puede obtener ORO PURO."""
        mock_http.return_value = {
            "facts": {
                "us-gaap": {
                    "NetCashProvidedByUsedInOperatingActivities": {
                        "units": {"USD": [{"form": "10-K", "start": "2025-01-01", "end": "2025-12-31", "filed": "2026-02-20", "val": 10e9}]}
                    },
                    "PaymentsToAcquirePropertyPlantAndEquipment": {
                        "units": {"USD": [{"form": "10-K", "start": "2025-01-01", "end": "2025-12-31", "filed": "2026-02-20", "val": 2e9}]}
                    },
                }
            }
        }
        # SEC FCF = 8.0B, Yahoo FCF = 9.8B -> desviación 22.5% -> ACCEPTABLE (no MATCH)
        rec = {
            "ticker": "ACME", "region": "EE. UU.", "currency": "USD", "financial_currency": "USD",
            "fcf_latest_date": "2025-12-31", "fcf_latest_trade": 9.8e9, "owner_earnings_adjusted": False,
        }
        res = o10.verify_with_sec_edgar(rec)
        self.assertEqual(res["sec_status"], "ACCEPTABLE")
        self.assertIs(res["sec_verified"], True)
        self.assertEqual(res["sec_diff_pct"], 22.5)
        self.assertFalse(o10.is_oro_puro(dict(_base_oro_record(), sec_status=res["sec_status"], sec_verified=res["sec_verified"])))

    @patch.object(o10, "_load_sec_cik_map", return_value=({"ACME": "0000001234"}, None))
    @patch.object(o10, "_read_http_json")
    def test_sec_over_30_pct_returns_mismatch(self, mock_http, _mock_cik):
        """Desviación >30% debe devolver MISMATCH y sec_verified=False."""
        mock_http.return_value = {
            "facts": {
                "us-gaap": {
                    "NetCashProvidedByUsedInOperatingActivities": {
                        "units": {"USD": [{"form": "10-K", "start": "2025-01-01", "end": "2025-12-31", "filed": "2026-02-20", "val": 10e9}]}
                    },
                    "PaymentsToAcquirePropertyPlantAndEquipment": {
                        "units": {"USD": [{"form": "10-K", "start": "2025-01-01", "end": "2025-12-31", "filed": "2026-02-20", "val": 2e9}]}
                    },
                }
            }
        }
        rec = {
            "ticker": "ACME", "region": "EE. UU.", "currency": "USD", "financial_currency": "USD",
            "fcf_latest_date": "2025-12-31", "fcf_latest_trade": 11.0e9, "owner_earnings_adjusted": False,
        }
        res = o10.verify_with_sec_edgar(rec)
        self.assertEqual(res["sec_status"], "MISMATCH")
        self.assertIs(res["sec_verified"], False)

    @patch.object(o10, "_load_sec_cik_map", return_value=({"ACME": "0000001234"}, None))
    @patch.object(o10, "_read_http_json")
    def test_sec_fiscal_period_mismatch_blocked(self, mock_http, _mock_cik):
        mock_http.return_value = {
            "facts": {
                "us-gaap": {
                    "NetCashProvidedByUsedInOperatingActivities": {
                        "units": {"USD": [{"form": "10-K", "start": "2024-01-01", "end": "2024-12-31", "filed": "2025-02-20", "val": 10e9}]}
                    },
                    "PaymentsToAcquirePropertyPlantAndEquipment": {
                        "units": {"USD": [{"form": "10-K", "start": "2024-01-01", "end": "2024-12-31", "filed": "2025-02-20", "val": 2e9}]}
                    },
                }
            }
        }
        rec = {
            "ticker": "ACME", "region": "EE. UU.", "currency": "USD", "financial_currency": "USD",
            "fcf_latest_date": "2025-12-31", "fcf_latest_trade": 8e9, "owner_earnings_adjusted": False,
        }
        res = o10.verify_with_sec_edgar(rec)
        self.assertEqual(res["sec_status"], "PERIOD_MISMATCH")
        self.assertIsNone(res["sec_verified"])

    @patch.object(o10, "_read_http_json", side_effect=RuntimeError("HTTP 403 Forbidden"))
    def test_sec_cik_download_failure_returns_error_not_no_cik(self, _mock_http):
        rec = {"ticker": "ADBE", "region": "EE. UU.", "currency": "USD", "financial_currency": "USD"}
        res = o10.verify_with_sec_edgar(rec)
        self.assertEqual(res["sec_status"], "ERROR")
        self.assertIsNone(res["sec_verified"])

    # --- 4) GEMINI: VALIDACIÓN ESTRICTA DE ENTERO Y CAMPOS OBLIGATORIOS ---
    def test_gemini_decimal_and_incomplete_payload_rejected(self):
        base_ok = {
            "fiscal": "Riesgo real de márgenes a la baja por competencia.",
            "abogado": "Marca líder con caja neta y clientes recurrentes.",
            "comprobar_5min": "Revisar guía trimestral y crecimiento orgánico.",
            "riesgo_trampa": 3,
            "veredicto": "TEMPORAL",
        }
        # Decimal 3.7 debe ser rechazado
        v_dec, e_dec = o10.validate_tribunal_payload(dict(base_ok, riesgo_trampa=3.7))
        self.assertIsNone(v_dec)
        self.assertIn("entero", e_dec)

        # Float entero 3.0 o booleano o string también rechazados
        for bad_val in (3.0, True, "3", None, -1, 11):
            v_b, e_b = o10.validate_tribunal_payload(dict(base_ok, riesgo_trampa=bad_val))
            self.assertIsNone(v_b, f"Debió rechazar riesgo_trampa={bad_val!r}")
            self.assertIsNotNone(e_b)

        # Incompleto (sin campos de texto)
        v_inc, e_inc = o10.validate_tribunal_payload({"veredicto": "TEMPORAL", "riesgo_trampa": 3})
        self.assertIsNone(v_inc)
        self.assertIsNotNone(e_inc)

        # Completo y entero exacto
        v_ok, e_ok = o10.validate_tribunal_payload(base_ok)
        self.assertIsNone(e_ok)
        self.assertEqual(v_ok["veredicto"], "TEMPORAL")
        self.assertEqual(v_ok["riesgo_trampa"], 3)

    # --- 5) ORO PURO TOLERANCIA CERO ---
    def test_oro_puro_strict_gates(self):
        good = _base_oro_record()
        self.assertTrue(o10.is_oro_puro(good))
        self.assertFalse(o10.is_oro_puro(dict(good, sec_verified=None)))
        self.assertFalse(o10.is_oro_puro(dict(good, sec_status="ACCEPTABLE")))
        self.assertFalse(o10.is_oro_puro(dict(good, sec_status="PERIOD_MISMATCH")))
        self.assertFalse(o10.is_oro_puro(dict(good, forward_ok=None)))
        self.assertFalse(o10.is_oro_puro(dict(good, forward_ok=False)))
        self.assertFalse(o10.is_oro_puro(dict(good, trap_risk_score=None)))
        self.assertFalse(o10.is_oro_puro(dict(good, trap_risk_score=3.5)))
        self.assertFalse(o10.is_oro_puro(dict(good, piotroski_score=6, piotroski_max=6)))
        self.assertFalse(o10.is_oro_puro(dict(good, ai_verdict="EVIDENCIA_INSUFICIENTE")))

    # --- 6) PRUEBA DE INTEGRACIÓN COMPLETA (ESCANEO -> ORÁCULO -> JSON PUBLICADO) ---
    def test_end_to_end_scan_to_published_json(self):
        """Prueba de integración de principio a fin: analyze_ticker -> main() -> JSONs coherentes v10.3."""
        try:
            import yfinance  # noqa: F401
        except ImportError:
            stub_yf = types.ModuleType("yfinance")
            stub_yf.Ticker = object
            sys.modules["yfinance"] = stub_yf
        try:
            import universe  # noqa: F401
        except ImportError:
            stub_u = types.ModuleType("universe")
            stub_u.detect_region = lambda t: "EE. UU."
            stub_u.get_universe = lambda *a, **k: ["ACME"]
            stub_u.parse_custom_tickers = lambda s: ["ACME"]
            sys.modules["universe"] = stub_u

        import buffett_scanner as bs

        class MockYFTicker:
            def __init__(self, sym: str):
                self.sym = sym
                self.info = {
                    "shortName": "Acme Quality Corp",
                    "currency": "USD",
                    "financialCurrency": "USD",
                    "currentPrice": 50.0,
                    "marketCap": 50e9,
                    "sharesOutstanding": 1e9,
                    "fiftyTwoWeekHigh": 85.0,
                    "fiftyTwoWeekLow": 48.0,
                    "sector": "Technology",
                    "industry": "Software",
                    "operatingMargins": 0.30,
                    "revenueGrowth": 0.12,
                    "earningsGrowth": 0.12,
                    "trailingPE": 14.0,
                    "forwardPE": 12.0,
                    "trailingEps": 3.5,
                    "forwardEps": 4.1,
                    "totalDebt": 2e9,
                    "totalCash": 6e9,
                    "ebitda": 8e9,
                }
                self.financials = pd.DataFrame({
                    "2025-12-31": [4.0e9, 20e9, 15e9, 6.0e9, 0.2e9, 1.0e9, 5.0e9, 1.0e9],
                    "2024-12-31": [3.5e9, 18e9, 13.4e9, 5.3e9, 0.2e9, 1.02e9, 4.4e9, 0.9e9],
                    "2023-12-31": [3.1e9, 16e9, 11.8e9, 4.7e9, 0.2e9, 1.05e9, 3.9e9, 0.8e9],
                }, index=[
                    "Net Income", "Total Revenue", "Gross Profit", "EBIT", "Interest Expense",
                    "Diluted Average Shares", "Pretax Income", "Tax Provision",
                ])
                self.balance_sheet = pd.DataFrame({
                    "2025-12-31": [30e9, 2e9, 2e9, 18e9, 6e9, 12e9, 4e9],
                    "2024-12-31": [28e9, 2.5e9, 2.5e9, 16e9, 5e9, 10e9, 4.2e9],
                }, index=[
                    "Total Assets", "Total Debt", "Long Term Debt", "Stockholders Equity",
                    "Cash And Cash Equivalents", "Current Assets", "Current Liabilities",
                ])
                self.cashflow = pd.DataFrame({
                    "2025-12-31": [5.5e9, -0.5e9, 5.0e9, 0.1e9],
                    "2024-12-31": [5.1e9, -0.5e9, 4.6e9, 0.1e9],
                    "2023-12-31": [4.7e9, -0.5e9, 4.2e9, 0.1e9],
                }, index=["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow", "Stock Based Compensation"])
                self.revenue_estimate = _make_rev_estimate(growth_0y=0.09, growth_1y=0.10)
                self.insider_transactions = pd.DataFrame()
                self.news = [{"title": "Acme posts strong cloud growth despite macro pullback", "publisher": "Reuters"}]

            def history(self, period="5d"):
                return pd.DataFrame({"Close": [0.92]})

        sec_ok = {
            "sec_status": "MATCH",
            "sec_verified": True,
            "sec_fiscal_end": "2025-12-31",
            "sec_note": "Verificado 10-K SEC 2025-12-31 (desviación 2% en caja)",
        }
        tribunal_ok = (
            "FISCAL: Competencia creciente.\nABOGADO: Caja neta y márgenes 30%.\n"
            "COMPROBAR (5 min): Guía Q1.\nRIESGO_TRAMPA: 3/10\nVEREDICTO: TEMPORAL",
            "TEMPORAL",
            3,
            None,
        )

        with tempfile.TemporaryDirectory() as td:
            tmp_dir = Path(td)
            with (
                patch.object(bs, "DATA_DIR", tmp_dir),
                patch.object(bs, "OUT_LATEST_JSON", tmp_dir / "latest-scan.json"),
                patch.object(bs, "OUT_DAILY_JSON", tmp_dir / "daily-scan.json"),
                patch.object(bs, "OUT_WEEKLY_JSON", tmp_dir / "weekly-scan.json"),
                patch.object(bs, "OUT_HISTORY_JSON", tmp_dir / "history.json"),
                patch.object(bs, "UNIVERSE_MODE", "rotacion_diaria"),
                patch.object(bs, "SHARD_TOTAL", 1),
                patch.object(bs, "MERGE_DIR", ""),
                patch.object(bs, "URGENT_ONLY", False),
                patch.object(bs, "GEMINI_API_KEY", "fake-key"),
                patch.object(bs, "get_universe", return_value=["ACME"]),
                patch.object(bs, "get_market_context", return_value={"vix": 18.0, "us10y_pct": 4.2, "sp500_drawdown_pct": -3.0, "regime": "NORMAL", "summary": "OK"}),
                patch.object(bs, "verify_with_sec_edgar", return_value=sec_ok),
                patch.object(bs, "run_tribunal_ia", return_value=tribunal_ok),
                patch.object(bs, "send_telegram", return_value=None),
                patch.object(bs.yf, "Ticker", side_effect=MockYFTicker),
                patch.object(bs.time, "sleep", return_value=None),
            ):
                bs.main()

            latest_path = tmp_dir / "latest-scan.json"
            daily_path = tmp_dir / "daily-scan.json"
            hist_path = tmp_dir / "history.json"
            self.assertTrue(latest_path.exists())
            self.assertTrue(daily_path.exists())
            self.assertTrue(hist_path.exists())

            published = json.loads(latest_path.read_text(encoding="utf-8"))
            self.assertEqual(published["scanner_version"], "v10.5-oraculo")
            self.assertIn("fx_rates_eur", published)
            self.assertEqual(len(published["companies"]), 1)

            c = published["companies"][0]
            self.assertEqual(c["ticker"], "ACME")
            self.assertEqual(c["oracle_version"], "v10.5-oraculo")
            self.assertEqual(c["fcf_latest_date"], "2025-12-31")
            self.assertIs(c["forward_ok"], True)
            self.assertEqual(c["forward_rev_growth_pct"], 9.0)
            self.assertEqual(c["piotroski_score"], 9)
            self.assertEqual(c["piotroski_max"], 9)
            self.assertTrue(c["oro_puro"])


    # --- 7) PIOTROSKI CLÁSICO (ACTIVOS MEDIOS Y TOLERANCIA CERO A AUSENCIAS) ---
    def test_piotroski_classic_mean_assets_and_missing_data(self):
        t_3y = DummyTickerFull()
        t_3y.balance_sheet = pd.DataFrame({
            "2025-12-31": [1000.0, 100.0, 400.0, 150.0],
            "2024-12-31": [1000.0, 120.0, 360.0, 160.0],
            "2023-12-31": [1000.0, 140.0, 340.0, 170.0],
        }, index=["Total Assets", "Long Term Debt", "Current Assets", "Current Liabilities"])
        chk = o10.compute_oracle_checks(
            t_3y,
            {"trailingEps": 5.0, "forwardEps": 5.5},
            {"sector_profile": "general", "status": "OPORTUNIDAD"},
        )
        self.assertEqual(chk["piotroski_classic_max"], 9)
        self.assertEqual(chk["piotroski_classic_score"], 9)
        self.assertEqual(chk["piotroski_classic_coverage"], "9/9 señales disponibles")

        # Si falta Current Liabilities en balance, no se puntúa ni penaliza y el denominador baja a 8/9
        t_missing = DummyTickerFull()
        t_missing.balance_sheet = pd.DataFrame({
            "2025-12-31": [1000.0, 100.0, 400.0],
            "2024-12-31": [1000.0, 120.0, 360.0],
            "2023-12-31": [1000.0, 140.0, 340.0],
        }, index=["Total Assets", "Long Term Debt", "Current Assets"])
        t_missing.cashflow = pd.DataFrame({
            "2025-12-31": [140.0],
            "2024-12-31": [115.0],
            "2023-12-31": [105.0],
        }, index=["Operating Cash Flow"])
        chk_m = o10.compute_oracle_checks(
            t_missing,
            {"trailingEps": 5.0, "forwardEps": 5.5},
            {"sector_profile": "general", "status": "OPORTUNIDAD"},
        )
        self.assertEqual(chk_m["piotroski_classic_max"], 8)
        self.assertEqual(chk_m["piotroski_classic_coverage"], "8/9 señales disponibles")

    # --- 8) CUADRANTES CALIDAD VS VALORACIÓN ---
    def test_quality_valuation_quadrants(self):
        q1 = o10.classify_quality_valuation_quadrant({
            "quality_score": 90, "passes_quality": True, "hard_fails": [],
            "sec_status": "MATCH", "sec_verified": True, "in_dcf_zone": True,
            "discount_vs_base_pct": 28.0, "forward_pe": 12.0, "p_fcf": 11.0,
        })
        self.assertEqual(q1["quadrant_code"], "EXCELENTE_PRECIO_RAZONABLE")

        q2 = o10.classify_quality_valuation_quadrant({
            "quality_score": 92, "passes_quality": True, "hard_fails": [],
            "sec_status": "MATCH", "sec_verified": True, "in_dcf_zone": False,
            "discount_vs_base_pct": -15.0, "forward_pe": 34.0, "p_fcf": 32.0,
        })
        self.assertEqual(q2["quadrant_code"], "EXCELENTE_CARA")

        q3 = o10.classify_quality_valuation_quadrant({
            "quality_score": 65, "passes_quality": False, "hard_fails": [],
            "sec_status": "MATCH", "sec_verified": True, "in_dcf_zone": True,
            "discount_vs_base_pct": 40.0, "forward_pe": 8.0, "p_fcf": 7.0,
        })
        self.assertEqual(q3["quadrant_code"], "MEDIOCRE_APARENTEMENTE_BARATA")

        q4 = o10.classify_quality_valuation_quadrant({
            "quality_score": 45, "passes_quality": False, "hard_fails": ["Deuda elevada"],
            "sec_status": "MISMATCH", "sec_verified": False, "in_dcf_zone": False,
            "discount_vs_base_pct": 10.0, "forward_pe": 14.0, "p_fcf": 13.0,
        })
        self.assertEqual(q4["quadrant_code"], "PROBLEMAS_FUNDAMENTALES")

    # --- 9) HISTORIAL PROSPECTIVO: CONSERVA FECHA/PRECIO ORIGINAL Y RELLENA BENCHMARKS ---
    def test_history_preserves_original_alert_and_populates_benchmarks(self):
        try:
            import yfinance  # noqa: F401
        except ImportError:
            stub_yf = types.ModuleType("yfinance")
            stub_yf.Ticker = object
            sys.modules["yfinance"] = stub_yf
        try:
            import universe  # noqa: F401
        except ImportError:
            stub_u = types.ModuleType("universe")
            stub_u.detect_region = lambda t: "EE. UU."
            stub_u.get_universe = lambda *a, **k: []
            stub_u.parse_custom_tickers = lambda s: []
            sys.modules["universe"] = stub_u

        import buffett_scanner as bs

        class MockHistTicker:
            def __init__(self, sym):
                self.sym = sym
                self.dividends = pd.Series(dtype=float)

            def history(self, period="5d", start=None, auto_adjust=False):
                if start is not None:
                    return pd.DataFrame({"Close": [100.0, 112.0]})
                return pd.DataFrame({"Close": [240.0]})

        with tempfile.TemporaryDirectory() as td:
            hist_file = Path(td) / "history.json"
            hist_file.write_text(json.dumps({
                "updated_at": "2026-10-01T10:00:00+00:00",
                "entries": [{
                    "ticker": "EXPE",
                    "name": "Expedia Group",
                    "currency": "USD",
                    "tier": "ORO_PURO",
                    "alert_date": "2026-10-01",
                    "alert_price": 200.0,
                    "quality_score": 96,
                    "last_price": 200.0,
                    "return_pct": 0.0,
                }]
            }), encoding="utf-8")

            with (
                patch.object(bs, "OUT_HISTORY_JSON", hist_file),
                patch.object(bs.yf, "Ticker", side_effect=MockHistTicker),
            ):
                bs.update_alert_history([])

            saved = json.loads(hist_file.read_text(encoding="utf-8"))
            e = saved["entries"][0]
            self.assertEqual(e["alert_date"], "2026-10-01")
            self.assertEqual(e["alert_price"], 200.0)
            self.assertEqual(e["last_price"], 240.0)
            self.assertEqual(e["return_pct"], 20.0)
            self.assertEqual(e["benchmarks"]["sp500_pct"], 12.0)
            self.assertEqual(e["benchmarks"]["msci_world_pct"], 12.0)

    # --- 10) FCF NORMALIZACIÓN: CRECIMIENTO SOSTENIDO (NVDA) VS PICO DE CIRCULANTE (EXPE) ---
    def test_fcf_sustained_growth_backed_by_ni_not_capped_vs_spike_capped(self):
        try:
            import yfinance  # noqa: F401
        except ImportError:
            stub_yf = types.ModuleType("yfinance")
            stub_yf.Ticker = object
            sys.modules["yfinance"] = stub_yf
        try:
            import universe  # noqa: F401
        except ImportError:
            stub_u = types.ModuleType("universe")
            stub_u.detect_region = lambda t: "EE. UU."
            stub_u.get_universe = lambda *a, **k: []
            stub_u.parse_custom_tickers = lambda s: []
            sys.modules["universe"] = stub_u

        import buffett_scanner as bs

        class GrowthTicker:
            def __init__(self):
                self.financials = pd.DataFrame({
                    "2026-01-31": [72e9],
                    "2025-01-31": [30e9],
                    "2024-01-31": [10e9],
                }, index=["Net Income"])
                self.balance_sheet = pd.DataFrame()
                self.cashflow = pd.DataFrame({
                    "2026-01-31": [60e9],
                    "2025-01-31": [27e9],
                    "2024-01-31": [8e9],
                    "2023-01-31": [4e9],
                }, index=["Free Cash Flow"])

        res_growth = bs.compute_roic_and_statements(GrowthTicker(), {})
        self.assertFalse(res_growth["fcf_spike_capped"])
        self.assertEqual(res_growth["fcf_normalized"], 60e9)

        class SpikeTicker:
            def __init__(self):
                self.financials = pd.DataFrame({
                    "2025-12-31": [1.2e9],
                    "2024-12-31": [1.0e9],
                    "2023-12-31": [0.9e9],
                }, index=["Net Income"])
                self.balance_sheet = pd.DataFrame()
                self.cashflow = pd.DataFrame({
                    "2025-12-31": [4.5e9],
                    "2024-12-31": [1.5e9],
                    "2023-12-31": [1.6e9],
                    "2022-12-31": [1.4e9],
                }, index=["Free Cash Flow"])

        res_spike = bs.compute_roic_and_statements(SpikeTicker(), {})
        self.assertTrue(res_spike["fcf_spike_capped"])
        self.assertLess(res_spike["fcf_normalized"], 4.5e9)

    # --- 11) REINTENTO ANTE ERROR TRANSITORIO 401 INVALID CRUMB ---
    def test_401_invalid_crumb_retried_successfully(self):
        try:
            import yfinance  # noqa: F401
        except ImportError:
            stub_yf = types.ModuleType("yfinance")
            stub_yf.Ticker = object
            sys.modules["yfinance"] = stub_yf
        try:
            import universe  # noqa: F401
        except ImportError:
            stub_u = types.ModuleType("universe")
            stub_u.detect_region = lambda t: "EE. UU."
            stub_u.get_universe = lambda *a, **k: []
            stub_u.parse_custom_tickers = lambda s: []
            sys.modules["universe"] = stub_u

        import buffett_scanner as bs

        calls = {"n": 0}
        def fake_analyze(sym):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("HTTP Error 401: Invalid Crumb")
            return {"ticker": sym, "status": "VIGILAR"}

        with (
            patch.object(bs, "analyze_ticker", side_effect=fake_analyze),
            patch.object(bs.time, "sleep", return_value=None),
        ):
            out = bs.analyze_ticker_with_retry("NVDA", retries=2)
        self.assertIsNotNone(out)
        self.assertEqual(out["ticker"], "NVDA")
        self.assertEqual(calls["n"], 2)

    # --- 12) NOTIFICACIONES PUSH FIREBASE FCM (HTTP v1) ---
    def test_fcm_urgent_push_payload_and_dispatch(self):
        try:
            import yfinance  # noqa: F401
        except ImportError:
            stub_yf = types.ModuleType("yfinance")
            stub_yf.Ticker = object
            sys.modules["yfinance"] = stub_yf
        try:
            import universe  # noqa: F401
        except ImportError:
            stub_u = types.ModuleType("universe")
            stub_u.detect_region = lambda t: "EE. UU."
            stub_u.get_universe = lambda *a, **k: []
            stub_u.parse_custom_tickers = lambda s: []
            sys.modules["universe"] = stub_u

        import buffett_scanner as bs

        captured = []

        class FakeResp:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b'{"name":"projects/cocodrilopro-64220/messages/1"}'

        def fake_urlopen(req, timeout=15):
            captured.append((req.full_url, dict(req.header_items()), json.loads(req.data.decode("utf-8"))))
            return FakeResp()

        fake_sa = json.dumps({"project_id": "cocodrilopro-64220", "client_email": "bot@test.iam.gserviceaccount.com"})
        sample_rec = {
            "ticker": "DECK",
            "name": "Deckers Outdoor Corporation",
            "price": 86.08,
            "currency": "USD",
            "drawdown_52w_pct": -30.0,
            "quality_score": 100,
            "discount_vs_base_pct": 57.8,
            "ai_verdict": "TEMPORAL",
            "oro_puro": True,
        }

        with (
            patch.object(bs, "FIREBASE_SERVICE_ACCOUNT", fake_sa),
            patch.object(bs, "FIREBASE_DEVICE_TOKEN", "tok_phone_1, tok_phone_2"),
            patch.object(bs, "_get_fcm_access_token", return_value="fake_oauth_bearer"),
            patch.object(bs.urllib.request, "urlopen", side_effect=fake_urlopen),
        ):
            sent = bs.send_fcm_urgent_alerts([sample_rec])

        self.assertEqual(sent, 2)
        self.assertEqual(len(captured), 2)
        url0, hdrs0, body0 = captured[0]
        self.assertIn("/projects/cocodrilopro-64220/messages:send", url0)
        self.assertEqual(body0["message"]["token"], "tok_phone_1")
        self.assertIn("DECK", body0["message"]["notification"]["title"])
        self.assertEqual(body0["message"]["data"]["ticker"], "DECK")
        self.assertEqual(body0["message"]["data"]["url"], "./?ticker=DECK")

if __name__ == "__main__":
    unittest.main()
