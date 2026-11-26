"""Batería de pruebas automatizadas — Radar Cocodrilo v10.2 (Oráculo Tolerancia Cero).

Ejecutar desde la raíz del repo:
    python -m unittest scanner/test_oracle_v10.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

SCANNER_DIR = Path(__file__).resolve().parent
if str(SCANNER_DIR) not in sys.path:
    sys.path.insert(0, str(SCANNER_DIR))

import oracle_v10 as o10


class DummyTickerFull:
    def __init__(self):
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


def _base_oro_record() -> dict:
    return {
        "ticker": "ACME",
        "region": "EE. UU.",
        "currency": "USD",
        "financial_currency": "USD",
        "status": "OPORTUNIDAD",
        "opportunity_tier": "GANGA",
        "sec_status": "VERIFIED",
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


class TestOracleV102(unittest.TestCase):
    def setUp(self):
        o10._sec_cik_map = None
        o10._sec_cik_error = None

    def test_piotroski_no_rescaling_blocks_oro_puro(self):
        chk = o10.compute_oracle_checks(
            DummyTickerPartialPiotroski(),
            {"trailingEps": 5.0, "forwardEps": 5.5, "revenueGrowth": 0.10},
            {"sector_profile": "general"},
        )
        self.assertEqual(chk["piotroski_score"], 6)
        self.assertEqual(chk["piotroski_max"], 6)
        rec = dict(_base_oro_record(), piotroski_score=chk["piotroski_score"], piotroski_max=chk["piotroski_max"])
        self.assertFalse(o10.is_oro_puro(rec))

    def test_piotroski_full_passes(self):
        chk = o10.compute_oracle_checks(
            DummyTickerFull(),
            {"trailingEps": 5.0, "forwardEps": 5.5, "revenueGrowth": 0.10},
            {"sector_profile": "general"},
        )
        self.assertEqual(chk["piotroski_score"], 9)
        self.assertEqual(chk["piotroski_max"], 9)
        self.assertIs(chk["forward_ok"], True)

    def test_forward_check_missing_estimates_is_none_and_blocks(self):
        chk = o10.compute_oracle_checks(DummyTickerFull(), {}, {"sector_profile": "general"})
        self.assertIsNone(chk["forward_ok"])
        self.assertEqual(chk["forward_status"], "MISSING")
        rec = dict(_base_oro_record(), forward_ok=chk["forward_ok"])
        self.assertFalse(o10.is_oro_puro(rec))

    def test_forward_check_negative_eps_fails(self):
        chk = o10.compute_oracle_checks(
            DummyTickerFull(),
            {"trailingEps": 5.0, "forwardEps": 4.0, "revenueGrowth": 0.05},
            {"sector_profile": "general"},
        )
        self.assertIs(chk["forward_ok"], False)
        self.assertEqual(chk["forward_status"], "FAIL")

    @patch.object(o10, "_load_sec_cik_map", return_value=({"ACME": "0000001234"}, None))
    @patch.object(o10, "_read_http_json")
    def test_sec_missing_capex_returns_incomplete(self, mock_http, _mock_cik):
        mock_http.return_value = {
            "facts": {
                "us-gaap": {
                    "NetCashProvidedByUsedInOperatingActivities": {
                        "units": {"USD": [{"form": "10-K", "start": "2025-01-01", "end": "2025-12-31", "val": 10e9}]}
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
    def test_sec_fiscal_period_mismatch_blocked(self, mock_http, _mock_cik):
        mock_http.return_value = {
            "facts": {
                "us-gaap": {
                    "NetCashProvidedByUsedInOperatingActivities": {
                        "units": {"USD": [{"form": "10-K", "start": "2024-01-01", "end": "2024-12-31", "val": 10e9}]}
                    },
                    "PaymentsToAcquirePropertyPlantAndEquipment": {
                        "units": {"USD": [{"form": "10-K", "start": "2024-01-01", "end": "2024-12-31", "val": 2e9}]}
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

    @patch.object(o10, "_load_sec_cik_map", return_value=({"ACME": "0000001234"}, None))
    @patch.object(o10, "_read_http_json")
    def test_sec_aligned_fiscal_year_verifies(self, mock_http, _mock_cik):
        mock_http.return_value = {
            "facts": {
                "us-gaap": {
                    "NetCashProvidedByUsedInOperatingActivities": {
                        "units": {"USD": [{"form": "10-K", "start": "2025-01-01", "end": "2025-12-31", "val": 10e9}]}
                    },
                    "PaymentsToAcquirePropertyPlantAndEquipment": {
                        "units": {"USD": [{"form": "10-K", "start": "2025-01-01", "end": "2025-12-31", "val": 2e9}]}
                    },
                }
            }
        }
        rec = {
            "ticker": "ACME", "region": "EE. UU.", "currency": "USD", "financial_currency": "USD",
            "fcf_latest_date": "2025-12-31", "fcf_latest_trade": 7.8e9, "owner_earnings_adjusted": False,
        }
        res = o10.verify_with_sec_edgar(rec)
        self.assertEqual(res["sec_status"], "VERIFIED")
        self.assertIs(res["sec_verified"], True)
        self.assertEqual(res["sec_fiscal_end"], "2025-12-31")

    @patch.object(o10, "_read_http_json", side_effect=RuntimeError("HTTP 403 Forbidden"))
    def test_sec_cik_download_failure_returns_error_not_no_cik(self, _mock_http):
        rec = {"ticker": "ADBE", "region": "EE. UU.", "currency": "USD", "financial_currency": "USD"}
        res = o10.verify_with_sec_edgar(rec)
        self.assertEqual(res["sec_status"], "ERROR")
        self.assertIsNone(res["sec_verified"])

    def test_gemini_incomplete_payload_rejected(self):
        v1, e1 = o10.validate_tribunal_payload({"veredicto": "TEMPORAL"})
        self.assertIsNone(v1)
        self.assertIsNotNone(e1)

        v2, e2 = o10.validate_tribunal_payload({
            "fiscal": "Riesgo real de márgenes a la baja por competencia.",
            "abogado": "Marca líder con caja neta y clientes recurrentes.",
            "comprobar_5min": "Revisar guía trimestral y crecimiento orgánico.",
            "riesgo_trampa": None,
            "veredicto": "TEMPORAL",
        })
        self.assertIsNone(v2)
        self.assertIn("riesgo_trampa", e2)

        v3, e3 = o10.validate_tribunal_payload({
            "fiscal": "Riesgo real de márgenes a la baja por competencia.",
            "abogado": "Marca líder con caja neta y clientes recurrentes.",
            "comprobar_5min": "Revisar guía trimestral y crecimiento orgánico.",
            "riesgo_trampa": 3,
            "veredicto": "TEMPORAL",
        })
        self.assertIsNone(e3)
        self.assertEqual(v3["veredicto"], "TEMPORAL")
        self.assertEqual(v3["riesgo_trampa"], 3)

    def test_oro_puro_strict_gates(self):
        good = _base_oro_record()
        self.assertTrue(o10.is_oro_puro(good))
        self.assertFalse(o10.is_oro_puro(dict(good, sec_verified=None)))
        self.assertFalse(o10.is_oro_puro(dict(good, sec_status="PERIOD_MISMATCH")))
        self.assertFalse(o10.is_oro_puro(dict(good, forward_ok=None)))
        self.assertFalse(o10.is_oro_puro(dict(good, trap_risk_score=None)))
        self.assertFalse(o10.is_oro_puro(dict(good, piotroski_score=6, piotroski_max=6)))
        self.assertFalse(o10.is_oro_puro(dict(good, ai_verdict="EVIDENCIA_INSUFICIENTE")))


if __name__ == "__main__":
    unittest.main()
