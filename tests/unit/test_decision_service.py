from __future__ import annotations

import unittest
from datetime import timedelta
from typing import Any

from decision_testdata import NOW, make_roles, make_twin
from src.orchestrator.decision_service import DecisionService
from src.orchestrator.transaction_manager import TransactionManager
from src.twin.outcome.predictor import PostMigrationOutcomePrediction


class FakeStore:
    def __init__(self, twin: Any = None) -> None:
        self.twin = twin

    def twin_state(self) -> Any:
        return self.twin

    def role_matrix(self) -> Any:
        return make_roles(self.twin) if self.twin is not None else {}


class FakePredictor:
    def __init__(self, dst: float = 0.50) -> None:
        self.dst = dst
        self.calls = 0

    def predict_batch(self, twin_state: Any, actions: Any, *, observe: bool = True) -> Any:
        self.calls += 1
        assert observe is False
        return [
            PostMigrationOutcomePrediction(
                snapshot_id=twin_state["snapshot_id"], switch_id=a.switch_id,
                source_controller=a.source_controller, target_controller=a.target_controller,
                model_name="post-migration-outcome", model_version="post-migration-v1",
                predicted_src_utilization=0.80, predicted_dst_utilization=self.dst,
                predicted_src_response_p95_ms=0.8, predicted_dst_response_p95_ms=0.8,
                predicted_worst_response_p95_ms=0.8, predicted_imbalance=0.05,
                dst_utilization_margin=0.037, response_margin_ms=0.009,
                inference_time_ms=1.0, status="READY",
            )
            for a in actions
        ]


def forecast(c1: bool | None = True, c2: bool | None = False, **over: Any) -> dict[str, Any]:
    def entry(cid: str, risk: bool | None) -> dict[str, Any]:
        status = "READY" if risk is not None else "NOT_READY"
        return {"controller_id": cid, "status": status, "overload_risk": risk}

    body: dict[str, Any] = {
        "status": "OK", "snapshot_id": "snap-1", "model_version": "forecast-models-v3",
        "controllers": [entry("c1", c1), entry("c2", c2)],
    }
    body.update(over)
    return body


def service(fc: Any = None, twin: Any = "default", predictor: Any = None,
            transactions: TransactionManager | None = None) -> DecisionService:
    twin = make_twin() if twin == "default" else twin
    body = forecast() if fc is None else fc
    store: Any = FakeStore(twin)
    return DecisionService(
        store, lambda: body, predictor or FakePredictor(),
        transactions or TransactionManager(), clock=lambda: NOW,
    )


class DecisionServiceTest(unittest.TestCase):
    def test_safe_funnel(self) -> None:
        out = service().evaluate()
        self.assertEqual(out["status"], "OK")
        self.assertEqual(out["at_risk"], ["c1"])
        self.assertEqual(out["funnel"], {"raw": 4, "eligible": 4, "safe": 4, "dropped": 0})
        self.assertEqual(set(out["timings_ms"]), {"t_candidate", "t_outcome", "t_safety"})
        self.assertTrue(all(e["safe"] for e in out["candidates"]))

    def test_unsafe_destination_is_rejected_by_the_gate(self) -> None:
        out = service(predictor=FakePredictor(dst=0.99)).evaluate()
        self.assertEqual(out["funnel"]["eligible"], 4)
        self.assertEqual(out["funnel"]["safe"], 0)
        self.assertTrue(out["rejections"]["safety"])

    def test_nothing_at_risk_skips_the_predictor(self) -> None:
        predictor = FakePredictor()
        out = service(forecast(c1=False), predictor=predictor).evaluate()
        self.assertEqual(out["funnel"], {"raw": 0, "eligible": 0, "safe": 0, "dropped": 0})
        self.assertEqual(predictor.calls, 0)

    def test_not_ready_reasons(self) -> None:
        self.assertEqual(service(twin=None).evaluate()["reason"], "NO_TWIN_STATE")
        not_ready = service(fc={"status": "NOT_READY"}).evaluate()
        self.assertEqual(not_ready["reason"], "FORECAST_NOT_READY")
        stale = service(fc=forecast(snapshot_id="snap-0")).evaluate()
        self.assertEqual(stale["reason"], "FORECAST_SNAPSHOT_MISMATCH")

    def test_unknown_destination_forecast_is_skipped_not_passed(self) -> None:
        out = service(fc=forecast(c2=None)).evaluate()
        skipped = out["candidates"][0]["safety"]["skipped_checks"]
        self.assertIn(["destination_not_at_risk", "NOT_AVAILABLE"], skipped)

    def test_cooldown_blocks_before_the_predictor(self) -> None:
        manager = TransactionManager()
        tx = manager.create("s9", "c1", "c2")
        manager.finish(tx)
        tx.finished_at = NOW - timedelta(seconds=5)
        predictor = FakePredictor()
        out = service(transactions=manager, predictor=predictor).evaluate()
        self.assertEqual(out["funnel"]["eligible"], 0)
        self.assertIn("COOLDOWN_ACTIVE", out["rejections"]["precheck"])
        self.assertEqual(predictor.calls, 0)

    def test_evaluate_has_no_side_effect(self) -> None:
        manager = TransactionManager()
        service(transactions=manager).evaluate()
        self.assertEqual(manager.list_all(), [])
        self.assertEqual(manager.active_count(), 0)


if __name__ == "__main__":
    unittest.main()
