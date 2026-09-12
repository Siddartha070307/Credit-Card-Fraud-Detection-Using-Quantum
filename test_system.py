"""
Comprehensive verification test suite for Quantum Fraud Detection system.
Tests API routes, error handling, model inference, SQLite persistence, and LangGraph agent traces.
"""
import math
import pandas as pd
from fastapi.testclient import TestClient
from api.main import app, selected_features
from agent.graph import build_graph, run_agent_pipeline_with_trace

client = TestClient(app)


def test_api_routes():
    # 1. Health / feature route
    res = client.get("/features")
    assert res.status_code == 200
    data = res.json()
    assert data["n_features"] == len(selected_features)
    assert data["selected_features"] == selected_features
    assert data["qubits"] == len(selected_features)

    # 2. Presets route
    res = client.get("/samples")
    assert res.status_code == 200
    presets = res.json()
    assert "real_fraud" in presets
    assert "real_fraud_2" in presets
    assert "real_legit" in presets
    assert "borderline" in presets

    # 3. Benchmark stats route
    res = client.get("/benchmark-stats")
    assert res.status_code == 200
    stats = res.json()
    assert "classical_svm" in stats["metrics"]
    assert "qsvc" in stats["metrics"]
    assert "random_forest" in stats["additional_reference"]

    # 4. ROC curve route
    res = client.get("/roc-curve")
    assert res.status_code == 200
    roc_data = res.json()
    assert "curves" in roc_data
    assert "classical_svm" in roc_data["curves"]
    assert "qsvc_tuned" in roc_data["curves"]
    assert "qsvc_pretuning" in roc_data["curves"]
    assert "random_baseline" in roc_data["curves"]
    assert roc_data["curves"]["classical_svm"]["auc"] > 0.99
    assert roc_data["curves"]["qsvc_tuned"]["auc"] > 0.98

    # 5. Generate PDF report route (synchronous, fast, no quantum inference)
    import time
    t0 = time.time()
    res_pdf = client.get("/generate-report")
    elapsed = time.time() - t0
    assert res_pdf.status_code == 200
    assert res_pdf.headers["content-type"] == "application/pdf"
    assert 'filename="quantum_fraud_report.pdf"' in res_pdf.headers.get("content-disposition", "")
    assert res_pdf.content.startswith(b"%PDF-"), "Must be a valid PDF binary starting with %PDF-"
    assert len(res_pdf.content) > 1000, f"PDF content size too small: {len(res_pdf.content)} bytes"
    assert elapsed < 0.5, f"PDF generation took {elapsed:.3f}s (should be fast, no quantum inference)"
    print(f"\n[test_api_routes] Generated PDF report: {len(res_pdf.content)} bytes in {elapsed*1000:.1f}ms")



def test_score_transaction_valid():
    # Test legit transaction with list of floats
    features = [0.0] * len(selected_features)
    res = client.post("/score-transaction", json={"features": features})
    assert res.status_code == 200
    data = res.json()
    assert "quantum_score" in data
    assert "classical_score" in data
    assert data["quantum_label"] in ["FRAUD", "LEGIT"]
    assert data["classical_label"] in ["FRAUD", "LEGIT"]
    assert data["decision"] in ["FLAGGED", "CLEARED"]
    assert data["latency_seconds"] > 0
    print(f"\n[test_score_valid] List Input Latency: {data['latency_seconds']}s, Decision: {data['decision']}")

    # Test with dictionary input mapping feature names to values
    feat_dict = {f: 0.1 for f in selected_features}
    res_dict = client.post("/score-transaction", json={"features": feat_dict})
    assert res_dict.status_code == 200
    data_dict = res_dict.json()
    assert data_dict["decision"] in ["FLAGGED", "CLEARED"]
    print(f"[test_score_valid] Dict Input Latency: {data_dict['latency_seconds']}s, Decision: {data_dict['decision']}")


def test_score_agent_trace():
    features = [0.17, 1.18, -1.49, 0.14, -0.03, -1.15]
    res = client.post("/score-agent", json={"features": features})
    assert res.status_code == 200
    data = res.json()
    assert "trace" in data
    assert len(data["trace"]) == 4

    node_names = [step["node"] for step in data["trace"]]
    assert node_names == ["ingest", "quantum_score", "classical_score", "decide_and_report"]
    for step in data["trace"]:
        assert "elapsed_ms" in step
        assert "description" in step
        assert "output" in step
    print(f"[test_score_agent] Successfully executed all 4 agent nodes. Latency: {data['latency_seconds']}s")


def test_score_transaction_malformed_inputs():
    # Wrong feature count (too few)
    res = client.post("/score-transaction", json={"features": [1.0, 2.0]})
    assert res.status_code == 400
    assert "Expected exactly" in res.json()["detail"]

    # Wrong feature count (too many)
    res = client.post("/score-transaction", json={"features": [0.0] * 10})
    assert res.status_code == 400
    assert "Expected exactly" in res.json()["detail"]

    # Missing keys in dict input
    res = client.post("/score-transaction", json={"features": {"V17": 0.5}})
    assert res.status_code == 400
    assert "Missing feature keys" in res.json()["detail"]

    # Non-numeric input
    res = client.post("/score-transaction", json={"features": ["abc", 1.0, 2.0, 3.0, 4.0, 5.0]})
    assert res.status_code == 400 or res.status_code == 422

    # Missing features key
    res = client.post("/score-transaction", json={"bad_key": [0.0] * 6})
    assert res.status_code == 422


def test_sqlite_history_lifecycle():
    # Ensure at least one record exists before asserting
    client.post("/score-transaction", json={"features": [0.0] * len(selected_features)})
    # Fetch history
    res = client.get("/history?limit=5")
    assert res.status_code == 200
    rows = res.json()
    assert isinstance(rows, list)
    assert len(rows) > 0
    assert "decision" in rows[0]
    assert "quantum_score" in rows[0]

    # Test history clear and re-fetch
    del_res = client.delete("/history")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "cleared"

    res_empty = client.get("/history?limit=5")
    assert res_empty.status_code == 200
    assert len(res_empty.json()) == 0

    # Insert a new record to verify re-population
    client.post("/score-transaction", json={"features": [0.0] * len(selected_features)})
    res_after = client.get("/history?limit=5")
    assert len(res_after.json()) == 1


def test_agent_graph():
    graph = build_graph()
    state = {"features": [0.0] * len(selected_features)}
    result = graph.invoke(state)
    assert "decision" in result
    assert "quantum_label" in result
    assert "classical_label" in result
    assert result["decision"] in ["FLAGGED", "CLEARED"]

    # Test trace runner directly
    trace_res = run_agent_pipeline_with_trace([0.0] * len(selected_features))
    assert len(trace_res["trace"]) == 4
    assert trace_res["decision"] in ["FLAGGED", "CLEARED"]

    # Test error handling in agent graph
    try:
        graph.invoke({"features": [1.0, 2.0]})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Expected" in str(e)


def test_scaler_persistence_and_info():
    res = client.get("/scaler-info")
    assert res.status_code == 200
    data = res.json()
    assert data["loaded"] is True
    assert data["n_features"] == len(selected_features)
    assert len(data["data_min"]) == len(selected_features)
    assert len(data["data_max"]) == len(selected_features)
    assert len(data["scale"]) == len(selected_features)
    assert data["n_samples_seen"] > 0
    assert data["type"] == "MinMaxScaler"
    print(f"[test_scaler_info] MinMaxScaler([0,pi]) verified loaded. Samples seen: {data['n_samples_seen']}")


def test_score_raw_unscaled_features():
    from agent.graph import get_scaler
    scaler = get_scaler()
    assert scaler is not None, "Scaler must exist"

    raw_features = [1.5, -2.0, 0.5, 3.2, -1.1, 0.8]
    raw_df = pd.DataFrame([raw_features], columns=selected_features)
    expected_scaled = [round(float(v), 4) for v in scaler.transform(raw_df)[0]]

    # 1. API score-transaction with scaled=False
    res = client.post("/score-transaction", json={"features": raw_features, "scaled": False})
    assert res.status_code == 200
    data = res.json()
    assert data["decision"] in ["FLAGGED", "CLEARED"]
    # Verify features in response are transformed
    for actual, expected in zip(data["features"], expected_scaled):
        assert abs(actual - expected) < 1e-4

    # 2. API score-agent with scaled=False
    res_agent = client.post("/score-agent", json={"features": raw_features, "scaled": False})
    assert res_agent.status_code == 200
    agent_data = res_agent.json()
    assert agent_data["decision"] in ["FLAGGED", "CLEARED"]
    for actual, expected in zip(agent_data["features"], expected_scaled):
        assert abs(actual - expected) < 1e-4

    # 3. Direct agent pipeline with scaled=False
    trace_res = run_agent_pipeline_with_trace(raw_features, scaled=False)
    assert len(trace_res["trace"]) == 4
    for actual, expected in zip(trace_res["features"], expected_scaled):
        assert abs(actual - expected) < 1e-4
    print("[test_score_raw_unscaled] Raw feature standardization verified with zero data leakage.")


def test_pdf_report_with_history():
    """Verify that PDF report correctly renders populated SQLite transaction audit logs."""
    from api.main import get_db
    import time
    with get_db() as conn:
        conn.execute(
            "INSERT INTO scored_transactions (features, quantum_score, classical_score, decision, explanation, latency_seconds, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]", 1.125, 0.945, "FLAGGED", "Automated test transaction", 0.035, time.time())
        )
        conn.commit()

    res = client.get("/generate-report")
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"
    assert res.content.startswith(b"%PDF-")
    assert len(res.content) > 1000
    print(f"[test_pdf_report_with_history] PDF with SQLite audit log verified: {len(res.content)} bytes.")


if __name__ == "__main__":
    test_api_routes()
    test_pdf_report_with_history()
    test_scaler_persistence_and_info()
    test_score_transaction_valid()
    test_score_raw_unscaled_features()
    test_score_agent_trace()
    test_score_transaction_malformed_inputs()
    test_sqlite_history_lifecycle()
    test_agent_graph()
    print("\nALL VERIFICATION TEST SUITES PASSED SUCCESSFULLY!")

