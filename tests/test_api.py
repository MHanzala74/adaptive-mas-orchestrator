from fastapi.testclient import TestClient
from api.main import app

c = TestClient(app)

def test_health_and_archs():
    assert c.get("/health").json()["status"] == "ok"
    assert set(c.get("/architectures").json()["architectures"]) == {"single", "debate", "planner_worker", "solver_critic", "debate_judge"}

def test_solve_with_fault_and_repair():
    body = {"question": "(3 + 4) * 5", "answer": "35", "architecture": "single",
            "faults": [{"agent": "solo", "kind": "loop"}], "use_router": False}
    r = c.post("/solve", json=body).json()
    assert r["attempts"] >= 2 and r["meta"]["history"][0]["repair"] in {"replace", "reprompt", "switch"}

def test_solve_validation():
    assert c.post("/solve", json={"question": "1+1", "architecture": "nope"}).status_code == 422
    assert c.post("/solve", json={"question": "1+1", "budget": -5}).status_code == 422

def test_train_router_then_route():
    info = c.post("/router/train", json={"n_tasks": 40, "lam": 0.5}).json()
    assert info["n_tasks"] == 40
    r = c.post("/solve", json={"question": "2 + 3 * 4", "answer": "14"}).json()
    assert r["meta"]["final_arch"] in {"single", "debate", "planner_worker", "solver_critic", "debate_judge"}

def test_sweep():
    rows = c.post("/faults/sweep", json={"n_tasks": 10}).json()["rows"]
    assert len(rows) == 15

def test_save_router_and_run_log(tmp_path, monkeypatch):
    import api.main as m
    monkeypatch.setattr(m, "ROUTER_PATH", str(tmp_path / "router.json"))
    monkeypatch.setattr(m, "RUN_LOG", str(tmp_path / "runs.jsonl"))
    c.post("/router/train", json={"n_tasks": 30})
    assert c.post("/router/save").status_code == 200 and (tmp_path / "router.json").exists()
    c.post("/solve", json={"question": "2 + 2", "answer": "4"})
    r = c.get("/runs").json()
    assert r["n"] == 1 and r["runs"][0]["tokens"] > 0

def test_solve_accepts_verify_and_margin():
    r = c.post("/solve", json={"question": "(3 + 4) * 5", "answer": "35", "architecture": "single", "use_router": False,
                               "verify": True, "min_agreement": 0.66, "faults": [{"agent": "solo", "kind": "wrong"}]})
    assert r.status_code == 200 and r.json()["attempts"] >= 1
    assert c.post("/solve", json={"question": "1+1", "min_agreement": 0}).status_code == 422

def test_effective_budget_caps_real_backend(monkeypatch):
    from api.main import effective_budget
    monkeypatch.delenv("MAX_BUDGET_PER_REQUEST", raising=False)
    monkeypatch.setenv("LLM_BACKEND", "mock")
    assert effective_budget(None) is None and effective_budget(500) == 500          # mock: uncapped
    monkeypatch.setenv("LLM_BACKEND", "openai")
    assert effective_budget(None) == 4000 and effective_budget(99999) == 4000 and effective_budget(500) == 500
    monkeypatch.setenv("MAX_BUDGET_PER_REQUEST", "800")
    assert effective_budget(None) == 800

def test_api_key_protects_endpoints_but_not_health(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    assert c.get("/health").status_code == 200
    assert c.post("/solve", json={"question": "1+1"}).status_code == 401
    assert c.get("/runs").status_code == 401
    ok = c.post("/solve", json={"question": "1+1", "answer": "2"}, headers={"x-api-key": "secret"})
    assert ok.status_code == 200
    monkeypatch.delenv("API_KEY")
    assert c.post("/solve", json={"question": "1+1"}).status_code == 200
