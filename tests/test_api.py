from fastapi.testclient import TestClient
from api.main import app

c = TestClient(app)

def test_health_and_archs():
    assert c.get("/health").json()["status"] == "ok"
    assert set(c.get("/architectures").json()["architectures"]) == {"single", "debate", "planner_worker"}

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
    assert r["meta"]["final_arch"] in {"single", "debate", "planner_worker"}

def test_sweep():
    rows = c.post("/faults/sweep", json={"n_tasks": 10}).json()["rows"]
    assert len(rows) == 15
