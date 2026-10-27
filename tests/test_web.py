"""Web portal: demo seeding, role separation and the review -> release flow (no models needed)."""
import importlib
import sys
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "web"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    import os

    os.environ["CARDIOSENTINEL_DATA"] = str(tmp_path_factory.mktemp("webdata"))
    sys.path.insert(0, str(WEB))
    for m in ("store", "demo", "inference", "server"):
        sys.modules.pop(m, None)
    server = importlib.import_module("server")
    server.init(reseed=True)
    from fastapi.testclient import TestClient

    return TestClient(server.app)


def login(c, email):
    r = c.post("/api/auth/login", json={"email": email, "password": "demo1234"})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_roles_are_separated(client):
    hp = login(client, "patient@cardiosentinel.demo")
    hc = login(client, "clinician@cardiosentinel.demo")
    assert client.get("/api/patients", headers=hp).status_code == 403
    assert client.get("/api/me/overview", headers=hc).status_code == 403
    assert client.get("/api/patients").status_code == 401
    assert client.post("/api/auth/login", json={"email": "patient@cardiosentinel.demo", "password": "x"}).status_code == 401


def test_patient_sees_only_released_results(client):
    hp = login(client, "patient@cardiosentinel.demo")
    hc = login(client, "clinician@cardiosentinel.demo")
    mine = client.get("/api/me/ecgs", headers=hp).json()
    hidden = [e for e in mine if not e["released"]]
    assert hidden, "seed gives the demo patient one unreleased ECG"
    eid = hidden[0]["id"]
    assert client.get(f"/api/ecgs/{eid}", headers=hp).status_code == 404
    assert client.get(f"/api/ecgs/{eid}/signal", headers=hp).status_code == 404
    r = client.post(f"/api/ecgs/{eid}/review", headers=hc,
                    json={"final_labels": {"NORM": 1}, "notes": "", "release": True})
    assert r.json()["status"] == "reviewed"
    v = client.get(f"/api/ecgs/{eid}", headers=hp).json()
    assert v["findings"] == ["NORM"] and "prediction" not in v  # no raw model internals for patients
    assert len(client.get(f"/api/ecgs/{eid}/signal", headers=hp).json()["data"]) == 12


def test_clinician_views(client):
    hc = login(client, "clinician@cardiosentinel.demo")
    d = client.get("/api/dashboard", headers=hc).json()
    assert d["n_patients"] == 36 and d["new_submissions"] >= 1
    q = client.get("/api/ecgs?status=needs_review", headers=hc).json()
    n_ind = [len(e["prediction"]["indeterminate"]) for e in q]
    assert n_ind == sorted(n_ind, reverse=True) and all(n > 0 for n in n_ind)
    assert client.get("/api/model/performance", headers=hc).status_code == 404  # no model internals in the portal


def test_signup_selfcheck_privacy_and_share(client):
    r = client.post("/api/auth/signup", json={"name": "New Person", "email": "new@example.com", "password": "longenough"})
    assert r.status_code == 200
    hp = {"Authorization": f"Bearer {r.json()['token']}"}
    assert client.post("/api/auth/signup", json={"name": "x", "email": "new@example.com", "password": "longenough"}).status_code == 400
    hc = login(client, "clinician@cardiosentinel.demo")
    samples = client.get("/api/me/samples", headers=hp).json()
    if not samples:
        pytest.skip("PTB-XL samples need prepared data")
    chk = client.post(f"/api/me/analyze/sample/{samples[0]['ecg_id']}", headers=hp).json()
    assert chk["self_check"] and not chk["shared"] and chk["ai"]["level"] in {"reassuring", "check", "see_doctor"}
    assert client.get(f"/api/ecgs/{chk['id']}", headers=hc).status_code == 404  # private until shared
    note = client.post("/api/me/notes", headers=hp, json={"feeling": 2, "symptoms": ["Palpitations"], "body": "racing"}).json()
    png = bytes([0x89]) + b"PNG" + bytes([13, 10, 26, 10]) + b"0" * 64
    att = client.post("/api/me/attachments", headers=hp, files={"file": ("photo.png", png, "image/png")}).json()
    assert client.post("/api/me/attachments", headers=hp, files={"file": ("x.png", b"nope", "image/png")}).status_code == 400
    assert client.get(f"/api/attachments/{att['id']}", headers=hc).status_code == 404  # not shared yet
    sub = client.post("/api/me/submissions", headers=hp, json={"ecg_id": chk["id"], "note_ids": [note["id"]],
                                                               "attachment_ids": [att["id"]], "message": "please look"}).json()
    assert client.get(f"/api/ecgs/{chk['id']}", headers=hc).status_code == 200
    assert client.get(f"/api/attachments/{att['id']}", headers=hc).content == png
    detail = client.get(f"/api/submissions/{sub['id']}", headers=hc).json()
    assert detail["status"] == "seen" and len(detail["notes"]) == 1 and len(detail["attachments"]) == 1
    client.post(f"/api/submissions/{sub['id']}/reply", headers=hc, json={"body": "All good"})
    assert client.get(f"/api/me/submissions/{sub['id']}", headers=hp).json()["reply"] == "All good"


def test_assistant_answers_from_own_records(client, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)   # offline mode: no network in tests
    hp = login(client, "patient@cardiosentinel.demo")
    info = client.get("/api/me/assistant", headers=hp).json()
    assert info["mode"] == "offline" and info["suggestions"]
    r = client.post("/api/me/assistant", headers=hp, json={"question": "When is my next appointment?"}).json()
    assert r["mode"] == "offline" and any(s["kind"] == "appointment" for s in r["sources"])
    assert client.post("/api/me/assistant", headers=hp, json={"question": "I have chest pain"}).json()["urgent"]
    hc = login(client, "clinician@cardiosentinel.demo")
    assert client.post("/api/me/assistant", headers=hc, json={"question": "hi"}).status_code == 403
