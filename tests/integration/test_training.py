# tests/test_training.py
import pytest
from httpx import AsyncClient
from src.api.main import app   # <-- adjust if your app lives elsewhere
import asyncio

@pytest.mark.asyncio
async def test_train_single():
    async with AsyncClient(app=app, base_url="http://test") as client:
        # 1. start
        resp = await client.post(
            "/api/train/xgboost",
            json={"retrain": True}
        )
        assert resp.status_code == 200
        data = resp.json()
        job_id = data["job_id"]
        assert data["status"] == "started"
        assert data["model_type"] == "xgboost"

        # 2. poll until finished (max 60 s)
        for _ in range(30):
            status_resp = await client.get(f"/api/train/status/{job_id}")
            assert status_resp.status_code == 200
            status = status_resp.json()
            if status["status"] in {"completed", "failed"}:
                break
            await asyncio.sleep(2)

        assert status["status"] == "completed"
        assert status["model_path"] is not None

        # 3. model really exists on disk
        from pathlib import Path
        assert Path(status["model_path"]).exists()

@pytest.mark.asyncio
async def test_train_all():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post(
            "/api/train",
            json={"model_type": "all", "use_cv": False}
        )
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        # poll parent job
        for _ in range(40):
            r = await client.get(f"/api/train/status/{job_id}")
            s = r.json()
            if s["status"] in {"completed", "failed"}:
                break
            await asyncio.sleep(3)

        assert s["status"] == "completed"

@pytest.mark.asyncio
async def test_list_models():
    async with AsyncClient(app=app, base_url="http://test") as client:
        r = await client.get("/api/train/models/available")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data["available_models"], dict)