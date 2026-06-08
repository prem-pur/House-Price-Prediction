"""
test_api.py — Unit tests for FastAPI prediction endpoint
"""
import sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
os.environ["MODEL_DIR"] = os.path.join(ROOT, "models")

import unittest
from fastapi.testclient import TestClient
from api.main import app

VALID_PAYLOAD = {
    "district": "Colombo", "area": "Borella",
    "perch": 10, "bedrooms": 3, "bathrooms": 2,
    "kitchen_area_sqft": 150, "parking_spots": 1,
    "has_garden": True, "has_ac": False,
    "water_supply": "Pipe-borne", "electricity": "Single phase",
    "floors": 2, "year_built": 2010
}

class TestHealthEndpoint(unittest.TestCase):
    def test_health_returns_200(self):
        with TestClient(app) as client:
            res = client.get("/health")
            self.assertEqual(res.status_code, 200)

    def test_health_model_loaded(self):
        with TestClient(app) as client:
            res = client.get("/health")
            self.assertTrue(res.json()["model_loaded"])

    def test_health_status_ok(self):
        with TestClient(app) as client:
            res = client.get("/health")
            self.assertEqual(res.json()["status"], "ok")


class TestPredictEndpoint(unittest.TestCase):
    def test_predict_returns_200(self):
        with TestClient(app) as client:
            res = client.post("/predict", json=VALID_PAYLOAD)
            self.assertEqual(res.status_code, 200)

    def test_predict_has_price_field(self):
        with TestClient(app) as client:
            res = client.post("/predict", json=VALID_PAYLOAD)
            self.assertIn("predicted_price_lkr", res.json())

    def test_predict_has_formatted_field(self):
        with TestClient(app) as client:
            res = client.post("/predict", json=VALID_PAYLOAD)
            self.assertIn("predicted_price_lkr_formatted", res.json())

    def test_predict_price_is_positive(self):
        with TestClient(app) as client:
            res = client.post("/predict", json=VALID_PAYLOAD)
            self.assertGreater(res.json()["predicted_price_lkr"], 0)

    def test_predict_formatted_starts_with_lkr(self):
        with TestClient(app) as client:
            res = client.post("/predict", json=VALID_PAYLOAD)
            self.assertTrue(res.json()["predicted_price_lkr_formatted"].startswith("LKR"))

    def test_predict_invalid_perch(self):
        with TestClient(app) as client:
            res = client.post("/predict", json={**VALID_PAYLOAD, "perch": 999})
            self.assertEqual(res.status_code, 422)

    def test_predict_missing_field(self):
        with TestClient(app) as client:
            payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "district"}
            res = client.post("/predict", json=payload)
            self.assertEqual(res.status_code, 422)

    def test_predict_different_districts(self):
        with TestClient(app) as client:
            for district, area in [("Kandy","Kandy City"),("Galle","Galle Fort"),("Jaffna","Jaffna Town")]:
                res = client.post("/predict", json={**VALID_PAYLOAD, "district": district, "area": area})
                self.assertEqual(res.status_code, 200)
                self.assertGreater(res.json()["predicted_price_lkr"], 0)

    def test_large_vs_small_land_price(self):
        with TestClient(app) as client:
            big   = client.post("/predict", json={**VALID_PAYLOAD, "perch": 50, "bedrooms": 6}).json()
            small = client.post("/predict", json={**VALID_PAYLOAD, "perch": 3, "bedrooms": 1}).json()
            self.assertGreater(big["predicted_price_lkr"], small["predicted_price_lkr"])


class TestRootEndpoint(unittest.TestCase):
    def test_root_returns_200(self):
        with TestClient(app) as client:
            res = client.get("/")
            self.assertEqual(res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
