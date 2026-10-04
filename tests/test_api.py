import unittest

import httpx

from arabic_sentiment.api import create_app


class FakePredictionService:
    model_version = "test-arabert-seed-42"

    def predict(self, review: str) -> dict[str, str | float]:
        self.last_review = review
        return {"label": "positive", "confidence": 0.875}


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service = FakePredictionService()
        transport = httpx.ASGITransport(app=create_app(self.service))
        self.client = httpx.AsyncClient(transport=transport, base_url="http://testserver")

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_health_success(self):
        response = await self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    async def test_predict_success_and_response_fields(self):
        review = "الخدمة ممتازة والغرفة نظيفة"
        response = await self.client.post("/predict", json={"review": review})
        body = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["label"], "positive")
        self.assertIn(body["label"], {"negative", "positive"})
        self.assertGreaterEqual(body["confidence"], 0.0)
        self.assertLessEqual(body["confidence"], 1.0)
        self.assertEqual(body["model_version"], self.service.model_version)
        self.assertEqual(self.service.last_review, review)

    async def test_predict_rejects_missing_empty_and_whitespace_review(self):
        for payload in ({}, {"review": ""}, {"review": "  \n\t"}):
            with self.subTest(payload=payload):
                response = await self.client.post("/predict", json=payload)
                self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
