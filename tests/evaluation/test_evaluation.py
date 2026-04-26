import json
import os
import pytest
from pathlib import Path

pytestmark = pytest.mark.evaluation

DATA_FILE = Path(__file__).parent / "data.json"

@pytest.fixture(autouse=True)
def require_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set — skipping evaluation tests")

def load_cases():
    with open(DATA_FILE) as f:
        return json.load(f)

@pytest.mark.parametrize("case", load_cases(), ids=[f"case_{i}" for i in range(len(load_cases()))])
def test_rag_quality(case):
    """
    Evaluates each case for:
    - FaithfulnessMetric >= 0.8: answer is grounded in retrieval_context
    - AnswerRelevancyMetric >= 0.7: answer addresses the question
    """
    from deepeval import assert_test
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

    test_case = LLMTestCase(
        input=case["input"],
        actual_output=case["actual_output"],
        retrieval_context=case["retrieval_context"],
        expected_output=case.get("expected_output"),
    )

    assert_test(test_case, [
        FaithfulnessMetric(threshold=0.8),
        AnswerRelevancyMetric(threshold=0.7),
    ])
