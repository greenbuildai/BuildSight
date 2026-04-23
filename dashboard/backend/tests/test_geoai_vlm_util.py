import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import geoai_vlm_util  # noqa: E402


class RuleBasedVLMDescriptionTests(unittest.TestCase):
    def test_answers_site_condition_question_directly(self) -> None:
        description = geoai_vlm_util._rule_based_description(  # type: ignore[attr-defined]
            {
                "scene": "S2_dusty",
                "total_workers": 0,
            },
            "What is the current site condition?",
        )

        self.assertIn("dust", description.lower())
        self.assertNotIn("site appears clear", description.lower())

    def test_answers_activity_question_with_worker_and_ppe_summary(self) -> None:
        description = geoai_vlm_util._rule_based_description(  # type: ignore[attr-defined]
            {
                "scene": "S4_crowded",
                "total_workers": 3,
                "helmets_detected": 2,
                "vests_detected": 1,
                "proximity_violations": 2,
            },
            "What is happening on the site right now?",
        )

        lowered = description.lower()
        self.assertIn("3 worker", lowered)
        self.assertIn("helmet", lowered)
        self.assertIn("vest", lowered)
        self.assertIn("crowded", lowered)


if __name__ == "__main__":
    unittest.main()
