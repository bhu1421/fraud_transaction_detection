import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ApplicationSmokeTest(unittest.TestCase):
    def test_application_sources_parse(self):
        for relative_path in (
            "app/streamlit_app.py",
            "src/fraud_detection/inference.py",
        ):
            with self.subTest(path=relative_path):
                ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
