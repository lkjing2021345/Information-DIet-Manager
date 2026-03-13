"""Refactor consistency checks for sentiment modules.

These tests are static contract checks (file content level) to avoid importing
heavy ML dependencies in CI smoke environments.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "src" / "algorithms" / "sentiment_train.py"
INFER_FILE = ROOT / "src" / "algorithms" / "sentiment.py"


def test_sentiment_train_contract_exports() -> None:
    content = TRAIN_FILE.read_text(encoding="utf-8")
    assert "class TrainConfig" in content
    assert "class TrainResult" in content
    assert "def train(config_path: Path, resume_from: Optional[Path] = None) -> TrainResult:" in content
    assert "def finetune(base_model_path: Path, new_data_path: Path, output_dir: Path) -> Path:" in content


def test_sentiment_inference_contract_exports() -> None:
    content = INFER_FILE.read_text(encoding="utf-8")
    assert "class SentimentAnalyzer" in content
    assert "def load(cls, model_path: Path) -> \"SentimentAnalyzer\":" in content


def test_train_inference_dependency_direction() -> None:
    train_content = TRAIN_FILE.read_text(encoding="utf-8")
    infer_content = INFER_FILE.read_text(encoding="utf-8")

    # train module should not depend on inference module
    assert "import sentiment" not in train_content
    assert "from . import sentiment" not in train_content

    # inference module should not import training module directly in runtime paths
    assert "from .sentiment_train" not in infer_content
