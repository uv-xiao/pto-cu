import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
GOAL_ROOT = ROOT / "docs" / "in_progress" / "nvidia_backend_paper_ready"
DISPATCH_ROOT = GOAL_ROOT / "dispatch_log"


def markdown_links(text):
    return set(re.findall(r"\]\(([^)]+\.md)\)", text))


def test_dispatch_log_is_split_for_human_review():
    landing = GOAL_ROOT / "dispatch_log.md"
    index = DISPATCH_ROOT / "index.md"
    entries_root = DISPATCH_ROOT / "entries"

    assert landing.is_file()
    assert index.is_file()
    assert entries_root.is_dir()
    assert len(landing.read_text(encoding="utf-8").splitlines()) <= 120

    entry_files = sorted(entries_root.glob("*.md"))
    assert entry_files
    index_links = markdown_links(index.read_text(encoding="utf-8"))
    expected_links = {
        f"entries/{path.name}"
        for path in entry_files
    }
    assert index_links == expected_links

    latest_text = entry_files[-1].read_text(encoding="utf-8")
    assert "Qwen Proxy Decode Loop Live Reuse" in latest_text
    for path in entry_files:
        text = path.read_text(encoding="utf-8")
        assert len(text.splitlines()) <= 300, path
        assert re.search(r"^### 2026-", text, flags=re.MULTILINE), path
