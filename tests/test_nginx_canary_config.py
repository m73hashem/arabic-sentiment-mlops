import re
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "deploy" / "nginx" / "canary.conf"


def test_nginx_routes_weighted_traffic_to_explicit_stable_and_canary_upstreams():
    config = CONFIG.read_text(encoding="utf-8")
    servers = re.findall(r"server\s+([\w.-]+):(\d+)\s+weight=(\d+);", config)

    assert servers == [
        ("arabic-sentiment-stable", "8000", "95"),
        ("arabic-sentiment-canary", "3000", "5"),
    ]
    assert sum(int(weight) for _, _, weight in servers) == 100
    assert "proxy_pass http://arabic_sentiment_versions;" in config
    assert "listen 8080;" in config
    assert "proxy_next_upstream off;" in config


def test_nginx_config_documents_progressive_rollout_and_stable_rollback_in_readme():
    readme = (CONFIG.parents[2] / "README.md").read_text(encoding="utf-8")

    for stage in ("95/5", "75/25", "50/50", "100%"):
        assert stage in readme
    assert "stable-only" in readme
    assert "nginx -t -c" in readme
