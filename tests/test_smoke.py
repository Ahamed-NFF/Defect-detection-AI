"""Smoke tests — expand as modules get implemented."""

def test_classifier_builds():
    from src.classifier.model import build_classifier
    net = build_classifier("resnet50")
    assert net is not None
