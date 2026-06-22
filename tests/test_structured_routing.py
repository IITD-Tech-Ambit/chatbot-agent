"""Tests for the structured-query router."""

import pytest

from agent.routing.structured import match_structured, RouteMatch


class TestMatchStructured:
    @pytest.mark.parametrize("msg,handler,capture", [
        ("h-index of Amit Kumar", "get_h_index", "Amit Kumar"),
        ("h index for Dr Sharma", "get_h_index", "Dr Sharma"),
        ("citations of Prof Kumar", "get_citations", "Prof Kumar"),
        ("citation count for Gupta", "get_citations", "Gupta"),
        ("faculty in Computer Science dept", "get_faculty_by_dept", "Computer Science"),
        ("professors from Electrical Engineering", "get_faculty_by_dept", "Electrical Engineering"),
        ("papers by Amit Kumar", "get_papers_by_author", "Amit Kumar"),
    ])
    def test_match(self, msg, handler, capture):
        result = match_structured(msg)
        assert result is not None
        assert result.handler == handler
        assert result.capture == capture

    @pytest.mark.parametrize("msg", [
        "What research is done on machine learning?",
        "Tell me about deep learning",
        "How many papers were published in 2023?",
        "Which professor works on NLP",
        "",
    ])
    def test_no_match(self, msg):
        assert match_structured(msg) is None
