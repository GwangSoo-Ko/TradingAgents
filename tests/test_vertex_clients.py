"""Vertex provider layer: auth helper + the three Vertex client classes.

All Vertex SDK imports are lazy, so these tests install lightweight fake
modules / mock the token fetch — no langchain-google-vertexai, google-auth,
or network required.
"""
import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
class TestVertexAuth:
    def test_openapi_base_url_global_drops_location_prefix(self):
        from tradingagents.llm_clients import vertex_auth
        assert vertex_auth.openapi_base_url("tpmn-dev", "global") == (
            "https://aiplatform.googleapis.com/v1/projects/tpmn-dev/"
            "locations/global/endpoints/openapi"
        )

    def test_openapi_base_url_regional_keeps_prefix(self):
        from tradingagents.llm_clients import vertex_auth
        assert vertex_auth.openapi_base_url("p", "us-east5") == (
            "https://us-east5-aiplatform.googleapis.com/v1/projects/p/"
            "locations/us-east5/endpoints/openapi"
        )

    def test_resolve_location_prefers_explicit_then_env_then_global(self, monkeypatch):
        from tradingagents.llm_clients import vertex_auth
        monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
        assert vertex_auth.resolve_location(None) == "global"
        assert vertex_auth.resolve_location("us-central1") == "us-central1"
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west1")
        assert vertex_auth.resolve_location(None) == "europe-west1"

    def test_resolve_project_prefers_explicit_then_env(self, monkeypatch):
        from tradingagents.llm_clients import vertex_auth
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "envproj")
        assert vertex_auth.resolve_project(None) == "envproj"
        assert vertex_auth.resolve_project("explicit") == "explicit"
