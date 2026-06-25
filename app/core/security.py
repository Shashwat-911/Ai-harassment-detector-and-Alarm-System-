"""
Security utilities for the Harassment Detection API.

Provides an API-key middleware and helper functions.
Expand with JWT / OAuth2 as needed for production.
"""

from __future__ import annotations

import secrets
from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

# ---------------------------------------------------------------------------
# API Key scheme
# ---------------------------------------------------------------------------

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# In production, load from env / vault.  This is a placeholder.
_VALID_API_KEYS: set[str] = {
    "dev-key-change-me-in-production",
}


async def verify_api_key(
    api_key: Optional[str] = Security(API_KEY_HEADER),
) -> str:
    """
    Dependency that validates the ``X-API-Key`` header.

    Raises:
        HTTPException 401 if the key is missing or invalid.
    """
    if api_key is None or api_key not in _VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
    return api_key


def generate_api_key() -> str:
    """Generate a cryptographically secure API key."""
    return secrets.token_urlsafe(32)
