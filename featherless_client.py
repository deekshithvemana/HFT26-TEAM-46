"""Root-level export for Featherless client wrapper (Handoff to P1 & P2).

Conforms to Section 4 Task 1 and Section 6.2 of the Hackathon Plan.
"""

from src.intelligence.featherless_client import (
    FeatherlessClient,
    encode_image_to_base64,
    default_client,
    main,
    FORENSIC_SYSTEM_PROMPT,
)

__all__ = [
    "FeatherlessClient",
    "encode_image_to_base64",
    "default_client",
    "main",
    "FORENSIC_SYSTEM_PROMPT",
]

if __name__ == "__main__":
    main()
