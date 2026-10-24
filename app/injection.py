# Heuristic flag for a human to review, not a defence: the real protection is
# strict validation of the model's reply (see classifier.validate_reply).
INJECTION_PHRASES: tuple[str, ...] = (
    "ignore all previous instructions",
    "ignore previous instructions",
    "ignore the above",
    "disregard previous instructions",
    "you are now",
    "system prompt",
    "classify it as",
    "classify this as",
)


def looks_like_injection(subject: str, body: str) -> bool:
    text = f"{subject}\n{body}".lower()
    return any(phrase in text for phrase in INJECTION_PHRASES)
