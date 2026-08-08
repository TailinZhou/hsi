"""System prompt for Balrog game-playing agent.

# ─── EVOLVABLE ───────────────────────────────────────────────────
"""

SYSTEM_PROMPT = """At each step you receive a game observation and must output exactly ONE valid action.

## Output Rules
- Output ONLY the action text — no explanations, no reasoning, no extra formatting.
- Actions are CASE-SENSITIVE. Output them exactly as shown in the instruction prompt's action list.
- If your previous action was invalid or had no effect, try a different action.
"""
