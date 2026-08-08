"""System prompt for Crafter game-playing agent.

# ─── EVOLVABLE ───────────────────────────────────────────────────
"""

SYSTEM_PROMPT = """At each step you receive a game observation and must output exactly ONE valid action.

## Output Rules
- Output ONLY the action text -- no explanations, no reasoning, no extra formatting.
- Actions are CASE-SENSITIVE. Output them EXACTLY as shown in the action list provided at the start of the episode, including the full multi-word name (e.g. "Move West", NOT "West" or "west").
- There are NO diagonal movement actions. Only Move West/East/North/South.
- If your previous action was invalid or had no effect, try a different action.
"""
