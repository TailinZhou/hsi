#!/usr/bin/env python3
"""
Test script for React Loop Agent framework.

Run this to verify the installation and basic functionality.
"""

import os
import sys
from pathlib import Path

# Windows Unicode fix: force UTF-8 for stdout/stderr before any output
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from src.react_loop import GodelAgent, AgentState, AgentAction, ActionType
        print("  ✓ Core imports successful")
    except Exception as e:
        print(f"  ✗ Core imports failed: {e}")
        return False

    try:
        from src.react_loop.state import EvolutionPhase
        print("  ✓ State module imports successful")
    except Exception as e:
        print(f"  ✗ State module imports failed: {e}")
        return False

    try:
        from src.react_loop.utils.tools import build_openai_tools, scan_external_tools
        tools = build_openai_tools([])
        print(f"  ✓ Tools module loaded ({len(tools)} tools)")
    except Exception as e:
        print(f"  ✗ Tools module imports failed: {e}")
        return False

    try:
        from src.react_loop.git_version import GitController, EvolutionTracker
        print("  ✓ Git version module imports successful")
    except Exception as e:
        print(f"  ✗ Git version module imports failed: {e}")
        return False

    try:
        from src.react_loop.utils import HarnessLoader, CodeValidator
        print("  ✓ Utils module imports successful")
    except Exception as e:
        print(f"  ✗ Utils module imports failed: {e}")
        return False

    try:
        from src.react_loop.evolution_prompt import (
            get_base_system_prompt,
            get_iteration_begin_prompt,
            YOUR_TASK_TEXT,
        )
        print("  ✓ Prompts module imports successful")
    except Exception as e:
        print(f"  ✗ Prompts module imports failed: {e}")
        return False

    return True


def test_state():
    """Test state management."""
    print("\nTesting state management...")

    from src.react_loop.state import AgentState, AgentAction, ActionType

    # Create state
    state = AgentState(iteration=0, goal="Test goal")
    print(f"  Created state with goal: {state.goal}")

    # Test action recording (using BASH instead of old INTROSPECT)
    action = AgentAction(action_type=ActionType.BASH)
    state.record_action(action)
    print(f"  Recorded action: {action.action_type.value}")

    # Test π_t collection
    state.update_pi({"solver.py": "test code"}, "/path/to/code.py")
    print(f"  π_t codes: {len(state.pi_codes)} file(s)")

    # Test S_t collection
    state.update_environment("Test environment")
    print(f"  S_t summary: {state.environment_summary}")

    # Test r_t collection
    state.update_reward(0.75)
    print(f"  r_t reward: {state.reward:.4f}")

    # Test iteration NOT complete yet (needs compact_context)
    is_not_complete = not state.is_iteration_complete()
    print(f"  Iteration not complete (before compact): {is_not_complete}")

    # Test mark_iteration_ended
    state.mark_iteration_ended(summary="Test iteration", reason="testing")
    is_complete = state.is_iteration_complete()
    print(f"  Iteration complete (after compact): {is_complete}")

    # Test mark_evolution_ended
    state2 = AgentState(iteration=1, goal="Test goal 2")
    state2.mark_evolution_ended(summary="Done", reason="perfect")
    assert state2.iteration_ended
    assert state2.evolution_ended
    print(f"  mark_evolution_ended works: {state2.evolution_ended}")

    return is_complete and is_not_complete and state2.evolution_ended


def test_git_controller():
    """Test git controller."""
    print("\nTesting git controller...")

    from src.react_loop.git_version import GitController

    controller = GitController(".")
    is_repo = controller.is_git_repo()
    print(f"  Is git repo: {is_repo}")

    if is_repo:
        commit = controller.get_current_commit()
        print(f"  Current commit: {commit[:7] if commit else 'None'}")

    return True


def test_tools():
    """Test tools module."""
    print("\nTesting tools...")

    from src.react_loop.utils.tools import build_openai_tools

    tools = build_openai_tools([])
    tool_names = [t['function']['name'] for t in tools]
    print(f"  Loaded {len(tools)} tools:")
    for name in tool_names:
        print(f"    - {name}")

    # Verify new control tools exist
    required_tools = ["compact_context", "end_evolution"]
    missing = [t for t in required_tools if t not in tool_names]
    if missing:
        print(f"  ✗ Missing tools: {missing}")
        return False
    print(f"  ✓ Control tools registered: {required_tools}")
    return True


def test_action_types():
    """Test ActionType enum."""
    print("\nTesting ActionType enum...")

    from src.react_loop.state import ActionType

    expected_types = ["bash", "read_history_self", "evaluate", "external_tool",
                      "get_historic_version",
                      "read_file", "edit_file", "write_file",
                      "compact_context", "end_evolution"]

    for t in expected_types:
        try:
            at = ActionType(t)
            print(f"  ✓ ActionType.{at.name} = '{t}'")
        except ValueError:
            print(f"  ✗ ActionType '{t}' not found")
            return False

    return True


def test_lcb():
    """Test Lower-Confidence-Bound reward helpers (uncertainty-aware selection)."""
    print("\nTesting LCB reward helpers...")
    from src.react_loop.state import lower_confidence_bound

    ok = True

    # n < 2 → mean (stdev undefined, no penalty)
    ok &= abs(lower_confidence_bound([0.5]) - 0.5) < 1e-9
    print("  ✓ n<2 returns mean (no penalty)")

    # empty → 0.0
    ok &= lower_confidence_bound([]) == 0.0
    print("  ✓ empty returns 0.0")

    # all-identical → std=0 → mean
    ok &= abs(lower_confidence_bound([0.8, 0.8, 0.8]) - 0.8) < 1e-9
    print("  ✓ all-identical returns mean")

    # [0.9, 0.7], z=1.0 → mean 0.8, std 0.14142 → 0.8 - 0.14142/√2 = 0.7
    val = lower_confidence_bound([0.9, 0.7], z=1.0)
    ok &= abs(val - 0.7) < 1e-6
    print(f"  ✓ [0.9,0.7] z=1.0 → {val:.4f} (≈0.7)")

    # Higher z penalizes harder (more conservative)
    lo = lower_confidence_bound([0.9, 0.7], z=0.5)
    hi = lower_confidence_bound([0.9, 0.7], z=2.0)
    ok &= hi < lo < 0.8
    print(f"  ✓ higher z → lower bound (z=0.5:{lo:.4f} > z=2.0:{hi:.4f})")

    # Core jackpot-rejection: jittery LCB < stable LCB at the SAME mean.
    jittery = lower_confidence_bound([0.9, 0.5])   # mean 0.7, high variance
    stable = lower_confidence_bound([0.7, 0.7])    # mean 0.7, no variance
    ok &= jittery < stable
    print(f"  ✓ jittery {jittery:.4f} < stable {stable:.4f} at same mean 0.7")

    if not ok:
        print("  ✗ LCB unit checks failed")
    return ok


def main():
    """Run all tests."""
    print("=" * 60)
    print("React Loop Agent - Test Suite")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("State", test_state),
        ("Git Controller", test_git_controller),
        ("Tools", test_tools),
        ("ActionTypes", test_action_types),
        ("LCB Reward", test_lcb),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ Test failed with exception: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)

    passed = 0
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")
        if result:
            passed += 1

    print(f"\nTotal: {passed}/{len(tests)} tests passed")

    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
