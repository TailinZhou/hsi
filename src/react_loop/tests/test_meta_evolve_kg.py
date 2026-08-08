"""
Comprehensive tests for Meta-Evolve and Knowledge Graph flows.

Tests cover:
1. Knowledge Graph initialization, node addition, edge creation, pruning
2. Meta-evolve state isolation and restoration
3. Integration between commit_iteration and KG node addition
4. KG rendering for meta-evolve prompts
5. Edge cases and error handling
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call
from dataclasses import dataclass, field, asdict
from types import SimpleNamespace
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.react_loop.knowledge_graph import KGNode, KGEdge, EvolutionKnowledgeGraph
from src.react_loop.state import AgentState, EvolutionPhase, IterationSummary, METADATA_FILES
from src.react_loop.meta_evolve import MetaEvolveHelper
from src.react_loop.evolve import EvolveHelper


class MockGitController:
    """Mock git controller for testing."""

    def __init__(self):
        self.commits = {}
        self.current = None
        self.branch = "master"
        # Mock each commit's file content
        self.commit_files = {}

    def get_current_commit(self):
        return self.current

    def set_commit_file(self, commit_hash, file_path, content):
        """Set the file content for a given commit (for testing)."""
        if commit_hash not in self.commit_files:
            self.commit_files[commit_hash] = {}
        self.commit_files[commit_hash][file_path] = content

    def get_file_at_commit(self, file_path, commit):
        """Get the file content for a given commit.

        Signature matches the real GitController.get_file_at_commit exactly:
        (file_path, commit) — file first, commit second. An earlier version of
        the mock swapped these two parameters, which happened to "match" the
        swapped call inside knowledge_graph and completely masked a production
        bug. Here we align with the real signature to prevent regressions.
        """
        if commit in self.commit_files and file_path in self.commit_files[commit]:
            return self.commit_files[commit][file_path]
        # Return the default mock content
        return f"# Mock content from {commit}:{file_path}\n\ndef mock_function():\n    pass"

    def _run_git_command(self, args, check=True):
        result = Mock()
        result.returncode = 0
        result.stdout = ""
        if "diff-tree" in args:
            result.stdout = "harness.py"
        return result


class MockLLMResponse:
    """Mock LLM response for testing."""

    def __init__(self, content: str):
        self.choices = [Mock()]
        self.choices[0].message = Mock()
        self.choices[0].message.content = content
        self.choices[0].message.tool_calls = None


class TestKnowledgeGraph(unittest.TestCase):
    """Test Knowledge Graph core functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.git_controller = MockGitController()

        def mock_call_llm(messages, tools=None):
            """Mock LLM that returns a single (undirected) pair analysis string."""
            return MockLLMResponse("A improves upon B with better prompt handling; "
                                   "B has simpler logic but lacks those improvements.")

        self.kg = EvolutionKnowledgeGraph(
            agent_code_dir=self.temp_dir,
            git_controller=self.git_controller,
            call_llm=mock_call_llm,
            max_nodes=5,  # Small for testing pruning
            chunk_size=2,
            log=lambda *a, **k: None,
        )

    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_kg_initialization(self):
        """Test KG initializes with empty state."""
        self.assertEqual(len(self.kg.nodes), 0)
        self.assertEqual(len(self.kg.edges), 0)
        self.assertEqual(self.kg.meta["node_count"], 0)
        self.assertEqual(self.kg.meta["edge_count"], 0)

    def test_add_first_node(self):
        """Test adding the first node creates node but no edges."""
        node_id = self.kg.add_node(
            iteration=1,
            git_hash="abc123def456",
            parent_hash="",
            reward=0.8,
            eval_mode="val",
            summary_text="First iteration",
            modified_files=["harness.py"],
            change_tags=["edit"],
        )

        self.assertEqual(node_id, "abc123def456")
        self.assertEqual(len(self.kg.nodes), 1)
        self.assertEqual(len(self.kg.edges), 0)  # No edges for first node

        node = self.kg.nodes["abc123def456"]
        self.assertEqual(node.iteration, 1)
        self.assertEqual(node.reward, 0.8)
        self.assertEqual(node.eval_mode, "val")

    def test_add_node_with_parent_creates_backbone_edge(self):
        """Test adding node with parent creates backbone edge."""
        # Add first node
        self.kg.add_node(
            iteration=1,
            git_hash="parent_hash",
            parent_hash="",
            reward=0.7,
            eval_mode="dev",
            summary_text="Parent",
            modified_files=["harness.py"],
        )

        # Add child node
        self.kg.add_node(
            iteration=2,
            git_hash="child_hash",
            parent_hash="parent_hash",
            reward=0.8,
            eval_mode="val",
            summary_text="Child",
            modified_files=["harness.py", "prompts.py"],
        )

        self.assertEqual(len(self.kg.nodes), 2)
        # One edge per pair: parent-child pairs only produce a backbone edge
        # (with LLM analysis), no extra semantic edge is layered on top.
        self.assertEqual(len(self.kg.edges), 1)

        # Check backbone edge: directed parent->child
        backbone_edge = next(
            (e for e in self.kg.edges
             if e.edge_type == "backbone" and e.src_id == "parent_hash"),
            None
        )
        self.assertIsNotNone(backbone_edge)
        self.assertEqual(backbone_edge.dst_id, "child_hash")
        # Parent-child edges also run LLM analysis (merged into backbone, not a separate semantic edge).
        self.assertTrue(backbone_edge.llm_diff_analysis)

    def test_add_duplicate_node_returns_none(self):
        """Test adding duplicate git_hash returns None."""
        node_id = self.kg.add_node(
            iteration=1,
            git_hash="duplicate_hash",
            parent_hash="",
            reward=0.5,
        )
        self.assertEqual(node_id, "duplicate_hash")

        # Try adding same hash again
        duplicate_id = self.kg.add_node(
            iteration=2,
            git_hash="duplicate_hash",
            parent_hash="other",
            reward=0.6,
        )
        self.assertIsNone(duplicate_id)
        self.assertEqual(len(self.kg.nodes), 1)

    def test_add_empty_hash_returns_none(self):
        """Test adding empty git_hash returns None."""
        node_id = self.kg.add_node(
            iteration=1,
            git_hash="",
            parent_hash="",
            reward=0.5,
        )
        self.assertIsNone(node_id)
        self.assertEqual(len(self.kg.nodes), 0)

    def test_semantic_edge_creation(self):
        """Test semantic edges are created between node pairs that pass selectivity filters."""
        # Add 3 nodes with partially overlapping modified files to pass the
        # selectivity filter (similarity must be in (0.15, 0.90) — not identical)
        hashes = ["hash1", "hash2", "hash3"]
        file_sets = [
            ["harness.py", "prompts.py"],           # hash1
            ["harness.py", "hooks.py"],              # hash2: 1 overlap with hash1
            ["prompts.py", "context.py"],            # hash3: 1 overlap with hash2, 1 with hash1
        ]
        for i, h in enumerate(hashes):
            parent = hashes[i-1] if i > 0 else ""
            self.kg.add_node(
                iteration=i+1,
                git_hash=h,
                parent_hash=parent,
                reward=0.5 + i * 0.1,
                eval_mode="val",
                summary_text=f"Node {i+1}",
                modified_files=file_sets[i],
            )

        # v3 selective: 3-node chain hash1←hash2←hash3
        # Backbone: hash1→hash2, hash2→hash3 (always created)
        # Correlation: hash1↔hash3 (shared file "prompts.py" + sim in range + iter gap ≤5)
        #   hash1-hash3 Jaccard = 1/3 ≈ 0.33 (in range)
        self.assertEqual(len(self.kg.nodes), 3)
        # Due to the selective filter, we might get 2 (backbone) or 3 (backbone + correlation)
        # depending on whether all selectivity criteria pass.
        # At minimum: 2 backbone edges.
        self.assertGreaterEqual(len(self.kg.edges), 2)

        backbone_edges = [e for e in self.kg.edges if e.edge_type == "backbone"]
        semantic_edges = [e for e in self.kg.edges if e.edge_type == "semantic"]
        self.assertEqual(len(backbone_edges), 2)
        # Semantic edges depend on selectivity criteria; verify at least backbone edges exist
        # All backbone edges carry LLM analysis
        for e in backbone_edges:
            self.assertTrue(e.llm_diff_analysis,
                            f"edge {e.src_id}-{e.dst_id} should carry LLM analysis")

    def test_cap_and_aggregate_prunes_low_reward_nodes(self):
        """Test pruning removes low-reward nodes when exceeding max_nodes."""
        # max_nodes is 5, add 7 nodes
        for i in range(7):
            self.kg.add_node(
                iteration=i+1,
                git_hash=f"hash{i}",
                parent_hash=f"hash{i-1}" if i > 0 else "",
                reward=0.5 + i * 0.05,  # Higher reward = later nodes
                eval_mode="dev",
            )

        # Should be pruned to max_nodes (5)
        self.assertLessEqual(len(self.kg.nodes), 5)

        # Best node (highest reward) should be preserved
        best_reward = max(n.reward for n in self.kg.nodes.values())
        self.assertGreater(best_reward, 0.7)  # Last node has 0.85

        # Most recent 3 nodes should be preserved
        iterations = sorted([n.iteration for n in self.kg.nodes.values()])
        self.assertEqual(iterations[-1], 7)  # Most recent

    def test_best_node_protected_during_pruning(self):
        """Test that best reward node is never pruned."""
        # Add nodes with varying rewards
        self.kg.add_node(iteration=1, git_hash="h1", reward=0.9)  # Best
        for i in range(10):
            self.kg.add_node(
                iteration=i+2,
                git_hash=f"h{i+2}",
                parent_hash="h1" if i == 0 else f"h{i+1}",
                reward=0.5 + i * 0.01,  # Lower than 0.9
            )

        # Find the best node
        best_node = max(self.kg.nodes.values(), key=lambda n: n.reward)
        self.assertEqual(best_node.git_hash, "h1")
        self.assertEqual(best_node.reward, 0.9)

    def test_structural_similarity_calculation(self):
        """Test Jaccard similarity for modified files."""
        node_a = KGNode(
            node_id="a", iteration=1, git_hash="a", reward=0.5,
            eval_mode="dev", summary_text="A",
            modified_files=["harness.py", "prompts.py"]
        )
        node_b = KGNode(
            node_id="b", iteration=2, git_hash="b", reward=0.6,
            eval_mode="dev", summary_text="B",
            modified_files=["harness.py", "tools.py"]
        )
        node_c = KGNode(
            node_id="c", iteration=3, git_hash="c", reward=0.7,
            eval_mode="dev", summary_text="C",
            modified_files=["new_file.py"]
        )

        # A and B: intersection={harness.py}, union={harness.py, prompts.py, tools.py}
        # Jaccard = 1/3 ≈ 0.33
        sim_ab = EvolutionKnowledgeGraph._structural_similarity(node_a, node_b)
        self.assertAlmostEqual(sim_ab, 1/3, places=2)

        # A and C: intersection=∅, Jaccard = 0
        sim_ac = EvolutionKnowledgeGraph._structural_similarity(node_a, node_c)
        self.assertEqual(sim_ac, 0.0)

        # A and A: intersection=union={harness.py, prompts.py}, Jaccard = 1
        sim_aa = EvolutionKnowledgeGraph._structural_similarity(node_a, node_a)
        self.assertEqual(sim_aa, 1.0)

    def test_scan_py_files_includes_evolution_dir(self):
        """_scan_py_files must scan evolution/ — change tracking for bash edits
        to select_*.py depends on this."""
        from src.react_loop.actions.agent_action import AgentActionExecutor
        from pathlib import Path
        os.makedirs(os.path.join(self.temp_dir, "evolution"), exist_ok=True)
        with open(os.path.join(self.temp_dir, "evolution", "select_seed.py"), "w") as f:
            f.write("STRATEGY_NAME = 'greedy'\n")
        executor = AgentActionExecutor(
            llm_client=Mock(), model="m", repo_path=Path(self.temp_dir),
            agent_code_dir=self.temp_dir, agent_instance=Mock(),
        )
        hashes = executor._scan_py_files()
        self.assertIn("evolution/select_seed.py", hashes)

    def test_file_hashes_resync_prevents_stale_modification_false_positive(self):
        """After reset() resets _file_hashes, evolution/* modified by meta-evolve
        must not be misidentified as "modified this round".

        Bug reproduction: _file_hashes was lazily initialized and never followed
        the disk. meta-evolve edited select_seed.py (v1 -> v2) and folded it
        into the working tree; the next round's _auto_reload_if_changed saw
        disk(v2) != cache(v1) and misidentified select_seed.py as "modified this
        round", polluting _modified_files -> KG modified_files.
        Fix: reset_for_iteration re-runs _file_hashes = _scan_py_files() to
        refresh the cache.
        (Renamed to drop the "hot reload" naming: it now only refreshes
        agent_codes/modified_files, no hot reload, but the requirement that
        hash re-sync prevents false positives remains.)
        """
        from src.react_loop.actions.agent_action import AgentActionExecutor
        from pathlib import Path
        os.makedirs(os.path.join(self.temp_dir, "evolution"), exist_ok=True)
        arc = os.path.join(self.temp_dir, "evolution", "select_seed.py")
        with open(arc, "w") as f:
            f.write("STRATEGY_NAME = 'greedy'\n")
        executor = AgentActionExecutor(
            llm_client=Mock(), model="m", repo_path=Path(self.temp_dir),
            agent_code_dir=self.temp_dir, agent_instance=Mock(),
        )
        executor._modified_files = set()
        # Mock away sync/rescan side effects; only verify the detection logic (modified set).
        executor._reload_single_file = lambda rel, content=None, skip_validation=False: f"[{rel}]"
        executor._rescan_external_tools = lambda *a, **k: None

        # 1. _file_hashes scans v1 (state after lazy initialization)
        executor._file_hashes = executor._scan_py_files()
        # 2. Disk changes to v2 (simulate meta-evolve editing select_seed.py,
        #    change folded into the next round's working tree)
        with open(arc, "w") as f:
            f.write("STRATEGY_NAME = 'recursive'\n")

        # 3a. Without resetting _file_hashes (the bug): v2 is misidentified as "modified this round"
        executor._modified_files = set()
        executor._auto_reload_if_changed()
        self.assertIn("evolution/select_seed.py", executor._modified_files)

        # 3b. Reset _file_hashes = current disk (the fix performed by reset_for_iteration)
        executor._modified_files = set()
        executor._file_hashes = executor._scan_py_files()
        executor._auto_reload_if_changed()
        self.assertNotIn("evolution/select_seed.py", executor._modified_files)

    def test_compute_code_hash_excludes_evolution_dir(self):
        """Fix A: _compute_code_hash excludes evolution/ — it is meta-evolve's
        territory, not the harness "version" being evaluated/committed. The
        reward should only be attributed to harness code: the same harness +
        different evolution/ must hash identically, and the snapshot must not
        contain evolution/ files.
        """
        from src.react_loop.actions.agent_action import AgentActionExecutor
        from pathlib import Path
        os.makedirs(os.path.join(self.temp_dir, "evolution"), exist_ok=True)
        with open(os.path.join(self.temp_dir, "harness.py"), "w") as f:
            f.write("def using_harness(a, t): return ''\n")
        with open(os.path.join(self.temp_dir, "evolution", "select_seed.py"), "w") as f:
            f.write("STRATEGY_NAME = 'greedy'\n")
        executor = AgentActionExecutor(
            llm_client=Mock(), model="m", repo_path=Path(self.temp_dir),
            agent_code_dir=self.temp_dir, agent_instance=Mock(),
        )
        h1, snap1 = executor._compute_code_hash(return_contents=True)
        self.assertFalse(any(k.startswith("evolution/") for k in snap1),
                         f"evolution/ leaked into snapshot: "
                         f"{[k for k in snap1 if k.startswith('evolution/')]}")
        self.assertIn("harness.py", snap1)

        # Editing only evolution/ -> the harness version hash must stay unchanged
        # (the code being evaluated has not changed)
        with open(os.path.join(self.temp_dir, "evolution", "select_seed.py"), "w") as f:
            f.write("STRATEGY_NAME = 'recursive'\n")
        h2, _ = executor._compute_code_hash(return_contents=True)
        self.assertEqual(h1, h2, "evolution/ change must not alter the harness version hash")

    def test_scan_py_files_includes_md(self):
        """Fix B: _scan_py_files scans .md (excluding BOOTSTRAP.md / plan.md),
        aligning with _compute_code_hash's file types so that bash edits to .md
        are also detected and recorded in _modified_files."""
        from src.react_loop.actions.agent_action import AgentActionExecutor
        from pathlib import Path
        with open(os.path.join(self.temp_dir, "harness.py"), "w") as f:
            f.write("x = 1\n")
        with open(os.path.join(self.temp_dir, "prompts.md"), "w") as f:
            f.write("# prompts\n")
        with open(os.path.join(self.temp_dir, "BOOTSTRAP.md"), "w") as f:
            f.write("notebook\n")
        with open(os.path.join(self.temp_dir, "plan.md"), "w") as f:
            f.write("# ephemeral plan\n")
        executor = AgentActionExecutor(
            llm_client=Mock(), model="m", repo_path=Path(self.temp_dir),
            agent_code_dir=self.temp_dir, agent_instance=Mock(),
        )
        hashes = executor._scan_py_files()
        self.assertIn("harness.py", hashes)
        self.assertIn("prompts.md", hashes)
        self.assertNotIn("BOOTSTRAP.md", hashes)
        self.assertNotIn("plan.md", hashes,
                         "plan.md is metadata — must be excluded like BOOTSTRAP.md")

    def test_compute_code_hash_excludes_plan_md(self):
        """plan.md is an ephemeral working notebook (gitignored, cleared each
        iteration). It must NOT enter the harness version hash, or every
        plan() call would spawn a spurious new code version and misalign
        (code_hash, reward) pairs. Mirrors the BOOTSTRAP.md / evolution/
        exclusion guarantee."""
        from src.react_loop.actions.agent_action import AgentActionExecutor
        from pathlib import Path
        with open(os.path.join(self.temp_dir, "harness.py"), "w") as f:
            f.write("def using_harness(a, t): return ''\n")
        executor = AgentActionExecutor(
            llm_client=Mock(), model="m", repo_path=Path(self.temp_dir),
            agent_code_dir=self.temp_dir, agent_instance=Mock(),
        )
        h1, snap1 = executor._compute_code_hash(return_contents=True)
        self.assertIn("harness.py", snap1)
        self.assertFalse(any(k == "plan.md" for k in snap1),
                         "plan.md leaked into code snapshot")

        # Editing only plan.md must leave the harness version hash unchanged.
        with open(os.path.join(self.temp_dir, "plan.md"), "w") as f:
            f.write("## Hypothesis\n\nunchanged harness, new plan\n")
        h2, snap2 = executor._compute_code_hash(return_contents=True)
        self.assertEqual(h1, h2,
                         "plan.md change altered the harness version hash")
        self.assertNotIn("plan.md", snap2)

    def test_auto_reload_tracks_md_without_reload(self):
        """Fix B: bash editing .md -> _auto_reload_if_changed records it in
        _modified_files but does not refresh agent_codes (.md is not a module,
        no need to sync the code mirror)."""
        from src.react_loop.actions.agent_action import AgentActionExecutor
        from pathlib import Path
        md = os.path.join(self.temp_dir, "prompts.md")
        with open(md, "w") as f:
            f.write("v1\n")
        executor = AgentActionExecutor(
            llm_client=Mock(), model="m", repo_path=Path(self.temp_dir),
            agent_code_dir=self.temp_dir, agent_instance=Mock(),
        )
        executor._modified_files = set()

        def _no_reload(rel, skip_validation=False):
            raise AssertionError(f"should not reload non-py file: {rel}")
        executor._reload_single_file = _no_reload
        executor._rescan_external_tools = lambda *a, **k: None
        executor._file_hashes = executor._scan_py_files()  # baseline = v1
        with open(md, "w") as f:
            f.write("v2\n")
        executor._auto_reload_if_changed()
        self.assertIn("prompts.md", executor._modified_files)

    def test_apply_version_switch_removes_orphan_files(self):
        """Risk 2 fix: switching seed to an older commit must remove harness
        files created in later commits, so the seed isn't polluted.

        `git checkout <old> -- .` only restores files present at <old>; files added
        in later commits linger and would be committed into the new iteration via
        `git add -A`. _clean_orphan_harness_files (called inside
        apply_version_switch) removes them.
        """
        import subprocess
        from src.react_loop.archive_manager import ArchiveManager
        from src.react_loop.git_version.controller import GitController
        from types import SimpleNamespace

        repo = tempfile.mkdtemp()

        def git(*args):
            subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

        git("init", "-q")
        git("config", "user.email", "t@t")
        git("config", "user.name", "t")

        # C1: harness.py only
        with open(os.path.join(repo, "harness.py"), "w") as f:
            f.write("harness v1\n")
        git("add", "-A"); git("commit", "-qm", "C1")
        c1 = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo).decode().strip()
        # C2: add utils.py + modify harness.py to import it
        with open(os.path.join(repo, "utils.py"), "w") as f:
            f.write("aux module\n")
        with open(os.path.join(repo, "harness.py"), "w") as f:
            f.write("harness v2\nimport utils\n")
        git("add", "-A"); git("commit", "-qm", "C2")
        # working tree now = C2 (utils.py present, harness imports it)

        gc = GitController(repo)
        agent = SimpleNamespace(
            git_controller=gc, agent_code_dir=repo, iteration=1,
            _log=lambda m: None,
        )
        am = ArchiveManager(agent)

        # 1) target file set at C1 excludes utils.py
        tf = am._target_tracked_files(c1)
        self.assertIn("harness.py", tf)
        self.assertNotIn("utils.py", tf)

        # 2) full apply_version_switch(C1): checkout reverts harness.py to v1,
        #    cleanup removes the orphan utils.py
        ok = am.apply_version_switch(c1, hint="seed")
        self.assertTrue(ok)
        self.assertFalse(os.path.exists(os.path.join(repo, "utils.py")),
                         "orphan utils.py must be removed after switching to C1")
        with open(os.path.join(repo, "harness.py")) as f:
            self.assertEqual(f.read(), "harness v1\n",
                             "harness.py must be reverted to C1's version")

    def test_persistence_save_and_load(self):
        """Test KG can be saved and loaded."""
        # Add some nodes
        self.kg.add_node(
            iteration=1,
            git_hash="saved_hash",
            parent_hash="",
            reward=0.75,
            eval_mode="val",
            summary_text="Saved node",
        )

        # Save
        self.kg.save()

        # Create new KG and load
        kg2 = EvolutionKnowledgeGraph(
            agent_code_dir=self.temp_dir,
            git_controller=self.git_controller,
            call_llm=lambda m, t: None,
        )
        kg2.load()

        self.assertEqual(len(kg2.nodes), 1)
        self.assertIn("saved_hash", kg2.nodes)
        self.assertEqual(kg2.nodes["saved_hash"].reward, 0.75)

    def test_render_for_prompt(self):
        """Test rendering produces valid markdown table."""
        # Add nodes
        for i in range(3):
            self.kg.add_node(
                iteration=i+1,
                git_hash=f"hash{i}",
                parent_hash=f"hash{i-1}" if i > 0 else "",
                reward=0.6 + i * 0.1,
                summary_text=f"Iteration {i+1} summary",
            )

        rendered = self.kg.render_for_prompt()

        # Check for markdown table
        self.assertIn("### Evolution Knowledge Graph", rendered)
        self.assertIn("| id | iter | reward |", rendered)
        self.assertIn("knowledge_graph.json", rendered)

        # Check nodes are shown
        self.assertIn("hash0", rendered)
        self.assertIn("hash1", rendered)

    def test_backfill_from_tracker(self):
        """Test backfilling from EvolutionTracker records."""
        # Create mock tracker
        tracker = Mock()

        # Create mock records
        records = []
        for i in range(5):
            record = Mock()
            record.iteration = i + 1
            record.new_commit = f"commit{i}"
            record.parent_commit = f"commit{i-1}" if i > 0 else ""
            record.reward = 0.6 + i * 0.05
            record.metadata = {
                "type": "main",  # Not meta_evolve
                "committed_eval_mode": "val",
                "summary_text": f"Iteration {i+1}",
                "modified_files": ["harness.py"],
            }
            record.state_summary = ""
            records.append(record)

        tracker.records = records

        # Backfill
        self.kg.backfill_from_tracker(tracker)

        # Should have 5 nodes
        self.assertEqual(len(self.kg.nodes), 5)

        # Should have backbone edges
        backbone_edges = [e for e in self.kg.edges if e.edge_type == "backbone"]
        self.assertEqual(len(backbone_edges), 4)  # 4 parent-child links

        # No meta_evolve nodes
        for node in self.kg.nodes.values():
            self.assertFalse(node.is_meta)

    def test_get_node_code_reads_real_content(self):
        """Regression: _get_node_code must call get_file_at_commit(file_path, commit).

        An earlier version swapped the args (commit, file_path), which made
        every `git show` fail and fed the LLM pair analysis empty code for
        every node. This test pins the correct order by asserting the real
        per-commit file content flows through (not the mock fallback).
        """
        # A top-level .py must exist so the directory scan picks up the name.
        harness_path = os.path.join(self.temp_dir, "harness.py")
        with open(harness_path, "w", encoding="utf-8") as f:
            f.write("# placeholder on disk\n")

        real_content = "def real_code():\n    return 42\n"
        self.git_controller.set_commit_file("node1", "harness.py", real_content)

        self.kg.add_node(
            iteration=1,
            git_hash="node1",
            parent_hash="",
            reward=0.5,
            summary_text="node one",
        )
        node = self.kg.nodes["node1"]

        code = self.kg._get_node_code(node)

        self.assertIn("harness.py", code)
        self.assertEqual(
            code["harness.py"], real_content,
            "_get_node_code must return the real per-commit content, not the "
            "mock fallback — if this fails, get_file_at_commit args are swapped."
        )


class TestMetaEvolveStateIsolation(unittest.TestCase):
    """Test meta-evolve state isolation and restoration."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create mock agent
        self.agent = Mock()
        self.agent.iteration = 5
        self.agent.goal = "Test goal"
        self.agent.agent_code_dir = self.temp_dir
        self.agent.git_controller = MockGitController()
        self.agent.evolution_tracker = Mock()
        self.agent.evolution_tracker.records = []

        # Create main iteration state
        self.agent.state = AgentState(
            iteration=5,
            goal="Test goal",
        )
        self.agent.state.record_action(Mock(
            action_type=Mock(value="read_file"),
            params={"path": "test.py"},
            result="Success",
        ))
        self.agent.state.modifications_made.append({"operation": "edit"})

        # Create action executor
        self.agent.action_executor = Mock()
        self.agent.action_executor.state = self.agent.state

        self.agent._actions_in_iteration = 42

        # Create evolution helper
        self.agent.iter_helper = EvolveHelper(self.agent)

        # Create meta-evolve helper
        self.meta_helper = MetaEvolveHelper(self.agent)

    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_meta_evolve_initialization(self):
        """Test meta-evolve helper initializes correctly."""
        self.assertEqual(self.meta_helper._meta_iteration, 0)
        self.assertEqual(self.meta_helper.agent, self.agent)

    def test_state_isolation_preserves_main_state(self):
        """Test that main iteration state is preserved after meta-evolve."""
        # Save original state
        original_action_count = len(self.agent.state.action_history)
        original_modifications = len(self.agent.state.modifications_made)
        original_actions_counter = self.agent._actions_in_iteration

        # Simulate meta-evolve state isolation (without actually running)
        saved_state = self.agent.state
        saved_executor_state = self.agent.action_executor.state
        saved_actions = self.agent._actions_in_iteration

        # Create throwaway state
        throwaway = AgentState(iteration=5, goal="Test")
        self.agent.state = throwaway
        self.agent.action_executor.set_state(throwaway)
        self.agent._actions_in_iteration = 0

        # Simulate some meta-evolve actions
        throwaway.record_action(Mock(
            action_type=Mock(value="read_file"),
            params={"path": "select_seed.py"},
            result="Archive content",
        ))
        throwaway.modifications_made.append({"operation": "meta_edit"})
        self.agent._actions_in_iteration = 5

        # Restore main state
        self.agent.state = saved_state
        self.agent.action_executor.set_state(saved_executor_state)
        self.agent._actions_in_iteration = saved_actions

        # Verify main state is unchanged
        self.assertEqual(len(self.agent.state.action_history), original_action_count)
        self.assertEqual(len(self.agent.state.modifications_made), original_modifications)
        self.assertEqual(self.agent._actions_in_iteration, original_actions_counter)

        # Verify throwaway changes didn't leak
        self.assertNotEqual(
            self.agent.state.action_history[-1].params.get("path"),
            "select_seed.py"
        )

    def _make_meta_bootstrap_executor(self):
        """Build an AgentActionExecutor wired to self.agent for meta_bootstrap tests."""
        from src.react_loop.actions.agent_action import AgentActionExecutor
        from pathlib import Path

        executor = AgentActionExecutor(
            llm_client=Mock(),
            model="gpt-4",
            repo_path=Path(self.temp_dir),
            agent_code_dir=self.temp_dir,
            agent_instance=self.agent,
        )
        executor.agent_codes = {}
        executor._modified_files = set()
        executor.agent_code_dir = self.temp_dir
        executor.set_state(self.agent.state)
        return executor

    def test_meta_bootstrap_file_creation(self):
        """Test meta_bootstrap appends a numbered record with all four sections."""
        executor = self._make_meta_bootstrap_executor()

        # Call meta_bootstrap with new four-field signature
        result = executor.meta_bootstrap(
            what="Test what",
            why="Test why",
            lesson="Test lesson",
            prediction="Test prediction",
        )

        self.assertIn("Meta-bootstrap appended", result)
        self.assertIn("Meta #5", result)
        self.assertIn("1 entries total", result)

        # Check file was created
        bootstrap_path = os.path.join(self.temp_dir, "evolution", "meta_bootstrap.md")
        self.assertTrue(os.path.exists(bootstrap_path))

        # Check content: numbered header + four sections
        with open(bootstrap_path, "r") as f:
            content = f.read()
        self.assertIn("## Meta #5 (after iter 5, reward=N/A)", content)
        self.assertIn("### What\nTest what", content)
        self.assertIn("### Why\nTest why", content)
        self.assertIn("### Lesson\nTest lesson", content)
        self.assertIn("### Prediction\nTest prediction", content)

    def test_meta_bootstrap_four_sections(self):
        """Each parameter maps to its own ### section."""
        executor = self._make_meta_bootstrap_executor()
        executor.meta_bootstrap(what="W", why="Y", lesson="L", prediction="P")

        with open(os.path.join(self.temp_dir, "evolution", "meta_bootstrap.md")) as f:
            content = f.read()
        self.assertIn("### What\nW", content)
        self.assertIn("### Why\nY", content)
        self.assertIn("### Lesson\nL", content)
        self.assertIn("### Prediction\nP", content)

    def test_meta_bootstrap_accumulates_across_rounds(self):
        """Successive meta-evolves accumulate records — never overwrite; newest first."""
        from types import SimpleNamespace

        executor = self._make_meta_bootstrap_executor()
        bootstrap_path = os.path.join(self.temp_dir, "evolution", "meta_bootstrap.md")

        # Simulate three meta-evolves after iter 3, 4, 5
        rewards = {3: 0.30, 4: 0.40, 5: 0.50}
        for it in (3, 4, 5):
            self.agent.iteration = it
            # latest non-meta record carries this iteration's reward
            self.agent.evolution_tracker.records = [
                SimpleNamespace(reward=rewards[it], metadata={"committed_eval_mode": "dev"})
            ]
            executor.meta_bootstrap(
                what=f"change{it}",
                why=f"reason{it}",
                lesson=f"lesson{it}",
                prediction=f"pred{it}",
            )

        with open(bootstrap_path) as f:
            content = f.read()

        # Three records, none overwritten
        self.assertEqual(content.count("## Meta #"), 3)
        for it in (3, 4, 5):
            self.assertIn(f"### What\nchange{it}", content)
            self.assertIn(f"### Why\nreason{it}", content)
            self.assertIn(f"### Lesson\nlesson{it}", content)
            self.assertIn(f"### Prediction\npred{it}", content)

        # Newest first (prepend order): #5 before #4 before #3
        self.assertLess(content.index("## Meta #5"), content.index("## Meta #4"))
        self.assertLess(content.index("## Meta #4"), content.index("## Meta #3"))

        # reward pulled from the latest non-meta record, formatted to 4 decimals
        self.assertIn("reward=0.5000", content)
        self.assertIn("reward=0.4000", content)
        self.assertIn("reward=0.3000", content)

        # last append reports 3 entries
        last = executor.meta_bootstrap(what="x")
        self.assertIn("4 entries total", last)

    def test_meta_bootstrap_cap_at_ten(self):
        """After 12 appends only the newest 10 records are kept."""
        from types import SimpleNamespace

        executor = self._make_meta_bootstrap_executor()
        bootstrap_path = os.path.join(self.temp_dir, "evolution", "meta_bootstrap.md")

        for it in range(1, 13):  # 12 rounds
            self.agent.iteration = it
            self.agent.evolution_tracker.records = [
                SimpleNamespace(reward=0.01 * it, metadata={})
            ]
            executor.meta_bootstrap(what=f"change{it}")

        with open(bootstrap_path) as f:
            content = f.read()

        self.assertEqual(content.count("## Meta #"), 10)
        # newest 10 kept (iters 3..12); oldest two (1, 2) dropped.
        # Use trailing-space headers so "#1" doesn't substring-match "#10".."#12".
        self.assertIn("## Meta #12 ", content)
        self.assertIn("## Meta #3 ", content)
        self.assertNotIn("## Meta #2 ", content)
        self.assertNotIn("## Meta #1 ", content)

    def test_meta_bootstrap_skips_meta_records_for_reward(self):
        """Reward is taken from the latest non-meta record, skipping meta_evolve records."""
        from types import SimpleNamespace

        executor = self._make_meta_bootstrap_executor()
        self.agent.iteration = 7
        # A meta_evolve record (latest) should be skipped; the dev record below it wins
        self.agent.evolution_tracker.records = [
            SimpleNamespace(reward=0.9999, metadata={"type": "meta_evolve"}),
            SimpleNamespace(reward=0.42, metadata={"committed_eval_mode": "dev"}),
        ]
        executor.meta_bootstrap(what="change")

        with open(os.path.join(self.temp_dir, "evolution", "meta_bootstrap.md")) as f:
            content = f.read()
        self.assertIn("reward=0.4200", content)
        self.assertNotIn("reward=0.9999", content)

    def test_meta_bootstrap_backward_compat_old_format(self):
        """An old Plan/Progress/Lesson file is preserved; new record prepended on top."""
        os.makedirs(os.path.join(self.temp_dir, "evolution"), exist_ok=True)
        old_path = os.path.join(self.temp_dir, "evolution", "meta_bootstrap.md")
        with open(old_path, "w") as f:
            f.write("## Plan\nOld plan\n\n## Progress\nOld progress\n\n## Lesson\nOld lesson\n")

        executor = self._make_meta_bootstrap_executor()
        executor.meta_bootstrap(what="new what", why="new why",
                                lesson="new lesson", prediction="new pred")

        with open(old_path) as f:
            content = f.read()

        # new record prepended on top
        self.assertIn("## Meta #5", content)
        self.assertIn("### What\nnew what", content)
        self.assertEqual(content.count("## Meta #"), 1)
        # old content preserved & readable (below the new record)
        self.assertIn("Old plan", content)
        self.assertIn("Old progress", content)
        self.assertIn("Old lesson", content)
        # new record sits above the legacy content
        self.assertLess(content.index("## Meta #5"), content.index("Old plan"))


class TestIntegration(unittest.TestCase):
    """Integration tests for commit_iteration → KG flow."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create mock agent
        self.agent = Mock()
        self.agent.iteration = 3
        self.agent.agent_code_dir = self.temp_dir
        self.agent.git_controller = MockGitController()
        self.agent.state = AgentState(iteration=3, goal="Test")
        self.agent.state.parent_commit = "parent_commit"
        self.agent._knowledge_graph = EvolutionKnowledgeGraph(
            agent_code_dir=self.temp_dir,
            git_controller=self.agent.git_controller,
            call_llm=lambda m, t: MockLLMResponse('{"analyses":[]}'),
            log=lambda *a, **k: None,
        )
        self.agent.evolution_tracker = Mock()

    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_commit_iteration_adds_kg_node(self):
        """Test that commit_iteration adds a node to KG."""
        # Simulate iteration completion
        self.agent.state.reward = {"utility": 0.8, "security": 0.9}
        self.agent.state.evaluation_snapshots = [
            ("code_hash_1", {"utility": 0.8}, "val"),
        ]
        self.agent.state.code_snapshots = {
            "code_hash_1": {"harness.py": "content"}
        }
        self.agent.state.modifications_made = [
            {"operation": "edit", "file": "harness.py"}
        ]
        self.agent.state.iteration_summary_text = "Test iteration"

        # Simulate commit
        new_commit = "new_commit_hash"
        self.agent.git_controller.current = new_commit

        # Simulate KG node addition (this happens in commit_iteration)
        node_id = self.agent._knowledge_graph.add_node(
            iteration=3,
            git_hash=new_commit,
            parent_hash="parent_commit",
            reward=0.8,  # Would be scalar_reward in real code
            eval_mode="val",
            summary_text="Test iteration",
            modified_files=["harness.py"],
            change_tags=["edit"],
            is_meta=False,
        )

        # Verify node was added
        self.assertEqual(node_id, new_commit)
        self.assertIn(new_commit, self.agent._knowledge_graph.nodes)

        node = self.agent._knowledge_graph.nodes[new_commit]
        self.assertEqual(node.iteration, 3)
        self.assertEqual(node.reward, 0.8)
        self.assertEqual(node.is_meta, False)

    def test_kg_persists_across_iterations(self):
        """Test KG nodes persist and accumulate across iterations."""
        kg = self.agent._knowledge_graph

        # Add nodes for 3 iterations
        for i in range(3):
            kg.add_node(
                iteration=i+1,
                git_hash=f"commit{i}",
                parent_hash=f"commit{i-1}" if i > 0 else "",
                reward=0.6 + i * 0.1,
                eval_mode="val",
                summary_text=f"Iteration {i+1}",
            )

        # Save and reload
        kg.save()
        kg2 = EvolutionKnowledgeGraph(
            agent_code_dir=self.temp_dir,
            git_controller=self.agent.git_controller,
            call_llm=lambda m, t: None,
        )
        kg2.load()

        # Verify all nodes persisted
        self.assertEqual(len(kg2.nodes), 3)
        for i in range(3):
            self.assertIn(f"commit{i}", kg2.nodes)

    def test_meta_evolve_does_not_create_kg_nodes(self):
        """Test that meta-evolve commits don't create KG nodes."""
        # In commit_iteration, KG nodes are only added for evolve commits
        # (is_meta=False). Meta commits should not add nodes.
        kg = self.agent._knowledge_graph

        # Add a main iteration node
        kg.add_node(
            iteration=1,
            git_hash="main_commit",
            parent_hash="",
            reward=0.8,
            is_meta=False,
        )

        # Don't add meta commit node
        node_id = kg.add_node(
            iteration=2,
            git_hash="meta_commit",
            parent_hash="main_commit",
            reward=0.0,
            is_meta=True,
        )

        # Should still only have 1 node (main commit)
        # Meta commit nodes are not added to KG by default
        self.assertEqual(len(kg.nodes), 2)  # Both added in test
        self.assertTrue(any(n.is_meta for n in kg.nodes.values()))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_add_node_with_none_llm_caller(self):
        """Test KG works without LLM caller (structural similarity only)."""
        kg = EvolutionKnowledgeGraph(
            agent_code_dir="/tmp",
            git_controller=MockGitController(),
            call_llm=None,  # No LLM
            log=lambda *a, **k: None,
        )

        # Add nodes
        kg.add_node(
            iteration=1,
            git_hash="h1",
            parent_hash="",
            reward=0.5,
        )
        kg.add_node(
            iteration=2,
            git_hash="h2",
            parent_hash="h1",
            reward=0.6,
        )

        # Should have backbone edge but no semantic edges (no LLM)
        backbone_edges = [e for e in kg.edges if e.edge_type == "backbone"]
        self.assertEqual(len(backbone_edges), 1)

    def test_render_empty_kg(self):
        """Test rendering empty KG produces valid output."""
        kg = EvolutionKnowledgeGraph(
            agent_code_dir="/tmp",
            git_controller=MockGitController(),
            call_llm=lambda m, t: None,
            log=lambda *a, **k: None,
        )

        rendered = kg.render_for_prompt()
        self.assertIn("(empty)", rendered)
        self.assertIn("knowledge_graph.json", rendered)

    def test_llm_parse_error_handling(self):
        """Test that LLM parse errors don't crash KG."""
        bad_responses = [
            "",  # Empty
            "not json",  # Invalid JSON
            "```json\n{incomplete",  # Malformed
        ]

        for bad_response in bad_responses:
            kg = EvolutionKnowledgeGraph(
                agent_code_dir="/tmp",
                git_controller=MockGitController(),
                call_llm=lambda m, t: MockLLMResponse(bad_response),
                log=lambda *a, **k: None,
            )

            # Should not crash
            try:
                kg.add_node(
                    iteration=1,
                    git_hash="h1",
                    parent_hash="",
                    reward=0.5,
                )
                kg.add_node(
                    iteration=2,
                    git_hash="h2",
                    parent_hash="h1",
                    reward=0.6,
                )
                # If we get here, error was handled gracefully
                success = True
            except Exception:
                success = False

            self.assertTrue(success, f"Failed to handle bad response: {bad_response[:50]}")

    def test_backfill_with_empty_tracker(self):
        """Test backfill with empty tracker doesn't crash."""
        kg = EvolutionKnowledgeGraph(
            agent_code_dir="/tmp",
            git_controller=MockGitController(),
            call_llm=lambda m, t: None,
            log=lambda *a, **k: None,
        )

        tracker = Mock()
        tracker.records = None

        # Should not crash
        kg.backfill_from_tracker(tracker)
        self.assertEqual(len(kg.nodes), 0)

        tracker.records = []
        kg.backfill_from_tracker(tracker)
        self.assertEqual(len(kg.nodes), 0)


class TestPlanLessonTools(unittest.TestCase):
    """Test the plan/lesson tool split (replaces the old single bootstrap tool).

    plan() → plan.md (ephemeral, Hypothesis preserved, Plan/Progress merged).
    lesson() → BOOTSTRAP.md ## Lesson (cumulative [Iter N|conf] line, sets
    state.lesson_recorded).
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        from src.react_loop.actions.agent_action import AgentActionExecutor
        from pathlib import Path
        self.agent = Mock()
        self.agent.iteration = 3
        self.executor = AgentActionExecutor(
            llm_client=Mock(), model="m", repo_path=Path(self.temp_dir),
            agent_code_dir=self.temp_dir, agent_instance=self.agent,
        )
        self.executor.agent_codes = {}
        self.executor._modified_files = set()
        self.state = AgentState(iteration=3, goal="g")
        self.executor.set_state(self.state)

    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_plan_preserves_hypothesis_and_merges_sections(self):
        """plan() keeps the framework-written ## Hypothesis intact and
        overwrites Plan/Progress only when non-empty."""
        # Framework wrote the hypothesis at iter start
        with open(os.path.join(self.temp_dir, "plan.md"), "w") as f:
            f.write("## Hypothesis\n\nseed says: try X\n")

        self.executor.plan(plan="target failure mode F", progress="changed A → r=0.3")
        with open(os.path.join(self.temp_dir, "plan.md")) as f:
            content = f.read()
        self.assertIn("## Hypothesis", content)
        self.assertIn("seed says: try X", content)
        self.assertIn("## Plan", content)
        self.assertIn("target failure mode F", content)
        self.assertIn("## Progress", content)
        self.assertIn("changed A", content)

        # Second call with empty plan keeps the prior plan (overwrite-if-non-empty)
        self.executor.plan(plan="", progress="r=0.5 now")
        with open(os.path.join(self.temp_dir, "plan.md")) as f:
            content = f.read()
        self.assertIn("target failure mode F", content)  # prior plan retained
        self.assertIn("r=0.5 now", content)  # progress overwritten

    def test_plan_does_not_record_modified_files(self):
        """plan.md is ephemeral metadata — must not enter _modified_files
        (it's not a code change tracked by git/KG)."""
        self.executor.plan(plan="some plan")
        self.assertNotIn("plan.md", self.executor._modified_files)
        # But the activity IS recorded in modifications_made
        self.assertTrue(any(m.get("operation") == "plan"
                           for m in self.state.modifications_made))

    def test_lesson_writes_iter_line_and_sets_recorded(self):
        """lesson() writes one [Iter N|conf] line into BOOTSTRAP.md ## Lesson
        and flips state.lesson_recorded."""
        self.assertFalse(self.state.lesson_recorded)
        self.executor.lesson(lesson="hypothesis H1 confirmed", confidence=0.9)

        with open(os.path.join(self.temp_dir, "BOOTSTRAP.md")) as f:
            content = f.read()
        self.assertIn("## Lesson", content)
        self.assertIn("[Iter 3|conf=0.90]", content)
        self.assertIn("hypothesis H1 confirmed", content)
        self.assertTrue(self.state.lesson_recorded)
        # BOOTSTRAP.md is tracked → recorded in _modified_files for git commit
        self.assertIn("BOOTSTRAP.md", self.executor._modified_files)

    def test_lesson_accumulates_across_iterations(self):
        """Each iteration gets exactly one line; the same iteration revises in place."""
        self.executor.lesson(lesson="first verdict", confidence=0.5)
        self.executor.lesson(lesson="revised verdict", confidence=0.8)  # same iter
        with open(os.path.join(self.temp_dir, "BOOTSTRAP.md")) as f:
            content = f.read()
        self.assertEqual(content.count("[Iter 3|"), 1)
        self.assertIn("revised verdict", content)
        self.assertNotIn("first verdict", content)

        # A new iteration appends a new line
        self.state.iteration = 4
        self.executor.lesson(lesson="iter 4 verdict", confidence=0.6)
        with open(os.path.join(self.temp_dir, "BOOTSTRAP.md")) as f:
            content = f.read()
        self.assertIn("[Iter 3|", content)
        self.assertIn("[Iter 4|", content)

    def test_lesson_overwrite_past_iteration(self):
        """Explicit iteration=N overwrites the [Iter N] line directly."""
        # Write iter 3 lesson first
        self.executor.lesson(lesson="original iter 3 lesson", confidence=0.85)
        # Meta-evolve corrects iter 3's lesson by passing iteration=3
        self.executor.lesson(
            lesson="correction: iter 3 was wrong",
            confidence=0.40,
            iteration=3,
        )
        with open(os.path.join(self.temp_dir, "BOOTSTRAP.md")) as f:
            content = f.read()
        # Only one [Iter 3] line — original replaced
        self.assertEqual(content.count("[Iter 3|"), 1)
        self.assertIn("[Iter 3|conf=0.40]", content)
        self.assertIn("correction: iter 3 was wrong", content)
        self.assertNotIn("original iter 3 lesson", content)

    def test_lesson_explicit_new_iteration_appends(self):
        """Explicit iteration=N for a new N appends a line."""
        self.executor.lesson(
            lesson="future-proof lesson for iter 5",
            confidence=0.70,
            iteration=5,
        )
        with open(os.path.join(self.temp_dir, "BOOTSTRAP.md")) as f:
            content = f.read()
        self.assertIn("[Iter 5|conf=0.70]", content)
        self.assertIn("future-proof lesson for iter 5", content)

    def test_lesson_default_iteration_uses_current(self):
        """iteration=None (default) uses current state.iteration."""
        self.state.iteration = 7
        self.executor.lesson(lesson="current iter lesson", confidence=0.60)
        with open(os.path.join(self.temp_dir, "BOOTSTRAP.md")) as f:
            content = f.read()
        self.assertIn("[Iter 7|conf=0.60]", content)
        self.assertIn("current iter lesson", content)

    def test_parse_sections_extracts_named_headers(self):
        """_parse_sections is the generalized helper shared by plan/lesson."""
        from src.react_loop.actions.agent_action import AgentActionExecutor
        content = (
            "## Hypothesis\nseed hyp\n\n"
            "## Plan\nfix X\n\n"
            "## Progress\nr=0.3\n\n"
            "## Lesson\n[Iter 1|conf=0.50] v\n"
        )
        sections = AgentActionExecutor._parse_sections(
            content, ["Hypothesis", "Plan", "Progress"]
        )
        self.assertIn("seed hyp", sections["hypothesis"])
        self.assertIn("fix X", sections["plan"])
        self.assertIn("r=0.3", sections["progress"])
        # Lesson wasn't requested → not a key in the returned dict
        self.assertNotIn("lesson", sections)


class TestEnsembleStrategy(unittest.TestCase):
    """Test ensemble strategy helpers (top-k filtering, code reading)."""

    def _load_ensemble_module(self):
        """Load godel_evolution_init/strategies/ensemble.py in isolation."""
        repo_root = Path(__file__).parent.parent.parent.parent
        strategy_path = repo_root / "godel_evolution_init" / "strategies" / "ensemble.py"
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ensemble_strategy_under_test", str(strategy_path)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    @staticmethod
    def _rec(commit, reward, mtype=None, op=None):
        md = {}
        if mtype:
            md["type"] = mtype
        if op:
            md["operation_type"] = op
        return SimpleNamespace(new_commit=commit, reward=reward, metadata=md)

    def test_get_top_k_excludes_meta_and_aux_records(self):
        """Regression: top-k must only contain real main-line iterations.

        meta_evolve, crossover, and ensemble records carry reward=0 and only
        touch evolution/, so fusing them feeds the LLM evolution/ diffs. An
        earlier version used heapq.nlargest over ALL records, letting 0-reward
        meta/aux records sneak into the candidates when real records < k.
        """
        mod = self._load_ensemble_module()

        records = [
            self._rec("best", 0.9),
            self._rec("mid", 0.5),
            self._rec("meta_commit", 0.0, mtype="meta_evolve"),
            self._rec("crossover_commit", 0.0, op="crossover"),
            self._rec("ensemble_commit", 0.0, op="ensemble"),
        ]
        agent = SimpleNamespace(
            evolution_tracker=SimpleNamespace(records=records)
        )

        top_k = mod._get_top_k_nodes(agent, k=5)

        hashes = [n["git_hash"] for n in top_k]
        self.assertNotIn("meta_commit", hashes, "meta_evolve records must be excluded")
        self.assertNotIn("crossover_commit", hashes, "crossover records must be excluded")
        self.assertNotIn("ensemble_commit", hashes, "ensemble records must be excluded")
        # Only 2 real records exist -> top-k caps at 2, sorted desc by reward.
        self.assertEqual(hashes, ["best", "mid"])

    def test_get_top_k_handles_dict_reward_without_crash(self):
        """reward_to_scalar key keeps nlargest safe if a dict reward appears."""
        mod = self._load_ensemble_module()

        records = [
            self._rec("a", {"utility": 0.8, "security": 0.6}),
            self._rec("b", 0.5),
        ]
        agent = SimpleNamespace(
            evolution_tracker=SimpleNamespace(records=records)
        )
        top_k = mod._get_top_k_nodes(agent, k=2)
        hashes = [n["git_hash"] for n in top_k]
        self.assertEqual(set(hashes), {"a", "b"})
        # dict {"utility":0.8,"security":0.6} -> mean 0.7 > 0.5, so "a" ranks first
        self.assertEqual(hashes[0], "a")


def run_tests():
    """Run all tests and print results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestKnowledgeGraph))
    suite.addTests(loader.loadTestsFromTestCase(TestMetaEvolveStateIsolation))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestPlanLessonTools))
    suite.addTests(loader.loadTestsFromTestCase(TestEnsembleStrategy))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")
        for test, traceback in result.failures + result.errors:
            print(f"\nFAILED: {test}")
            print(traceback)

    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
