"""
Evolution Knowledge Graph — a fully-connected graph of evolution history.

Each commit at the end of an iteration is promoted to a graph node (node =
commit code reference + iteration summary + derived features), and edges
between nodes encode their relationship:

- backbone edges: git parent-child relationships (the skeleton; free and accurate).
- semantic edges: LLM-generated diff analyses (fully connected: a single
  **undirected** edge per node pair, with one LLM call per pair).

Code is not stored in the node (to avoid bloat); it is lazily loaded via
git_hash using git_controller.get_file_at_commit().

Stored at evolution/knowledge_graph.json (evolution/ is writable, so the file
rides along with git and survives resume).
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional


@dataclass
class KGNode:
    """Knowledge-graph node: a commit at the end of an iteration."""
    node_id: str            # = git_hash (full, unique)
    iteration: int
    git_hash: str           # Full hash, used for lazy `git show` of the code
    reward: float           # Scalar
    eval_mode: str          # "val" | "dev"
    summary_text: str
    modified_files: List[str] = field(default_factory=list)
    change_tags: List[str] = field(default_factory=list)
    is_meta: bool = False   # commit_iteration is evolve-only; defaults to False

    def short(self) -> str:
        return self.git_hash[:7] if self.git_hash else self.node_id[:7]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KGNode":
        return cls(
            node_id=data.get("node_id", ""),
            iteration=int(data.get("iteration", 0)),
            git_hash=data.get("git_hash", ""),
            reward=float(data.get("reward", 0.0)),
            eval_mode=data.get("eval_mode", "dev"),
            summary_text=data.get("summary_text", ""),
            modified_files=list(data.get("modified_files", [])),
            change_tags=list(data.get("change_tags", [])),
            is_meta=bool(data.get("is_meta", False)),
        )


@dataclass
class KGEdge:
    """Knowledge-graph edge: a relationship between two nodes."""
    src_id: str
    dst_id: str
    edge_type: str          # "backbone" (git parent-child) | "semantic" (LLM diff)
    structural_similarity: float  # 0..1, Jaccard over modified files
    llm_diff_analysis: str = ""   # LLM-generated similarity/difference analysis; may be ""

    def key(self) -> str:
        return f"{self.src_id}->{self.dst_id}|{self.edge_type}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KGEdge":
        return cls(
            src_id=data.get("src_id", ""),
            dst_id=data.get("dst_id", ""),
            edge_type=data.get("edge_type", "semantic"),
            structural_similarity=float(data.get("structural_similarity", 0.0)),
            llm_diff_analysis=data.get("llm_diff_analysis", ""),
        )


class EvolutionKnowledgeGraph:
    """
    Fully-connected evolution knowledge graph.

    Usage:
        kg = EvolutionKnowledgeGraph(agent_code_dir, git_controller, agent.call_llm, ...)
        kg.load()                       # Restore across resume
        kg.add_node(iteration=..., git_hash=..., parent_hash=..., ...)  # After commit_iteration
        kg.render_for_prompt(token_budget=2000)  # Feed into the meta-evolve / evolve prompt
    """

    VERSION = 3
    DEFAULT_FILENAME = "knowledge_graph.json"

    def __init__(
        self,
        agent_code_dir: str,
        git_controller,
        call_llm: Optional[Callable] = None,
        storage_path: Optional[str] = None,
        max_nodes: int = 100,
        concurrency: int = 2,
        chunk_size: int = None,  # Parameter kept for backward compatibility; no longer used
        log: Callable = print,
    ):
        self.agent_code_dir = agent_code_dir
        self.git_controller = git_controller
        self.call_llm = call_llm      # agent.call_llm(messages, tools=None) -> response
        self.max_nodes = max_nodes
        self.concurrency = max(1, concurrency)
        # chunk_size is no longer used (switched to per-pair analysis)
        self._log = log or (lambda *a, **k: None)

        self.storage_path = storage_path or os.path.join(agent_code_dir, self.DEFAULT_FILENAME)

        self.nodes: Dict[str, KGNode] = {}      # node_id -> KGNode
        self.edges: List[KGEdge] = []           # legacy v2 full-connection edges (kept for compat)
        # v3 split storage
        self.lineage_edges: Dict[str, KGEdge] = {}   # "parent->child" -> backbone edge
        self.correlation_edges: Dict[str, KGEdge] = {}  # canonical "src->dst" -> semantic edge
        self.meta: Dict[str, Any] = {
            "last_node_iteration": 0,
            "node_count": 0,
            "edge_count": 0,
            "lineage_count": 0,
            "correlation_count": 0,
            "pruned": [],   # Summaries of pruned nodes (cap_and_aggregate)
        }

    # -----------------------------------------------------------------
    # Node / edge construction
    # -----------------------------------------------------------------

    def add_node(
        self,
        iteration: int,
        git_hash: str,
        parent_hash: str = "",
        reward: float = 0.0,
        eval_mode: str = "dev",
        summary_text: str = "",
        modified_files: Optional[List[str]] = None,
        change_tags: Optional[List[str]] = None,
        is_meta: bool = False,
        skip_save: bool = False,
    ) -> Optional[str]:
        """
        Add a node and build its relationship to the existing graph (one edge per pair).

        Called after commit_iteration. Returns the new node_id; returns None
        (skipping the add) if git_hash already exists.

        Builds exactly one edge per existing node; edge_type encodes the relationship:
        - Parent-child pair (existing node == parent_hash) → backbone edge
          (directed parent→child, with LLM analysis).
        - Non-parent-child pair → semantic edge (undirected, with LLM analysis).
        Each pair triggers exactly one LLM call.
        """
        if not git_hash:
            return None
        if git_hash in self.nodes:
            # Same code already has a node — don't add a duplicate
            return None

        node = KGNode(
            node_id=git_hash,
            iteration=iteration,
            git_hash=git_hash,
            reward=reward,
            eval_mode=eval_mode,
            summary_text=summary_text or "",
            modified_files=list(modified_files or []),
            change_tags=list(change_tags or []),
            is_meta=is_meta,
        )
        self.nodes[git_hash] = node
        self.meta["last_node_iteration"] = max(
            self.meta.get("last_node_iteration", 0), iteration
        )

        # v3: selective edge building (backbone always, correlation only for meaningful pairs)
        try:
            self.relate_selective(git_hash, parent_hash)
        except Exception as e:
            self._log(f"  KG: relate_selective failed (structural-only fallback): {e}")

        # Cost guardrail
        try:
            self.cap_and_aggregate()
        except Exception as e:
            self._log(f"  KG: cap_and_aggregate failed: {e}")

        self._update_counts()
        if not skip_save:
            try:
                self.save()
            except Exception as e:
                self._log(f"  KG: save failed: {e}")
        return git_hash

    def relate_semantic_all(self, new_id: str, parent_hash: str = "") -> None:
        """
        Full-connection edge building: new_id gets exactly one edge per other node.

        Each pair triggers exactly one LLM call (analyzing the pair's
        similarities and differences), producing one edge:
        - Parent-child pair (other == parent_hash) → backbone (directed
          parent→child, with LLM analysis).
        - Non-parent-child pair → semantic (undirected, src/dst normalized
          via _canonical_pair, with LLM analysis).
        Cost guardrail: concurrent processing (concurrency); on a failed pair
        the edge degrades to structural-similarity-only.

        When there is no LLM caller, the fallback is a structural-only
        backbone edge for the parent-child pair (no analysis) — it never
        regresses to "no edge at all".
        """
        new_node = self.nodes.get(new_id)
        if new_node is None:
            return

        # Candidates: all nodes except self (including the parent)
        others = [n for nid, n in self.nodes.items() if nid != new_id]
        if not others:
            return

        # No LLM caller: fallback to a parent-child backbone edge
        # (structural-only); no other edges are added.
        if self.call_llm is None:
            if parent_hash and parent_hash in self.nodes and parent_hash != new_id:
                self._add_edge(KGEdge(
                    src_id=parent_hash, dst_id=new_id, edge_type="backbone",
                    structural_similarity=self._structural_similarity(
                        self.nodes[parent_hash], new_node),
                    llm_diff_analysis="",
                ))
            return

        # With an LLM: per-pair concurrent analysis; one edge per pair
        # (parent-child = backbone, non-parent-child = semantic).
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = [
                executor.submit(self._analyze_pair, new_node, other, parent_hash)
                for other in others
            ]
            for future in futures:
                try:
                    edge = future.result()
                    if edge is not None:
                        self._add_edge(edge)
                except Exception as e:
                    self._log(f"  KG: Pair analysis failed: {e}")

    def relate_selective(self, new_id: str, parent_hash: str = "") -> None:
        """
        Selective edge building (v3): backbone edges for git parent→child always,
        correlation (semantic) edges only for MEANINGFUL node pairs.

        Correlation edge conditions (all must be met):
        1. structural_similarity in (0.15, 0.90) — exclude identical and completely different
        2. At least one shared modified file
        3. Iteration gap ≤ 5 — recent correlations are more meaningful

        This replaces the old O(n²) full-connection relate_semantic_all.
        """
        new_node = self.nodes.get(new_id)
        if new_node is None:
            return

        # ── Backbone edge: parent→child, always created ──
        if parent_hash and parent_hash in self.nodes and parent_hash != new_id:
            parent_node = self.nodes[parent_hash]
            sim = self._structural_similarity(parent_node, new_node)
            # Build backbone edge with LLM analysis if available
            if self.call_llm is not None:
                try:
                    code_a = self._get_node_code(parent_node)
                    code_b = self._get_node_code(new_node)
                    analysis = self._llm_analyze_pair(parent_node, new_node, code_a, code_b)
                except Exception:
                    analysis = ""
            else:
                analysis = ""
            edge = KGEdge(
                src_id=parent_hash, dst_id=new_id, edge_type="backbone",
                structural_similarity=sim,
                llm_diff_analysis=analysis or "",
            )
            self._add_edge(edge)
            self.lineage_edges[f"{parent_hash}->{new_id}"] = edge

        # ── Correlation edges: only for meaningful pairs ──
        if self.call_llm is None:
            return

        others = [n for nid, n in self.nodes.items() if nid != new_id]
        # Filter candidates: only those that pass the selectivity criteria
        candidates = []
        for other in others:
            if other.node_id == parent_hash:
                continue  # already handled as backbone
            sim = self._structural_similarity(new_node, other)
            # Condition 1: similarity in (0.15, 0.90)
            if sim <= 0.15 or sim >= 0.90:
                continue
            # Condition 2: at least one shared modified file
            if not (set(new_node.modified_files) & set(other.modified_files)):
                continue
            # Condition 3: iteration gap ≤ 5
            if abs(new_node.iteration - other.iteration) > 5:
                continue
            candidates.append(other)

        if not candidates:
            return

        # Concurrent LLM analysis for correlation candidates
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = [
                executor.submit(self._analyze_pair, new_node, other, "")
                for other in candidates
            ]
            for future in futures:
                try:
                    edge = future.result()
                    if edge is not None:
                        self._add_edge(edge)
                        # Canonical key for correlation storage
                        src, dst = self._canonical_pair(
                            self.nodes[edge.src_id], self.nodes[edge.dst_id]
                        )
                        self.correlation_edges[f"{src.node_id}->{dst.node_id}"] = edge
                except Exception as e:
                    self._log(f"  KG: Selective pair analysis failed: {e}")

    def get_lineage_tree(self) -> Dict[str, Any]:
        """Return the lineage subgraph for seed selection display.

        Returns:
            {"roots": [root_hashes], "edges": {"parent->child": edge_dict, ...}}
        """
        roots = []
        child_set = set()
        for key, edge in self.lineage_edges.items():
            child_set.add(edge.dst_id)
        for key, edge in self.lineage_edges.items():
            if edge.src_id not in child_set:
                if edge.src_id not in roots:
                    roots.append(edge.src_id)

        # Also check nodes without incoming lineage edges
        for nid in self.nodes:
            if nid not in child_set and nid not in roots:
                roots.append(nid)

        return {
            "roots": roots,
            "edges": {k: e.to_dict() for k, e in self.lineage_edges.items()},
        }

    def get_correlations(self, node_id: str) -> List[KGEdge]:
        """Return all correlation edges for a given node."""
        result = []
        for edge in self.correlation_edges.values():
            if edge.src_id == node_id or edge.dst_id == node_id:
                result.append(edge)
        return result

    def render_lineage_for_prompt(self, token_budget: int = 2000) -> str:
        """Render the lineage tree as indented text for LLM prompt injection.

        Reads from v3 lineage_edges if available; falls back to filtering v2 edges
        list for backward compatibility.
        """
        if not self.nodes:
            return ""

        # Build parent→children map
        children: Dict[str, List[str]] = {}
        all_parents: set = set()

        if self.lineage_edges:
            for key, edge in self.lineage_edges.items():
                children.setdefault(edge.src_id, []).append(edge.dst_id)
                all_parents.add(edge.src_id)
        else:
            # v2 fallback: filter backbone edges from self.edges
            for edge in self.edges:
                if edge.edge_type == "backbone":
                    children.setdefault(edge.src_id, []).append(edge.dst_id)
                    all_parents.add(edge.src_id)

        if not children:
            return ""

        # Find roots (nodes not appearing as children)
        all_children = set()
        for clist in children.values():
            all_children.update(clist)
        roots = [nid for nid in all_parents if nid not in all_children]
        # Also include nodes not referenced in any lineage edge
        for nid in self.nodes:
            if nid not in all_parents and nid not in all_children:
                roots.append(nid)

        lines = ["## Evolution Lineage\n"]
        rendered = set()
        budget_remaining = token_budget

        def _render_node(nid: str, indent: int, prefix: str = ""):
            nonlocal budget_remaining
            if budget_remaining <= 0 or nid in rendered:
                return
            rendered.add(nid)
            node = self.nodes.get(nid)
            if node is None:
                return
            short = node.short()
            indent_str = "  " * indent
            line = f"{indent_str}{prefix}`{short}` (iter{node.iteration}, {node.reward:.4f})"
            lines.append(line)
            budget_remaining -= len(line)
            if budget_remaining <= 0:
                return
            # Render children
            kids = children.get(nid, [])
            for i, kid in enumerate(kids):
                is_last = (i == len(kids) - 1)
                kid_prefix = "└── " if is_last else "├── "
                _render_node(kid, indent + 1, kid_prefix)

        for root in roots:
            if budget_remaining <= 0:
                break
            _render_node(root, 0, "")

        return "\n".join(lines)

    def render_correlations_for_prompt(self, node_ids: List[str] = None,
                                        token_budget: int = 1500) -> str:
        """Render correlation edges as text for LLM prompt injection.

        If node_ids is None, renders all correlations. Reads from v3
        correlation_edges if available; falls back to filtering v2 edges
        list for backward compatibility.
        """
        if node_ids is None:
            if self.correlation_edges:
                corr_edges = list(self.correlation_edges.values())
            else:
                # v2 fallback
                corr_edges = [e for e in self.edges if e.edge_type == "semantic"]
        else:
            id_set = set(node_ids)
            if self.correlation_edges:
                corr_edges = [
                    e for e in self.correlation_edges.values()
                    if e.src_id in id_set or e.dst_id in id_set
                ]
            else:
                corr_edges = [
                    e for e in self.edges
                    if e.edge_type == "semantic"
                    and (e.src_id in id_set or e.dst_id in id_set)
                ]

        if not corr_edges:
            return ""

        # Sort by similarity desc
        corr_edges.sort(key=lambda e: e.structural_similarity, reverse=True)

        lines = ["## Cross-Version Correlations\n"]
        budget_remaining = token_budget
        for e in corr_edges:
            if budget_remaining <= 0:
                lines.append("... (truncated)")
                break
            src_node = self.nodes.get(e.src_id)
            dst_node = self.nodes.get(e.dst_id)
            src_label = f"iter{src_node.iteration}" if src_node else "?"
            dst_label = f"iter{dst_node.iteration}" if dst_node else "?"
            analysis = (e.llm_diff_analysis or "").strip()
            if len(analysis) > 200:
                analysis = analysis[:200] + "..."
            line = (
                f"- `{e.src_id[:7]}`({src_label}) ↔ `{e.dst_id[:7]}`({dst_label}) "
                f"(sim={e.structural_similarity:.2f})"
            )
            if analysis:
                line += f": {analysis}"
            lines.append(line)
            budget_remaining -= len(line)

        return "\n".join(lines)

    @staticmethod
    def _canonical_pair(a: KGNode, b: KGNode):
        """Deterministic normalization for undirected edges: order ascending by (iteration, git_hash); the older one is src."""
        if (a.iteration, a.git_hash) <= (b.iteration, b.git_hash):
            return a, b
        return b, a

    def _analyze_pair(
        self, node_a: KGNode, node_b: KGNode, parent_hash: str = ""
    ) -> Optional[KGEdge]:
        """
        Analyze the similarities and differences of a node pair; return one edge
        (parent-child = directed backbone, non-parent-child = undirected semantic).

        Flow:
        1. Determine whether the pair is a parent-child relationship (other.git_hash == parent_hash).
        2. Fetch the full harness code for both nodes.
        3. Call the LLM once to analyze the pair's similarities and differences.
        4. Return a single edge: parent-child → backbone (src=parent, dst=child);
           non-parent-child → semantic (src/dst normalized via _canonical_pair).
           Both carry structural similarity + LLM analysis.

        On LLM-analysis failure, degrades to a structural-similarity-only edge
        (empty analysis).
        """
        is_parent = bool(parent_hash) and parent_hash in (node_a.git_hash, node_b.git_hash)
        if is_parent:
            # Parent-child edge: directed parent→child
            parent = node_a if node_a.git_hash == parent_hash else node_b
            child = node_b if node_a.git_hash == parent_hash else node_a
            src, dst, edge_type = parent, child, "backbone"
        else:
            # Non-parent-child: undirected, deterministic normalization
            src, dst = self._canonical_pair(node_a, node_b)
            edge_type = "semantic"
        sim = self._structural_similarity(node_a, node_b)
        try:
            code_a = self._get_node_code(node_a)
            code_b = self._get_node_code(node_b)
            analysis = self._llm_analyze_pair(node_a, node_b, code_a, code_b)
        except Exception as e:
            self._log(f"  KG: Pair analysis {node_a.short()}-{node_b.short()} failed: {e}")
            analysis = ""

        return KGEdge(
            src_id=src.node_id,
            dst_id=dst.node_id,
            edge_type=edge_type,
            structural_similarity=sim,
            llm_diff_analysis=analysis or "",
        )

    def _get_node_code(self, node: KGNode) -> Dict[str, str]:
        """Get the node's full harness code (all files).

        Scans all .py and key .md files under agent_code_dir and fetches their
        full content from the corresponding commit via git show. No truncation,
        no summarization.

        Returns:
            {"harness.py": "...", "prompts.py": "...", "hooks.py": "...", ...}
        """
        code = {}
        # Scan all top-level .py files + evolution_base_prompt.md.
        # Excludes the evolution/ subdirectory (that's meta-evolve territory,
        # not harness code).
        try:
            files_to_read = []
            if os.path.isdir(self.agent_code_dir):
                for fname in os.listdir(self.agent_code_dir):
                    full_path = os.path.join(self.agent_code_dir, fname)
                    if os.path.isfile(full_path) and fname.endswith(".py"):
                        files_to_read.append(fname)

            for fname in files_to_read:
                try:
                    # get_file_at_commit(file_path, commit) — file first,
                    # commit second (matches GitController signature). An
                    # earlier version swapped these, silently running
                    # `git show <fname>:<hash>` (always failing) and feeding
                    # every LLM pair analysis empty code.
                    content = self.git_controller.get_file_at_commit(
                        fname, node.git_hash
                    )
                    if content:
                        code[fname] = content
                except Exception:
                    pass
        except Exception:
            pass

        return code

    def _llm_analyze_pair(
        self, node_a: KGNode, node_b: KGNode,
        code_a: Dict[str, str], code_b: Dict[str, str]
    ) -> str:
        """
        Call the LLM once to analyze the similarities and differences of a node
        pair; returns a snippet of analysis text.

        A single call produces a direction-agnostic analysis (no direction
        distinction), used on the unique semantic edge for that pair.
        """
        # Build the prompt
        system = (
            "You are a code evolution analyst comparing two versions of an agent harness system. "
            "Analyze the similarities AND differences between the two versions:\n"
            "- Key architectural changes\n"
            "- Strategy differences (prompting, action execution, validation, etc.)\n"
            "- Which version is stronger and why (tie it to reward when relevant)\n"
            "- If the harness code is identical, say so explicitly\n\n"
            "Be concise but specific. Return plain prose (no JSON, no markdown code blocks)."
        )

        # Build the code-comparison text: each node's full harness code
        def format_code(code_dict: Dict[str, str]) -> str:
            parts = []
            # Sort by filename for stable output
            for fname in sorted(code_dict.keys()):
                content = code_dict[fname].strip()
                if content:
                    parts.append(f"### {fname}\n{content}")
            return "\n\n".join(parts) if parts else "(no code available)"

        user = f"""## Version A (iteration={node_a.iteration}, reward={node_a.reward:.4f}, mode={node_a.eval_mode})
Modified files: {', '.join(node_a.modified_files) or '(none)'}

{format_code(code_a)}

## Version B (iteration={node_b.iteration}, reward={node_b.reward:.4f}, mode={node_b.eval_mode})
Modified files: {', '.join(node_b.modified_files) or '(none)'}

{format_code(code_b)}

Analyze the similarities and differences between A and B."""

        response = self.call_llm(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=None,
        )

        try:
            content = response.choices[0].message.content.strip()
            # Strip markdown code-block wrapping
            if content.startswith("```"):
                lines = content.split("\n")
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines).strip()

            return content
        except Exception as e:
            self._log(f"  KG: LLM analysis parse error: {e}")

        return ""

    # -----------------------------------------------------------------
    # Structural similarity / edge management / cost guardrail
    # -----------------------------------------------------------------

    @staticmethod
    def _structural_similarity(a: KGNode, b: KGNode) -> float:
        """Jaccard similarity over modified files, 0..1."""
        sa = set(a.modified_files)
        sb = set(b.modified_files)
        if not sa and not sb:
            # Neither has modified-file info: fall back to a mild similarity based on iteration proximity
            return 0.0 if a.node_id == b.node_id else 0.1
        union = sa | sb
        if not union:
            return 0.0
        return len(sa & sb) / len(union)

    def _add_edge(self, edge: KGEdge) -> None:
        # One edge per node pair: dedupe by unordered endpoint pair (no longer
        # split by edge_type). Backbone (directed parent→child) uses the same
        # unordered dedupe — a given pair can only have one relationship
        # (parent-child or non-parent-child), never both.
        for e in self.edges:
            if {e.src_id, e.dst_id} == {edge.src_id, edge.dst_id}:
                return
        self.edges.append(edge)

    def cap_and_aggregate(self) -> None:
        """
        Prune the lowest-value old nodes when the node count exceeds max_nodes.

        Protected: the highest-reward node + the 3 most recent nodes. Pruned
        nodes have their summary recorded in meta['pruned'].
        """
        if len(self.nodes) <= self.max_nodes:
            return
        keep = self.max_nodes
        ids = list(self.nodes.keys())
        # Protected set
        best_id = max(ids, key=lambda i: self.nodes[i].reward) if ids else None
        recent_ids = sorted(ids, key=lambda i: self.nodes[i].iteration)[-3:]
        protected = {best_id} | set(recent_ids)

        # Pruning candidates: non-protected nodes, sorted ascending by
        # (reward, iteration) — low reward + older pruned first.
        candidates = [i for i in ids if i not in protected]
        candidates.sort(key=lambda i: (self.nodes[i].reward, self.nodes[i].iteration))
        num_to_prune = len(self.nodes) - keep
        to_prune = candidates[:max(0, num_to_prune)]

        for nid in to_prune:
            node = self.nodes.pop(nid)
            self.meta["pruned"].append({
                "id": node.short(),
                "iteration": node.iteration,
                "reward": node.reward,
                "summary": (node.summary_text or "")[:160],
            })
            self.edges = [e for e in self.edges if e.src_id != nid and e.dst_id != nid]
            self.lineage_edges = {
                k: e for k, e in self.lineage_edges.items()
                if e.src_id != nid and e.dst_id != nid
            }
            self.correlation_edges = {
                k: e for k, e in self.correlation_edges.items()
                if e.src_id != nid and e.dst_id != nid
            }
        # Cap the pruned-history length
        self.meta["pruned"] = self.meta["pruned"][-50:]

    # -----------------------------------------------------------------
    # Prompt rendering (a condensed subset, fed to the LLM)
    # -----------------------------------------------------------------

    def render_for_prompt(
        self, focus_id: Optional[str] = None
    ) -> str:
        """
        Full overview: a table of all nodes + the knowledge_graph.json path +
        a JSON-schema hint. Edges / diff analyses are left for the agent to
        pull itself via bash on the JSON (active exploration). No truncation;
        all nodes are rendered.
        """
        path_line = f"Full graph (read via bash): `{self.storage_path}`"
        if not self.nodes:
            return f"### Evolution Knowledge Graph (empty)\n{path_line}"

        total_edges = len(self.edges) + len(self.lineage_edges) + len(self.correlation_edges)
        lines = [
            f"### Evolution Knowledge Graph "
            f"({len(self.nodes)} nodes, {total_edges} edges — full data in the JSON file)",
            "",
            "| id | iter | reward | mode | summary |",
            "|----|------|--------|------|---------|",
        ]
        ordered = sorted(self.nodes.values(), key=lambda n: n.iteration, reverse=True)
        if focus_id and focus_id in self.nodes:
            ordered = [self.nodes[focus_id]] + [n for n in ordered if n.node_id != focus_id]
        for n in ordered:
            summ = (n.summary_text or "").strip().replace("\n", " ").replace("|", "/")
            lines.append(f"| {n.short()} | {n.iteration} | {n.reward:.4f} | {n.eval_mode} | {summ} |")
        lines.append("")
        lines.append(path_line)
        lines.append(
            "JSON schema (v3): `{version: 3, nodes: {<git_hash>: {...}}, "
            "lineage: {edges: {\"parent->child\": {...}}, roots: [...]}, "
            "correlations: {edges: {\"src->dst\": {...}}}}`. "
            "Inspect edges/analyses with `jq`; "
            "fetch code with `git show <hash>:<file>`; diff with `git diff <a> <b> -- <file>` "
            "(read-only — do NOT checkout)."
        )
        return "\n".join(lines)

    # -----------------------------------------------------------------
    # Persistence / backfill
    # -----------------------------------------------------------------

    def _update_counts(self) -> None:
        self.meta["node_count"] = len(self.nodes)
        self.meta["edge_count"] = len(self.edges)
        self.meta["lineage_count"] = len(self.lineage_edges)
        self.meta["correlation_count"] = len(self.correlation_edges)

    def to_dict(self) -> Dict[str, Any]:
        self._update_counts()
        return {
            "version": self.VERSION,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "lineage": {
                "edges": {k: e.to_dict() for k, e in self.lineage_edges.items()},
                "roots": self.get_lineage_tree()["roots"],
            },
            "correlations": {
                "edges": {k: e.to_dict() for k, e in self.correlation_edges.items()},
            },
            "edges": [e.to_dict() for e in self.edges],  # legacy: kept for v2 compat reads
            "meta": self.meta,
        }

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        # Try new location first; if not found, try old location (evolution/)
        # and migrate to new location on next save.
        load_path = self.storage_path
        if not os.path.exists(load_path):
            old_path = os.path.join(
                os.path.dirname(self.storage_path), "evolution", self.DEFAULT_FILENAME
            )
            if os.path.exists(old_path):
                load_path = old_path
                self._log(f"  KG: migrating from legacy path: {old_path}")
            else:
                return
        try:
            with open(load_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self._log(f"  KG: load failed (starting fresh): {e}")
            return

        version = data.get("version", 2)
        self.nodes = {
            nid: KGNode.from_dict(nd) for nid, nd in data.get("nodes", {}).items()
        }
        self.edges = [KGEdge.from_dict(e) for e in data.get("edges", [])]
        self.meta = data.get("meta", self.meta)

        # v3 format: split lineage + correlations
        if version >= 3:
            lineage_data = data.get("lineage", {})
            corr_data = data.get("correlations", {})
            self.lineage_edges = {
                k: KGEdge.from_dict(e) for k, e in lineage_data.get("edges", {}).items()
            }
            self.correlation_edges = {
                k: KGEdge.from_dict(e) for k, e in corr_data.get("edges", {}).items()
            }
        else:
            # v2 format: migrate edges into lineage/correlations containers
            for e in self.edges:
                if e.edge_type == "backbone":
                    key = f"{e.src_id}->{e.dst_id}"
                    self.lineage_edges[key] = e
                elif e.edge_type == "semantic":
                    # Canonical key for correlation storage
                    src_node = self.nodes.get(e.src_id)
                    dst_node = self.nodes.get(e.dst_id)
                    if src_node and dst_node:
                        a, b = self._canonical_pair(src_node, dst_node)
                        key = f"{a.node_id}->{b.node_id}"
                    else:
                        key = f"{e.src_id}->{e.dst_id}"
                    self.correlation_edges[key] = e

        self._update_counts()

    def backfill_from_tracker(self, tracker) -> None:
        """
        First-time enable: rebuild historical nodes from EvolutionTracker.records
        (semantic edges left empty — backbone + structural similarity only).
        Records with metadata['type']=='meta_evolve' are skipped.

        Pool-aware: each pool entry (one commit per record) becomes its own
        KGNode, with parent → entry as a backbone edge.
        """
        if not tracker or not getattr(tracker, "records", None):
            return
        records = [r for r in tracker.records if r.metadata.get("type") != "meta_evolve"]
        # Sort by iteration so backbone edges are correct
        records.sort(key=lambda r: r.iteration)
        for r in records:
            for entry in r.iter_pool():
                commit_hash = entry["new_commit"]
                if not commit_hash or commit_hash in self.nodes:
                    continue
                self.nodes[commit_hash] = KGNode(
                    node_id=commit_hash,
                    iteration=r.iteration,
                    git_hash=commit_hash,
                    reward=float(entry["reward"]) if entry["reward"] is not None else 0.0,
                    eval_mode=entry.get("committed_eval_mode") or "dev",
                    summary_text=r.metadata.get("summary_text", "") or r.state_summary or "",
                    modified_files=list(r.metadata.get("modified_files", [])),
                    change_tags=[],
                    is_meta=False,
                )
                if r.parent_commit and r.parent_commit in self.nodes and r.parent_commit != commit_hash:
                    edge = KGEdge(
                        src_id=r.parent_commit, dst_id=commit_hash, edge_type="backbone",
                        structural_similarity=self._structural_similarity(
                            self.nodes[r.parent_commit], self.nodes[commit_hash]),
                        llm_diff_analysis="",
                    )
                    self._add_edge(edge)
                    self.lineage_edges[f"{r.parent_commit}->{commit_hash}"] = edge
        self.meta["last_node_iteration"] = max(
            (n.iteration for n in self.nodes.values()), default=0
        )
        self._update_counts()
        try:
            self.save()
        except Exception as e:
            self._log(f"  KG: backfill save failed: {e}")
