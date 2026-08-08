"""
Collect visualization data from EvolutionTracker + GitController.

Builds a JSON-ready dict of nodes, edges, diffs, and metadata
for the HTML renderer to consume.

Supports three node types:
- iteration_final: main iteration commit (existing)
- eval_snapshot: evaluate() snapshots within an iteration (new)
- meta_evolve: meta-evolution commits (new)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..git_version.controller import EvolutionTracker, GitController

_EMPTY_DIFF = {"files_changed": 0, "insertions": 0, "deletions": 0, "files": []}


def _classify_records(records) -> Tuple[List, List]:
    """Split records into main_records and meta_records.

    Identifies meta-evolve records via metadata.type == "meta_evolve".
    """
    main_records = []
    meta_records = []
    for record in records:
        metadata = record.metadata or {}
        if metadata.get("type") == "meta_evolve":
            meta_records.append(record)
        else:
            main_records.append(record)
    return main_records, meta_records


def _find_selected_snapshot_index(
    reward_history: List[Dict],
    metadata: Dict[str, Any],
) -> Optional[int]:
    """Find which evaluate snapshot corresponds to the final committed code.

    Matches by eval_mode and reward proximity (tolerance < 0.001).
    Returns None if no match found.
    """
    if not reward_history:
        return None

    committed_mode = metadata.get("committed_eval_mode")
    committed_reward = metadata.get("committed_code_reward")

    if committed_reward is None:
        return None

    # committed_code_reward may be a dict with scalar_reward
    if isinstance(committed_reward, dict):
        committed_reward = committed_reward.get("scalar_reward", committed_reward)

    if not isinstance(committed_reward, (int, float)):
        return None

    best_idx = None
    best_diff = float("inf")

    for i, entry in enumerate(reward_history):
        entry_reward = entry.get("reward")
        if entry_reward is None:
            continue

        # Match eval_mode if committed_mode is available
        if committed_mode is not None:
            entry_mode = entry.get("eval_mode", "")
            if entry_mode != committed_mode:
                continue

        diff = abs(entry_reward - committed_reward)
        if diff < best_diff:
            best_diff = diff
            best_idx = i

    # Tolerance check
    if best_idx is not None and best_diff < 0.001:
        return best_idx
    return None


def build_visualization_data(
    tracker: EvolutionTracker,
    git_controller: GitController,
    goal: str = "",
) -> Dict[str, Any]:
    """Build visualization-ready data from tracker + git history.

    Args:
        tracker: EvolutionTracker with recorded iterations.
        git_controller: GitController for the evolution repo.
        goal: The evolution goal text.

    Returns:
        Dict with "meta", "nodes", "edges", "iteration_groups" keys.
    """
    best_version = tracker.get_best_version("highest_reward")
    best_commit = best_version[0] if best_version else None
    best_reward = best_version[1] if best_version else None

    main_records, meta_records = _classify_records(tracker.records)

    # Build meta_by_main_iter: {main_iteration: meta_record}
    meta_by_main_iter: Dict[int, Any] = {}
    for meta_rec in meta_records:
        main_iter = meta_rec.metadata.get("main_iteration", meta_rec.iteration)
        meta_by_main_iter[main_iter] = meta_rec

    commit_meta = _batch_commit_meta(git_controller, tracker.records)

    nodes = []
    edges = []
    iteration_groups = []

    iter_final_ids: Dict[int, str] = {}  # iter -> primary (best-reward) commit short hash
    record_by_iter: Dict[int, Any] = {r.iteration: r for r in main_records}

    for record in main_records:
        parent_hash = record.parent_commit
        metadata = record.metadata or {}

        # Pool-aware: each pool entry becomes one node. iter_final_ids keeps
        # the primary (best-reward) commit's short hash for cross-iteration edges.
        pool_entries = list(record.iter_pool())
        if not pool_entries:
            continue

        # First (primary) entry drives iteration-level linkage.
        primary_entry = max(
            pool_entries,
            key=lambda e: e.get("reward", 0.0),
        )
        primary_id = primary_entry["new_commit"][:7]
        iter_final_ids[record.iteration] = primary_id

        reward_history = metadata.get("reward_history", [])
        selected_idx = _find_selected_snapshot_index(reward_history, metadata)

        for entry in pool_entries:
            git_hash = entry["new_commit"]
            meta_info = commit_meta.get(git_hash, {})
            final_id = git_hash[:7]

            ccr = entry.get("committed_code_reward")
            nodes.append({
                "id": final_id,
                "node_type": "iteration_final",
                "full_hash": git_hash,
                "iteration": record.iteration,
                "parent_id": parent_hash[:7] if parent_hash else None,
                "reward": entry["reward"],
                "reward_detail": ccr if isinstance(ccr, (dict, list)) else None,
                "timestamp": meta_info.get("timestamp", ""),
                "commit_message": meta_info.get("message", ""),
                "summary_text": metadata.get("summary_text", ""),
                "action_count": record.action_count,
                "modified_files": metadata.get("modified_files", []),
                "diff_stats": _get_diff_stats(git_controller, parent_hash, git_hash),
                "reward_history": reward_history,
                "end_reason": metadata.get("iteration_end_reason", ""),
                "is_best": best_commit is not None and git_hash == best_commit,
                "selected_snapshot_index": selected_idx,
                "is_pool_entry": len(pool_entries) > 1,
            })

        eval_snapshots = []
        if reward_history:
            prev_snap_id = None
            for snap_idx, snap in enumerate(reward_history):
                snap_id = f"snap_{record.iteration}_{snap_idx}"
                snap_reward = snap.get("reward")
                snap_mode = snap.get("eval_mode", "")
                snap_code_hash = snap.get("code_hash", "")

                nodes.append({
                    "id": snap_id,
                    "node_type": "eval_snapshot",
                    "iteration": record.iteration,
                    "snapshot_index": snap_idx,
                    "reward": snap_reward,
                    "eval_mode": snap_mode,
                    "code_hash": snap_code_hash,
                    "is_selected": selected_idx is not None and snap_idx == selected_idx,
                })

                eval_snapshots.append({
                    "id": snap_id,
                    "snapshot_index": snap_idx,
                    "reward": snap_reward,
                    "eval_mode": snap_mode,
                    "is_selected": selected_idx is not None and snap_idx == selected_idx,
                })

                if prev_snap_id is not None:
                    edges.append({
                        "source": prev_snap_id,
                        "target": snap_id,
                        "type": "within_iteration",
                    })

                prev_snap_id = snap_id

            if prev_snap_id is not None:
                edges.append({
                    "source": prev_snap_id,
                    "target": primary_id,
                    "type": "within_iteration",
                })

        # Edge: parent → each pool entry
        if parent_hash:
            for entry in pool_entries:
                eh = entry["new_commit"][:7]
                edges.append({
                    "source": parent_hash[:7],
                    "target": eh,
                    "type": "refine",
                })

        group = {
            "iteration": record.iteration,
            "final_commit_id": primary_id,
            "eval_snapshots": eval_snapshots,
        }

        meta_rec = meta_by_main_iter.get(record.iteration)
        if meta_rec:
            meta_hash = meta_rec.primary_commit()
            meta_short = meta_hash[:7]
            group["meta_evolve_id"] = meta_short
            group["meta_summary"] = meta_rec.metadata.get("summary_text", "")
            group["meta_modifications_count"] = meta_rec.metadata.get("modifications_count", 0)
            group["next_start_commit"] = meta_hash

        iteration_groups.append(group)

    for meta_rec in meta_records:
        main_iter = meta_rec.metadata.get("main_iteration", meta_rec.iteration)
        meta_hash = meta_rec.primary_commit()
        meta_short = meta_hash[:7]
        parent_hash = meta_rec.parent_commit
        meta_info = commit_meta.get(meta_hash, {})
        summary = meta_rec.metadata.get("summary_text", "")

        nodes.append({
            "id": meta_short,
            "node_type": "meta_evolve",
            "full_hash": meta_hash,
            "iteration": meta_rec.iteration,
            "meta_main_iteration": main_iter,
            "reward": meta_rec.primary_reward(),
            "meta_summary": summary,
            "meta_modifications_count": meta_rec.metadata.get("modifications_count", 0),
            "timestamp": meta_info.get("timestamp", ""),
            "commit_message": meta_info.get("message", ""),
            "action_count": meta_rec.action_count,
            "diff_stats": _get_diff_stats(git_controller, parent_hash, meta_hash),
            "is_best": False,
            "is_noop": meta_hash == parent_hash,
        })

        main_final_id = iter_final_ids.get(main_iter)
        if main_final_id:
            edges.append({
                "source": main_final_id,
                "target": meta_short,
                "type": "meta_evolve_bridge",
            })

    # Connect meta -> next round's starting point
    main_iters_sorted = sorted(iter_final_ids.keys())
    for i, main_iter in enumerate(main_iters_sorted):
        meta_rec = meta_by_main_iter.get(main_iter)
        if not meta_rec:
            continue

        meta_short = meta_rec.primary_commit()[:7]

        if i + 1 < len(main_iters_sorted):
            next_iter = main_iters_sorted[i + 1]
            next_final_id = iter_final_ids[next_iter]
            next_record = record_by_iter.get(next_iter)

            if next_record:
                next_parent = next_record.parent_commit[:7] if next_record.parent_commit else None
                edge_type = "meta_evolve_bridge"

                if (next_parent
                    and next_parent != meta_short
                    and next_parent != iter_final_ids.get(main_iter)):
                    edge_type = "archive_switch"

                edges.append({
                    "source": meta_short,
                    "target": next_parent if next_parent else next_final_id,
                    "type": edge_type,
                })

    return {
        "meta": {
            "goal": goal,
            "total_iterations": len(main_records),
            "total_meta_evolutions": len(meta_records),
            "best_version": {
                "commit": best_commit[:7] if best_commit else None,
                "full_hash": best_commit,
                "reward": best_reward,
            },
            "generated_at": datetime.now().isoformat(),
        },
        "nodes": nodes,
        "edges": edges,
        "iteration_groups": iteration_groups,
    }


def _batch_commit_meta(
    git_controller: GitController,
    records,
) -> Dict[str, Dict[str, str]]:
    """Fetch commit message + timestamp for all records in one git call."""
    if not records:
        return {}

    hashes = [entry["new_commit"]
              for r in records
              for entry in r.iter_pool()
              if entry["new_commit"]]
    if not hashes:
        return {}

    try:
        result = git_controller._run_git_command(
            ["log", "--format=%H|%s|%ci", "--no-walk"] + hashes,
            check=False,
        )
        if result.returncode != 0:
            return {}
        meta = {}
        for line in result.stdout.strip().splitlines():
            if not line:
                continue
            parts = line.split("|", 2)
            if len(parts) >= 3:
                meta[parts[0]] = {"message": parts[1], "timestamp": parts[2]}
        return meta
    except Exception:
        return {}


def _get_diff_stats(
    git_controller: GitController,
    parent_hash: str,
    commit_hash: str,
) -> Dict[str, Any]:
    if not parent_hash or not commit_hash:
        return _EMPTY_DIFF

    try:
        result = git_controller._run_git_command(
            ["diff", "--numstat", parent_hash, commit_hash], check=False
        )
        if result.returncode != 0:
            return _EMPTY_DIFF

        files = []
        total_add = 0
        total_del = 0
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                added = int(parts[0]) if parts[0] != "-" else 0
                removed = int(parts[1]) if parts[1] != "-" else 0
                total_add += added
                total_del += removed
                files.append({"path": parts[2], "added": added, "removed": removed})

        return {
            "files_changed": len(files),
            "insertions": total_add,
            "deletions": total_del,
            "files": files,
        }
    except Exception:
        return _EMPTY_DIFF
