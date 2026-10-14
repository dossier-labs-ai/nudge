"""Full-sample, outcome-supervised routing-factor audit; no model calls.

Greedy forward selection partitions prompts by observable Boolean factors and
assigns each cell its utility-maximizing model. This fits the supplied sample;
it neither estimates generalization nor proves a minimum-size factor set.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .routerbench_data import fingerprint


PATTERNS = {
    "asks for code/function": r"\b(write|implement)\b.{0,40}\b(function|program|code)\b",
    "asks for numerical math answer": r"provide a numerical answer",
    "two answer choices": r'only a single choice\s+from "A" or "B" without',
    "four answer choices": r'only a single choice\s+from "A" or "B" or "C" or "D"',
    "letter-answer preamble": r"^please answer with the letter of the correct answer",
    "how-to markup": r"\[(header|title|step|substeps)\]",
    "blank to resolve": r"\b_\b",
    "explicit question mark": r"\?",
    "negation or exception": r"\b(not|except|never|least|incorrect|false)\b",
    "comparison or superlative": r"\b(more|less|most|least|larger|smaller|greatest|smallest|best|worst)\b",
    "causal reasoning words": r"\b(because|cause|causes|caused|why|therefore|hence|result)\b",
    "conditional reasoning words": r"\b(if|unless|assuming|suppose|given that)\b",
    "probability or statistics": r"\b(probability|likely|variance|standard deviation|median|mean|random|sample|distribution)\b",
    "money or prices": r"\$|\b(dollars?|price|cost|profit|buys?|sells?|salary|interest|discount)\b",
    "percent or fractions": r"%|\b(percent|percentage|half|quarter|fraction|ratio|third)\b|\d/\d",
    "time or rates": r"\b(hours?|minutes?|seconds?|days?|weeks?|months?|years?|speed|rate|per)\b",
    "geometry or measurement": r"\b(area|volume|angle|triangle|circle|radius|diameter|length|width|distance|meters?|miles?|feet)\b",
    "algebra or arithmetic": r"\b(equation|solve|sum|product|multiply|divide|subtract|integer|number|numbers|total)\b",
    "medicine or biology": r"\b(patient|disease|symptom|blood|gene|genes|genetic|cell|cells|organism|organisms|plant|plants|animal|animals|protein|bacteria|tissues?|treatment)\b",
    "physics or engineering": r"\b(force|energy|gravity|motion|electric|electrical|current|voltage|heat|temperature|light|sound|wave|waves|magnetic)\b",
    "chemistry": r"\b(atom|atoms|molecule|molecules|chemical|reaction|acid|oxygen|hydrogen|carbon|element|elements|compound|solution)\b",
    "earth or astronomy": r"\b(earth|sun|moon|planet|planets|rock|rocks|weather|climate|water|ocean|star|stars|soil)\b",
    "law or government": r"\b(law|legal|court|contract|rights|government|constitution|congress|president|policy|state|states)\b",
    "economics or business": r"\b(economic|economy|market|demand|supply|trade|investment|firm|company|business|inflation|tax|income)\b",
    "history or religion": r"\b(war|century|historical|history|empire|king|religion|religious|church|god|islam|christian|muslims)\b",
    "psychology or social behavior": r"\b(psychology|psychological|memory|behavior|behaviour|learning|social|emotion|emotional|personality)\b",
    "strings or text processing": r"\b(strings?|characters?|substring|palindrome|anagram|words?|text|regex)\b",
    "arrays or collections": r"\b(arrays?|lists?|tuples?|dictionary|dictionaries|sets?|matrix|matrices|elements?)\b",
    "sorting or searching": r"\b(sort|sorted|sorting|search|searching|minimum|maximum|largest|smallest|heap|binary)\b",
    "number theory": r"\b(prime|factorial|fibonacci|divisible|divisors?|factors?|even|odd|gcd|lcm)\b",
    "graphs or recursion": r"\b(graph|tree|nodes?|edges?|recursive|recursion|path|paths)\b",
    "physical activity": r"\b(man|woman|person|people|boy|girl|walking|running|holding|playing|sitting|standing|riding)\b",
    "household or food": r"\b(kitchen|food|cook|cooking|bowl|pan|water|clean|cleaning|room|house|table)\b",
    "pronoun relationships": r"\b(he|she|him|her|his|hers|they|their|them)\b",
    "URL present": r"https?://",
}


def task_text(state):
    """Exclude fixed worked math examples and answer-format boilerplate."""
    if "provide a numerical answer" in state.lower():
        state = state.rsplit("\n\n\n\n", 1)[-1]
    return re.split(r"\nPrint only a single choice", state, maxsplit=1)[0]


def extract(states, *, lexical=False):
    """Input is prompt text only: no family, ID, outcomes, or model answers."""
    names = list(PATTERNS)
    contents = [task_text(s) for s in states]
    columns = [
        [bool(re.search(pattern, s, re.I | re.S)) for s in
         (states if name in list(PATTERNS)[:7] else contents)]
        for name, pattern in PATTERNS.items()
    ]
    for threshold in (100, 200, 400, 800, 1200, 2000):
        names.append(f"task text length >= {threshold} characters")
        columns.append([len(s) >= threshold for s in contents])
    for threshold in (1, 2, 3, 5, 8):
        names.append(f"task text has >= {threshold} numbers")
        columns.append([len(re.findall(r"\d+(?:\.\d+)?", s)) >= threshold for s in contents])
    if lexical:
        tokens = [set(re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", s.lower())) for s in contents]
        counts = Counter(token for row in tokens for token in row)
        # Include singleton words: the report exposes the complexity required
        # for exact sample reconstruction instead of hiding rare distinctions.
        for token in sorted(counts):
            if counts[token] < len(states):
                names.append(f"contains token {token!r}")
                columns.append([token in row for row in tokens])
    values = np.asarray(columns, dtype=bool).T
    varying = (values.sum(axis=0) > 0) & (values.sum(axis=0) < len(states))
    return [name for name, keep in zip(names, varying) if keep], values[:, varying]


def metrics(quality, cost, actions, penalty):
    rows = np.arange(len(quality))
    q, c = quality[rows, actions], cost[rows, actions]
    return {"quality": float(q.mean()), "total_cost_usd": float(c.sum()),
            "mean_utility": float((q - penalty * c).mean())}


def fit(values, rewards, *, max_factors=None):
    """Greedy joint factor selection by total utility, with deterministic ties.

    A zero-gain step is allowed if it separates unresolved outcomes (e.g. XOR).
    Stop at the oracle or when no remaining feature separates unresolved cells.
    """
    selected, history = [], []
    groups = [np.arange(len(rewards))]
    actions = np.zeros(len(rewards), dtype=int)
    remaining = np.ones(values.shape[1], dtype=bool)
    oracle = rewards.max(axis=1)
    while True:
        gains = np.zeros(values.shape[1])
        separations = np.zeros(values.shape[1])
        total = 0.0
        for indices in groups:
            totals = rewards[indices].sum(axis=0)
            action = int(totals.argmax())
            actions[indices] = action
            best = totals[action]
            total += best
            if oracle[indices].sum() - best <= 1e-12:
                continue
            x = values[indices].astype(float)
            yes = x.T @ rewards[indices]
            no = totals - yes
            counts = x.sum(axis=0)
            viable = (counts > 0) & (counts < len(indices))
            gains += np.where(viable, yes.max(axis=1) + no.max(axis=1) - best, 0)
            separations += np.where(viable, np.minimum(counts, len(indices) - counts), 0)
        history.append({"factors": len(selected), "cells": len(groups),
                        "mean_utility": float(total / len(rewards)),
                        "actions": actions.copy()})
        if oracle.sum() - total <= 1e-10:
            break
        if max_factors is not None and len(selected) >= max_factors:
            break
        eligible = remaining & (separations > 0)
        if not eligible.any():
            break
        gains[~eligible] = -np.inf
        top = gains.max()
        # Prefer the earliest readable factor for numerically tied gains.
        feature = int(np.flatnonzero(eligible & (gains >= top - 1e-12))[0])
        if top <= 1e-12:
            feature = int(np.argmax(np.where(eligible, separations, -1)))
        selected.append(feature)
        remaining[feature] = False
        split = []
        for indices in groups:
            mask = values[indices, feature]
            split.extend(part for part in (indices[~mask], indices[mask]) if len(part))
        groups = split
    return selected, history, groups


def partition(values, selected, rewards):
    if selected:
        _, inverse = np.unique(values[:, selected], axis=0, return_inverse=True)
    else:
        inverse = np.zeros(len(values), dtype=int)
    groups = [np.flatnonzero(inverse == i) for i in range(inverse.max() + 1)]
    actions = np.zeros(len(values), dtype=int)
    for group in groups:
        actions[group] = rewards[group].sum(axis=0).argmax()
    return actions, groups


def prune(values, selected, rewards):
    """Remove factors while preserving attained utility; no global minimality claim."""
    selected = list(selected)
    actions, _ = partition(values, selected, rewards)
    target = rewards[np.arange(len(values)), actions].sum()
    for feature in reversed(selected.copy()):
        candidate = [i for i in selected if i != feature]
        actions, _ = partition(values, candidate, rewards)
        score = rewards[np.arange(len(values)), actions].sum()
        if score >= target - 1e-10:
            selected = candidate
    return selected


def audit(prepared, *, lexical=False, penalty=10.0, max_factors=None):
    rows = prepared["train"] + prepared["test"]
    quality = np.array([r["quality"] for r in rows])
    cost = np.array([r["cost"] for r in rows])
    rewards = quality - penalty * cost
    names, values = extract([r["state"] for r in rows], lexical=lexical)
    selected, history, groups = fit(values, rewards, max_factors=max_factors)
    forward_factors = [names[i] for i in selected]
    selected = prune(values, selected, rewards)
    final_actions, groups = partition(values, selected, rewards)
    steps = []
    for h in history:
        steps.append({k: v for k, v in h.items() if k != "actions"} |
                     metrics(quality, cost, h["actions"], penalty))
    factors = [names[i] for i in selected]
    cells = []
    for group in groups:
        action = int(rewards[group].sum(axis=0).argmax())
        cells.append({"factor_bits": values[group[0], selected].astype(int).tolist(),
                      "prompts": len(group), "model": prepared["models"][action],
                      **metrics(quality[group], cost[group], np.full(len(group), action), penalty)})
    fixed = int(rewards.sum(axis=0).argmax())
    return {"sample_fingerprint": fingerprint(prepared), "prompts": len(rows),
            "scope": "fit and evaluate on all supplied prompts", "penalty": penalty,
            "search": "greedy forward partition refinement; not a minimum-set proof",
            "candidate_factors": len(names), "lexical": lexical,
            "oracle": metrics(quality, cost, rewards.argmax(axis=1), penalty),
            "best_fixed_model": {"model": prepared["models"][fixed],
                                 **metrics(quality, cost, np.full(len(rows), fixed), penalty)},
            "forward_selected_factors": forward_factors,
            "selected_factors": factors, "steps": steps, "cells": cells,
            "pruned": {"factors": len(selected), "cells": len(groups),
                       **metrics(quality, cost, final_actions, penalty)},
            "unresolved_prompts": int(np.sum(
                rewards.max(axis=1) - rewards[np.arange(len(rows)), history[-1]["actions"]] > 1e-12))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--lexical", action="store_true")
    parser.add_argument("--penalty", type=float, default=10.0)
    parser.add_argument("--max-factors", type=int)
    args = parser.parse_args()
    if args.penalty < 0 or not np.isfinite(args.penalty):
        parser.error("penalty must be finite and nonnegative")
    if args.max_factors is not None and args.max_factors < 1:
        parser.error("max-factors must be positive")
    result = audit(json.loads(args.prepared.read_text()), lexical=args.lexical,
                   penalty=args.penalty, max_factors=args.max_factors)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("cells", "steps")}, indent=2))
    print(json.dumps(result["steps"], indent=2))


if __name__ == "__main__":
    main()
