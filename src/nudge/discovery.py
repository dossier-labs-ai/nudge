"""Bounded, exploratory question selection; never a promotion certification."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np

from .math import predict
from .replay import SamplingReplayBuffer
from .types import Proposal, ProposalContext, Question, canonical, validate_answer
from .question_evidence import summarize, update_evidence


def question_key(question: Question) -> str:
    return hashlib.sha256(canonical(question.to_dict()).encode()).hexdigest()


class BudgetExhausted(RuntimeError):
    pass


class CallBudget:
    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0

    def take(self) -> None:
        if self.used >= self.limit:
            raise BudgetExhausted("learning model-call budget exhausted")
        self.used += 1


def hydrate(rows, questions, *, store, answerer, choice, budget):
    # Concurrency is an explicit adapter opt-in; ordinary/stateful answerers
    # retain serial execution. SQLite and call-budget ownership stay here.
    concurrency = getattr(answerer, 'concurrency', 1)
    if type(concurrency) is not int or not 1 <= concurrency <= 128:
        raise ValueError('answerer concurrency must be between 1 and 128')
    started = time.monotonic()
    calls_before = budget.used
    result = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for offset in range(0, len(rows), concurrency):
            pending = []
            exhausted = False
            for row in rows[offset:offset + concurrency]:
                values = {q.name: store.cached(row['id'], question_key(q)) for q in questions}
                missing = [q for q in questions if values[q.name] is None]
                future = None
                if missing:
                    try:
                        budget.take()
                    except BudgetExhausted:
                        exhausted = True
                        break
                    # All missing heads for this prompt remain one request.
                    if concurrency > 1:
                        future = pool.submit(answerer.ask, row['state'], missing)
                pending.append((row, values, missing, future))
            for row, values, missing, future in pending:
                if missing:
                    raw = future.result() if future is not None else answerer.ask(row['state'], missing)
                    answer = validate_answer(raw, choice, missing)
                    store.cache(row['id'], {question_key(q): answer.features[q.name] for q in missing})
                    # First writer wins; apply in replay order, never worker order.
                    values.update({q.name: store.cached(row['id'], question_key(q)) for q in missing})
                result.append({**row, 'features': values})
            if exhausted:
                raise BudgetExhausted('learning model-call budget exhausted')
    if budget.used > calls_before:
        print(json.dumps({'stage': 'hydrate', 'rows': len(rows), 'heads': len(questions),
                          'calls': budget.used - calls_before, 'concurrency': concurrency,
                          'elapsed_seconds': round(time.monotonic() - started, 3)}), flush=True)
    return result


def edited(questions, proposal: Proposal, limit: int):
    if proposal.operation == 'add_bundle':
        result = list(questions)
        for question in proposal.questions:
            result = edited(result, Proposal('add', question), limit)
        return result
    names = {q.name for q in questions}
    if proposal.operation == "drop":
        if proposal.target not in names:
            raise ValueError("drop target does not exist")
        return [q for q in questions if q.name != proposal.target]
    if proposal.operation == "revise":
        if proposal.target not in names:
            raise ValueError("revise target does not exist")
        if proposal.question.name in names - {proposal.target}:
            raise ValueError("revised question name collides")
        return [proposal.question if q.name == proposal.target else q for q in questions]
    if proposal.question.name in names or len(questions) >= limit:
        raise ValueError("duplicate question or question limit reached")
    return [*questions, proposal.question]


def examples_for(rows, critics, questions, limit, round_number=1):
    result = []
    for row in rows:
        if row["weight"] == 0 or row['split'] != 'train':
            continue
        residual = row["reward"] - predict(critics, row, questions)
        result.append(
            {
                "id": row['id'],
                "split": row['split'],
                "version": row['version'],
                "propensity": row['propensity'],
                "state": row["state"],
                "action": row["action"],
                "fields": row["fields"],
                "reward": row["reward"],
                "residual": residual,
                "features": row["features"],
                "created_at": row["created_at"],
                "weight": row["weight"],
            }
        )
    return tuple(SamplingReplayBuffer(
        result, questions=questions, round_number=round_number).sample(limit))


def answer_statistics(question, rows):
    values = np.array([question.feature_vector(r['features'][question.name]) for r in rows])
    weights = np.array([r['weight'] for r in rows])
    mean = np.average(values, axis=0, weights=weights)
    std = float(np.sqrt(np.average(np.sum((values - mean)**2, axis=1), weights=weights)))
    return {'answer_mean': mean.tolist(), 'answer_std': std}


def bank_examples(rows, questions, limit, round_number):
    # All observed prompts belong to the discovery pool. Surprise is relative
    # to a simple action average, independent of the serving regressor.
    means = {a: float(np.mean([r['reward'] for r in rows if r['action'] == a]))
             for a in {r['action'] for r in rows}}
    examples = [{**r, 'residual': r['reward'] - means[r['action']]}
                for r in rows]
    sampled = SamplingReplayBuffer(examples, questions=questions,
                                   round_number=round_number, training_only=False).sample(limit)
    # Serving scores, coefficient contributions and full action distributions
    # dominated the live proposer context. Discovery needs observed evidence,
    # not the serving regressor's explanation of its own decisions. Project
    # after sampling so priority, membership and replay provenance stay intact.
    fields = ('id', 'split', 'version', 'propensity', 'state', 'action', 'fields',
              'reward', 'residual', 'features', 'created_at', 'weight', 'replay_source')
    return tuple({key: row[key] for key in fields} for row in sampled)


def check_answer(question, train, *, store, answerer, choice, budget, expert):
    result = {'question': question.name, **answer_statistics(question, train)}
    if result['answer_std'] <= expert.min_feature_std:
        return {**result, 'status': 'flat'}
    if not question.paraphrase or question.paraphrase == question.prompt:
        return {**result, 'status': 'missing_distinct_paraphrase'}
    alternate = replace(question, prompt=question.paraphrase, paraphrase=None)
    sample = train[:expert.stability_samples]
    alternates = hydrate(sample, [alternate], store=store, answerer=answerer,
                         choice=choice, budget=budget)
    error = float(np.average([
        np.sum(np.abs(np.array(question.feature_vector(a['features'][question.name])) -
                      np.array(alternate.feature_vector(b['features'][question.name])))) /
        (2 if question.kind == 'categorical' else 1)
        for a, b in zip(sample, alternates)], weights=[r['weight'] for r in sample]))
    return {**result, 'paraphrase_mae': error,
            'status': 'unstable' if error > expert.max_paraphrase_mae else 'measured'}


def discover(rows, questions, *, choice, expert, answerer, proposer, store, budget, run,
             evidence_history=None):
    questions = list(questions)
    events = []
    if proposer is None or expert.max_rounds == 0:
        return questions, rows, events
    if sum(r['weight'] for r in rows) < expert.min_split:
        return questions, rows, [{"status": "insufficient_data"}]
    evidence_history = evidence_history or {}
    for round_number in range(1, expert.max_rounds + 1):
        diagnostics = update_evidence(rows, questions, list(choice.actions), evidence_history,
                                      choice.to_dict())
        context = ProposalContext(
            choice.to_dict(),
            tuple(questions),
            bank_examples(rows, questions, expert.proposer_examples, round_number),
            round_number,
            history=tuple(store.proposal_history(choice.to_dict())),
            pending=tuple(store.pending_proposals(choice.to_dict())),
            question_diagnostics=tuple(summarize(d) for d in diagnostics.values()),
        )
        try:
            budget.take()
        except BudgetExhausted:
            events.append({"status": "budget_exhausted"})
            break
        proposals = list(proposer.propose(context))
        if len(proposals) > max(16, 2 * expert.max_questions):
            raise ValueError("proposer returned too many proposals")
        if not all(isinstance(p, Proposal) for p in proposals):
            raise ValueError("proposer must return Proposal objects")
        record = store.proposal(
            run,
            round_number,
            proposer.identity,
            context.to_dict(),
            [p.to_dict() for p in proposals],
            [],
        )
        results = []
        exhausted = False
        for proposal in proposals:
            result = {"round": round_number, "proposal": proposal.to_dict()}
            try:
                try:
                    candidate_questions = edited(questions, proposal, expert.max_questions)
                except ValueError as error:
                    result.update(status="invalid", reason=str(error))
                    continue
                candidate_rows = hydrate(
                    rows,
                    candidate_questions,
                    store=store,
                    answerer=answerer,
                    choice=choice,
                    budget=budget,
                )
                incoming = (proposal.questions if proposal.operation == 'add_bundle' else
                            (proposal.question,) if proposal.question is not None else ())
                checks = [check_answer(q, candidate_rows, store=store, answerer=answerer,
                                       choice=choice, budget=budget, expert=expert) for q in incoming]
                result['answer_checks'] = checks
                if len(checks) == 1:
                    result.update({k: v for k, v in checks[0].items() if k != 'question'})
                failed = next((c for c in checks if c['status'] != 'measured'), None)
                if failed:
                    result.update(status=failed['status'])
                    continue
                measured = update_evidence(candidate_rows, candidate_questions,
                                           list(choice.actions), diagnostics, choice.to_dict())
                result.update(status='kept', retention_basis='proposer_hypothesis_and_answer_checks',
                              question_evidence=[summarize(measured[question_key(q)]) for q in incoming])
                questions, rows = candidate_questions, candidate_rows
                evidence_history = measured
            except BudgetExhausted:
                result.update(status="budget_exhausted")
                exhausted = True
                break
            finally:
                result['retained_question_names'] = [q.name for q in questions]
                results.append(result)
                store.finish_proposal(record, results)
        events.extend(results)
        # Rejections are useful feedback: permit a revision on the next bounded round.
        if not proposals or exhausted:
            break
    return questions, rows, events
