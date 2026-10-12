"""
Lifecycle state machines (State pattern).

All status changes of result rows and processing runs go through
``transition()``; invalid transitions raise. Every transition is recorded by
the repository in ``state_transitions`` (actor + reason + timestamp).
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Set


class InvalidStateTransitionError(ValueError):
    pass


class ResultStatus(str, Enum):
    EXTRACTED = "EXTRACTED"  # candidate row built from OCR evidence
    VALIDATED = "VALIDATED"  # no blocking issue
    NEEDS_REVIEW = "NEEDS_REVIEW"  # at least one blocking issue
    REVIEWED = "REVIEWED"  # human decisions recorded, awaiting re-validation
    PUBLISHED = "PUBLISHED"  # copied to the published dataset
    SUPERSEDED = "SUPERSEDED"  # replaced by a publication of a newer run
    REJECTED = "REJECTED"  # discarded by a reviewer


class RunStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


RESULT_TRANSITIONS: Dict[ResultStatus, Set[ResultStatus]] = {
    ResultStatus.EXTRACTED: {ResultStatus.VALIDATED, ResultStatus.NEEDS_REVIEW, ResultStatus.REJECTED},
    ResultStatus.VALIDATED: {ResultStatus.PUBLISHED, ResultStatus.NEEDS_REVIEW, ResultStatus.REJECTED},
    # NEEDS_REVIEW -> VALIDATED only happens during re-validation when no blocking issue
    # remains (e.g. gazetteer supplied, a dependency row corrected); it is logged with reason.
    ResultStatus.NEEDS_REVIEW: {ResultStatus.REVIEWED, ResultStatus.REJECTED, ResultStatus.VALIDATED},
    ResultStatus.REVIEWED: {ResultStatus.VALIDATED, ResultStatus.NEEDS_REVIEW, ResultStatus.REJECTED},
    ResultStatus.PUBLISHED: {ResultStatus.SUPERSEDED},
    ResultStatus.SUPERSEDED: set(),
    ResultStatus.REJECTED: set(),
}

RUN_TRANSITIONS: Dict[RunStatus, Set[RunStatus]] = {
    RunStatus.CREATED: {RunStatus.RUNNING, RunStatus.FAILED},
    RunStatus.RUNNING: {RunStatus.COMPLETED, RunStatus.FAILED},
    RunStatus.COMPLETED: set(),
    RunStatus.FAILED: set(),
}


def transition(current: Enum, target: Enum, table: Dict) -> Enum:
    if current == target:
        return current
    allowed = table.get(current, set())
    if target not in allowed:
        raise InvalidStateTransitionError(
            f"invalid transition {current.value} -> {target.value}; allowed: {sorted(s.value for s in allowed)}"
        )
    return target


def result_transition(current: ResultStatus, target: ResultStatus) -> ResultStatus:
    return ResultStatus(transition(current, target, RESULT_TRANSITIONS))


def run_transition(current: RunStatus, target: RunStatus) -> RunStatus:
    return RunStatus(transition(current, target, RUN_TRANSITIONS))
