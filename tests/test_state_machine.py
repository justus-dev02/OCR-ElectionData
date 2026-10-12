import pytest

from election_ocr.pipeline.state_machine import (
    InvalidStateTransitionError,
    ResultStatus,
    RunStatus,
    result_transition,
    run_transition,
)


def test_publish_only_from_validated():
    assert result_transition(ResultStatus.VALIDATED, ResultStatus.PUBLISHED) == ResultStatus.PUBLISHED
    for s in (ResultStatus.EXTRACTED, ResultStatus.NEEDS_REVIEW, ResultStatus.REVIEWED, ResultStatus.REJECTED):
        with pytest.raises(InvalidStateTransitionError):
            result_transition(s, ResultStatus.PUBLISHED)


def test_review_cycle():
    s = result_transition(ResultStatus.EXTRACTED, ResultStatus.NEEDS_REVIEW)
    s = result_transition(s, ResultStatus.REVIEWED)
    assert result_transition(s, ResultStatus.VALIDATED) == ResultStatus.VALIDATED


def test_terminal_states():
    with pytest.raises(InvalidStateTransitionError):
        result_transition(ResultStatus.REJECTED, ResultStatus.VALIDATED)
    with pytest.raises(InvalidStateTransitionError):
        run_transition(RunStatus.COMPLETED, RunStatus.RUNNING)
