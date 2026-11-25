"""Unit tests for the pure logic of the LastHavenPlus PR-status digest workflow."""

from worker.workflows.pr_status_email import (
    CHECKS_FAILING,
    CHECKS_NONE,
    CHECKS_PASSING,
    CHECKS_PENDING,
    PrStatusDigestRequest,
    PrStatusDigestResult,
    PullRequestStatus,
    _aggregate_check_state,
    _diff_entries,
    _render_html,
    _render_text,
    _report_rows,
    _snapshot_entries,
    _subject,
)


def _status(
    number: int, checks: str = CHECKS_PASSING, title: str | None = None
) -> PullRequestStatus:
    return PullRequestStatus(
        number=number,
        title=title or f"PR {number}",
        head_ref=f"feature/{number}",
        base_ref="beta",
        head_sha=f"sha-{number}",
        draft=False,
        updated_at="2026-11-25T00:00:00Z",
        html_url=f"https://github.com/iNeonyne/LastHavenPlus/pull/{number}",
        checks=checks,
    )


def test_aggregate_check_state_buckets() -> None:
    assert _aggregate_check_state([]) == CHECKS_NONE
    assert (
        _aggregate_check_state([{"status": "completed", "conclusion": "success"}])
        == CHECKS_PASSING
    )
    assert (
        _aggregate_check_state([{"status": "completed", "conclusion": "neutral"}])
        == CHECKS_PASSING
    )
    assert (
        _aggregate_check_state([{"status": "completed", "conclusion": "skipped"}])
        == CHECKS_PASSING
    )
    assert (
        _aggregate_check_state([{"status": "completed", "conclusion": "failure"}])
        == CHECKS_FAILING
    )
    assert (
        _aggregate_check_state([{"status": "completed", "conclusion": "timed_out"}])
        == CHECKS_FAILING
    )
    assert (
        _aggregate_check_state([{"status": "in_progress", "conclusion": None}])
        == CHECKS_PENDING
    )
    assert (
        _aggregate_check_state(
            [
                {"status": "completed", "conclusion": "success"},
                {"status": "in_progress", "conclusion": None},
            ]
        )
        == CHECKS_PENDING
    )
    assert (
        _aggregate_check_state(
            [
                {"status": "completed", "conclusion": "success"},
                {"status": "completed", "conclusion": "failure"},
            ]
        )
        == CHECKS_FAILING
    )
    assert (
        _aggregate_check_state([{"status": "unknown", "conclusion": "weird"}])
        == CHECKS_PENDING
    )


def test_diff_entries_detects_add_update_and_removal() -> None:
    current = _snapshot_entries([_status(2), _status(3)])
    previous = {**current, "3": {**current["3"], "checks": CHECKS_NONE}, "1": {}}
    previous.pop("2")

    changed, removed = _diff_entries(current, previous)

    assert changed == [2, 3]
    assert removed == [1]


def test_diff_entries_ignores_unchanged() -> None:
    entries = _snapshot_entries([_status(5)])
    changed, removed = _diff_entries(dict(entries), dict(entries))

    assert changed == []
    assert removed == []


def test_snapshot_entries_shape_is_stable() -> None:
    entries = _snapshot_entries([_status(9, checks=CHECKS_FAILING, title="T")])

    assert entries == {
        "9": {
            "number": 9,
            "title": "T",
            "head_ref": "feature/9",
            "base_ref": "beta",
            "head_sha": "sha-9",
            "draft": False,
            "updated_at": "2026-11-25T00:00:00Z",
            "html_url": "https://github.com/iNeonyne/LastHavenPlus/pull/9",
            "checks": CHECKS_FAILING,
        }
    }


def test_report_rows_filters_to_changed_when_present() -> None:
    statuses = [_status(1), _status(2), _status(3)]

    assert [s.number for s in _report_rows(statuses, [2])] == [2]
    assert [s.number for s in _report_rows([], [])] == []
    assert [s.number for s in _report_rows(statuses, [])] == [1, 2, 3]


def test_subject_and_reports_cover_failing_and_removed() -> None:
    statuses = [_status(1, checks=CHECKS_FAILING), _status(2)]

    subject = _subject(statuses, baseline=False)
    text = _render_text(statuses, removed_numbers=[7])
    body_html = _render_html(statuses, removed_numbers=[7])

    assert "2 update(s)" in subject
    assert "1 failing" in subject
    assert "LastHavenPlus" in subject
    assert "#7" in text and "Closed/merged" in text
    assert "❌" in text and "#1" in text
    assert "https://github.com/iNeonyne/LastHavenPlus/pulls" in text
    assert "#7" in body_html and "Closed/merged" in body_html
    assert "<table" in body_html and "#1" in body_html
    assert "<b>" not in body_html


def test_request_and_result_models_defaults() -> None:
    request = PrStatusDigestRequest()
    result = PrStatusDigestResult(
        open_pull_requests=0,
        failing=0,
        pending=0,
        changed=False,
        changed_numbers=[],
        removed_numbers=[],
        email_sent=False,
        reason="no_change",
    )

    assert request.recipient is None
    assert request.force_email is False
    assert request.dry_run is False
    assert result.email_sent is False
