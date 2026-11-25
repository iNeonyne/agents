"""Email digest of open pull-request statuses for LastHavenPlus.

Runs on demand (manual executions). Each run fetches every open PR of the source
repository with its aggregate check state, compares it against the "last seen"
snapshot stored as a file in a state repository on GitHub, and emails a report
through the Gmail connector only when something changed (unless forced).
"""

import base64
import html
import json
import urllib.parse
from datetime import UTC, datetime, timedelta
from typing import Any

import pydantic
from env.pr_status import pr_status_env
from mistralai import workflows
from mistralai.workflows import Depends
from mistralai.workflows.plugins.mistralai.connectors import (
    ToolCallClient,
    connector,
    uses_connectors,
)

GITHUB_API_BASE_URL = "https://api.github.com"

CHECKS_PASSING = "passing"
CHECKS_FAILING = "failing"
CHECKS_PENDING = "pending"
CHECKS_NONE = "none"

_FAILURE_CONCLUSIONS = frozenset(
    {"failure", "cancelled", "timed_out", "startup_failure", "action_required"}
)
_PASS_CONCLUSIONS = frozenset({"success", "neutral", "skipped"})

CHECK_EMOJI = {
    CHECKS_PASSING: "✅",
    CHECKS_FAILING: "❌",
    CHECKS_PENDING: "⏳",
    CHECKS_NONE: "🚫",
}

GITHUB_UNAUTHENTICATED_RATE_LIMIT = workflows.RateLimit(
    time_window_in_sec=1,
    max_execution=5,
    key="github-public-api",
)

gmail_connector = connector("gmail")
state_github_connector = connector("github_app")


class PrStatusDigestRequest(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(hide_input_in_errors=True)

    recipient: str | None = None
    force_email: bool = False
    dry_run: bool = False


class PullRequestStatus(pydantic.BaseModel):
    number: int
    title: str
    head_ref: str
    base_ref: str
    head_sha: str
    draft: bool
    updated_at: str
    html_url: str
    checks: str


class SnapshotRef(pydantic.BaseModel):
    blob_sha: str | None
    entries: dict[str, dict[str, Any]]


class PrStatusDigestResult(pydantic.BaseModel):
    open_pull_requests: int
    failing: int
    pending: int
    changed: bool
    changed_numbers: list[int]
    removed_numbers: list[int]
    email_sent: bool
    reason: str


def _require_ok_response(status_code: int) -> None:
    if 200 <= status_code < 300:
        return
    retryable = status_code in {429, 502, 503, 504}
    raise workflows.WorkflowError(
        f"github_http_{status_code}",
        non_retryable=not retryable,
        next_retry_delay=None,
    )


def _aggregate_check_state(check_runs: list[dict[str, Any]]) -> str:
    if not check_runs:
        return CHECKS_NONE
    conclusions = [(run.get("status"), run.get("conclusion")) for run in check_runs]
    if any(conclusion in _FAILURE_CONCLUSIONS for _status, conclusion in conclusions):
        return CHECKS_FAILING
    if any(status != "completed" for status, _conclusion in conclusions):
        return CHECKS_PENDING
    if all(
        status == "completed" and conclusion in _PASS_CONCLUSIONS
        for status, conclusion in conclusions
    ):
        return CHECKS_PASSING
    return CHECKS_PENDING


def _snapshot_entries(statuses: list[PullRequestStatus]) -> dict[str, dict[str, Any]]:
    return {
        str(status.number): {
            "number": status.number,
            "title": status.title,
            "head_ref": status.head_ref,
            "base_ref": status.base_ref,
            "head_sha": status.head_sha,
            "draft": status.draft,
            "updated_at": status.updated_at,
            "html_url": status.html_url,
            "checks": status.checks,
        }
        for status in statuses
    }


def _diff_entries(
    current: dict[str, dict[str, Any]],
    previous: dict[str, dict[str, Any]],
) -> tuple[list[int], list[int]]:
    changed = sorted(
        int(number)
        for number, entry in current.items()
        if number not in previous or previous[number] != entry
    )
    removed = sorted(int(number) for number in previous if number not in current)
    return changed, removed


def _report_rows(
    statuses: list[PullRequestStatus], changed_numbers: list[int]
) -> list[PullRequestStatus]:
    if changed_numbers:
        wanted = set(changed_numbers)
        return [status for status in statuses if status.number in wanted]
    return statuses


def _subject(statuses: list[PullRequestStatus], baseline: bool) -> str:
    repo = f"{pr_status_env.lasthaven_owner}/{pr_status_env.lasthaven_repo}"
    failing = sum(status.checks == CHECKS_FAILING for status in statuses)
    pending = sum(status.checks == CHECKS_PENDING for status in statuses)
    counts = f"{len(statuses)} update(s)"
    detail = f", {failing} failing" if failing else ""
    detail += f", {pending} pending" if pending else ""
    opener = "PR baseline" if baseline else "PR status changed"
    return f"[{repo}] {opener}: {counts}{detail}"


def _render_text(statuses: list[PullRequestStatus], removed_numbers: list[int]) -> str:
    lines = [
        f"{CHECK_EMOJI[status.checks]} #{status.number} {status.title} "
        f"({status.head_ref} -> {status.base_ref}, checks: {status.checks})"
        for status in statuses
    ]
    if removed_numbers:
        lines.append(
            "Closed/merged since last report: "
            + ", ".join(f"#{number}" for number in removed_numbers)
        )
    lines.append(
        "Open PRs: "
        f"https://github.com/{pr_status_env.lasthaven_owner}/"
        f"{pr_status_env.lasthaven_repo}/pulls"
    )
    return "\n".join(lines)


def _render_html(statuses: list[PullRequestStatus], removed_numbers: list[int]) -> str:
    def escapes(value: str) -> str:
        return html.escape(value, quote=True)

    owner_repo = f"{pr_status_env.lasthaven_owner}/{pr_status_env.lasthaven_repo}"
    rows = [
        (
            "<tr>"
            f"<td>{CHECK_EMOJI[status.checks]}</td>"
            f'<td><a href="{escapes(status.html_url)}">#{status.number}</a></td>'
            f"<td>{escapes(status.title)}</td>"
            f"<td>{escapes(status.head_ref)} &rarr; {escapes(status.base_ref)}</td>"
            f"<td>{escapes(status.checks)}{escapes(' (draft)' if status.draft else '')}</td>"
            f"<td>{escapes(status.updated_at)}</td>"
            "</tr>"
        )
        for status in statuses
    ]
    removed_section = ""
    if removed_numbers:
        removed_section = "<p>Closed/merged since last report: {}</p>".format(
            ", ".join(f"#{number}" for number in removed_numbers)
        )
    return (
        "<html><body>"
        f'<p><a href="https://github.com/{owner_repo}/pulls">{escapes(owner_repo)}</a>'
        " open pull-request report</p>"
        '<table border="1" cellpadding="6" cellspacing="0" '
        'style="border-collapse:collapse">'
        "<tr><th>Checks</th><th>PR</th><th>Title</th><th>Branch</th>"
        "<th>Status</th><th>Updated</th></tr>"
        + "".join(rows)
        + "</table>"
        + removed_section
        + '<p style="color:#666">Sent by the lasthaven-pr-status-email workflow.</p>'
        "</body></html>"
    )


@workflows.activity(
    name="pr-status-fetch-open-prs",
    start_to_close_timeout=timedelta(seconds=30),
    retry_policy_max_attempts=3,
    retry_policy_backoff_coefficient=2.0,
    rate_limit=GITHUB_UNAUTHENTICATED_RATE_LIMIT,
)
async def fetch_open_prs(owner: str, repo: str) -> list[PullRequestStatus]:
    import httpx

    async with httpx.AsyncClient(
        base_url=GITHUB_API_BASE_URL,
        timeout=20,
        follow_redirects=False,
        headers={"Accept": "application/vnd.github+json"},
    ) as client:
        pulls_response = await client.get(
            f"/repos/{owner}/{repo}/pulls",
            params={"state": "open", "per_page": 100, "page": 1},
        )
        _require_ok_response(pulls_response.status_code)
        pulls = pulls_response.json()
        statuses: list[PullRequestStatus] = []
        for pull in pulls[:100]:
            sha = pull["head"]["sha"]
            checks_response = await client.get(
                f"/repos/{owner}/{repo}/commits/{sha}/check-runs",
                params={"per_page": 100},
            )
            if checks_response.status_code == 404:
                check_runs: list[dict[str, Any]] = []
            else:
                _require_ok_response(checks_response.status_code)
                check_runs = checks_response.json().get("check_runs", [])
            statuses.append(
                PullRequestStatus(
                    number=pull["number"],
                    title=pull["title"],
                    head_ref=pull["head"]["ref"],
                    base_ref=pull["base"]["ref"],
                    head_sha=sha,
                    draft=bool(pull.get("draft", False)),
                    updated_at=pull["updated_at"],
                    html_url=pull["html_url"],
                    checks=_aggregate_check_state(check_runs),
                )
            )
        return statuses


@workflows.activity(
    name="pr-status-load-snapshot",
    start_to_close_timeout=timedelta(seconds=15),
    retry_policy_max_attempts=2,
    rate_limit=GITHUB_UNAUTHENTICATED_RATE_LIMIT,
)
async def load_snapshot(owner: str, repo: str, branch: str, path: str) -> SnapshotRef:
    import httpx

    encoded_path = urllib.parse.quote(path.lstrip("/"), safe="/")
    url_suffix = f"/repos/{owner}/{repo}/contents/{encoded_path}?ref={urllib.parse.quote(branch)}"
    async with httpx.AsyncClient(
        base_url=GITHUB_API_BASE_URL,
        timeout=15,
        follow_redirects=False,
        headers={"Accept": "application/vnd.github+json"},
    ) as client:
        response = await client.get(url_suffix)
    if response.status_code == 404:
        return SnapshotRef(blob_sha=None, entries={})
    _require_ok_response(response.status_code)
    body = response.json()
    blob_sha = body.get("sha")
    if blob_sha is None:
        return SnapshotRef(blob_sha=None, entries={})
    content = body.get("content", "")
    if body.get("encoding") == "base64" and content:
        content = base64.b64decode(content).decode("utf-8")
    try:
        parsed = json.loads(content)
        entries = parsed.get("entries", {})
    except (json.JSONDecodeError, AttributeError):
        entries = {}
    if not isinstance(entries, dict):
        entries = {}
    return SnapshotRef(blob_sha=blob_sha, entries=entries)


@workflows.activity(
    name="pr-status-send-email",
    start_to_close_timeout=timedelta(seconds=30),
    retry_policy_max_attempts=1,
)
async def send_report_email(
    to: list[str],
    subject: str,
    text: str,
    body_html: str,
    gmail: ToolCallClient = Depends(gmail_connector),  # noqa: B008
) -> None:
    await gmail.call_tool(
        tool_name="send_email",
        arguments={"to": to, "subject": subject, "body": text, "htmlBody": body_html},
    )


@workflows.activity(
    name="pr-status-save-snapshot",
    start_to_close_timeout=timedelta(seconds=30),
    retry_policy_max_attempts=1,
)
async def save_snapshot(
    owner: str,
    repo: str,
    branch: str,
    path: str,
    blob_sha: str | None,
    entries: dict[str, dict[str, Any]],
    state_github: ToolCallClient = Depends(state_github_connector),  # noqa: B008
) -> None:
    payload = {
        "schema": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "entries": entries,
    }
    arguments: dict[str, Any] = {
        "owner": owner,
        "repo": repo,
        "path": path,
        "content": json.dumps(payload, indent=2, sort_keys=True),
        "message": "chore(pr-status): refresh LastHavenPlus PR status snapshot",
        "branch": branch,
    }
    if blob_sha is not None:
        arguments["sha"] = blob_sha
    await state_github.call_tool(
        tool_name="create_or_update_file",
        arguments=arguments,
    )


@workflows.workflow.define(name="lasthaven-pr-status-email", on_behalf_of=True)
@uses_connectors(gmail_connector, state_github_connector)
class LastHavenPrStatusEmailWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, request: PrStatusDigestRequest) -> PrStatusDigestResult:
        statuses = await fetch_open_prs(
            pr_status_env.lasthaven_owner,
            pr_status_env.lasthaven_repo,
        )
        snapshot = await load_snapshot(
            pr_status_env.pr_status_state_owner,
            pr_status_env.pr_status_state_repo,
            pr_status_env.pr_status_state_branch,
            pr_status_env.pr_status_state_path,
        )
        current_entries = _snapshot_entries(statuses)
        changed_numbers, removed_numbers = _diff_entries(
            current_entries, snapshot.entries
        )
        baseline = snapshot.blob_sha is None
        failing = sum(status.checks == CHECKS_FAILING for status in statuses)
        pending = sum(status.checks == CHECKS_PENDING for status in statuses)
        changed = bool(changed_numbers or removed_numbers)

        if request.dry_run:
            return PrStatusDigestResult(
                open_pull_requests=len(statuses),
                failing=failing,
                pending=pending,
                changed=changed,
                changed_numbers=changed_numbers,
                removed_numbers=removed_numbers,
                email_sent=False,
                reason="dry_run",
            )

        if not changed and not request.force_email:
            return PrStatusDigestResult(
                open_pull_requests=len(statuses),
                failing=failing,
                pending=pending,
                changed=False,
                changed_numbers=changed_numbers,
                removed_numbers=removed_numbers,
                email_sent=False,
                reason="no_change",
            )

        recipient = request.recipient or pr_status_env.pr_status_email_recipient
        if not recipient:
            raise workflows.WorkflowError(
                "recipient_not_configured",
                non_retryable=True,
            )

        report_statuses = _report_rows(statuses, changed_numbers)
        subject = _subject(report_statuses, baseline=baseline)
        text = _render_text(report_statuses, removed_numbers)
        body_html = _render_html(report_statuses, removed_numbers)
        await send_report_email(
            [recipient],
            subject,
            text,
            body_html,
        )
        await save_snapshot(
            pr_status_env.pr_status_state_owner,
            pr_status_env.pr_status_state_repo,
            pr_status_env.pr_status_state_branch,
            pr_status_env.pr_status_state_path,
            snapshot.blob_sha,
            current_entries,
        )
        return PrStatusDigestResult(
            open_pull_requests=len(statuses),
            failing=failing,
            pending=pending,
            changed=changed,
            changed_numbers=changed_numbers,
            removed_numbers=removed_numbers,
            email_sent=True,
            reason="changed" if changed else "forced",
        )
