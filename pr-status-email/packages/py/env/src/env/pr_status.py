from env._base import BaseEnv


class PrStatusEnv(BaseEnv):
    lasthaven_owner: str = "iNeonyne"
    lasthaven_repo: str = "LastHavenPlus"
    pr_status_state_owner: str = "iNeonyne"
    pr_status_state_repo: str = "agents"
    pr_status_state_branch: str = "main"
    pr_status_state_path: str = ".state/lasthaven-pr-snapshot.json"
    pr_status_email_recipient: str | None = None


pr_status_env = PrStatusEnv()
