# zarr-python agent sandbox (devcontainer)

A sandboxed container for running a coding agent (e.g. Claude Code with
`--dangerously-skip-permissions`) on an isolated branch.

## Security model

The threat: a prompt-injected or runaway skip-permissions agent that could damage
the host or exfiltrate secrets. Defenses, in layers:

- **No host bind mounts.** The container clones its **own** copy of the repo
  (`onCreateCommand`); the agent's filesystem cannot reach the host repo,
  `~/.ssh`, `~/.aws`, or the host `~/.gitconfig` (which may carry tokens).
- **A separate GitHub identity.** Use a **dedicated agent GitHub account** with a
  **fork-only, fine-grained, expiring** Personal Access Token — never your
  personal account or a broad token. The agent works on its **fork**; you review
  and merge PRs from your real account. A leaked token can only touch the agent's
  forks, not your canonical repos.
- **Default-deny egress.** `init-firewall.sh` drops all outbound traffic except
  an allowlist (PyPI, GitHub, the Anthropic API, the VS Code marketplace), so a
  leaked secret has nowhere to go.
- **Fresh Claude identity.** `~/.claude` is a container-only volume; your host
  credentials and memories are not mounted.

The sandbox limits *local* blast radius; the scoped agent account limits *GitHub*
blast radius. Use both.

## Prerequisites

- Docker running on the host.
- The devcontainer CLI: `npm install -g @devcontainers/cli`.
- A **dedicated agent GitHub account** that has **forked** the repo. Create a
  fine-grained PAT for it scoped to that fork only (`contents: read/write`,
  `pull requests: write`; no admin/workflow/org scopes; short expiry).
- In `devcontainer.json`, set `ZARR_REPO_URL` to the agent account's fork and
  `GIT_AUTHOR_EMAIL`/`GIT_AUTHOR_NAME` to the agent's identity.

## 1. Start the container

```bash
cd /path/to/checkout      # any dir containing .devcontainer/ (NOT a worktree)
devcontainer up --workspace-folder .
```

The container clones `ZARR_REPO_URL` into its own `/workspace` — it does **not**
use the folder you launched from for code, only for the `.devcontainer/` config.

This builds the image, applies the bind mount, runs `postCreateCommand`
(`uv sync --group dev`, installs the `prek` binary), then `postStartCommand`
(brings up the firewall). First run takes a few minutes.

## 2. Open a shell and start the agent

```bash
devcontainer exec --workspace-folder . zsh
```

You land as the `node` user in `/workspace`.

**First time only — authenticate GitHub as the agent account.** This is done
*inside* the container so the token lives only in the container's `gh` config
volume, never on the host environment:

```bash
gh auth login    # log in as the AGENT account, paste its fine-grained PAT
```

This persists across rebuilds (the `~/.config/gh` named volume) and is what the
agent uses to push to its fork and open PRs.

Start the agent:

```bash
claude --dangerously-skip-permissions
```

`claude` refuses to run as root by design; `node` is the non-root `remoteUser`,
so this is fine. Run `devcontainer exec ... zsh` again in another terminal for
additional shells into the same container.

The first time, `claude` will also prompt you to log in: the container's
`~/.claude` is a **fresh, container-only identity**, not your host's — your host
credentials and memories are not mounted. The login (and any memories the agent
makes) persist in a named volume across rebuilds. Project knowledge the agent
should have lives in the repo's `CLAUDE.md` at `/workspace/CLAUDE.md`.

## 3. Connect an editor

The container is just a running Docker container; attach however you like.

### VS Code

Open the repo in VS Code and run **Dev Containers: Attach to Running
Container...**, or **Reopen in Container** (it reuses the container started in
step 1).

### Emacs (TRAMP)

Find the container name, then open files over the `docker` TRAMP method:

```bash
docker ps --filter "label=devcontainer.local_folder=$(pwd)" --format '{{.Names}}'
# (run from the folder you launched the container from)
```

```
C-x C-f /docker:CONTAINER_NAME:/workspace/src/zarr/core/group.py
```

(`/docker:` is built into Emacs 29+; older Emacs needs the `docker-tramp`
package.) `dired` on `/docker:CONTAINER_NAME:/workspace/` works the same way.

### Plain `docker exec` (any editor / terminal)

```bash
docker exec -it -u node CONTAINER_NAME zsh
```

The `-u node` matters: a bare `docker exec` lands you as root, where `claude`
won't run.

## Remote workflow (laptop → SSH → desktop → container)

When the Docker host is a remote desktop you reach over SSH, and you want to
drive the agent and browse the code from a laptop:

```
[laptop: VS Code UI] --Remote-SSH--> [desktop: Docker host] --> [container]
```

**One-time, on the desktop:** clone a *normal* checkout (not a worktree) and
install the CLI:

```bash
git clone https://github.com/<you>/zarr-python.git ~/dev/zarr-python-sandbox
cd ~/dev/zarr-python-sandbox && git checkout devcontainer   # until merged to main
npm install -g @devcontainers/cli
```

**Start the container (over SSH on the desktop)** — it runs independently of any
editor:

```bash
cd ~/dev/zarr-python-sandbox
devcontainer up --workspace-folder .
```

**Run the agent in tmux** so it survives SSH drops and laptop sleep:

```bash
devcontainer exec --workspace-folder . zsh
# inside the container:
tmux new -s agent
claude --dangerously-skip-permissions
# detach with Ctrl-b d; reattach later with: tmux attach -t agent
```

**Attach an editor from the laptop:** in VS Code, *Remote-SSH: Connect to Host*
(the desktop), then *Dev Containers: Attach to Running Container* and pick the
running container. The editor now browses/edits files inside the container; its
integrated terminal is inside too, so `tmux attach -t agent` reaches the agent.

**Reconnect after a drop:** SSH back to the desktop and `tmux attach -t agent`
(or reattach VS Code) — the agent and container keep running throughout.

## Lifecycle

```bash
# stop (keeps the container; restart is fast)
docker stop CONTAINER_NAME

# rebuild after changing the Dockerfile or firewall script
devcontainer up --workspace-folder . --remove-existing-container
# (use --build-no-cache if a cached COPY layer serves a stale init-firewall.sh)
```

## Notes

- The container works on its **own clone** (`/workspace`), not your host repo.
  Edits never touch host files. The agent commits to a branch and pushes to its
  fork; you open/merge a PR to your canonical repo from your real account.
  Pushes go out through the firewall allowlist (GitHub is allowed).
- The clone is a real repo, so `prek install` runs and git hooks work normally.
- To change the firewall allowlist, edit `init-firewall.sh` and rebuild — the
  script is baked into the image (`COPY`), so edits need a rebuild to take effect.
- `--build-no-cache` if a cached `COPY` layer serves a stale `init-firewall.sh`.
