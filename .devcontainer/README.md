# zarr-python agent sandbox (devcontainer)

A sandboxed container for running a coding agent (e.g. Claude Code with
`--dangerously-skip-permissions`) on an isolated branch. Isolation comes from two
layers: the container boundary, and an outbound-network firewall
(`init-firewall.sh`) that allowlists only the hosts the dev workflow needs
(PyPI, GitHub, the Anthropic API, the VS Code marketplace).

The container **bind-mounts the folder you open** (`${localWorkspaceFolder}`) at
`/workspace`. Open the **main repo** (or any normal clone) — its real `.git`
directory makes git, `prek`, and `hatch-vcs` version detection work inside the
container. Do **not** open a git *worktree*: a worktree's `.git` metadata lives
in the main repo outside the mount, which breaks git inside the container.
Branch isolation comes from the container + firewall; the agent works on its own
branch in here. The config has no hardcoded host paths, so it is portable.

## Prerequisites

- Docker running on the host.
- The devcontainer CLI: `npm install -g @devcontainers/cli`
  (or invoke via `npx @devcontainers/cli ...`).
- Optional, for non-interactive auth: export `CLAUDE_CODE_OAUTH_TOKEN` (or
  `ANTHROPIC_API_KEY`) on the host before starting; it is passed through via
  `remoteEnv`. Without it, `claude` prompts for interactive login inside.

## 1. Start the container

```bash
cd /path/to/zarr-python      # a NORMAL clone (not a worktree) with .devcontainer/
devcontainer up --workspace-folder .
```

This builds the image, applies the bind mount, runs `postCreateCommand`
(`uv sync --group dev`, installs the `prek` binary), then `postStartCommand`
(brings up the firewall). First run takes a few minutes.

## 2. Open a shell and start the agent

```bash
devcontainer exec --workspace-folder . zsh
```

You land as the `node` user in `/workspace`. Start the agent:

```bash
claude --dangerously-skip-permissions
```

`claude` refuses to run as root by design; `node` is the non-root `remoteUser`,
so this is fine. Run `devcontainer exec ... zsh` again in another terminal for
additional shells into the same container.

The first time, `claude` will prompt you to log in: the container's `~/.claude`
is a **fresh, container-only identity**, not your host's. This is deliberate —
your host credentials and memories are not mounted, so a skip-permissions agent
can't read or exfiltrate them. The login (and any memories the agent makes)
persist in a named volume across rebuilds. Project knowledge the agent should
have lives in the repo's `CLAUDE.md`, which is visible at `/workspace/CLAUDE.md`.

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

## Lifecycle

```bash
# stop (keeps the container; restart is fast)
docker stop CONTAINER_NAME

# rebuild after changing the Dockerfile or firewall script
devcontainer up --workspace-folder . --remove-existing-container
# (use --build-no-cache if a cached COPY layer serves a stale init-firewall.sh)
```

## Notes

- Edits inside the container write straight to the bind-mounted main repo, so
  branch/commit from inside the container; pushes go out through the firewall
  allowlist (GitHub is allowed).
- `.git/config` and `.git/hooks` are mounted **read-only**, so the agent cannot
  rewrite your git remote or install hooks. Your host's existing pre-commit hooks
  are visible through that mount; `prek install` is therefore not run in-container
  (it would fail on the read-only mount and is unnecessary). The `prek` binary is
  installed so you can run `prek run --all-files` manually.
- To change the firewall allowlist, edit `init-firewall.sh` and rebuild — the
  script is baked into the image (`COPY`), so edits need a rebuild to take effect.
