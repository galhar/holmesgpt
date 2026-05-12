# Serving an LLM on Newton and tunneling it to localhost

A reproducible recipe: run Ollama on a Newton GPU compute node, expose it
locally as `http://localhost:11434` for any application that speaks the
Ollama / OpenAI-compatible API.

The non-obvious bits this guide handles:

- Newton compute nodes have **no public hostname** — traffic from your laptop
  has to pipe through the login node via a reverse tunnel.
- The shared login node's port 11434 is often **already taken by another user's
  Ollama** — so this setup uses **login-node port 11435** to stay out of the way.
- Ollama **rejects HTTP requests whose `Host:` header doesn't match its bind
  address** — must start it with `OLLAMA_HOST=0.0.0.0:11434` or the tunnel fails with 403.
- Backgrounded SSH tunnels from Claude Code's Bash tool get killed unless launched via
  `run_in_background: true` (or in a real terminal, via `ssh -f -N`).

Prerequisites: Technion VPN connected, `ssh newton` works.

---

## Check what's already running

Always start here. Often nothing new needs to be set up.

```bash
# (1) Newton side: is there an ollama sbatch job running?
ssh newton "squeue | grep galhar | grep ollama"

# (2) Newton side: is the reverse tunnel exposing Ollama on login:11435?
ssh newton "curl -sf --max-time 3 http://127.0.0.1:11435/api/tags | head -c 200"

# (3) Laptop side: is the local forward open?
ss -tln | grep :11434 && curl -sf --max-time 3 http://localhost:11434/api/tags | jq -r '.models[].name'
```

- If (1), (2), (3) all succeed → you're done. Use `http://localhost:11434`.
- If (1) and (2) are fine but (3) is empty → skip to **[Open the local tunnel](#open-the-local-tunnel)**.
- If (1) is missing → skip to **[Submit the sbatch job](#submit-the-sbatch-job)**.
- If (1) and (2) run but tunnel is broken somehow → `scancel` the job and
  re-submit, then re-open the local tunnel.

---

## One-time setup (first run ever)

Only do these once per Newton account.

### 1. Allow the login node to accept passwordless SSH from compute nodes

The sbatch script needs to open a reverse tunnel `compute → login`. Since Newton's
`$HOME` is shared via NFS, authorizing Newton's own pubkey on itself makes this
passwordless:

```bash
ssh newton "cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys"
```

(If you don't have an ed25519 key on Newton yet: `ssh newton "ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519"` first.)

### 2. Write the sbatch script on Newton

Save this at `~/ollama_server.sh` on Newton. It installs Ollama (user-space, no
sudo), pulls the model, and opens the reverse tunnel.

```bash
ssh newton "cat > ~/ollama_server.sh" <<'SBATCH'
#!/bin/bash
#SBATCH -A cs
#SBATCH -p public
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --time=08:00:00
#SBATCH --job-name=ollama-benchmark
#SBATCH --output=/home/galhar/ollama_job_%j.log

set -e
echo "Job started on node: $(hostname)"
nvidia-smi -L || true

# Install ollama (tarball is .tar.zst, NOT .tgz — the old URL 404s)
if [ ! -f ~/bin/ollama ]; then
    mkdir -p ~/bin ~/ollama_lib
    cd /tmp
    curl -fsSL -o ollama.tar.zst \
        https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst
    tar --zstd -xf ollama.tar.zst -C ~/ollama_lib
    ln -sf ~/ollama_lib/bin/ollama ~/bin/ollama
    rm ollama.tar.zst
fi
export PATH="$HOME/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/ollama_lib/lib/ollama:$LD_LIBRARY_PATH"

# MUST bind to 0.0.0.0 — Ollama rejects Host-header mismatches with 403 otherwise
OLLAMA_MODELS="$HOME/.ollama/models" OLLAMA_HOST=0.0.0.0:11434 \
    ~/bin/ollama serve > ~/ollama_serve.log 2>&1 &
OLLAMA_PID=$!
sleep 10

curl -sf http://127.0.0.1:11434/api/tags >/dev/null || {
  echo "ERROR: Ollama did not start"
  cat ~/ollama_serve.log
  exit 1
}
echo "Ollama is up."

# Pull your model if not cached
MODEL="${OLLAMA_MODEL:-qwen3.5:35b}"
if ! ~/bin/ollama list | grep -q "$MODEL"; then
    echo "Pulling $MODEL..."
    ~/bin/ollama pull "$MODEL"
fi
~/bin/ollama list

# Reverse tunnel compute:11434 → login:11435 (11434 is often taken by others)
ssh -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=60 -o ServerAliveCountMax=3 \
    -N -R 11435:127.0.0.1:11434 galhar@132.68.39.200 &
TUNNEL_PID=$!

sleep 5
echo "=== READY ==="
echo "compute node: $(hostname)"
echo "exposed on Newton login at 127.0.0.1:11435"
wait $OLLAMA_PID
SBATCH
```

> Edit `OLLAMA_MODEL` to pick a different model, or pass it at submit time:
> `ssh newton "OLLAMA_MODEL=llama3.3:70b sbatch ~/ollama_server.sh"`.
>
> **GPU choice:** `PRO6000` (~48–97 GB VRAM) fits models up to ~70B quantized.
> For smaller GPUs, change `--gres=gpu:PRO6000:1` → `--gres=gpu:L40:1` etc.
>
> **Partition:** `-A cs -p public` → public partition, 8 h time limit.
> Don't use `-A espresso -p espresso` for personal projects — that's lab-reserved.

### 3. Make it executable

```bash
ssh newton "chmod +x ~/ollama_server.sh"
```

---

## Submit the sbatch job

```bash
ssh newton "sbatch ~/ollama_server.sh"
# → Submitted batch job 68xxxxxx
```

Watch it come up (model pull is ~20 GB at ~130 MB/s → a few minutes on first run,
instant afterward since models are cached in `~/.ollama/models`):

```bash
JOB=<job-id-from-above>
ssh newton "tail -20 ~/ollama_job_${JOB}.log | tr '\r' '\n' | grep -v '^pulling\|^verifying\|^$' | tail -10"
```

Wait for `=== READY ===`. Confirm the reverse tunnel landed:

```bash
ssh newton "curl -sf http://127.0.0.1:11435/api/tags | head -c 200"
```

---

## Open the local tunnel

This forwards Newton-login:11435 → your laptop:11434.

**In a normal terminal:**
```bash
ssh -f -g -N -L 11434:127.0.0.1:11435 \
    -o ExitOnForwardFailure=no \
    -o ServerAliveInterval=20 \
    -o ServerAliveCountMax=3 \
    newton
```

- `-g` binds to all interfaces — required so Docker / kind containers can reach
  Ollama via `host.docker.internal` or the bridge gateway IP.
- `ExitOnForwardFailure=no` is important: your `~/.ssh/config` may have a
  `LocalForward` that fails to bind IPv6 (`bind [::1]:8085: Cannot assign
  requested address`). Without `=no`, that benign IPv6 failure kills the whole
  SSH session.

**From Claude Code's Bash tool:** use `run_in_background: true` with the same
command (without `-f`) — manual `&` backgrounding exits with code 144.

### Verify

```bash
# local loopback
curl -sf http://localhost:11434/api/tags | jq -r '.models[].name'
# from inside a container on a docker network (e.g. from kind)
# the bridge gateway IP is typically 172.XX.0.1 — find with:
#   docker network inspect <network> --format '{{(index .IPAM.Config 0).Gateway}}'
```

Both should return the model name. First inference is slow (~60 s warm-up for
35B on PRO6000); subsequent calls are fast.

---

## Stop / cleanup

```bash
# Kill the local tunnel
pkill -f 'ssh .* -L 11434:127.0.0.1:11435'

# Cancel the Newton job (frees the GPU)
ssh newton "scancel <job-id>"
# or find all your ollama jobs:
ssh newton "squeue | awk '/galhar.*ollama/ {print \$1}' | xargs -r scancel"
```

The model weights persist in `~/.ollama/models` on Newton — next submit is fast.

---

## Troubleshooting

| Symptom                                    | Cause                                                 | Fix                                                      |
|--------------------------------------------|-------------------------------------------------------|----------------------------------------------------------|
| `ssh: connect to host 132.68.39.200 port 22: Connection timed out` | Technion VPN disconnected               | Reconnect VPN                                            |
| Tunnel process exits 255 soon after opening | VPN dropped, or idle connection timed out            | Reconnect VPN + re-run tunnel; keepalives (above) help   |
| `curl: Recv failure: Connection reset by peer` against local 11434 | The local port is listening but the SSH forward failed to establish a channel (login → 11435 was down) | `ssh newton "curl 127.0.0.1:11435/api/tags"` to confirm; usually the sbatch job was cancelled / timed out |
| Reverse tunnel log: `bind: Address already in use` | Another user has port 11434 on login                 | Already handled — we use 11435. If 11435 is also taken, try 11436 etc. |
| 403 Forbidden when curling Ollama via the tunnel | Ollama started with `OLLAMA_HOST=127.0.0.1:11434` instead of `0.0.0.0:11434`; rejects non-localhost Host headers | Fix `OLLAMA_HOST` in the sbatch script, resubmit        |
| Ollama download fails with 404             | Using old `.tgz` URL                                  | Use `.tar.zst` URL (already in script)                   |
| sbatch job shows `PENDING` forever         | No free PRO6000 on public partition                   | `ssh newton "sinfo --Format=nodehost,gres,gresused,statelong \| grep PRO6000"` to find free nodes; try `--gres=gpu:L40:1` instead |
| Job killed after 8 h                       | Public-partition time limit (enforced)                | Re-submit; model is cached so restart is ~15 s           |

---

## Appendix: useful one-liners

```bash
# How much VRAM is the model using?
ssh newton "ssh -o StrictHostKeyChecking=no \$(squeue | awk '/galhar.*ollama/ {print \$8; exit}') 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv'"

# List models cached on Newton
ssh newton "~/bin/ollama list 2>/dev/null || ls ~/.ollama/models/manifests/registry.ollama.ai/library/ 2>/dev/null"

# Pull an additional model without restarting the job (run inside the running job)
ssh newton "ssh \$(squeue | awk '/galhar.*ollama/ {print \$8; exit}') '~/bin/ollama pull llama3.3:70b'"
```
