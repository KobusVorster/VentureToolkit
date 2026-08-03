# venture-toolkit

VC 2.0 venture analysis tooling, set up as Claude Code commands and skills.
Open this repo in Claude Code and the slash commands are available immediately —
nothing to install.

| Command | What it does |
|---|---|
| `/score-opportunity` | EPAC-validated 10-dimension venture scorecard |
| `/generate-safer` | Safer term sheet with VREC provisions |
| `/simulate-safer` | Safer scenario simulation → HTML report |
| `/simulate-fund` | Monte Carlo fund simulation |

The `studio-os` skill loads on its own whenever scorecard work comes up, so you
can also just describe a deal in plain language instead of using a command.

## Layout

```
.claude/commands/           the four slash commands
.claude/skills/studio-os/   scorecard framework (auto-loads by description)
tools/safer.py              Safer scenario simulator — stdlib only
tools/safer_monte_carlo.py  fund Monte Carlo — needs requirements.txt
docs/                       upstream README and plugin manifest
```

## Running the simulators directly

`safer.py` needs nothing beyond Python 3:

```bash
python tools/safer.py --scenario steady-growth-acquisition --output report.html
```

`safer_monte_carlo.py` needs numpy, pandas, and matplotlib:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python tools/safer_monte_carlo.py --scenario <preset>
```

## Provenance

Unpacked from the `vc2-essentials.plugin` bundle (v2.2.0) in
[next-wave-partners/vc2-essentials](https://github.com/next-wave-partners/vc2-essentials),
by Next Wave Partners, based on *Venture Capital 2.0* by John Cowan.

Two changes were made so the commands work as repo config rather than as an
installed plugin:

- Upstream paths used `${CLAUDE_PLUGIN_ROOT}`, which is only defined for
  installed plugins. Rewritten to repo-relative paths.
- `/generate-safer` opened by reading a `capital-structurer` skill that the free
  release does not ship. That read is now optional, with a fallback to the
  command's own inline parameter defaults.

To update: re-download the upstream `.plugin` (it is a zip), unpack, copy
`commands/` → `.claude/commands/`, `skills/` → `.claude/skills/`, `tools/` →
`tools/`, then redo those two changes.

## Licensing

Prompts and methodology are **CC BY-NC 4.0** (non-commercial). The simulator
scripts are covered by `tools/SAFER-LICENSE`. Review both before any commercial
use.
