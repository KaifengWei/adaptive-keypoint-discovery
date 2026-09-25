# ARIS-Code in this project

The supplied `aris-code-windows-x64.zip` is the official ARIS-Code **v0.4.27 Windows CLI** release, not a standalone `SKILL.md`. Its SHA-256 is `0ee17ce9bae1a3b3f691b1c18184be5bee96884da0c257a94b803f4784e45dba`, matching the GitHub release asset digest. The archive contains one file, `aris.exe`, installed locally at `.aris/bin/aris.exe`. The binary, sessions, credentials, and any temporary upstream checkout are ignored by Git.

Two relevant Markdown skills from the ARIS bundled-skills source pin `4734364` are installed for project-local Codex use at `.agents/skills/analyze-results/SKILL.md` and `.agents/skills/experiment-plan/SKILL.md`, with the three output protocol references required by `experiment-plan`. They support evidence-led result analysis and a claim-to-experiment map. The CLI itself contains the full bundled skill set; open `/skills` inside ARIS to see it.

The copied Markdown retains its upstream content. The upstream MIT license is at `.agents/skills/ARIS_LICENSE`. Skill SHA-256 values: `analyze-results` `2b97b8ae1a279ac8a7db14e7d7bea67934c88352d55c95e20fcfb681a8640930`; `experiment-plan` `c5b53692ff95b0b55e80702e33afc79d9fe8d4f55a69e2a39714013f97ac595e`.

From PowerShell in this repository:

```powershell
.\tools\aris.ps1 --version
.\tools\aris.ps1 doctor
.\tools\aris.ps1
```

ARIS is installed and responds to `--version`, `--help`, and `doctor`. At installation time, `doctor` reported no executor API authorization, so the interactive research agent and its cross-model review were **not** run. Configure an authorized executor through `aris setup` if desired; keep API keys and runtime settings outside Git. A healthy Codex CLI reviewer bridge alone does not supply an executor. The local Markdown skills can be used in Codex without ARIS executor authorization. A formal ARIS `experiment-audit` verdict requires an independent reviewer and must not be represented as completed here.

Project `AGENTS.md`, `ACCOUNT_HANDOFF.md`, `PROJECT_STATE.md`, and the locked Pipeline V2 preregistration take precedence over generic ARIS workflow suggestions. In particular: Student-B epoch 53 and the GT are frozen; V4 test is locked; the single Pipeline V2 has a preregistered design but needs separate implementation authorization. Do not let a broad ARIS research prompt train, tune, edit frozen method components, read V4 test, or upload ignored human GT. Use `read-only` permission mode for exploratory ARIS sessions until a specific action is authorized.

Source: https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep (release `v0.4.27`; skill source pin `4734364`).
