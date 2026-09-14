---
name: repo-self-review
description: Read-only, repository-wide audit for bugs, regressions, wrong assumptions, integration breaks, weak tests, stale code and doc/code mismatches, ending in a severity-ranked report. Use when asked to self-review, audit or sanity-check the whole repo before building the next feature. Never edits project files.
argument-hint: "[optional focus, e.g. 'perception' or 'since <commit>'; the whole repo is still mapped]"
---

# Repo Self-Review

A systematic audit of the **entire** repository. The output is a report, not a change.

Focus argument (optional): `$ARGUMENTS`. A focus sets where to dig deepest. It never shrinks the audit: every stage below still covers the whole repo.

## Hard rules

- **Read-only.** Do not edit, create, delete, format, stage or commit anything in the repository. That includes `CLAUDE.md`, tests, configs and generated files. Fixes are suggested in the report, never applied.
- **Experiments go in the scratchpad.** Probe scripts run from the session scratchpad directory. Run the test suite without writing caches into the repo: `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider`. If a command would write into the repo, don't run it; say what it would have checked.
- **Evidence or it is not a finding.** Every finding cites code, an output you actually ran, or a documented claim it contradicts. Never fabricate test results, measurements or line numbers.
- **Trace before judging.** Before calling a function broken, read its callers and callees and the data that actually reaches it. A bug in unreachable code is dead code, not a pipeline bug.
- **Unconventional is not wrong.** Check `CLAUDE.md`, docs and comments for a stated rationale first. A documented, deliberate choice that works is not a bug. Only flag it if the rationale is false or no longer holds.
- **Uncertain means a question.** If you can't confirm something, list it as an assumption or question with LOW confidence. Do not promote it to a bug.
- **Missing is not broken.** A pipeline stage that was never implemented (for example, perception on a milestone not yet started) is a gap. Report it as a gap, not as a bug.
- **Finish the audit.** Do not stop after the first few findings. Every stage must be completed and every source file visited.
- **Delegate code style.** Leave style to the `clean-code` skill's rules. Here, report a code-quality issue only when it creates real bug or maintenance risk.

## Stage 1: Repository mapping (do this before judging any file)

1. Read `CLAUDE.md`, `README.md` and any docs. Record the claimed architecture, milestones, measured results, parameters and limitations. These are the claims the audit tests.
2. List the tree with `git ls-files`, plus untracked and ignored files (`git status --ignored`). Classify each file:
   - entry point (console scripts in `pyproject.toml`, `run_scripts/`, `__main__`)
   - core module
   - test
   - scene/XML
   - asset
   - config
   - Docker/infra
   - docs
   - generated or temporary
3. Check submodule state with `git submodule status`. Record which submodule files are actually referenced; the submodules themselves are not audited.
4. Build the module dependency graph: imports, XML `include`s, config → code, script → module. Note cycles and orphans.
5. Write down the intended end-to-end pipeline and the actual call path that implements each stage today. Mark each stage as implemented, partial, ground-truth stand-in or missing.
6. Read `git log` to see recent changes. Recently touched areas are regression candidates, but the audit still covers everything.

Keep this map; it goes in the report's executive summary.

## Stage 2: Functional bug review (every source and XML file)

Check each file for:

- **Logic and control flow.** Inverted or unreachable conditions, off-by-one errors, loops that never terminate or exit early, phases or state machines that skip or repeat steps, per-phase clocks mistaken for global time.
- **Frames and conventions:**
  - world vs body vs site vs camera frames
  - MuJoCo quaternion order (w, x, y, z) vs scipy/mink (x, y, z, w)
  - `xmat` row-major 3×3 layout
  - camera looking down −z with +y up (OpenCV uses +z forward, y down)
  - `xpos` vs `xipos`
  - `site_xpos` vs `body` position
  - sign of constraint forces (`efc_force` is a reaction)
- **Units and signs.** m vs mm, rad vs deg, N vs N·m, gripper ctrl 0–255 vs metres, depth in metres vs normalized buffer values.
- **MuJoCo API:**
  - `mj_forward` missing after writing `qpos`/`mocap`
  - `mj_step` vs `mj_step1/2` ordering
  - reading `efc_*` beyond `nefc`
  - `qpos` vs `qvel` addresses (free joints are 7 vs 6)
  - `jnt_qposadr` / `jnt_dofadr` / `actuator_trnid` mix-ups
  - `eq_active` vs `eq_active0` across resets
  - renderer buffer size vs offscreen size
  - named lookups that silently resolve to id −1 or the wrong object
- **Indexing.** Hardcoded joint, actuator, body, geom or equality indices that break if the XML order changes. Slices that assume the robot's qpos comes first.
- **State mutation.** Shared `MjData` mutated by helpers or IK scratch copies, arrays aliased instead of copied (`data.qpos` without `.copy()`), mutable default arguments, defaults bound at import time, class-level mutable attributes.
- **Reset and physics.**
  - timers, welds, grasp state, controller state or RNG not restored on reset
  - keyframe padding for free joints
  - `time` rewinding
  - warm-start leakage between episodes
- **Collision.** `contype`/`conaffinity` bitmasks. Which pairs can and cannot touch, including excluded parents, `<exclude>`s and geoms that should collide but don't. Collision-vs-visual group mix-ups.
- **Hardcoded values.** Magic numbers tuned to one weed, pose, species or bed that are used as if general. Thresholds with no margin.
- **Failure handling.** Silent `except`, IK that returns stale targets when it fails to converge, grasps "confirmed" without a physical check, success reported without verification, timeouts missing.
- **Determinism.** Unseeded RNG, ordering over sets or dicts that affects behaviour, dependence on wall-clock time.
- **Resources and paths.**
  - paths relative to the working directory instead of the file
  - `include`/`meshdir` resolution (see `CLAUDE.md` path rule)
  - missing assets
  - renderers and viewers not closed
  - GL backend assumptions
  - Docker paths vs native paths
  - dependency pins disagreeing between `pyproject.toml`, `Dockerfile` and docs

Where a claim is cheap to check, verify it with a scratchpad probe: load the model, look up the name, step the sim, print the frame.

## Stage 3: Pipeline and integration review

Trace the real execution path, stage by stage, following the actual function calls:

camera → perception → 3D localization (deprojection) → IK → approach → grasp → extraction → retract → verification

At every boundary, check:

- **Contract match.** Does the output of one stage match what the next stage expects: shape, dtype, frame, units, ID type (body id vs name vs species), ordering?
- **Ground-truth leakage.** Does the control path read simulator truth it would not have in reality? Examples: `data.xpos` of the weed, body names encoding species, `eq_active` as a success signal, collision geometry used for aiming. Separate a *declared* bootstrap stand-in (acceptable if documented) from an *undeclared* leak (a finding).
- **Test code in production.** Test helpers, fixtures, debug constants or `slow`-test shortcuts reachable from launch or control code.
- **Disagreements between modules.** Frames, coordinate conventions, weed/species naming, weed indexing, gripper open/closed values, grasp-site definition. Compare the env, IK model, uprooter, launcher, configs and XML.
- **Plant-free copies.** Does the IK model built from a modified scene still match the full scene's robot: joint order, limits, sites, base pose?
- **Regressions.** Use `git log -p` on core files to check that previously documented working behaviour still holds. Run the relevant tests or probes where feasible.

## Stage 4: Robustness review

- Identify every place behaviour is tuned to a single weed, species, bed, position or orientation. Probe others where cheap: another weed, the far bed, a rotated plant, a base offset.
- **Species.** Does the grasp and extraction logic handle every species it claims to? Compare with the documented species limitations.
- **Target selection.** Is it ambiguous? Could it select a crop plant or a weed that was already uprooted? Is the order stable?
- **Collision edge cases.** Fingers against the bed, arm against the beds or its own links, leaves jamming the jaws, plants against the base.
- **Reset and reinit.** Run two episodes back to back. Check that the second behaves like the first: roots re-armed, grasp welds released, timers cleared, IK warm start reset.
- **Grasp and extraction conditions.** Can success be declared when the plant was not actually held or removed? Are the thresholds marginal (for example, a load that reaches 1.19 but is not sustained)?
- **Randomized tests.** Check that they really randomize the thing that matters, seed reproducibly, report failures with the seed, and are not filtered to the easy cases.

## Stage 5: Tests and regressions

1. Inventory every test and validation script (`tests/`, `run_scripts/`, markers such as `slow`). Run the suite read-only (command in Hard rules). Record the exact pass, fail and skip counts, and compare them with the count documented in `CLAUDE.md`.
2. For each test, ask: **does it assert the behaviour its name claims?** Look for:
   - assertions that are trivially true
   - loose tolerances
   - asserting on the setpoint instead of the outcome
   - tests that pass when the stage under test is skipped
   - mocked-away physics
3. Look for tests that can pass while the real pipeline is broken. Examples: success read from ground truth, tests that pass with the IK replaced by a no-op, a `slow` marker that means the test never runs.
4. Compare current measured behaviour against the documented baselines: timings, forces, iteration counts, uproot times, which weeds come out. Report drift.
5. List important behaviours with no coverage, ranked by pipeline impact.

## Stage 6: Clean code (risk-focused)

Apply the `clean-code` skill's checklist to the core modules, but report only issues with concrete bug or maintenance risk:

- duplicated logic that must change in lockstep
- misleading names that invite misuse (for example, a name suggesting a frame or unit it doesn't have)
- god functions that hide phase bugs
- hidden coupling through shared state

Pure style goes in a single short line at most.

## Stage 7: Documentation vs implementation

Check every factual claim in `CLAUDE.md`, the README, docstrings, comments and config comments against the code and, where cheap, a run. The claims to check:

- parameter values
- counts (tests, geoms, nq)
- file lists and the repository-structure tree
- milestone status
- "works" rows
- measured numbers
- stated limitations that were since fixed (or fixes that never landed)

Flag claims that are false, stale, or stronger than the implementation supports.

## Stage 8: Dead and stale code

Find:

- unused modules, functions, constants, configs and fixtures
- scripts that reference removed APIs
- temporary or probe files committed to the repo
- duplicate implementations of the same step
- old approaches still imported or documented
- console-script entry points that no longer exist
- assets nothing references

Confirm with a repo-wide search before calling anything unused, including XML and YAML, not just Python.

## Severity and confidence

- **CRITICAL.** Breaks the end-to-end pipeline, silently produces a wrong result that looks right, or invalidates reported results.
- **HIGH.** Wrong behaviour in a realistic case (another weed, a reset, a second episode), or a test that masks a real failure.
- **MEDIUM.** Fragility that is likely to break with the next change, a narrow edge case, or a misleading doc that would cause a wrong decision.
- **LOW.** Minor inconsistency, dead code, or maintenance friction.

Confidence:

- **HIGH.** Verified by running code or an unambiguous trace.
- **MEDIUM.** Strong reading of the code, not executed.
- **LOW.** Suspicion or open question.

## Report format

Deliver the report in the conversation. Write it to a file only if the user asks, and never inside the repo unless told to.

Use this format for every finding:

```
### [ID] Short title
- Path: file (function / class / XML element / doc section)
- Problem: exact problem
- Why it matters: consequence
- Evidence: code quote, command output, or contradicting claim
- Severity: CRITICAL | HIGH | MEDIUM | LOW    Confidence: HIGH | MEDIUM | LOW
- Suggested fix: concrete, minimal
- Affects end-to-end pipeline: yes / no / only when <condition>
```

Sections, in order (write "None found" for an empty section rather than omitting it):

- **A. Executive summary.** The repository map, the pipeline stage status table, the test-run result and the headline risks.
- **B. Critical bugs**
- **C. High-risk bugs**
- **D. Medium-risk issues**
- **E. Low-risk issues**
- **F. Integration inconsistencies**
- **G. Missing tests**
- **H. Clean-code issues**
- **I. Dead/stale code**
- **J. Documentation mismatches**
- **K. Recommended fixes in priority order**
- **Assumptions and open questions.** Unverified items, each with LOW confidence.

A finding belongs to one section. Cross-reference it by ID elsewhere instead of repeating it.

End with:

1. **Top 10 issues to fix first**, by ID, with one line each.
2. **Is the repo internally consistent?** Yes, no or partly, with the main reasons.
3. **Is the end-to-end pipeline safe to continue building on?** Yes, no or conditionally, with the conditions.
4. **Fix before the next major feature.** The minimal blocking set.
