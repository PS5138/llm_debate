"""Streamlit demo for the medical debate pipeline.

Bring your own API key, pick how many cases to run, watch debates fill in
row by row, click a case to see its transcript and the judge's pick under
each arm. Fresh runs write to timestamped folders under exp/; resume reuses
the latest saved folder for the selected family. Temporary secrets and
process logs are isolated in a tempdir.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from app.runner import (
    CONDITIONS,
    FAMILY_MODELS,
    DebateRun,
    Phase,
    arm_judgement_is_complete,
    canonical_baselines_path,
    estimate_cost_usd,
    find_latest_run,
    pilot_case_ids,
    transcript_rounds,
)
from core.scoring.accuracy import find_answer

# Committed DDXPlus pilot size. Both the fresh-run and resume sliders cap here.
PILOT_MAX_CASES = 100
OPENAI_KEY_WIDGET = "openai_api_key"
ANTHROPIC_KEY_WIDGET = "anthropic_api_key"

CONDITION_LABEL = {
    "e1_info_asymmetry": "E1 · info asymmetry",
    "e2_double_asymmetry": "E2 · double asymmetry (headline)",
    "e3_capability_asymmetry": "E3 · capability asymmetry",
    "e4_full_symmetry": "E4 · full symmetry",
}
PHASE_LABEL = {
    Phase.PENDING: "Ready",
    Phase.BASELINES: "Running baselines",
    Phase.DEBATE: "Generating debates",
    Phase.JUDGING: "Judging",
    Phase.ANALYSIS: "Aggregating results",
    Phase.DONE: "Done",
    Phase.ERROR: "Error",
    Phase.STOPPED: "Stopped",
}
PHASE_PROGRESS = {
    Phase.PENDING: 0.0,
    Phase.BASELINES: 0.15,
    Phase.DEBATE: 0.45,
    Phase.JUDGING: 0.80,
    Phase.ANALYSIS: 0.95,
    Phase.DONE: 1.0,
    Phase.ERROR: 1.0,
    Phase.STOPPED: 1.0,
}


def _display_path(path: Path) -> str:
    """Show repo paths compactly and external paths without crashing."""
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- page

st.set_page_config(
    page_title="Medical Debate · Live Demo",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
      .small-muted { color: #666; font-size: 0.85rem; }
      .pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.8rem; }
      .pill-pending { background:#eee; color:#555; }
      .pill-active  { background:#fff7d6; color:#8a6d00; }
      .pill-done    { background:#d8f5d6; color:#1a6d1a; }
      .pill-error   { background:#f7d6d6; color:#8a1a1a; }
      pre.transcript { background:#fafafa; border:1px solid #eee; padding:12px;
                       border-radius:6px; white-space:pre-wrap; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- helpers


def _summarise_pick(raw: str) -> str:
    """Return 'A', 'B', or '' using the project's own answer parser."""
    if not isinstance(raw, str) or not raw.strip():
        return ""
    lowered = raw.lower()
    if "error message:" in lowered or "error occurred" in lowered:
        return "?"
    try:
        a = find_answer(raw, "A")
        b = find_answer(raw, "B")
    except (IndexError, TypeError, ValueError):
        return "?"
    if a and not b:
        return "A"
    if b and not a:
        return "B"
    return "?"


def _swap_averaged_verdict(orig: str, swap: str) -> str:
    """Truth is in A when swap=False, in B when swap=True. Both passes must
    point to the correct side to count as a clean correct judgement."""
    orig_pick = _summarise_pick(orig)
    swap_pick = _summarise_pick(swap)
    if not orig and not swap:
        return "—"
    if not orig or not swap:
        return "partial"
    if orig_pick not in {"A", "B"} or swap_pick not in {"A", "B"}:
        return "unparseable"
    if orig_pick == "A" and swap_pick == "B":
        return "✓ correct"
    if orig_pick == "B" and swap_pick == "A":
        return "✗ wrong"
    return "split"  # both parse, but agree on position rather than truth


def _arm_display_values(answers: object) -> tuple[str, str, str]:
    """Return live picks/verdict only for explicitly completed orderings."""
    if not isinstance(answers, dict):
        return "—", "—", "—"
    orig_complete = answers.get("orig_complete") is True
    swap_complete = answers.get("swap_complete") is True
    orig = str(answers.get("orig", ""))
    swap = str(answers.get("swap", ""))
    orig_pick = (_summarise_pick(orig) or "—") if orig_complete else "—"
    swap_pick = (_summarise_pick(swap) or "—") if swap_complete else "—"
    if not orig_complete and not swap_complete:
        verdict = "—"
    elif not orig_complete or not swap_complete:
        verdict = "partial"
    else:
        verdict = _swap_averaged_verdict(orig, swap)
    return orig_pick, swap_pick, verdict


# --------------------------------------------------------------------------- state


def _init_state() -> None:
    st.session_state.setdefault("run", None)
    st.session_state.setdefault("selected_case", None)
    # Two-click confirmation gate so an accidental ▶ Run click can't burn $$.
    # `confirm_armed` flips on after the first click; second click within the
    # same configuration actually starts the run.
    st.session_state.setdefault("confirm_armed", False)
    st.session_state.setdefault("confirm_signature", None)
    st.session_state.setdefault("resume_confirm_armed", False)
    st.session_state.setdefault("resume_confirm_signature", None)


_init_state()


def _cleanup_run() -> None:
    run: DebateRun | None = st.session_state.get("run")
    if run is not None:
        run.cleanup()
    st.session_state["run"] = None
    st.session_state["selected_case"] = None
    st.session_state["confirm_armed"] = False
    st.session_state["confirm_signature"] = None
    st.session_state["resume_confirm_armed"] = False
    st.session_state["resume_confirm_signature"] = None


def _clear_run() -> None:
    """Clear transient run state and both password-backed widget values."""
    _cleanup_run()
    st.session_state[OPENAI_KEY_WIDGET] = ""
    st.session_state[ANTHROPIC_KEY_WIDGET] = ""


# --------------------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Run settings")

    family = st.selectbox(
        "Model family",
        list(FAMILY_MODELS.keys()),
        index=0,
        help="Frontier debaters / preference judge come from this family. "
        "Fresh runs use gpt-4o-mini for concession judging.",
    )
    models = FAMILY_MODELS[family]
    st.caption(f"Frontier: `{models['frontier']}` · Weaker judge: `{models['weaker']}`")

    n_cases = st.slider(
        "Number of cases", min_value=1, max_value=PILOT_MAX_CASES, value=3
    )

    est = estimate_cost_usd(family, n_cases)
    if est >= 20:
        st.error(
            f"Debate/judging estimate on your own key: **~${est:0.2f}** "
            f"before optional baseline recomputation. "
            f"The OpenAI allowance follows gpt-5.5 pricing. The Anthropic "
            f"allowance is provisional: an Opus 4.7 smoke run was ~$4/case; "
            f"Opus 4.6 is budgeted lower because it has no adaptive-thinking "
            f"overhead. "
            f"Start with 1–3 cases to confirm the pipeline works on "
            f"your account before spending more."
        )
    elif est >= 5:
        st.warning(
            f"Debate/judging estimate on your own key: **~${est:0.2f}** "
            f"before optional baseline recomputation. "
            f"Real spend varies with case length and how chatty the "
            f"models feel."
        )
    else:
        st.info(
            f"Debate/judging estimate on your own key: **~${est:0.2f}** "
            f"before optional baseline recomputation. "
            f"Opus 4.6 lists at $5 / $25 per million; gpt-5.5 lists at "
            f"$5 / $30. Real spend varies with case length."
        )

    st.divider()
    st.subheader("Baselines")
    baseline_recompute_allowance = (
        estimate_cost_usd(family, n_cases, include_baseline_recompute=True) - est
    )
    canonical = canonical_baselines_path(
        family,
        min_rows=n_cases,
        expected_ids=pilot_case_ids(n_cases),
    )
    if canonical is not None:
        st.caption(
            f"Canonical baselines for **{family}** are committed at "
            f"`{_display_path(canonical)}` (100 cases, fixed "
            f"reference for PGR). By default the debate run skips the "
            f"baseline-API stage and reuses these numbers."
        )
        recompute_baselines = st.toggle(
            "Recompute baselines (don't use canonical)",
            value=False,
            help="Turn on to re-run blind + oracle baselines from scratch "
            "alongside the debate run, for BOTH the weaker and frontier "
            "judges (so every debate arm has a same-judge baseline for "
            "PGR). Useful if you've changed a judge model or prompt. "
            f"Adds a planning allowance of ~${baseline_recompute_allowance:0.2f} "
            "for this case count. For Anthropic this is deliberately "
            "conservative because the frontier baseline dominates spend.",
        )
    else:
        st.caption(
            f"No canonical baselines committed for **{family}** yet. "
            f"This run will compute them fresh."
        )
        recompute_baselines = True

    run_est = estimate_cost_usd(
        family,
        n_cases,
        include_baseline_recompute=bool(recompute_baselines),
    )

    st.divider()
    st.subheader("API keys (session only)")
    openai_key = st.text_input(
        "OpenAI key",
        type="password",
        key=OPENAI_KEY_WIDGET,
        help="Required for both supported model families because concession "
        "judging uses gpt-4o-mini.",
    )
    anthropic_key = st.text_input(
        "Anthropic key",
        type="password",
        key=ANTHROPIC_KEY_WIDGET,
        help="Required for the Anthropic family. Leave blank for OpenAI runs.",
    )

    st.divider()
    run: DebateRun | None = st.session_state.get("run")
    is_active = run is not None and run.snapshot().phase in {
        Phase.BASELINES,
        Phase.DEBATE,
        Phase.JUDGING,
        Phase.ANALYSIS,
    }

    # Two-click confirmation. The first click arms the button; the second
    # click (with the same family + n_cases + baseline choice) actually starts
    # the run. If the user changes settings in between, the armed state is
    # dropped explicitly.
    signature = (family, n_cases, bool(recompute_baselines))
    if st.session_state.get("confirm_signature") != signature:
        st.session_state["confirm_armed"] = False
        st.session_state["confirm_signature"] = None
    confirm_armed = (
        st.session_state.get("confirm_armed", False)
        and st.session_state.get("confirm_signature") == signature
    )

    run_label = (
        f"✅ Confirm: spend ~${run_est:0.2f}"
        if confirm_armed
        else f"▶ Run ({n_cases} case{'s' if n_cases != 1 else ''}, ~${run_est:0.2f})"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        start_clicked = st.button(
            run_label,
            type="primary",
            disabled=is_active,
            width="stretch",
        )
    with col_b:
        stop_clicked = st.button(
            "⏹ Stop",
            disabled=not is_active,
            width="stretch",
        )

    if confirm_armed:
        st.caption(
            "Click **Confirm** to start. Changing the family, case count, or "
            "baseline choice cancels confirmation."
        )

    # ---- Resume or extend previous run --------------------------------
    latest = find_latest_run(family=family) if not is_active else None
    resume_clicked = False
    resume_target_n = None
    resume_confirm_armed = False
    resume_signature = None
    if latest is not None:
        st.divider()
        st.subheader(f"Resume / extend previous {family} run")
        prior_n = int(latest["n_cases"])
        saved_at = (
            latest.get("last_updated_at_utc")
            or latest.get("recorded_at_utc")
            or "unknown"
        )
        st.caption(
            f"Most recent saved **{family}** run: `{latest['exp_dir'].name}` "
            f"(current n={prior_n}). Saved {saved_at}."
        )
        saved_frontier = latest.get("frontier_model")
        saved_weaker = latest.get("weaker_model")
        saved_concession = latest.get("concession_model")
        if saved_frontier and saved_weaker:
            model_note = (
                f"Resume keeps the recorded models: frontier `{saved_frontier}`, "
                f"weaker `{saved_weaker}`"
            )
            if saved_concession:
                model_note += f", concession `{saved_concession}`"
            st.caption(model_note + ".")
        # Surface the run's original baseline source so users know that
        # the "Recompute baselines" toggle above does NOT apply on
        # resume — the original choice is preserved to keep PGR
        # denominators consistent across the run's lifetime.
        if latest.get("baselines_dir"):
            source_label = (
                "canonical (committed reference)"
                if latest.get("using_canonical_baselines")
                else f"saved non-canonical source at `{_display_path(Path(latest['baselines_dir']))}`"
            )
            st.caption(
                f"This run's baseline source is fixed at **{source_label}**. "
                f"The Recompute toggle above only affects fresh Run clicks."
            )

        # A run already at the pilot ceiling has nothing to extend to, and
        # Streamlit raises if a slider's min equals its max — which crashed
        # the whole page for anyone whose most recent run was a full 100-case
        # one. Offer retry-only in that case instead of a degenerate slider.
        if prior_n >= PILOT_MAX_CASES:
            resume_target_n = prior_n
            st.caption(
                f"This run is already at the {PILOT_MAX_CASES}-case pilot "
                f"maximum, so there is nothing further to extend to. Resuming "
                f"will retry any cases that never completed and re-run the "
                f"analysis; finished cases are not paid for again."
            )
        else:
            resume_target_n = st.slider(
                "Total cases after this run",
                min_value=prior_n,
                max_value=PILOT_MAX_CASES,
                value=prior_n,
                key=f"resume_target_{family}_{latest['exp_dir'].name}",
                help="Set to the same n to just retry incomplete work. Set higher "
                "to keep the already-finished cases and add new ones up to the "
                "new total. The pipeline writes back to the same files and skips "
                "already-complete work within each stage; unfinished debate or "
                "judging work among earlier cases is still retried.",
            )

        added = resume_target_n - prior_n
        resume_baselines_writable = bool(
            latest.get(
                "baselines_writable",
                not latest.get("using_canonical_baselines", False),
            )
        )
        resume_planning_cost = estimate_cost_usd(
            latest["family"],
            int(resume_target_n),
            include_baseline_recompute=resume_baselines_writable,
        )
        resume_signature = (
            family,
            str(Path(latest["exp_dir"]).resolve()),
            int(resume_target_n),
            saved_at,
            saved_frontier,
            saved_weaker,
            saved_concession,
            str(latest.get("baselines_dir") or ""),
            resume_baselines_writable,
        )
        if st.session_state.get("resume_confirm_signature") != resume_signature:
            st.session_state["resume_confirm_armed"] = False
            st.session_state["resume_confirm_signature"] = None
        resume_confirm_armed = (
            st.session_state.get("resume_confirm_armed", False)
            and st.session_state.get("resume_confirm_signature") == resume_signature
        )

        if resume_confirm_armed:
            button_label = f"✅ Confirm: budget ~${resume_planning_cost:0.2f}"
        elif added == 0:
            button_label = (
                f"↻ Resume incomplete work at n={prior_n} "
                f"(budget ~${resume_planning_cost:0.2f})"
            )
        else:
            button_label = (
                f"➕ Extend from n={prior_n} to n={resume_target_n} "
                f"(budget ~${resume_planning_cost:0.2f})"
            )

        baseline_budget_note = (
            "This includes the baseline-recompute allowance because the saved "
            "in-run baseline source may still have incomplete work. "
            if resume_baselines_writable
            else "The saved baselines are read-only, so no baseline API spend is included. "
        )
        resume_warning = (
            f"Full-rerun planning estimate: ~${resume_planning_cost:0.2f} if no "
            f"saved work can be reused across the {resume_target_n} target cases. "
            + baseline_budget_note
            + f"Already-complete work is skipped, so resume spend should usually be "
            f"lower; token usage and retries can still move the actual total."
        )
        st.warning(resume_warning)

        st.caption(
            "Resume re-launches the pipeline against the existing folder. "
            "Already-complete rows (`complete=True`, `complete_judge=True`) "
            "are skipped automatically, so only new or incomplete cases hit "
            "the API. Extension grows the working CSVs in place rather than "
            "starting a fresh folder."
        )
        resume_button_clicked = st.button(
            button_label,
            disabled=is_active,
            width="stretch",
            key=f"resume_{family}_{latest['exp_dir'].name}",
        )
        if resume_confirm_armed:
            st.caption(
                "Click **Confirm** to resume this exact run and target. "
                "Changing the family, saved run, or target cancels confirmation."
            )
        if resume_button_clicked:
            if not resume_confirm_armed:
                st.session_state["resume_confirm_armed"] = True
                st.session_state["resume_confirm_signature"] = resume_signature
                st.session_state["confirm_armed"] = False
                st.session_state["confirm_signature"] = None
                st.rerun()
            else:
                st.session_state["resume_confirm_armed"] = False
                st.session_state["resume_confirm_signature"] = None
                resume_clicked = True
    else:
        st.session_state["resume_confirm_armed"] = False
        st.session_state["resume_confirm_signature"] = None

    st.caption(
        "Stop cancels the current run. If you click Run again afterwards, "
        "the app starts from the beginning in a fresh results folder rather "
        "than resuming partial work."
    )

    st.button(
        "Clear run (keep exp outputs)",
        width="stretch",
        on_click=_clear_run,
    )

    st.divider()
    st.caption(
        "Keys live in this session only. They are written to a "
        "tempfile inside the run tempdir. Clear run stops the run, clears "
        "both key fields, and deletes the temp secrets/logs, while leaving "
        "the run outputs under exp/. Nothing is sent anywhere except the "
        "chosen model providers."
    )


# --------------------------------------------------------------------------- actions

if stop_clicked and run is not None:
    run.stop()

if start_clicked:
    if not confirm_armed:
        # First click: arm the confirmation. Show a confirm button on the
        # next rerender.
        st.session_state["confirm_armed"] = True
        st.session_state["confirm_signature"] = signature
        st.session_state["resume_confirm_armed"] = False
        st.session_state["resume_confirm_signature"] = None
        st.rerun()
    else:
        # Second click: actually launch.
        st.session_state["confirm_armed"] = False
        st.session_state["confirm_signature"] = None
        _cleanup_run()
        try:
            new_run = DebateRun(
                family=family,
                n_cases=n_cases,
                openai_key=openai_key.strip(),
                anthropic_key=anthropic_key.strip(),
                use_canonical_baselines=not recompute_baselines,
            )
            new_run.start()
            st.session_state["run"] = new_run
            st.session_state["selected_case"] = None
            st.rerun()
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(f"Could not start the run: {exc}")

if resume_clicked and latest is not None:
    _cleanup_run()
    try:
        new_run = DebateRun(
            family=latest["family"],
            n_cases=(
                int(resume_target_n)
                if resume_target_n is not None
                else latest["n_cases"]
            ),
            openai_key=openai_key.strip(),
            anthropic_key=anthropic_key.strip(),
            frontier_model=latest["frontier_model"],
            weaker_model=latest["weaker_model"],
            concession_model=latest["concession_model"],
            resume_from=latest["exp_dir"],
            use_canonical_baselines=not recompute_baselines,
        )
        # Surface the extension report so the user sees what got grown.
        if new_run.extension_report:
            added_total = sum(v for v in new_run.extension_report.values() if v)
            if added_total > 0:
                st.success(
                    f"Extended {sum(1 for v in new_run.extension_report.values() if v)} "
                    f"CSV file(s); {added_total} row-extensions written."
                )
        new_run.start()
        st.session_state["run"] = new_run
        st.session_state["selected_case"] = None
        st.rerun()
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(f"Could not resume the run: {exc}")


# --------------------------------------------------------------------------- header

st.title("Medical Debate · Live Demo")
st.markdown(
    "Two AI debaters argue opposite diagnoses on a synthetic patient case. "
    "A judge picks A or B, sometimes without seeing the evidence. "
    "This page runs the pipeline live on your key and shows debates as they finish. "
    "[Project README](https://github.com/PS5138/llm_debate#readme) · methodology, arms, and caveats."
)
st.warning(
    "Research demo using synthetic patients. Not medical advice, a diagnostic "
    "system, or clinical decision support. Do not use it for real-patient decisions."
)


# --------------------------------------------------------------------------- live view

run = st.session_state.get("run")
if run is None:
    st.info(
        "Pick a family and a case count in the sidebar, paste your key(s), and hit **Run**. "
        "The default is 3 cases. OpenAI uses `gpt-5.5` for debaters and E1/E4, "
        "with `gpt-5.4-mini` as the E2/E3 judge."
    )
    st.stop()

snap = run.snapshot()

phase_label = PHASE_LABEL.get(snap.phase, snap.phase.value)
progress = PHASE_PROGRESS.get(snap.phase, 0.0)

c1, c2, c3 = st.columns([2, 4, 2])
with c1:
    st.metric("Phase", phase_label)
with c2:
    st.progress(progress, text=snap.message)
with c3:
    elapsed = (snap.finished_at or time.time()) - (snap.started_at or time.time())
    st.metric("Elapsed", f"{int(elapsed)}s")

if snap.phase == Phase.ERROR:
    st.error(snap.message)

if snap.exp_dir:
    st.caption(f"Run output folder: `{snap.exp_dir}`")


# ---- table ----------------------------------------------------------------

st.subheader("Cases")

case_rows = []
for c in snap.cases:
    if c.debate_complete:
        status = "debate ✓"
    elif snap.phase == Phase.BASELINES:
        status = "baselines…"
    elif snap.phase == Phase.DEBATE:
        status = "debating…"
    elif snap.phase in {Phase.JUDGING, Phase.ANALYSIS, Phase.DONE}:
        status = "debate ✓"
    else:
        status = "pending"

    arms_done = sum(
        1 for cond in CONDITIONS if arm_judgement_is_complete(c.arm_answers.get(cond))
    )
    case_rows.append(
        {
            "case_id": c.case_id,
            "status": status,
            "arms judged": f"{arms_done}/4",
            "correct (hidden)": c.correct,
            "distractor": c.distractor,
            "question": (
                (c.question[:120] + "…") if len(c.question) > 120 else c.question
            ),
        }
    )

if not case_rows:
    st.warning(
        "No cases loaded yet. If this persists after the baselines phase, check the log below."
    )
else:
    df = pd.DataFrame(case_rows)
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "case_id": st.column_config.TextColumn("case", width="small"),
            "status": st.column_config.TextColumn("status", width="small"),
            "arms judged": st.column_config.TextColumn("arms", width="small"),
            "question": st.column_config.TextColumn("question", width="large"),
        },
    )


# ---- detail pane ----------------------------------------------------------

st.subheader("Inspect a case")
case_ids = [c.case_id for c in snap.cases]
selected = (
    st.selectbox(
        "Pick a case to read its debate",
        case_ids,
        index=(
            case_ids.index(st.session_state["selected_case"])
            if st.session_state.get("selected_case") in case_ids
            else 0
        ),
        key="selected_case",
    )
    if case_ids
    else None
)

if selected:
    case = next((c for c in snap.cases if c.case_id == selected), None)
    if case is not None:
        st.markdown(f"**Case `{case.case_id}`**")
        st.markdown(f"_Question:_ {case.question}")
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f"**Correct (hidden during debate):** {case.correct}")
        with col_r:
            st.markdown(f"**Distractor:** {case.distractor}")

        if not case.debate_complete:
            st.info("Debate not finished yet. Come back once this row flips to ✓.")
        else:
            rounds = transcript_rounds(case.transcript_json or "")
            if not rounds:
                st.warning("Transcript not parseable yet. It may be mid-write.")
            else:
                last_round = max(r["round"] for r in rounds)
                for ri in range(1, last_round + 1):
                    st.markdown(f"##### Round {ri}")
                    for turn in [r for r in rounds if r["round"] == ri]:
                        avatar = "🟢" if turn["side"] == "correct" else "🔴"
                        with st.chat_message("assistant", avatar=avatar):
                            st.markdown(f"**{turn['speaker']}**")
                            st.markdown(turn["text"])

        # Per-arm answers
        st.markdown("---")
        st.markdown(
            "**Judge picks per arm** (✓ = picked the correct side after swap-averaging)"
        )
        rows = []
        for cond in CONDITIONS:
            answers = case.arm_answers.get(cond, {})
            orig_pick, swap_pick, verdict = _arm_display_values(answers)
            rows.append(
                {
                    "arm": CONDITION_LABEL[cond],
                    "orig pick": orig_pick,
                    "swap pick": swap_pick,
                    "result": verdict,
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ---- aggregate results ----------------------------------------------------

if snap.phase == Phase.DONE and snap.results_dir:
    st.subheader("Aggregate results")
    results_dir = Path(snap.results_dir)
    acc_path = results_dir / "accuracy_by_condition.csv"
    headline_path = results_dir / "headline_results.csv"
    plots_dir = results_dir / "plots"

    cols = st.columns(2)
    with cols[0]:
        if acc_path.exists():
            st.markdown("**Accuracy by condition**")
            st.dataframe(pd.read_csv(acc_path), width="stretch", hide_index=True)
    with cols[1]:
        if headline_path.exists():
            st.markdown("**PGR by condition**")
            # headline_results.csv is the single PGR table (it also carries the
            # blind/debate/oracle accuracies PGR is derived from). Show the
            # same-judge floor and ceiling next to PGR rather than all 17
            # columns, which would not fit this half-width pane.
            headline = pd.read_csv(headline_path)
            preferred = [
                "label",
                "judge_model",
                "blind_acc",
                "debate_acc",
                "oracle_acc",
                "pgr",
                "pgr_ci_lo",
                "pgr_ci_hi",
            ]
            shown = [c for c in preferred if c in headline.columns] or list(
                headline.columns
            )
            st.dataframe(headline[shown], width="stretch", hide_index=True)

    if plots_dir.exists():
        for png in sorted(plots_dir.glob("*.png")):
            st.image(str(png), width="stretch")


# ---- raw log --------------------------------------------------------------

with st.expander("Pipeline log (last 40 lines)"):
    if snap.log_tail:
        st.code("\n".join(snap.log_tail), language="bash")
    else:
        st.caption("No log lines yet.")


# ---- auto-refresh ---------------------------------------------------------

if snap.phase in {
    Phase.BASELINES,
    Phase.DEBATE,
    Phase.JUDGING,
    Phase.ANALYSIS,
}:
    time.sleep(2.0)
    st.rerun()
