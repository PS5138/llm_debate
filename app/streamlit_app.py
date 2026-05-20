"""Streamlit demo for the medical debate pipeline.

Bring your own API key, pick how many cases to run, watch debates fill in
row by row, click a case to see its transcript and the judge's pick under
each arm. Each run writes outputs to a fresh timestamped folder under exp/.
Temporary secrets and process logs are isolated in a tempdir.
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
    DebateRun,
    FAMILY_MODELS,
    Phase,
    estimate_cost_usd,
    find_latest_run,
    transcript_rounds,
)
from core.scoring.accuracy import find_answer

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
    a = find_answer(raw, "A")
    b = find_answer(raw, "B")
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
    if orig_pick == "A" and swap_pick == "B":
        return "✓ correct"
    if orig_pick == "B" and swap_pick == "A":
        return "✗ wrong"
    return "split"  # the two passes disagreed → A/B position bias


# --------------------------------------------------------------------------- state

def _init_state() -> None:
    st.session_state.setdefault("run", None)
    st.session_state.setdefault("selected_case", None)
    # Two-click confirmation gate so an accidental ▶ Run click can't burn $$.
    # `confirm_armed` flips on after the first click; second click within the
    # same configuration actually starts the run.
    st.session_state.setdefault("confirm_armed", False)
    st.session_state.setdefault("confirm_signature", None)  # (family, n_cases)


_init_state()


def _cleanup_run() -> None:
    run: DebateRun | None = st.session_state.get("run")
    if run is not None:
        run.cleanup()
    st.session_state["run"] = None
    st.session_state["selected_case"] = None


# --------------------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Run settings")

    family = st.selectbox(
        "Model family",
        list(FAMILY_MODELS.keys()),
        index=0,
        help="Frontier debaters / preference judge come from this family. "
             "The concession judge uses gpt-4o-mini regardless.",
    )
    models = FAMILY_MODELS[family]
    st.caption(f"Frontier: `{models['frontier']}` · Weaker judge: `{models['weaker']}`")

    n_cases = st.slider("Number of cases", min_value=1, max_value=100, value=3)

    est = estimate_cost_usd(family, n_cases)
    if est >= 20:
        st.error(
            f"Estimated cost on your own key: **~${est:0.2f}**. "
            f"Calibrated against a May 2026 Anthropic smoke run "
            f"(~$2.50/case ballpark on both Opus 4.6 and gpt-5.5). "
            f"Start with 1–3 cases to confirm the pipeline works on "
            f"your account before spending more."
        )
    elif est >= 5:
        st.warning(
            f"Estimated cost on your own key: **~${est:0.2f}**. "
            f"Real spend varies with case length and how chatty the "
            f"models feel."
        )
    else:
        st.info(
            f"Estimated cost on your own key: **~${est:0.2f}**. "
            f"Opus 4.6 lists at $5 / $25 per million; gpt-5.5 lists at "
            f"$5 / $30. Real spend varies with case length."
        )

    st.divider()
    st.subheader("API keys (session only)")
    openai_key = st.text_input(
        "OpenAI key",
        type="password",
        help="Always required — the concession judge uses gpt-4o-mini.",
    )
    anthropic_key = st.text_input(
        "Anthropic key",
        type="password",
        help="Required for the Anthropic family. Leave blank for OpenAI-only runs.",
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
    # click (with the same family + n_cases) actually starts the run. If
    # the user changes settings in between, the armed state is dropped.
    signature = (family, n_cases)
    confirm_armed = (
        st.session_state.get("confirm_armed", False)
        and st.session_state.get("confirm_signature") == signature
    )

    run_label = (
        f"✅ Confirm: spend ~${est:0.2f}"
        if confirm_armed
        else f"▶ Run ({n_cases} case{'s' if n_cases != 1 else ''}, ~${est:0.2f})"
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
            "Click **Confirm** to start. Change any setting above to cancel."
        )

    # ---- Resume or extend previous run --------------------------------
    latest = find_latest_run() if not is_active else None
    resume_clicked = False
    resume_target_n = None
    if latest is not None:
        st.divider()
        st.subheader("Resume / extend previous run")
        prior_n = int(latest["n_cases"])
        st.caption(
            f"Most recent run: `{latest['exp_dir'].name}` "
            f"({latest['family']}, current n={prior_n}). "
            f"Recorded {latest.get('recorded_at_utc', 'unknown')}."
        )

        resume_target_n = st.slider(
            "Total cases after this run",
            min_value=prior_n,
            max_value=100,
            value=prior_n,
            help="Set to the same n to just retry incomplete work. Set higher "
                 "to keep the already-finished cases and add new ones up to the "
                 "new total. The pipeline writes back to the same `data0.csv` "
                 "files, so cases 1..prior_n are not re-spent on.",
        )

        added = resume_target_n - prior_n
        cost_str = ""
        if added > 0:
            extra_cost = estimate_cost_usd(latest["family"], added)
            cost_str = f" (~${extra_cost:0.2f} for the {added} new case{'s' if added != 1 else ''})"

        if added == 0:
            button_label = f"↻ Resume incomplete work at n={prior_n}"
        else:
            button_label = f"➕ Extend from n={prior_n} to n={resume_target_n}{cost_str}"

        st.caption(
            "Resume re-launches the pipeline against the existing folder. "
            "Already-complete rows (`complete=True`, `complete_judge=True`) "
            "are skipped automatically, so only new or incomplete cases hit "
            "the API. Extension grows the working CSVs in place rather than "
            "starting a fresh folder."
        )
        resume_clicked = st.button(
            button_label,
            disabled=is_active,
            width="stretch",
        )

    st.caption(
        "Stop cancels the current run. If you click Run again afterwards, "
        "the app starts from the beginning in a fresh results folder rather "
        "than resuming partial work."
    )

    reset_clicked = st.button("Clear run (keep exp outputs)", width="stretch")

    st.divider()
    st.caption(
        "Keys live in this session only. They are written to a "
        "tempfile inside the run tempdir. Clear run stops the run and "
        "deletes the temp secrets/logs, while leaving the run outputs "
        "under exp/. Nothing is sent anywhere except the chosen model "
        "providers."
    )


# --------------------------------------------------------------------------- actions

if reset_clicked:
    _cleanup_run()
    st.rerun()

if stop_clicked and run is not None:
    run.stop()

if start_clicked:
    if not confirm_armed:
        # First click: arm the confirmation. Show a confirm button on the
        # next rerender.
        st.session_state["confirm_armed"] = True
        st.session_state["confirm_signature"] = signature
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
            )
            new_run.start()
            st.session_state["run"] = new_run
            st.session_state["selected_case"] = None
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

if resume_clicked and latest is not None:
    _cleanup_run()
    try:
        new_run = DebateRun(
            family=latest["family"],
            n_cases=int(resume_target_n) if resume_target_n is not None else latest["n_cases"],
            openai_key=openai_key.strip(),
            anthropic_key=anthropic_key.strip(),
            resume_from=latest["exp_dir"],
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
    except ValueError as exc:
        st.error(str(exc))


# --------------------------------------------------------------------------- header

st.title("Medical Debate · Live Demo")
st.markdown(
    "Two AI debaters argue opposite diagnoses on a synthetic patient case. "
    "A judge picks A or B, sometimes without seeing the evidence. "
    "This page runs the pipeline live on your key and shows debates as they finish. "
    "[Project README](https://github.com/PS5138/llm_debate#readme) · methodology, arms, and caveats."
)


# --------------------------------------------------------------------------- live view

run = st.session_state.get("run")
if run is None:
    st.info(
        "Pick a family and a case count in the sidebar, paste your key(s), and hit **Run**. "
        "Default is 3 cases on `gpt-5.4-mini` to keep cost predictable."
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

    arms_done = sum(1 for cond in CONDITIONS if c.arm_answers.get(cond))
    case_rows.append(
        {
            "case_id": c.case_id,
            "status": status,
            "arms judged": f"{arms_done}/4",
            "correct (hidden)": c.correct,
            "distractor": c.distractor,
            "question": (c.question[:120] + "…") if len(c.question) > 120 else c.question,
        }
    )

if not case_rows:
    st.warning("No cases loaded yet. If this persists after the baselines phase, check the log below.")
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
selected = st.selectbox(
    "Pick a case to read its debate",
    case_ids,
    index=case_ids.index(st.session_state["selected_case"]) if st.session_state.get("selected_case") in case_ids else 0,
    key="selected_case",
) if case_ids else None

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
        st.markdown("**Judge picks per arm** (✓ = picked the correct side after swap-averaging)")
        rows = []
        for cond in CONDITIONS:
            answers = case.arm_answers.get(cond, {})
            orig = answers.get("orig", "")
            swap = answers.get("swap", "")
            verdict = _swap_averaged_verdict(orig, swap)
            rows.append(
                {
                    "arm": CONDITION_LABEL[cond],
                    "orig pick": _summarise_pick(orig) or "—",
                    "swap pick": _summarise_pick(swap) or "—",
                    "result": verdict,
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ---- aggregate results ----------------------------------------------------

if snap.phase == Phase.DONE and snap.results_dir:
    st.subheader("Aggregate results")
    results_dir = Path(snap.results_dir)
    acc_path = results_dir / "accuracy_by_condition.csv"
    pgr_path = results_dir / "pgr_by_condition.csv"
    plots_dir = results_dir / "plots"

    cols = st.columns(2)
    with cols[0]:
        if acc_path.exists():
            st.markdown("**Accuracy by condition**")
            st.dataframe(pd.read_csv(acc_path), width="stretch", hide_index=True)
    with cols[1]:
        if pgr_path.exists():
            st.markdown("**PGR by condition**")
            st.dataframe(pd.read_csv(pgr_path), width="stretch", hide_index=True)

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
