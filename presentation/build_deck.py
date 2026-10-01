"""Builds the 14-slide presentation (.pptx), its outline with scripts and Q&A, and preview PNGs for layout QA.

Slide numbers are read from outputs/tables, so the deck always matches the pipeline.

Run:  python presentation/build_deck.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
from deck_engine import (ACCENT, INK2, TINT, Box, Img, Slide, Text, bullets, check, closing_slide,  # noqa: E402
                         header, preview, stats_image, stats_text, table_slide, title_slide, two_columns,
                         write_pptx, W, H)
from narration import SLIDES  # noqa: E402

TABLES, FIGS = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
OUT = HERE / "churn_prediction_presentation.pptx"


def t(name):
    return pd.read_csv(TABLES / f"{name}.csv")


def fig(name):
    return FIGS / f"{name}.png"


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


def flow_slide(title, kicker, steps, items=None):
    els = header(title, kicker)
    n, gap = len(steps), 0.25
    w = (W - 1.2 - gap * (n - 1)) / n
    y, h = 1.85, 2.35
    for i, (head, body) in enumerate(steps):
        x = 0.6 + i * (w + gap)
        els.append(Box(x, y, w, h, fill=TINT))
        els.append(Text(x + 0.12, y + 0.1, w - 0.24, 0.5, [f"{i + 1}  {head}"], size=15, bold=True, color=ACCENT))
        els.append(Text(x + 0.12, y + 0.62, w - 0.24, h - 0.7, [body], size=12.5, color=INK2))
    if items:
        els.append(Text(0.6, y + h + 0.35, W - 1.2, H - (y + h + 0.35) - 0.4, bullets(items, 15), size=15, space_after=6))
    return Slide(elements=els, title=title)


def slides():
    snap = t("snapshots_target_distribution").set_index("cutoff")
    off = snap.loc["2024-03-31"]
    summ = t("table_summary").set_index("table")
    fm = t("final_model_test_metrics").set_index("population")
    g = t("gains_active").set_index("targeted_share")
    ms = t("model_selection_summary").set_index("model")
    sc = t("retention_scenarios").set_index("scenario")
    tg = t("retention_target_group")
    leak = t("leakage_check")
    ab = t("ablation")
    ab = ab[ab.model == "logistic regression"].set_index("variant")
    grp = t("lr_driver_group_share").set_index("group")["active"]
    seg = t("eda_segment_rates")
    tier = seg[(seg.attribute == "MEMBERSHIP_TIER") & (seg.population == "active")].set_index("value").churn_rate
    traj = t("eda_trajectories")
    ds = pd.read_csv(ROOT / "data" / "processed" / "customer_modelling_dataset.csv", usecols=["snapshot"])
    lapsed_churners = round(off.lapsed_members * off.churn_rate_lapsed)
    all_churners = round(off.eligible_members * off.churn_rate_all)
    sel = json.loads((TABLES / "final_model_selection.json").read_text())
    boot = sel["bootstrap_active_population"]["PR-AUC diff (LR - LGBM)"]

    def tr(col, churn, m):
        r = traj[(traj.churn == churn) & (traj.m_before == m)]
        return float(r[col].iloc[0])

    model_rows = [["Model (active members, Apr-Jun 2024)", "PR-AUC (95% CI)", "ROC-AUC", "F1", "Brier", "PR-AUC, 3 quarters"]]
    for k, name in [("rule_recency", "Recency rule (baseline)"), ("logreg", "Logistic regression"),
                    ("logreg_balanced", "Logistic regression, class-weighted"), ("random_forest", "Random forest"),
                    ("lightgbm", "LightGBM"), ("lightgbm_balanced", "LightGBM, class-weighted")]:
        r = ms.loc[k]
        model_rows.append([name, f"{r['PR-AUC active (test)']:.3f} ({r['95% CI']})", f"{r['ROC-AUC active']:.3f}",
                           f"{r['F1 active']:.3f}", f"{r['Brier active']:.3f}",
                           r["stability: PR-AUC over 3 quarters (mean +/- sd)"].replace("not run", "-")])
    eng, cou, sup = (sc.loc[n] for n in ["C. engagement outreach (app + email)", "B. targeted coupon", "A. proactive support outreach"])
    tgt = tg.set_index("segment")

    return [
        title_slide("Finding loyalty members before they stop shopping",
                    "Churn prediction, drivers and retention scenarios for FreshBasket's loyalty programme",
                    "Tailwyndz Propel 2026  |  Assessment 4: Customer Churn Prediction  |  FreshBasket loyalty data, Jan 2023 - Jun 2024"),
        stats_text("The business problem", "One retention offer for everyone is expensive and mostly reaches members who would stay",
                   [(f"{summ.loc['fb Customer Profile', 'unique_customers']:,}", "loyalty members (Silver, Gold, Platinum)"),
                    ("18", "months of monthly activity, Jan 2023 - Jun 2024"),
                    ("3 months", "churn window: no purchase despite earlier purchases"),
                    (pct(off.churn_rate_active, 1), "of active members churn per quarter")],
                   ["Decision supported: whom to contact each month, with which action, instead of contacting everyone",
                    "Unit of analysis: a member at a quarter-end cutoff; output: churn probability for the next 3 months",
                    "Success measure: PR-AUC, precision and recall on active members - not accuracy (90% for 'nobody churns')",
                    "Plus: why each member is at risk, and which retention actions are worth testing"]),
        table_slide("The data", "Three tables, one workbook, joined on CUSTOMER_ID",
                    [["Table", "Rows", "Members", "What it holds"],
                     ["fb Customer Profile", f"{summ.loc['fb Customer Profile', 'rows']:,}", f"{summ.loc['fb Customer Profile', 'unique_customers']:,}",
                      "signup date, age, gender, city, tier, marketing opt-in, preferred store/category, store price tier"],
                     ["fb Monthly Activity", f"{summ.loc['fb Monthly Activity', 'rows']:,}", f"{summ.loc['fb Monthly Activity', 'unique_customers']:,}",
                      "transactions, spend, basket, categories, promo %, app sessions, emails, coupons, tickets, complaints"],
                     ["fb Churn Label", f"{summ.loc['fb Churn Label', 'rows']:,}", f"{summ.loc['fb Churn Label', 'unique_customers']:,}",
                      "CHURNED (the target) plus three columns computed over the label window"]],
                    [2.6, 1.1, 1.1, 7.3], size=13, row_h=0.5,
                    items=["Same 2,600 members in all three tables; 0 orphan IDs; 6 members have no activity at all",
                           "Activity is one row per member-month, so a raw merge would repeat each profile 18 times",
                           f"Modelling grain: one row per member per quarter-end snapshot ({len(ds):,} rows over 5 snapshots)",
                           "Features from the months up to the snapshot; label from the 3 months after it"]),
        stats_text("Data quality: diagnosed, not guessed", "Every finding is logged with evidence, severity and treatment",
                   [("99.8%", "of purchase months: spend = transactions x basket"), ("22", "negative spends: pure sign errors"),
                    ("13", "mistyped totals (e.g. 3,731 vs 297), rebuilt"), ("953", "missing member-months: deleted records")],
                   ["Duplicates in every table (8 profile, 58 activity, 5 label rows); 2 conflicting pairs differed only in the sign of spend",
                    "City 40 spellings -> 10 cities, tier 12 -> 3; missing age, gender and store tier kept as 'Unknown'",
                    "Missing months treated as unknown and decided from data up to each cutoff only",
                    "Label table: LAST_PURCHASE_DATE, MONTHS_OBSERVED, INSUFFICIENT_HISTORY_FLAG are computed over the label window -> never used"]),
        stats_image("Key insight: most 'churners' had already left", None,
                    [(pct(lapsed_churners / all_churners), f"of Apr-Jun churners bought nothing in Jan-Mar ({lapsed_churners} of {all_churners})"),
                     (pct(off.churn_rate_lapsed), "churn among already-lapsed members"),
                     (pct(off.churn_rate_active, 1), "churn among members still active"),
                     ("~10%", "active-member churn in every quarter since mid-2023")],
                    fig("eda_churn_by_population"),
                    caption="Headline churn rises (18% -> 34%) only because lapsed members accumulate. Every result is reported for both populations."),
        Slide(elements=header("Key insight: churners fade before they leave",
                              f"Active members at the March 2024 cutoff: transactions {tr('TRANSACTIONS', 1, 5):.1f} -> {tr('TRANSACTIONS', 1, 0):.1f} a month, "
                              f"app sessions {tr('APP_SESSIONS', 1, 5):.1f} -> {tr('APP_SESSIONS', 1, 0):.1f}, tickets "
                              f"{tr('SUPPORT_TICKETS', 1, 5):.2f} -> {tr('SUPPORT_TICKETS', 1, 0):.2f}")
              + [Img(0.6, 1.8, W - 1.2, H - 1.8 - 0.8, str(fig("eda_trajectories"))),
                 Text(0.6, H - 0.75, W - 1.2, 0.45, ["Retained members stay flat (about 4.7 transactions a month). There is a window to act, "
                                                     "and trend features should help."], size=12, color="898781")],
              title="Key insight: churners fade before they leave"),
        stats_text("36 leakage-safe features", "Feature window (6 months, data <= cutoff) -> cutoff -> label window (next 3 months)",
                   [("14", "behavioural: recency, purchases, spend, trends"), ("10", "engagement: app, email, coupons, trends"),
                    ("4 + 8", "support (tickets, trend, complaints) + demographics"),
                    (f"{leak.max_abs_difference.max():g}", f"feature change after deleting up to {leak.activity_rows_removed.max():,} future rows")],
                   ["Fixed 3- and 6-month windows, recency capped at 6 months: every quarter's features mean the same thing",
                    "Trends (last 3 months minus the 3 before) capture the fading pattern",
                    "A missing month is a deleted record or inactivity - decided from rows up to the cutoff only",
                    "Automated leakage test: every snapshot rebuilt from truncated data gives identical features"]),
        flow_slide("Method and out-of-time validation", "Learn from past quarters, predict the next one - no random splits",
                   [("Snapshots", "5 quarter-end cutoffs; 3 training quarters with labels rebuilt (99.65% agree with the given label)"),
                    ("Baseline", "transparent rule: risk rises with months since the last purchase"),
                    ("Candidates", "logistic regression, random forest, LightGBM; with and without class weights"),
                    ("Out-of-time test", "tune on Jan-Mar 2024; test once on Apr-Jun 2024 (the given CHURNED)"),
                    ("Robustness", "3-quarter rolling check, bootstrap CIs, calibration, ablation")],
                   ["No random splits: a member's future quarter must never help predict their past one",
                    "Imbalance (about 1 in 10 active members churn) handled by a validation-tuned threshold, not resampling",
                    "Regularisation checked on the validation quarter only: results flat across C = 0.01-10"]),
        table_slide("Results: a tie at the top, so the simpler model wins", None, model_rows, [3.9, 2.4, 1.3, 1.1, 1.1, 2.3],
                    highlight=2, size=12.5, row_h=0.42,
                    items=[f"Logistic regression vs LightGBM: bootstrap PR-AUC difference {boot[0]:+.3f} (95% CI {boot[1]:+.3f} to {boot[2]:+.3f}) - a tie",
                           "Selected: logistic regression - best calibrated, most stable, and every score splits into exact per-member reasons",
                           f"At the validation-tuned threshold: precision {fm.loc['active', 'precision']:.2f}, recall {fm.loc['active', 'recall']:.2f} on active members"]),
        stats_image("Targeting value", "Active members, Apr-Jun 2024 test quarter",
                    [(pct(g.loc[0.1, "share_of_churners_captured"]), "of churners reached by contacting the riskiest 10%"),
                     (pct(g.loc[0.1, "precision_in_target"]), "of those contacted really churn"),
                     (f"{g.loc[0.1, 'lift_vs_random']:.1f}x", "better than random targeting"),
                     (pct(g.loc[0.2, "share_of_churners_captured"]), "reached with the riskiest 20%")],
                    fig("gains_active"), caption="Calibrated: top risk decile predicted 72.4% churn, observed 72.7%."),
        Slide(elements=header("What drives churn", "Behaviour and engagement - not demographics or tier")
              + [Img(0.6, 1.7, W - 1.2, 3.65, str(fig("drivers"))),
                 Text(0.6, 5.45, W - 1.2, H - 5.45 - 0.3, bullets([
                     f"Attribution for active members: behavioural {pct(grp['behavioural'])}, engagement {pct(grp['engagement'])}, "
                     f"demographics {pct(grp['demographics'])}, support {pct(grp['support'])} - LightGBM SHAP points to the same drivers",
                     "Risk rises with falling transactions, fewer emails and app sessions, and more support tickets",
                     f"Tier makes no difference: Silver {pct(tier['Silver'], 1)}, Gold {pct(tier['Gold'], 1)}, Platinum {pct(tier['Platinum'], 1)}",
                     f"Ablation (ROC-AUC, active): demographics {ab.loc['1. demographics only', 'ROC-AUC active']:.2f} -> + behaviour "
                     f"{ab.loc['2. + behavioural', 'ROC-AUC active']:.2f} -> + engagement {ab.loc['3. + engagement', 'ROC-AUC active']:.2f}; "
                     "every member gets their own top reasons"], 13.5), size=13.5, space_after=3)],
              title="What drives churn"),
        stats_image("Retention what-ifs for the riskiest 20% (model-based)",
                    f"{int(tgt.loc['targeted top 20%', 'members'])} members holding {tgt.loc['targeted top 20%', 'expected_churners']:.0f} of "
                    f"{tgt.loc['all active members', 'expected_churners']:.0f} expected churners; each action closes half the retained-vs-churned behaviour gap",
                    [(f"{eng.change_pp_eligible:+.0f} pp", f"engagement outreach (n={eng.members_eligible_for_action}): ~{eng.expected_churners_avoided_model_based:.0f} fewer expected churners"),
                     (f"{sup.change_pp_eligible:+.1f} pp", f"proactive support for members with tickets (n={sup.members_eligible_for_action})"),
                     (f"{cou.change_pp_eligible:+.1f} pp", f"targeted coupon (n={cou.members_eligible_for_action})"),
                     ("25-100%", "of the gap closed: sensitivity shown on the right")],
                    fig("retention_scenarios"),
                    caption="Changes in predicted risk under stated assumptions: they rank what to test, not what the actions will achieve."),
        two_columns("Business decision and limitations", None, "What FreshBasket should do",
                    ["Monthly risk list of active members; contact the top 10% (~170) with their reasons",
                     "Test engagement outreach and proactive support with a 20-30% random hold-out",
                     "Report active-member churn as the KPI; run win-back separately for lapsed members",
                     "Capture offer cost and member value to size campaigns on expected value"],
                    "What not to assume",
                    ["The scores and what-ifs are associations, not causal effects of any action",
                     "Signals are unusually clean (likely synthetic data): re-validate on live data",
                     "One given label quarter; earlier labels rebuilt (~0.4% noise)",
                     "Threshold tuned for F1, not costs (no cost data supplied)"]),
        closing_slide("Takeaways", ["Most measured churn is members who had already left; active-member churn is a steady ~10% per quarter",
                                    "A transparent model finds 76% of next quarter's churners in the riskiest 10% of active members",
                                    "Target by behaviour, not tier - and prove the retention actions with a hold-out test"]),
    ]


def outline(deck) -> str:
    words = sum(len(m["script"].split()) for m in SLIDES)
    lines = ["# Presentation outline - FreshBasket loyalty churn prediction", "",
             f"Deck: `presentation/{OUT.name}` ({len(deck)} slides). Each script is also in the slide's speaker notes, so "
             f"PowerPoint's recording view shows it as a teleprompter. About {words / 140:.0f} minutes at 140 words a "
             "minute, plus a 2-minute live demo.", ""]
    for i, (s, m) in enumerate(zip(deck, SLIDES), 1):
        content, pending, numbers = [], None, []
        for e in s.elements:
            if isinstance(e, Img):
                numbers.append(f"chart `{Path(e.path).relative_to(ROOT).as_posix()}`")
            if not isinstance(e, Text):
                continue
            texts = [p if isinstance(p, str) else p[0] for p in e.paras]
            if texts and texts[0] == s.title:
                continue
            if e.size >= 28 and e.color == ACCENT:
                pending = texts[0]
                continue
            for tx in texts:
                if not tx:
                    continue
                if pending is not None:
                    content.append(f"- **{pending}** {tx}")
                    numbers.append(f"**{pending}**")
                    pending = None
                else:
                    content.append(f"- {tx}")
        for e in s.elements:
            if e.__class__.__name__ == "Table":
                content += ["", "| " + " | ".join(e.rows[0]) + " |", "|" + "---|" * len(e.rows[0])]
                content += ["| " + " | ".join(r) + " |" for r in e.rows[1:]]
                numbers.append("the table above")
        secs = len(m["script"].split()) / 140 * 60
        lines += [f"## Slide {i}. {s.title}", "", f"**Objective:** {m['objective']}", "", "**Exact content:**", ""] + content + [
            "", f"**Recommended visual:** {m['visual']}", "",
            f"**Number / chart to show:** {', '.join(numbers) if numbers else 'title only'}", "",
            f"**Speaking script (~{secs:.0f} s):** {m['script']}", "",
            f"**Likely evaluator question:** {m['question']}", "", f"**Answer:** {m['answer']}", ""]
    return "\n".join(lines)


def run(check_only=False):
    deck = slides()
    assert len(deck) == len(SLIDES), (len(deck), len(SLIDES))
    for s, m in zip(deck, SLIDES):
        s.notes = m["script"]
    problems = check(deck, "deck")
    if not check_only:
        write_pptx(deck, OUT)
        (HERE / "presentation_outline.md").write_text(outline(deck), encoding="utf-8")
        prev = HERE / "_preview"
        prev.mkdir(exist_ok=True)
        preview(deck, prev / "slide")
    print("\n".join(problems) if problems else f"layout check: no overflow or out-of-bounds elements ({len(deck)} slides)")
    return problems


if __name__ == "__main__":
    run("--check" in sys.argv)
