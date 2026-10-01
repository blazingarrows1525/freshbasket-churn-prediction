"""Per-slide presentation content: objective, recommended visual, spoken script, likely question and answer.

build_deck.py puts each script into the slide's speaker notes (PowerPoint shows them while recording) and
writes everything to presentation/presentation_outline.md. Every number quoted is on the slide or in
outputs/tables.
"""

DEMO_CUE = ("[DEMO, about 2 minutes: switch to the GitHub repository / README, then run notebooks/06_live_demo.ipynb; "
            "come back for the last slide.]")

SLIDES = [
    dict(objective="Introduce myself, the assessment chosen, the business problem and the objective.",
         visual="Title slide.",
         script="Hi, I'm [NAME], and this is my Tailwyndz Propel 2026 assessment submission. For this assessment, I chose "
                "Assessment 4: customer churn prediction for FreshBasket's loyalty programme. The business problem I focused "
                "on is that FreshBasket sends the same retention offer to every member, which is expensive and mostly reaches "
                "people who were never going to leave. My objective was to find the members who are likely to stop shopping "
                "next quarter, explain why, and work out which retention actions are worth testing. I'll briefly walk through "
                "the data, methodology, modelling approach, results, and the business implications.",
         question="Why did you choose this assessment over the other three?",
         answer="Its data matched the brief exactly, so every requirement could be met without workarounds, and it let me "
                "show what matters in a real project: finding leakage traps, designing an honest out-of-time validation and "
                "turning a model into a targeting decision. The whole pipeline also runs in under two minutes, so anyone can "
                "reproduce it."),
    dict(objective="Define the decision, the target, the unit of analysis and how success is measured.",
         visual="Stat cards plus four bullets: decision, target, unit, metric.",
         script="Let me start with the decision this supports. Today every member gets the same retention offer. The business "
                "wants a list instead: who is likely to stop shopping in the next three months, and why. The brief fixes the "
                "label: a member churns if they make no purchase in a three-month window despite having bought before. So the "
                "unit of analysis is a member at a point in time, and the output is a probability for the next quarter. One "
                "thing I decided early: churners are a minority, so I judge models on how well they find churners, using "
                "precision, recall and PR-AUC, not on accuracy, which looks great even for a model that predicts nobody churns.",
         question="Why not use accuracy?",
         answer="Among active members only about one in ten churns, so predicting 'no churn' for everyone is already 90% "
                "accurate and finds nobody. PR-AUC and recall measure what the business needs: how many churners we reach "
                "and how many contacts are wasted."),
    dict(objective="Show the three tables, their size and how they relate, and why I did not simply merge them.",
         visual="Table of the three sheets (rows, members, content).",
         script="The data is three tables from one workbook. A customer profile with demographics and tier: 2,608 rows. "
                "Monthly activity: 28,073 rows from January 2023 to June 2024, with transactions, spend, app sessions, emails "
                "opened, coupons, support tickets and complaints. And the churn label. They all join on customer ID and "
                "contain exactly the same 2,600 members. I didn't just merge them, though. Activity is one row per member per "
                "month, so a raw join would repeat the profile every month. Instead I built one row per member per quarterly "
                "snapshot, with features from before the snapshot date and the label from after it.",
         question="How did you check the joins?",
         answer="Before cleaning I checked keys and cardinality: zero orphan IDs, profile and label one row per member once "
                "exact duplicates are removed, activity one row per member-month. Six members have no activity at all, so "
                "they are not eligible for a prediction."),
    dict(objective="Show that the data was inspected properly and every issue was diagnosed and treated with evidence.",
         visual="Four stat cards (spend identity, sign errors, mistyped totals, missing months) and bullets.",
         script="The brief warned there would be data-quality problems, and they were all there. The most useful thing I "
                "found is that total spend equals transactions times average basket on 99.8% of purchase months. That let me "
                "diagnose errors instead of guessing. The 22 negative spends were pure sign errors. Thirteen extreme spends "
                "were mistyped totals, for example 3,731 where the arithmetic says 297, so I rebuilt them from the identity. "
                "There were duplicates in all three tables, city names in 40 spellings that were really 10 cities, and 953 "
                "missing member-months that look like deleted records rather than inactivity. And one trap: three columns in "
                "the label table are calculated over the label window itself, so using them would leak the answer. I never "
                "used them.",
         question="Why not just drop the outlier spends?",
         answer="The evidence showed they weren't real big shoppers: the transaction count and basket value were normal, "
                "only the total was mistyped. Rebuilding them from the identity keeps the member's real behaviour; dropping "
                "them would throw away valid months."),
    dict(objective="Present the finding that reframes the problem: most labelled churners had already left.",
         visual="Churn rate by quarter for lapsed vs active members, with four stat cards.",
         script="This was the finding that shaped everything. Eighty percent of the members labelled as churned in April to "
                "June hadn't bought anything in January to March. They had already left before the label window even "
                "started. In that group churn is 99%, so there's nothing to predict, and it's too late for a retention offer. "
                "Among members who were still buying, churn is 9.6%, and it has been about 10% in every quarter since "
                "mid-2023. So the headline churn rate is rising only because lapsed members pile up. That's why I report every "
                "result for two populations, and why the active members are the real early-warning problem.",
         question="Isn't excluding lapsed members cherry-picking?",
         answer="I don't exclude them: every metric is reported for both groups. Separating them stops a trivially "
                "predictable group from inflating the scores. The all-member ROC-AUC is 0.99, but the honest number for the "
                "business is the active-member PR-AUC."),
    dict(objective="Show that churners fade before they leave, which is the signal an early-warning model needs.",
         visual="Monthly transactions, app sessions, emails and tickets in the 6 months before the cutoff: churned vs retained.",
         script="The second insight is that churners fade before they leave. These are members who were still active at the "
                "end of March. The ones who churned in the next quarter had been cutting back for months. Transactions fell "
                "from 3.6 a month to 0.9, app sessions from 1.8 to 0.3, while support tickets went up from 0.17 to about 0.5. "
                "Members who stayed were flat, at about 4.7 transactions a month. That matters for two reasons: "
                "there's a window where the business can still act, and it tells me trends, not just levels, should be features.",
         question="Couldn't this just be the label definition showing up early?",
         answer="No. These are all months before the cutoff, and the label only looks at the three months after it. The fade "
                "is real behaviour before the outcome, which is exactly what an early-warning model needs."),
    dict(objective="Explain the feature groups, the design choices and the proof that nothing leaks from the future.",
         visual="Stat cards (36 features, group sizes, leakage-test result) and bullets.",
         script="So I engineered 36 features in four groups. Behavioural: recency, transactions and spend over three and six "
                "months, and their trends. Engagement: app sessions, emails opened and coupons, also with trends. Support: "
                "tickets, ticket trend and complaints. And demographics plus tenure. Two design choices matter. First, fixed "
                "three- and six-month windows, so every quarter's features mean the same thing. Second, everything comes only "
                "from months up to the cutoff, including the decision whether a missing month is a deleted record or real "
                "inactivity. I didn't just assume that. I rebuilt every snapshot after deleting all the data after its cutoff, "
                "up to about 19,000 rows, and not a single feature value changed.",
         question="How do you know there is no leakage?",
         answer="Three ways: the leaky label columns are excluded and the code asserts they can never become features; "
                "features are built only from rows up to the cutoff; and an automated test rebuilds each snapshot from "
                "truncated data and finds a maximum difference of exactly zero."),
    dict(objective="Explain the validation design, the baseline and the candidate models.",
         visual="Five-step flow: snapshots, baseline, candidates, out-of-time test, robustness checks.",
         script="For validation I copied how the model would actually be used: learn from past quarters, predict the next "
                "one. The label is only given for April to June 2024, so I rebuilt the same label for three earlier quarters. "
                "It matches the given label for 99.65% of members. I tuned on the January-to-March quarter and tested once on "
                "April to June. No random splits, because those would let a member's future quarter help predict their past. I "
                "started with a simple baseline, where risk rises with months since the last purchase, and compared it with "
                "logistic regression, random forest and LightGBM, with and without class weights. On top of that I re-ran the "
                "comparison across three consecutive quarters to check the result is stable.",
         question="Why not a random train/test split with cross-validation?",
         answer="Members appear in several quarters. A random split would put a member's March snapshot in training and their "
                "December snapshot in test, so the model would effectively see the future. Out-of-time snapshots are the "
                "honest version of how it runs in production."),
    dict(objective="Compare the models on more than one score and justify the final choice.",
         visual="Model comparison table: PR-AUC with 95% CI, ROC-AUC, F1, Brier, stability over 3 quarters.",
         script="Here are the results on the held-out quarter for active members. The baseline rule gets a PR-AUC of 0.51. "
                "Logistic regression gets 0.83, LightGBM 0.82 and random forest 0.82. The difference between logistic "
                "regression and LightGBM is a tie: a bootstrap puts it at plus 0.005, with an interval from minus 0.02 to "
                "plus 0.03. When models tie, I take the simpler one. The logistic regression is better calibrated and every "
                "score splits exactly into reasons per member. It's also the most stable: across three consecutive quarters it "
                "averages 0.82 and it's the best model in every one. Class weights didn't improve ranking and made calibration "
                "worse, so I handled the imbalance by tuning the threshold instead.",
         question="Isn't a more complex model always better?",
         answer="Only if the evidence says so. Here the boosting model and the logistic regression are statistically "
                "indistinguishable on the test quarter, and the logistic regression wins in every quarter of the rolling "
                "check. With a tie, transparency and calibration decide."),
    dict(objective="Translate the model into a business number: how many churners a targeted list reaches.",
         visual="Cumulative gains curve with four stat cards.",
         script="This is what it means for the business. If FreshBasket contacts the riskiest 10% of active members, about "
                "170 people, it reaches 76% of the quarter's churners, and 73% of the people it contacts really are about to "
                "churn. That's 7.6 times better than picking at random. The riskiest 20% reaches 91%. And the probabilities can "
                "be trusted: in the top risk decile the model predicted 72.4% churn and the actual rate was 72.7%.",
         question="How would you decide how many members to contact?",
         answer="From this gains curve plus the cost of the offer and the value of a retained member. The data has no costs, "
                "so I report the trade-off rather than claim an optimum; the riskiest 10-20% captures most churners with good "
                "precision."),
    dict(objective="Explain globally and locally what drives the predictions, and what the ablation shows.",
         visual="Driver chart: logistic contributions next to LightGBM SHAP, with bullets.",
         script="What drives the predictions? Mostly behaviour and engagement: about 40% of the model's attribution is "
                "behavioural and 37% engagement. Falling transactions, fewer emails opened, fewer app sessions, rising support "
                "tickets. LightGBM's SHAP values point to the same drivers. Demographics barely matter. On their own they score "
                "a ROC-AUC of 0.51 for active members, which is chance, and tier makes no difference: Silver, Gold and Platinum "
                "all churn at about 9.5%. In the ablation, adding behaviour lifts it to 0.93 and engagement to 0.96. And "
                "because it's a logistic regression, every member on the list comes with their own reasons, like 'support "
                "tickets up, transactions falling'. These are associations, not causes.",
         question="Does that mean support tickets cause churn?",
         answer="No. Tickets are a warning sign the model uses to rank members. Whether resolving tickets keeps people is a "
                "causal question this data can't answer; that's what the controlled test is for."),
    dict(objective="Run the retention scenario with data-based assumptions and state clearly that it is not causal.",
         visual="Scenario chart (risk before/after for each action; sensitivity to the assumed size) with stat cards.",
         script="Finally, the retention scenario. I took the riskiest 20% of members active in June 2024, 331 people holding "
                "131 of the 140 expected churners, and asked the model what happens to their risk if their behaviour moved part "
                "of the way towards a typical retained member. I didn't invent the size of the change. Each action closes half "
                "the observed gap between retained and churning members, with 25% and 100% as sensitivity. Engagement outreach "
                "lowers predicted risk by about 10 points, roughly 34 fewer expected churners. Coupons and support outreach move "
                "it by about 3 points each. But this is the model's association, not a causal effect, so these numbers tell us "
                "what to test first, not what will happen.",
         question="Why should anyone trust these scenario numbers?",
         answer="They shouldn't be trusted as effects, and I say so. They rank actions under stated, data-based assumptions. "
                "The real effect needs a randomised hold-out test, which is my main recommendation."),
    dict(objective="State the business decision, the recommendations and what the business must not assume.",
         visual="Two columns: recommendations and limitations.",
         script="So what should FreshBasket do? First, replace the blanket offer with a monthly risk list of active members, "
                "starting with the top 10%, each with their reasons attached. Second, test engagement outreach and proactive "
                "support with a 20 to 30% random hold-out, so we measure what actually works. Third, report churn among active "
                "members as the KPI and run a separate win-back programme for lapsed members. And capture offer costs and "
                "member value, so campaigns can be sized in money rather than F1. What the business should not assume: that "
                "these scores are causal, that accuracy will be this high on live data, because the signals here are unusually "
                "clean, or that one test quarter is enough. The model should be re-validated every quarter. Before I wrap up, "
                "let me show you the project itself. " + DEMO_CUE,
         question="How would you deploy and monitor this?",
         answer="Score members monthly with the saved pipeline and send the list with reasons to the retention team. Every "
                "quarter, when new labels arrive, check PR-AUC, calibration and the churn rate per risk band, then retrain. If "
                "performance drops or active-member churn shifts, investigate before using the list."),
    dict(objective="Summarise the project in three takeaways and the decision.",
         visual="Closing slide with three numbered takeaways.",
         script="To sum up. Most of the churn FreshBasket sees is members who had already left, while active-member churn is "
                "a steady 10% a quarter. A transparent, well-calibrated model finds about three quarters of next quarter's "
                "churners in the riskiest 10% of active members, validated on quarters it never saw. And the drivers are "
                "behaviour, not tier. My recommendation is a monthly targeted list plus a controlled test of the retention "
                "actions. Thanks for watching.",
         question="What would you do with one more month?",
         answer="Design the hold-out test with the business, add offer costs so the target size is chosen on expected value, "
                "and extend the validation over more quarters as new labels arrive."),
]
