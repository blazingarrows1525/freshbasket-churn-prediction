# Presentation outline - FreshBasket loyalty churn prediction

Deck: `presentation/churn_prediction_presentation.pptx` (14 slides). Each script is also in the slide's speaker notes, so PowerPoint's recording view shows it as a teleprompter. About 11 minutes at 140 words a minute, plus a 2-minute live demo.

## Slide 1. Finding loyalty members before they stop shopping

**Objective:** Introduce myself, the assessment chosen, the business problem and the objective.

**Exact content:**

- Churn prediction, drivers and retention scenarios for FreshBasket's loyalty programme
- Tailwyndz Propel 2026  |  Assessment 4: Customer Churn Prediction  |  FreshBasket loyalty data, Jan 2023 - Jun 2024

**Recommended visual:** Title slide.

**Number / chart to show:** title only

**Speaking script (~41 s):** Hi, I'm [NAME], and this is my Tailwyndz Propel 2026 assessment submission. For this assessment, I chose Assessment 4: customer churn prediction for FreshBasket's loyalty programme. The business problem I focused on is that FreshBasket sends the same retention offer to every member, which is expensive and mostly reaches people who were never going to leave. My objective was to find the members who are likely to stop shopping next quarter, explain why, and work out which retention actions are worth testing. I'll briefly walk through the data, methodology, modelling approach, results, and the business implications.

**Likely evaluator question:** Why did you choose this assessment over the other three?

**Answer:** Its data matched the brief exactly, so every requirement could be met without workarounds, and it let me show what matters in a real project: finding leakage traps, designing an honest out-of-time validation and turning a model into a targeting decision. The whole pipeline also runs in under two minutes, so anyone can reproduce it.

## Slide 2. The business problem

**Objective:** Define the decision, the target, the unit of analysis and how success is measured.

**Exact content:**

- One retention offer for everyone is expensive and mostly reaches members who would stay
- **2,600** loyalty members (Silver, Gold, Platinum)
- **18** months of monthly activity, Jan 2023 - Jun 2024
- **3 months** churn window: no purchase despite earlier purchases
- **9.6%** of active members churn per quarter
- Decision supported: whom to contact each month, with which action, instead of contacting everyone
- Unit of analysis: a member at a quarter-end cutoff; output: churn probability for the next 3 months
- Success measure: PR-AUC, precision and recall on active members - not accuracy (90% for 'nobody churns')
- Plus: why each member is at risk, and which retention actions are worth testing

**Recommended visual:** Stat cards plus four bullets: decision, target, unit, metric.

**Number / chart to show:** **2,600**, **18**, **3 months**, **9.6%**

**Speaking script (~50 s):** Let me start with the decision this supports. Today every member gets the same retention offer. The business wants a list instead: who is likely to stop shopping in the next three months, and why. The brief fixes the label: a member churns if they make no purchase in a three-month window despite having bought before. So the unit of analysis is a member at a point in time, and the output is a probability for the next quarter. One thing I decided early: churners are a minority, so I judge models on how well they find churners, using precision, recall and PR-AUC, not on accuracy, which looks great even for a model that predicts nobody churns.

**Likely evaluator question:** Why not use accuracy?

**Answer:** Among active members only about one in ten churns, so predicting 'no churn' for everyone is already 90% accurate and finds nobody. PR-AUC and recall measure what the business needs: how many churners we reach and how many contacts are wasted.

## Slide 3. The data

**Objective:** Show the three tables, their size and how they relate, and why I did not simply merge them.

**Exact content:**

- Three tables, one workbook, joined on CUSTOMER_ID
- Same 2,600 members in all three tables; 0 orphan IDs; 6 members have no activity at all
- Activity is one row per member-month, so a raw merge would repeat each profile 18 times
- Modelling grain: one row per member per quarter-end snapshot (10,776 rows over 5 snapshots)
- Features from the months up to the snapshot; label from the 3 months after it

| Table | Rows | Members | What it holds |
|---|---|---|---|
| fb Customer Profile | 2,608 | 2,600 | signup date, age, gender, city, tier, marketing opt-in, preferred store/category, store price tier |
| fb Monthly Activity | 28,073 | 2,594 | transactions, spend, basket, categories, promo %, app sessions, emails, coupons, tickets, complaints |
| fb Churn Label | 2,605 | 2,600 | CHURNED (the target) plus three columns computed over the label window |

**Recommended visual:** Table of the three sheets (rows, members, content).

**Number / chart to show:** the table above

**Speaking script (~44 s):** The data is three tables from one workbook. A customer profile with demographics and tier: 2,608 rows. Monthly activity: 28,073 rows from January 2023 to June 2024, with transactions, spend, app sessions, emails opened, coupons, support tickets and complaints. And the churn label. They all join on customer ID and contain exactly the same 2,600 members. I didn't just merge them, though. Activity is one row per member per month, so a raw join would repeat the profile every month. Instead I built one row per member per quarterly snapshot, with features from before the snapshot date and the label from after it.

**Likely evaluator question:** How did you check the joins?

**Answer:** Before cleaning I checked keys and cardinality: zero orphan IDs, profile and label one row per member once exact duplicates are removed, activity one row per member-month. Six members have no activity at all, so they are not eligible for a prediction.

## Slide 4. Data quality: diagnosed, not guessed

**Objective:** Show that the data was inspected properly and every issue was diagnosed and treated with evidence.

**Exact content:**

- Every finding is logged with evidence, severity and treatment
- **99.8%** of purchase months: spend = transactions x basket
- **22** negative spends: pure sign errors
- **13** mistyped totals (e.g. 3,731 vs 297), rebuilt
- **953** missing member-months: deleted records
- Duplicates in every table (8 profile, 58 activity, 5 label rows); 2 conflicting pairs differed only in the sign of spend
- City 40 spellings -> 10 cities, tier 12 -> 3; missing age, gender and store tier kept as 'Unknown'
- Missing months treated as unknown and decided from data up to each cutoff only
- Label table: LAST_PURCHASE_DATE, MONTHS_OBSERVED, INSUFFICIENT_HISTORY_FLAG are computed over the label window -> never used

**Recommended visual:** Four stat cards (spend identity, sign errors, mistyped totals, missing months) and bullets.

**Number / chart to show:** **99.8%**, **22**, **13**, **953**

**Speaking script (~54 s):** The brief warned there would be data-quality problems, and they were all there. The most useful thing I found is that total spend equals transactions times average basket on 99.8% of purchase months. That let me diagnose errors instead of guessing. The 22 negative spends were pure sign errors. Thirteen extreme spends were mistyped totals, for example 3,731 where the arithmetic says 297, so I rebuilt them from the identity. There were duplicates in all three tables, city names in 40 spellings that were really 10 cities, and 953 missing member-months that look like deleted records rather than inactivity. And one trap: three columns in the label table are calculated over the label window itself, so using them would leak the answer. I never used them.

**Likely evaluator question:** Why not just drop the outlier spends?

**Answer:** The evidence showed they weren't real big shoppers: the transaction count and basket value were normal, only the total was mistyped. Rebuilding them from the identity keeps the member's real behaviour; dropping them would throw away valid months.

## Slide 5. Key insight: most 'churners' had already left

**Objective:** Present the finding that reframes the problem: most labelled churners had already left.

**Exact content:**

- **80%** of Apr-Jun churners bought nothing in Jan-Mar (649 of 813)
- **99%** churn among already-lapsed members
- **9.6%** churn among members still active
- **~10%** active-member churn in every quarter since mid-2023
- Headline churn rises (18% -> 34%) only because lapsed members accumulate. Every result is reported for both populations.

**Recommended visual:** Churn rate by quarter for lapsed vs active members, with four stat cards.

**Number / chart to show:** **80%**, **99%**, **9.6%**, **~10%**, chart `outputs/figures/eda_churn_by_population.png`

**Speaking script (~46 s):** This was the finding that shaped everything. Eighty percent of the members labelled as churned in April to June hadn't bought anything in January to March. They had already left before the label window even started. In that group churn is 99%, so there's nothing to predict, and it's too late for a retention offer. Among members who were still buying, churn is 9.6%, and it has been about 10% in every quarter since mid-2023. So the headline churn rate is rising only because lapsed members pile up. That's why I report every result for two populations, and why the active members are the real early-warning problem.

**Likely evaluator question:** Isn't excluding lapsed members cherry-picking?

**Answer:** I don't exclude them: every metric is reported for both groups. Separating them stops a trivially predictable group from inflating the scores. The all-member ROC-AUC is 0.99, but the honest number for the business is the active-member PR-AUC.

## Slide 6. Key insight: churners fade before they leave

**Objective:** Show that churners fade before they leave, which is the signal an early-warning model needs.

**Exact content:**

- Active members at the March 2024 cutoff: transactions 3.6 -> 0.9 a month, app sessions 1.8 -> 0.3, tickets 0.17 -> 0.49
- Retained members stay flat (about 4.7 transactions a month). There is a window to act, and trend features should help.

**Recommended visual:** Monthly transactions, app sessions, emails and tickets in the 6 months before the cutoff: churned vs retained.

**Number / chart to show:** chart `outputs/figures/eda_trajectories.png`

**Speaking script (~41 s):** The second insight is that churners fade before they leave. These are members who were still active at the end of March. The ones who churned in the next quarter had been cutting back for months. Transactions fell from 3.6 a month to 0.9, app sessions from 1.8 to 0.3, while support tickets went up from 0.17 to about 0.5. Members who stayed were flat, at about 4.7 transactions a month. That matters for two reasons: there's a window where the business can still act, and it tells me trends, not just levels, should be features.

**Likely evaluator question:** Couldn't this just be the label definition showing up early?

**Answer:** No. These are all months before the cutoff, and the label only looks at the three months after it. The fade is real behaviour before the outcome, which is exactly what an early-warning model needs.

## Slide 7. 36 leakage-safe features

**Objective:** Explain the feature groups, the design choices and the proof that nothing leaks from the future.

**Exact content:**

- Feature window (6 months, data <= cutoff) -> cutoff -> label window (next 3 months)
- **14** behavioural: recency, purchases, spend, trends
- **10** engagement: app, email, coupons, trends
- **4 + 8** support (tickets, trend, complaints) + demographics
- **0** feature change after deleting up to 18,975 future rows
- Fixed 3- and 6-month windows, recency capped at 6 months: every quarter's features mean the same thing
- Trends (last 3 months minus the 3 before) capture the fading pattern
- A missing month is a deleted record or inactivity - decided from rows up to the cutoff only
- Automated leakage test: every snapshot rebuilt from truncated data gives identical features

**Recommended visual:** Stat cards (36 features, group sizes, leakage-test result) and bullets.

**Number / chart to show:** **14**, **10**, **4 + 8**, **0**

**Speaking script (~48 s):** So I engineered 36 features in four groups. Behavioural: recency, transactions and spend over three and six months, and their trends. Engagement: app sessions, emails opened and coupons, also with trends. Support: tickets, ticket trend and complaints. And demographics plus tenure. Two design choices matter. First, fixed three- and six-month windows, so every quarter's features mean the same thing. Second, everything comes only from months up to the cutoff, including the decision whether a missing month is a deleted record or real inactivity. I didn't just assume that. I rebuilt every snapshot after deleting all the data after its cutoff, up to about 19,000 rows, and not a single feature value changed.

**Likely evaluator question:** How do you know there is no leakage?

**Answer:** Three ways: the leaky label columns are excluded and the code asserts they can never become features; features are built only from rows up to the cutoff; and an automated test rebuilds each snapshot from truncated data and finds a maximum difference of exactly zero.

## Slide 8. Method and out-of-time validation

**Objective:** Explain the validation design, the baseline and the candidate models.

**Exact content:**

- Learn from past quarters, predict the next one - no random splits
- 1  Snapshots
- 5 quarter-end cutoffs; 3 training quarters with labels rebuilt (99.65% agree with the given label)
- 2  Baseline
- transparent rule: risk rises with months since the last purchase
- 3  Candidates
- logistic regression, random forest, LightGBM; with and without class weights
- 4  Out-of-time test
- tune on Jan-Mar 2024; test once on Apr-Jun 2024 (the given CHURNED)
- 5  Robustness
- 3-quarter rolling check, bootstrap CIs, calibration, ablation
- No random splits: a member's future quarter must never help predict their past one
- Imbalance (about 1 in 10 active members churn) handled by a validation-tuned threshold, not resampling
- Regularisation checked on the validation quarter only: results flat across C = 0.01-10

**Recommended visual:** Five-step flow: snapshots, baseline, candidates, out-of-time test, robustness checks.

**Number / chart to show:** title only

**Speaking script (~53 s):** For validation I copied how the model would actually be used: learn from past quarters, predict the next one. The label is only given for April to June 2024, so I rebuilt the same label for three earlier quarters. It matches the given label for 99.65% of members. I tuned on the January-to-March quarter and tested once on April to June. No random splits, because those would let a member's future quarter help predict their past. I started with a simple baseline, where risk rises with months since the last purchase, and compared it with logistic regression, random forest and LightGBM, with and without class weights. On top of that I re-ran the comparison across three consecutive quarters to check the result is stable.

**Likely evaluator question:** Why not a random train/test split with cross-validation?

**Answer:** Members appear in several quarters. A random split would put a member's March snapshot in training and their December snapshot in test, so the model would effectively see the future. Out-of-time snapshots are the honest version of how it runs in production.

## Slide 9. Results: a tie at the top, so the simpler model wins

**Objective:** Compare the models on more than one score and justify the final choice.

**Exact content:**

- Logistic regression vs LightGBM: bootstrap PR-AUC difference +0.005 (95% CI -0.019 to +0.029) - a tie
- Selected: logistic regression - best calibrated, most stable, and every score splits into exact per-member reasons
- At the validation-tuned threshold: precision 0.79, recall 0.73 on active members

| Model (active members, Apr-Jun 2024) | PR-AUC (95% CI) | ROC-AUC | F1 | Brier | PR-AUC, 3 quarters |
|---|---|---|---|---|---|
| Recency rule (baseline) | 0.514 (0.437-0.589) | 0.861 | 0.576 | 0.069 | 0.502 +/- 0.038 |
| Logistic regression | 0.828 (0.775-0.873) | 0.962 | 0.756 | 0.034 | 0.824 +/- 0.019 |
| Logistic regression, class-weighted | 0.816 (0.756-0.865) | 0.961 | 0.740 | 0.043 | - |
| Random forest | 0.817 (0.764-0.863) | 0.955 | 0.711 | 0.038 | 0.765 +/- 0.045 |
| LightGBM | 0.823 (0.769-0.872) | 0.960 | 0.749 | 0.034 | 0.804 +/- 0.025 |
| LightGBM, class-weighted | 0.827 (0.774-0.873) | 0.960 | 0.744 | 0.037 | - |

**Recommended visual:** Model comparison table: PR-AUC with 95% CI, ROC-AUC, F1, Brier, stability over 3 quarters.

**Number / chart to show:** the table above

**Speaking script (~50 s):** Here are the results on the held-out quarter for active members. The baseline rule gets a PR-AUC of 0.51. Logistic regression gets 0.83, LightGBM 0.82 and random forest 0.82. The difference between logistic regression and LightGBM is a tie: a bootstrap puts it at plus 0.005, with an interval from minus 0.02 to plus 0.03. When models tie, I take the simpler one. The logistic regression is better calibrated and every score splits exactly into reasons per member. It's also the most stable: across three consecutive quarters it averages 0.82 and it's the best model in every one. Class weights didn't improve ranking and made calibration worse, so I handled the imbalance by tuning the threshold instead.

**Likely evaluator question:** Isn't a more complex model always better?

**Answer:** Only if the evidence says so. Here the boosting model and the logistic regression are statistically indistinguishable on the test quarter, and the logistic regression wins in every quarter of the rolling check. With a tie, transparency and calibration decide.

## Slide 10. Targeting value

**Objective:** Translate the model into a business number: how many churners a targeted list reaches.

**Exact content:**

- Active members, Apr-Jun 2024 test quarter
- **76%** of churners reached by contacting the riskiest 10%
- **73%** of those contacted really churn
- **7.6x** better than random targeting
- **91%** reached with the riskiest 20%
- Calibrated: top risk decile predicted 72.4% churn, observed 72.7%.

**Recommended visual:** Cumulative gains curve with four stat cards.

**Number / chart to show:** **76%**, **73%**, **7.6x**, **91%**, chart `outputs/figures/gains_active.png`

**Speaking script (~32 s):** This is what it means for the business. If FreshBasket contacts the riskiest 10% of active members, about 170 people, it reaches 76% of the quarter's churners, and 73% of the people it contacts really are about to churn. That's 7.6 times better than picking at random. The riskiest 20% reaches 91%. And the probabilities can be trusted: in the top risk decile the model predicted 72.4% churn and the actual rate was 72.7%.

**Likely evaluator question:** How would you decide how many members to contact?

**Answer:** From this gains curve plus the cost of the offer and the value of a retained member. The data has no costs, so I report the trade-off rather than claim an optimum; the riskiest 10-20% captures most churners with good precision.

## Slide 11. What drives churn

**Objective:** Explain globally and locally what drives the predictions, and what the ablation shows.

**Exact content:**

- Behaviour and engagement - not demographics or tier
- Attribution for active members: behavioural 40%, engagement 37%, demographics 15%, support 7% - LightGBM SHAP points to the same drivers
- Risk rises with falling transactions, fewer emails and app sessions, and more support tickets
- Tier makes no difference: Silver 9.5%, Gold 9.8%, Platinum 9.4%
- Ablation (ROC-AUC, active): demographics 0.51 -> + behaviour 0.93 -> + engagement 0.96; every member gets their own top reasons

**Recommended visual:** Driver chart: logistic contributions next to LightGBM SHAP, with bullets.

**Number / chart to show:** chart `outputs/figures/drivers.png`

**Speaking script (~47 s):** What drives the predictions? Mostly behaviour and engagement: about 40% of the model's attribution is behavioural and 37% engagement. Falling transactions, fewer emails opened, fewer app sessions, rising support tickets. LightGBM's SHAP values point to the same drivers. Demographics barely matter. On their own they score a ROC-AUC of 0.51 for active members, which is chance, and tier makes no difference: Silver, Gold and Platinum all churn at about 9.5%. In the ablation, adding behaviour lifts it to 0.93 and engagement to 0.96. And because it's a logistic regression, every member on the list comes with their own reasons, like 'support tickets up, transactions falling'. These are associations, not causes.

**Likely evaluator question:** Does that mean support tickets cause churn?

**Answer:** No. Tickets are a warning sign the model uses to rank members. Whether resolving tickets keeps people is a causal question this data can't answer; that's what the controlled test is for.

## Slide 12. Retention what-ifs for the riskiest 20% (model-based)

**Objective:** Run the retention scenario with data-based assumptions and state clearly that it is not causal.

**Exact content:**

- 331 members holding 131 of 140 expected churners; each action closes half the retained-vs-churned behaviour gap
- **-10 pp** engagement outreach (n=331): ~34 fewer expected churners
- **-3.4 pp** proactive support for members with tickets (n=157)
- **-2.7 pp** targeted coupon (n=247)
- **25-100%** of the gap closed: sensitivity shown on the right
- Changes in predicted risk under stated assumptions: they rank what to test, not what the actions will achieve.

**Recommended visual:** Scenario chart (risk before/after for each action; sensitivity to the assumed size) with stat cards.

**Number / chart to show:** **-10 pp**, **-3.4 pp**, **-2.7 pp**, **25-100%**, chart `outputs/figures/retention_scenarios.png`

**Speaking script (~51 s):** Finally, the retention scenario. I took the riskiest 20% of members active in June 2024, 331 people holding 131 of the 140 expected churners, and asked the model what happens to their risk if their behaviour moved part of the way towards a typical retained member. I didn't invent the size of the change. Each action closes half the observed gap between retained and churning members, with 25% and 100% as sensitivity. Engagement outreach lowers predicted risk by about 10 points, roughly 34 fewer expected churners. Coupons and support outreach move it by about 3 points each. But this is the model's association, not a causal effect, so these numbers tell us what to test first, not what will happen.

**Likely evaluator question:** Why should anyone trust these scenario numbers?

**Answer:** They shouldn't be trusted as effects, and I say so. They rank actions under stated, data-based assumptions. The real effect needs a randomised hold-out test, which is my main recommendation.

## Slide 13. Business decision and limitations

**Objective:** State the business decision, the recommendations and what the business must not assume.

**Exact content:**

- What FreshBasket should do
- Monthly risk list of active members; contact the top 10% (~170) with their reasons
- Test engagement outreach and proactive support with a 20-30% random hold-out
- Report active-member churn as the KPI; run win-back separately for lapsed members
- Capture offer cost and member value to size campaigns on expected value
- What not to assume
- The scores and what-ifs are associations, not causal effects of any action
- Signals are unusually clean (likely synthetic data): re-validate on live data
- One given label quarter; earlier labels rebuilt (~0.4% noise)
- Threshold tuned for F1, not costs (no cost data supplied)

**Recommended visual:** Two columns: recommendations and limitations.

**Number / chart to show:** title only

**Speaking script (~66 s):** So what should FreshBasket do? First, replace the blanket offer with a monthly risk list of active members, starting with the top 10%, each with their reasons attached. Second, test engagement outreach and proactive support with a 20 to 30% random hold-out, so we measure what actually works. Third, report churn among active members as the KPI and run a separate win-back programme for lapsed members. And capture offer costs and member value, so campaigns can be sized in money rather than F1. What the business should not assume: that these scores are causal, that accuracy will be this high on live data, because the signals here are unusually clean, or that one test quarter is enough. The model should be re-validated every quarter. Before I wrap up, let me show you the project itself. [DEMO, about 2 minutes: switch to the GitHub repository / README, then run notebooks/06_live_demo.ipynb; come back for the last slide.]

**Likely evaluator question:** How would you deploy and monitor this?

**Answer:** Score members monthly with the saved pipeline and send the list with reasons to the retention team. Every quarter, when new labels arrive, check PR-AUC, calibration and the churn rate per risk band, then retrain. If performance drops or active-member churn shifts, investigate before using the list.

## Slide 14. Takeaways

**Objective:** Summarise the project in three takeaways and the decision.

**Exact content:**

- 1
- Most measured churn is members who had already left; active-member churn is a steady ~10% per quarter
- 2
- A transparent model finds 76% of next quarter's churners in the riskiest 10% of active members
- 3
- Target by behaviour, not tier - and prove the retention actions with a hold-out test

**Recommended visual:** Closing slide with three numbered takeaways.

**Number / chart to show:** title only

**Speaking script (~32 s):** To sum up. Most of the churn FreshBasket sees is members who had already left, while active-member churn is a steady 10% a quarter. A transparent, well-calibrated model finds about three quarters of next quarter's churners in the riskiest 10% of active members, validated on quarters it never saw. And the drivers are behaviour, not tier. My recommendation is a monthly targeted list plus a controlled test of the retention actions. Thanks for watching.

**Likely evaluator question:** What would you do with one more month?

**Answer:** Design the hold-out test with the business, add offer costs so the target size is chosen on expected value, and extend the validation over more quarters as new labels arrive.
