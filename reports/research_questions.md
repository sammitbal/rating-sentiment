# How the project answers the research questions

**Overall:** ratings and text agree far more often than sentiment tools suggest. Only about 1 in 125 1–2★ or 4–5★ reviews genuinely contradicts its stars. Those contradictions are mostly 4★ reviews reporting a problem, and readers find them *more* helpful. Over time, products with a bad start usually stay rough: they recover less often than chance would produce (Q5).

## Q1. How often does the rating disagree with the written sentiment?

**Rarely: about 0.8% of 1–2★ and 4–5★ reviews (95% interval 0.6–1.0%), roughly 5,000 of 638,532.**

- **The count depends on the tool.** Off-the-shelf sentiment tools flag anywhere from 0.6% to 9.3%, and a simple vote of all three flags 5.6%.
- **Most flags are wrong.** Reading 250 flagged reviews by hand showed most are tool errors or genuinely mixed reviews. The 0.8% estimate corrects for this.
- **Both directions are rare.** About 0.8% of 4–5★ reviews have negative text, and about 0.6% of 1–2★ reviews have positive text.
- **Moderate ratings disagree most.** Flag rates are 8.7% at 2★ and 3.7% at 4★, against 2.5% at 1★ and 0.6% at 5★.
- **The two directions have different causes.**
  - High stars with negative text mostly report a product failure (75%) or a shipping or seller problem (19%).
  - Low stars with positive text are mostly plain praise such as "Great. I love it" (80%), which suggests misclicks or a misread star scale.

## Q2. What factors predict disagreement?

**Above all, a moderate rating. A few review and product traits add to it.**

- **Moderate rating:** a 4★ review has about 5× the odds of negative text that a 5★ review has. A 2★ review has about 3× the odds of positive text that a 1★ review has.
- **Contrast words** ("but", "however") roughly double the odds in both directions.
- **Shipping, delivery or seller mentions** raise the odds, and more so under the strictest definition.
- **The crowd's rating:** disagreement is more likely when the product's average rating sides with the text rather than the stars.
- **For 4–5★ reviews:**
  - More likely: verified purchases (odds ratio 1.45) and more recent years.
  - Less likely: reviews with photos (0.56), reviews with exclamation marks (0.46), and more active reviewers.
- **Predictions are only moderately good** (AUC 0.73–0.79), so most disagreement is idiosyncratic.

## Q3. Are some reviewers systematically inconsistent?

**There is a real but small tendency. There is no distinct group of "inconsistent raters".**

- A review is about twice as likely to be flagged if the same reviewer has another flagged review (3.2% vs 1.6%).
- Chance alone would produce only about 1.1–1.3× (p < 0.001). The chance comparison keeps each reviewer's mix of ratings, lengths and review types fixed.
- The effect runs forward in time: an earlier inconsistent review raises the odds for the next one (odds ratio 1.8).
- In absolute terms it is tiny. Only 26 reviewers repeat it, against about 22 expected by chance.
- For reviewers with 5 or more reviews, how closely their stars track their words matches what chance alone would produce.
- **Limitation:** 93% of reviewers wrote only one review in this category, so this answer rests on a minority of the data.

## Q4. Are these reviews more or less helpful?

**It depends on the direction.**

- **High stars with negative text are more helpful.**
  - With controls, they receive 42% more helpful votes (rate ratio 1.42, CI 1.18–1.71) and have 15% higher odds of getting any vote.
  - Comparing reviews of the same product only, the gap is smaller (+1.4 percentage points, p = 0.055).
  - A plausible reading is that honest criticism inside a positive review is information shoppers value.
- **Low stars with positive text show no reliable difference.**
- **None of this improves prediction.** Adding the consistency measures does not help predict which reviews will be voted helpful.

## Q5. Do products get better over time, so that their negative reviews fade?

**Not as a general pattern. A few products do recover, but fewer than chance alone would produce.**

- **Scope:** 3,444 products with at least 30 reviews over at least a year (3.1% of products, 43% of all reviews).
- **Rough starts do not recover.**
  - 467 products start rough, meaning their first third of reviews averages 3.5★ or less. Only 40 of them improve by a star or more, against about 62 when each product's ratings are shuffled across its own timeline.
  - Rough starters gain +0.01★ on average. The shuffled data predicts +0.24★.
- **Negative reviews rarely disappear.** 9 products go from 30%+ 1–2★ reviews to 5% or less, against 8–21 under shuffling, so there is nothing beyond chance.
- **Real recoveries exist.** A lip gloss base climbs from 2.4★ to 4.1★ over 91 reviews, and a hair dryer from 2.8★ to 4.6★ over 40. They show what a recovery looks like, not how common it is.
- **Early reviews may be kinder than later ones (a hint, not a tested result).**
  - Fewer products start rough than shuffling predicts (467 against 678).
  - The most-reviewed product slips from 4.74★ to 4.52★, and its 1–2★ share rises from 3% to 9%.
  - Category-wide, the mean rating fell from 4.18★ in 2014 to 3.77★ in 2022, and the 1–2★ share rose from 14% to 26%. Different products are reviewed in different years, so this is not the same as products getting worse.
- **The words follow the stars.** Among rough starters the change in rating and the change in text sentiment correlate at ρ = 0.88, so a recovery shows in what people write and not only in the stars.
- **The finding holds under seven different definitions.** In none does the real data show more recovery than chance.

## What this adds to Kung et al. (2024)

- The paper never measured whether ratings and text agree. This project builds that measure and checks it by hand.
- **The paper's 0.969 accuracy is reproduced exactly, but it is an artefact of target leakage.** Its main feature includes the review being predicted. Using earlier reviews only, accuracy is 0.743, barely above the 0.733 you get by always predicting "no helpful vote".

## Caveats to state alongside these answers

- **Who checked the flags:** the 250-review check was done by an AI assistant, not by independent human annotators. `reports/validation_sample.csv` holds 300 reviews ready for human labelling.
- **Scope:** the results come from one category (All_Beauty).
- **Time (Q5):** the product analysis rests on the 3.1% of products with a long history. A change in who reviews cannot be told apart from a change in the product, and "fewer recoveries than chance" partly reflects that product quality persists. The safe reading is no evidence of more recovery than chance.
- **Helpfulness is observational:** vote counts also reflect how Amazon sorts and displays reviews, so the helpfulness results are associations, not causes.
