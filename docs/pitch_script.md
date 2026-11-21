# Boardroom briefing script (about 6.5 minutes, 14 slides + 4 appendix slides)

Suggested split: Speaker 1 (slides 1–4), Speaker 2 (5–8), Speaker 3 (9–11), Speaker 4 (12–14). Appendix slides A1–A4 are for Q&A only.

**1. Cover (20 s)**
Good morning. We are [team name]. Tindihan ni BERB is an online gift and party-supply store whose buyers are mostly resellers. Today we will show you what one year of its orders says about what to bundle, who the best customers are, when to stock up, and who is likely to cancel. We will finish with four concrete actions.

**2. The problem (30 s)**
Right now the owner decides what to stock and which promos to run by instinct. Every sale is recorded, but no one has analysed the records. We turned that into four business questions: what sells together, who the customers are, when sales happen, and who cancels. Each question is tied to a money decision.

**3. Methodology (35 s)**
We followed CRISP-DM. The data is the UCI Online Retail dataset, used as the store's order history. Cleaning kept 72% of rows: we set aside cancellations for their own analysis, removed duplicates, and dropped rows without a customer number instead of inventing IDs. Then we ran 60 logged experiments: association rules, clustering, four classifier families and a regression. Every model was tuned on validation data and tested once on data it had never seen.

**4. Executive dashboard (20 s)**
Here is the one-page view. Four numbers matter: 391,283 clean order lines, 4,334 customers profiled, 64% of revenue from 16% of customers, and a cancel-risk model with a test AUC of 0.77. The next slides walk through each panel.

**5. Seasonality – RQ3 (25 s)**
First, timing. Revenue is flat at roughly half a million a month until August, then climbs every month to a November peak of about 1.14 million. That is the pre-holiday buying season for resellers. The December dip is not real; the data stops on the ninth. So stock decisions must be made by the end of August.

**6. Top products – RQ3 (30 s)**
Which products bring in the money? The Regency 3-tier cakestand is first at about 142,000, then the white hanging heart T-light holder and the jumbo red retrospot bag. The cakestand and jumbo bag are also in our bundle families, so bundles can be built around items that already sell. One forensic note: a paper-craft item looked like the top seller, but all of it came from a single 81,000-unit order that was cancelled 12 minutes later, so we removed it.

**7. Bought together – RQ1 (30 s)**
Now, what sells together. We tuned Apriori over 16 settings and kept 54 strong rules. The pattern is clear: customers buy matching sets. 83% of orders with the pink Regency teacup also include the green one. Lunch boxes, kneeling pads and alarm-clock colours follow the same pattern. FP-Growth found exactly the same rules, so the result is not an accident of the algorithm.

**8. Product network – RQ1 (30 s)**
When we draw the rules as a network, they split into eight separate families. Each colour is a bundle we can sell today: the Regency tea set, the lunch-bag range, alarm clocks, kneeling pads, Christmas paper chains. The two biggest circles, the pink polkadot and red retrospot lunch bags, are the hubs that connect the most products, so they are where cross-sell suggestions should go first.

**9. Customer segments – RQ2 (35 s)**
Next, who the customers are. k-Means on recency, frequency and spend gives four segments, and they are stable: rerunning with ten random seeds gives almost the same groups. The headline: 684 loyal customers produce 64% of revenue. Losing just ten percent of them costs about as much as the entire inactive group spends in a year. Meanwhile 1,636 customers have not bought in about six months: a win-back opportunity.

**10. Cancellation models – RQ4 (30 s)**
Cancellations are common: almost 15% of invoices, and over a third of customers have cancelled at least once. We compared 21 models from four families. All beat the baseline clearly, with AUC around 0.8. We chose Logistic Regression with class balancing: it matches the best AUC and has the highest F1, because it catches far more of the customers who actually cancel. It is also the easiest to explain.

**11. Final test – RQ4 (30 s)**
On 867 customers the model had never seen, it caught 219 of the 311 who cancelled: 70 percent. The trade-off is 158 false alarms, so only about six in ten flagged customers really cancel. That is fine for a cheap action like a confirmation message, but it would be unfair to refuse anyone service on this score. Test AUC is 0.77, a little below cross-validation, which is normal for a test set this size.

**12. Recommendations (40 s)**
So, four actions. One: sell the product families as bundles and put suggestions on the hub items. Two: protect the 684 loyal buyers with VIP perks, and send lapsed buyers a win-back offer in August, before the peak. Three: have peak inventory in by the end of August. Four: use the risk score to send confirmations and stock checks, and add manual approval for very large orders, because the two biggest cancellations were single orders of about 80,000 units.

**13. Ethics and privacy (25 s)**
We also designed for responsible use. The data holds only customer numbers, no names. We used just five behaviour features, in line with the proportionality principle of the Data Privacy Act. The risk score is only for helpful actions, never to deny service. And every number in our report can be traced back: rows removed are counted, every experiment is logged, and results are reproducible.

**14. Close (20 s)**
To sum up: bundle the sets customers already collect, keep the 684 customers who bring in most of the money, stock up by August, and confirm risky orders before they become cancellations. The full report, code and experiment log are in our repository. Thank you; we are happy to take questions.

## Likely questions

- **Why k = 4 when silhouette is highest at k = 2?** Two groups only split active from inactive. k = 3 and 4 have the same silhouette (0.338), and k = 4 separates new low spenders from regular buyers, which needs a different promo.
- **Why not the neural network?** Its CV AUC (0.784) is lower than Logistic Regression (0.797), it is less stable across folds, and it is harder to explain.
- **Isn't Frequency just counting chances to cancel?** Partly, yes. The model flags customers to watch; it does not prove cause. An order-level model is our next step.
- **Why does the top-products chart leave out two orders?** (Appendix A4) Two bulk orders of 74,215 and 80,995 units were cancelled within minutes but the originals stayed in the sales table, adding 245,653 that was never received. Counting them would make a cancelled order the "top seller".
- **How did you choose the Apriori settings?** (Appendix A1) A 16-run grid; support 0.015 and confidence 0.6 give 54 readable rules with mean lift 12.3.
- **Is this real Philippine data?** No. It is a public UK dataset used as a stand-in, which is a stated limitation.
