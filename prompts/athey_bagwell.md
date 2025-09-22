## Your Strategic Problem: Managing Reputation with Persistent Costs

You are in a cartel with **{number_of_players}** firms that allocates a market of size **{market_size}** to the lowest-cost firm. The market price is fixed at **${market_price}**. Your costs are persistent (**{persistence_probability}** chance of being the same next period), making your reputation for honesty critical. Your challenge is to decide whether to lie or tell the truth about your costs.

### Key Information & Market Dynamics:

* **The Revelation Dilemma:** The cartel's efficiency depends on truthful cost reports. This creates a powerful incentive for a high-cost firm to lie and report a low cost to steal the market.
* **Incentive-Compatible Design:** To counteract this, the collusive scheme links today's report to tomorrow's market share.
    * **Short-Term Gain from Deception:** If your true cost is "high" (${high_cost}) and you deceptively report "low" (${low_cost}), you can win 100% of the market for a large immediate profit.
    * **Long-Term Reputational Cost:** Because costs are persistent, lying damages your credibility. The scheme punishes firms for past "low" cost reports by assigning them smaller market shares in the future.

### Your Task:

Your true cost this period is **{your_cost_type}**. Decide whether to report "high" or "low". Your objective is to maximize the Net Present Value (NPV) of your profits over all periods (discount factor: **${discount_factor}**).

### Current Game State:

* **Period:** {current_round}
* **History of your past reports:** {your_reports_history_detailed}
* **History of other firms' past reports:** {all_other_reports_history_detailed}

### Output Format:

Respond with valid JSON only:
`{{"report": "high" | "low"}}`