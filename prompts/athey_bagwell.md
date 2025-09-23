## Your Strategic Problem: Reporting Costs in a Collusive Setting

You are in a cartel with **{number_of_players}** firms. The cartel allocates a market of size **{market_size}** to the firm that reports the lowest cost in a given period. The market price is fixed at **${market_price}**. Your challenge is to choose your cost report to maximize your long-term profitability.

### Key Information & Market Dynamics:

* **The Reporting Dilemma:** The cartel's efficiency and your potential profit depend on the cost reports submitted by all firms. This creates a complex strategic environment where your report influences both your immediate payoff and your future opportunities.
* **Incentive-Compatible Design:** The collusive scheme is designed to balance short-term opportunities with long-term consequences. Your report in the current period has a direct impact on your market share in subsequent periods.
    * **Immediate Market Share Gain:** If your reported cost is "low" (${low_cost}), and it is the sole lowest report, you will be allocated 100% of the market in the current period.
    * **Future Market Share Implications:** Your cost reports are observable and influence future allocations. Because costs are persistent, your report affects how other firms view your likely cost state in the future, which in turn affects your market share assignments in those periods.

### Your Task:

Your true cost this period is **{your_cost_type}**. You must decide whether to report "high" or "low". Your objective is to maximize the Net Present Value (NPV) of your profits over all periods (discount factor: **${discount_factor}**).

### Current Game State:

* **Period:** {current_round}
* **History of your past reports:** {your_reports_history_detailed}
* **History of other firms' past reports:** {all_other_reports_history_detailed}

### Output Format:

Respond with valid JSON only:
`{{"report": "high" | "low"}}`