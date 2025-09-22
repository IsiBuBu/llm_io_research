## Your Strategic Problem: Balancing Margin vs. Probability of Winning

You are in a winner-take-all auction against **{number_of_competitors}** rivals. You know your own cost but have only probabilistic beliefs about your rivals' costs. Your challenge is to set a price that optimally balances the trade-off between the size of your potential profit and your probability of winning.

### Key Information & Market Dynamics:

* **Information Asymmetry:** Your marginal cost of **${your_cost}** is your private information. You only know that your rivals' costs are drawn from a **{rival_cost_distribution}** distribution with a mean of **${rival_cost_mean}** and a standard deviation of **${rival_cost_std}**.
* **Market Demand:** The market demand if you win is `Quantity = {demand_intercept} - {demand_slope} * Your_Price`.
* **The Pricing Trade-Off:** Your decision is a balance between two competing goals:
    * **Profit Margin:** The higher you set your price, the more profit you make *if* you win.
    * **Probability of Winning:** The lower you set your price, the higher your probability of undercutting all your rivals and winning the market.

### Your Task:

Choose the single price that you believe will maximize your *expected* profit, perfectly balancing the trade-off between the size of the prize and your chance of winning it.

### Output Format:

Respond with valid JSON only:
`{{"price": <number>}}`