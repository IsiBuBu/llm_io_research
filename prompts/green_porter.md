## Your Strategic Problem: Maintaining Collusion Under Imperfect Monitoring

You are in a cartel with **{number_of_players}** firms. Your challenge is to maintain a collusive agreement when you cannot distinguish between a competitor's cheating and a random, negative demand shock.

### Key Information & Market Dynamics:

* **Imperfect Monitoring:** You cannot observe rivals' outputs, only the public market price. The price is determined by the formula: `Price = {base_demand} - {demand_slope} * (Total Industry Quantity) + Demand Shock`. The hidden demand shock is drawn from a **{demand_shock_distribution}** distribution with a mean of **{demand_shock_mean}** and a standard deviation of **{demand_shock_std}**.
* **The Dilemma of Punishment:** The cartel uses a "trigger price" of **${trigger_price}** to enforce discipline. If the market price falls below this, a costly "price war" (producing the Cournot quantity of **{cournot_quantity}**) is triggered for **{punishment_duration}** periods.
* **Credible Threat:** Because of the random shocks, price wars will inevitably be triggered even when no one has cheated. This is a necessary cost of maintaining a credible deterrent.

### Your Task:

Choose your quantity for this period. The agreed-upon collusive quantity is **{collusive_quantity}**. Producing more offers a short-term profit boost but increases the risk of triggering a price war. Your marginal cost is **${marginal_cost}**. Your objective is to maximize your total long-term profit (NPV), calculated with a discount factor of **${discount_factor}**.

### Current Game State:

* **Period:** {current_round}
* **Current Market State:** {current_market_state}
* **Market prices from the last {price_history_length} periods:** {price_history}

### Output Format:

Respond with valid JSON only:
`{{"quantity": <number>}}`