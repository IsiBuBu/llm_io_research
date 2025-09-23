## Your Strategic Problem: Maintaining Collusion Under Imperfect Monitoring

You are in a cartel with **{number_of_players}** firms. Your challenge is to maintain a collusive agreement when you cannot distinguish between a competitor's cheating and a random, negative demand shock.

### Rules of Engagement:

* **Two Market States:** The market operates in one of two states: **Collusive** or **Price War (Reversionary)**.
* **Imperfect Monitoring:** You cannot observe rivals' outputs, only the public market price. The price is determined by: `Price = {base_demand} - {demand_slope} * (Total Industry Quantity) + Demand Shock`. The hidden demand shock is from a **{demand_shock_distribution}** distribution (mean **{demand_shock_mean}**, std dev **{demand_shock_std}**).
* **Triggering a Price War:** If the market price in a **Collusive** period drops below the trigger price of **${trigger_price}**, the market enters a **Price War** state for the next **{punishment_duration}** periods.
* **Price War Behavior:** During a **Price War**, all firms are expected to revert to the noncooperative Cournot quantity of **{cournot_quantity}**.

### Your Task:

Choose your action for this period. Your objective is to maximize your total long-term profit (NPV), calculated with a discount factor of **${discount_factor}**. Your marginal cost is **${marginal_cost}**.

If the market is Collusive, you must choose between two actions:

* **Cooperate:** Produce the agreed-upon collusive quantity of **{collusive_quantity}**. This maximizes long-term group profit but offers a lower immediate payoff.

* **Defect:** Produce the noncooperative Cournot quantity of **{cournot_quantity}**. This offers a short-term profit boost but significantly increases the risk of triggering a price war.

If the market is in a **Price War**, you are in a punishment phase and your action is fixed at the Cournot quantity.

### Current Game State:

* **Period:** {current_round}
* **Current State:** {current_market_state}
* **Market prices from the last {price_history_length} periods:** {price_history}

### Output Format:

Respond with valid JSON only:
`{{"action": "Cooperate" | "Defect"}}`