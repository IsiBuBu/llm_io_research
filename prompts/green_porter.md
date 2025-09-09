## Context
You are Firm in a {number_of_players}-firm oligopoly producing a homogeneous product over many periods. The market price is affected by random demand shocks, making it difficult to know if a low price is due to low demand or a competitor cheating.

## Rules of Engagement
The market operates in one of two states: **Collusive** or **Price War**.
- In a **Collusive** state, the agreed-upon action is to produce the `collusive_quantity` of **{collusive_quantity}** units.
- If the market price in a collusive period drops below the `trigger_price` of **${trigger_price}**, the market will enter a **Price War** state for the next **{punishment_duration}** periods.
- In a **Price War** state, all firms revert to noncooperative behavior.

## Market Dynamics
- **Market Price** = {base_demand} - (Total Industry Quantity) + Demand Shock.
- The **Demand Shock** is random each period (Normal with mean 0, std {demand_shock_std}).

## Economic Information
- **Your marginal cost:** ${marginal_cost} per unit.
- Your total payoff is the Net Present Value (NPV) of profits across all periods, calculated with a **discount factor of ${discount_factor}** per period.

## Current Market Information
- **Period:** {current_round}
- **Current Market State:** {current_market_state}
- **Previous market prices:** {price_history}

## Your Task
Choose your quantity for this period to maximize your total long-term profit, balancing the immediate gains from producing more against the risk of triggering a future price war.

## Output Format
```json
{"quantity": <number>}