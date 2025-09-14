## Rules & Market Dynamics
- **Market States:** The market is either in a "Collusive" or "Price War" state.
- **Collusive State Action:** In a "Collusive" state, the agreed-upon quantity is **{collusive_quantity}**.
- **Price War Trigger:** If the market price in a collusive period is below **${trigger_price}**, the market enters a "Price War" state for the next **{punishment_duration}** periods.
- **Price War State Action:** In a "Price War" state, firms revert to noncooperative behavior.
- **Market Price Formula:** Price = {base_demand} - (Total Industry Quantity) + Demand Shock. The Demand Shock is random.

## Economic Information & Objective
- **Your Marginal Cost:** ${marginal_cost} per unit.
- **Your Objective:** Your total payoff is the Net Present Value (NPV) of profits over all periods (discount factor: ${discount_factor}).

## Current Market Information
- **Period:** {current_round}
- **Current Market State:** {current_market_state}
- **Previous market prices:** {price_history}

## Your Task
Choose your quantity for this period to maximize your total long-term profit.

## Output Format
Respond with valid JSON only:
{{"quantity": <number>}}