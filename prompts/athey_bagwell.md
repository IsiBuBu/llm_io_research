## Context
You are Firm in a {number_of_players}-firm cartel operating over many periods. The cartel coordinates market shares to maximize joint profits, but each firm has private information about its costs.

## Information & Market Share Rules
- Each period, your marginal cost is either "high" (${cost_types[high]}) or "low" (${cost_types[low]}).
- Cost types are persistent ({persistence_probability} probability your cost stays the same next period).
- Other firms only see your public cost reports.
- **Market shares are allocated based on reports as follows:**
- If one firm reports "low" and all others report "high," the "low" firm gets 100% of the market.
- If multiple firms report "low," they split the market evenly.
- If all firms report "high," they split the market evenly.

## Economic Information
- **Market price:** Fixed at ${market_price} per unit.
- **Your profit is:** (${market_price} - Your True Cost) × Your Allocated Market Share.
- Your total payoff is the Net Present Value (NPV) of profits across all periods, calculated with a **discount factor of ${discount_factor}** per period.

## Current Game State
- **Period:** {current_round}
- **Your true cost this period:** {your_cost_type}
- **History of all firms' past reports:** {all_reports_history_detailed}

## Your Task
Decide whether to report your cost as "high" or "low" for this period to maximize your total long-term profit.

## Output Format
Respond with valid JSON only:
{{"report": "high"}} or {{"report": "low"}}