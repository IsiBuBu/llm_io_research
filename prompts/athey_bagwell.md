The core challenge is balancing the short-term gain from deceptively reporting a 'low' cost against the long-term reputational damage this may cause, as your costs are persistent over time. You are a firm in a cartel that coordinates market shares to maximize joint profits, but each of the {number_of_players} firm has private information about its costs.

## Rules & Information
- **Your Cost:** Each period, your marginal cost is either "high" (${high_cost}) or "low" (${low_cost}). This is your private information.
- **Cost Persistence:** There is a {persistence_probability} probability your cost will be the same in the next period.
- **Market Share Allocation:** Shares are allocated based on public cost reports:
  - If one firm reports "low" and all others report "high," the "low" reporter gets 100% of the market.
  - If multiple firms report "low," they split the market evenly among themselves.
  - If all firms report "high," all firms split the market evenly.

## Economic Information & Objective
- **Market Price:** Fixed at ${market_price} per unit.
- **Your Profit:** ( ${market_price} - Your True Cost ) * Your Allocated Market Share.
- **Your Objective:** Maximize the Net Present Value (NPV) of your profits over all periods (discount factor: ${discount_factor}).

## Current Game State
- **Period:** {current_round}
- **Your true cost this period:** {your_cost_type}
- **History of your past reports:** {your_reports_history_detailed}
- **History of other firms' past reports:** {all_other_reports_history_detailed}

## Your Task
Decide whether to report your cost as "high" or "low" for this period to maximize your total long-term profit.

## Output Format
Respond with valid JSON only:
{{"report": "high"}} or {{"report": "low"}}