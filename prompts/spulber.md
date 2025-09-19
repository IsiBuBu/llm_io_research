## Rules & Information Structure
- **Auction Type:** You are bidding in a winner-take-all price auction against {number_of_competitors} rivals. The lowest price wins the entire market. **If multiple firms submit the same lowest price, they split the market evenly.**
- **Information:** You know your marginal cost. Your rivals' costs are unknown and are drawn from a normal distribution (mean: ${rival_cost_mean}, std dev: ${rival_cost_std}).

## Economic Information & Objective
- **Your Marginal Cost:** ${your_cost} per unit.
- **Market Demand:** D(p) = {demand_intercept} - p.
- **Your Profit:** If you win, profit is (Your Price - ${your_cost}) * ({demand_intercept} - Your Price). If you lose, profit is $0.

## Your Task
Choose your price to maximize your expected profit.

## Output Format
Respond with valid JSON only:
{{"price": <number>}}