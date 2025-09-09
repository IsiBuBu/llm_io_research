## Context
You are bidding in a {number_of_players}-winner-take-all price auction. The firm with the lowest price wins the entire market. All losing firms earn zero profit.

## Information Structure
You know your own marginal cost, but your competitors' costs are unknown. They are independently drawn from a normal distribution with a mean of ${rival_cost_mean} and a standard deviation of ${rival_cost_std}.

## Economic Information
- **Your marginal cost:** ${your_cost} per unit.
- **Market demand at a given price `p` is `D(p) = {demand_intercept} - p`**.
- **Your profit if you win is:** (Your Price - ${your_cost}) × ({demand_intercept} - Your Price).
- **Your profit if you lose is:** $0.

## Your Task
Choose your price to maximize your expected profit, considering the uncertainty about your rivals' costs.

## Output Format
```json
{"price": <number>}