## Rules & Economic Information
- **Market:** You are one of {number_of_players} firms in a circular market of {market_size} customers.
- **Costs:** Your marginal cost is ${marginal_cost}. Your fixed cost is ${fixed_cost}.
- **Customer Behavior:** Customers buy from the firm with the lowest total cost (price + transportation cost). Transportation cost is ${transport_cost} per unit of distance. The maximum price any customer will pay is ${v}.
- **Objective:** Your profit is (Your Price - ${marginal_cost}) * Quantity Sold - ${fixed_cost}.

## Your Task
Choose your price to maximize your profit. All firms choose their prices simultaneously.

## Output Format
Respond with valid JSON only:
{{"price": <number>}}