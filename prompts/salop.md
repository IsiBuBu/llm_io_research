## Your Strategic Problem: Pricing for Two Markets at Once

You are one of {number_of_players} firms competing in a circular market with a circumference of **{circumference}** and a total of **{market_size}** customers. Your core challenge is to set a single price that balances the exploitation of your "local monopoly" power over nearby customers against the need to capture market share from direct competitors.

### Key Information & Market Dynamics:

* **Monopoly Segment:** For customers closest to you, your only competition is an "outside good." You can charge them a higher price, but their total cost is your price plus a travel cost. This travel cost is calculated as the **transportation cost of ${transport_cost} per unit of distance** multiplied by how far they are from you. If this total cost exceeds their effective **reservation price of ${reservation_price}**, they will buy nothing from your industry. (This reservation price is derived from a brand utility of **${max_brand_utility}** and an outside good surplus of **${outside_good_surplus}**).
* **Competitive Segment:** At the edges of your market, your territory overlaps with your neighbors. To win these customers, you must offer a better deal (your price + their travel cost) than your rivals. Lowering your price here directly steals their market share.
* **The Kink:** Your demand curve has a "kink" at the price where your market transitions from purely monopolistic to competitive. Your pricing decision revolves around this point.

### Your Task:

Choose your price for this period. Your objective is to maximize your total profit, calculated as `(Your Price - ${marginal_cost}) * Quantity Sold - ${fixed_cost}`.

### Output Format:

Respond with valid JSON only:
`{{"price": <number>}}`