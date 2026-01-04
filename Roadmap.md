### Project Description: Bid-Ask Consistent Arbitrage-Free Implied Volatility Surfaces with Heston Model Calibration for Enhanced Stability

This project involves developing a sophisticated data processing and analysis tool for option pricing data, specifically focused on detecting, diagnosing, and repairing logical inconsistencies (arbitrage violations) in market-quoted option prices to produce a clean, arbitrage-free implied volatility (IV) surface. Building on this foundation, the tool extends to calibrating the Heston stochastic volatility model to the surface both before and after repairs, demonstrating how the cleaning process improves model fit, parameter stability, and downstream applications like pricing and hedging. The project is implemented in Python using real market data (e.g., SPY options from public sources like CBOE or Yahoo Finance), emphasizing mathematical rigor through proofs, optimization techniques, and evaluation metrics.

At its core, the tool addresses a practical challenge in quantitative finance: real-world option quotes often contain noise, errors, or illiquidity-induced anomalies that violate no-arbitrage principles, leading to unreliable risk models or trading signals. By automating detection and minimal repairs while respecting bid-ask spreads, the system ensures the resulting IV surface is mathematically consistent and usable. The Heston integration adds a unique layer, bridging static data cleaning with dynamic stochastic modeling, allowing you to quantify how unrepaired surfaces lead to unphysical model parameters (e.g., negative volatility of volatility) versus stable, interpretable ones post-repair. This not only differentiates the project from basic volatility smile fitting demos but also showcases your ability to handle stochastic calculus concepts, which is crucial for MFE/MoF applications as a CS student.

The project title could be: "Arbitrage-Free Volatility Surface Repair with Bid-Ask Constraints and Heston Calibration: Ensuring Model Stability in Noisy Market Data." This signals depth in finance math, engineering, and practical quant workflows, making it an impressive portfolio piece for admissions committees who value projects that demonstrate ownership of theorems, code structure, and real-world applicability over novelty for its own sake.

### Key Features

The tool comprises three main modules (Detection, Smoothing, and Repair) from the original blueprint, enhanced with Heston calibration and supporting elements for uniqueness and math depth. Here's a breakdown:

1. **Detection Module (The Detective)**:

   - Reads raw option chain data (calls/puts across strikes and maturities) from CSV files or APIs.
   - Scans for arbitrage violations using discrete checks: monotonicity (call prices decreasing in strike), convexity (second differences non-negative), and calendar spreads (longer-dated options valued appropriately higher).
   - Incorporates bid-ask awareness: Treats quotes as intervals, flagging violations only if they persist across the spread (e.g., no feasible price within bid-ask satisfies rules).
   - Outputs: A diagnostic report summarizing violations (e.g., "23 monotonicity issues in 6 maturities, 7 convexity failures"), with heatmaps visualizing problem areas by strike/maturity for interpretability.

2. **Smoothing Module (The Smoother)**:

   - Fits a smooth IV curve (e.g., using Stochastic Volatility Inspired (SVI) parametrization or splines) to scattered quotes, enabling interpolation/extrapolation to a dense grid of strikes.
   - Handles uncertainty: Uses mid-prices as defaults but weights by liquidity (e.g., tighter spreads get higher fit priority).
   - Outputs: Plots of raw vs. smoothed IV smiles for each maturity, plus extracted risk-neutral densities to preview potential issues (e.g., negative densities indicating convexity problems).

3. **Repair Module (The Fixer)**:

   - Formulates a convex optimization problem to adjust prices minimally while enforcing no-arbitrage constraints and staying within bid-ask bounds where possible.
   - Objective: Minimize squared deviations from original prices; constraints include monotone decreasing calls, convex second differences, and calendar inequalities.
   - Feasibility analysis: If market data is inherently inconsistent (e.g., bid-ask too narrow for rules), reports "infeasible strikes" and applies soft penalties.
   - Outputs: Before/after comparisons (e.g., "47 violations reduced to 0"), adjusted IV surface CSV, and adjustment heatmaps showing where changes were made (e.g., larger tweaks in illiquid wings).

4. **Heston Calibration Extension (The Dynamic Validator)**:

   - Calibrates the Heston model to the IV surface pre- and post-repair, optimizing parameters (initial variance, mean reversion, long-run variance, volatility of variance, correlation) to minimize the difference between model-implied and market IVs.
   - Quantifies improvements: Metrics like root mean squared error (RMSE) of fit, parameter stability (e.g., variance across bootstrapped fits), and physical plausibility (e.g., ensuring Feller condition for positivity: \(2\kappa\theta > \xi^2\)).
   - Stress testing integration: Runs calibration on segmented data (e.g., high-vol vs. low-vol days) to show regime-specific benefits.
   - Outputs: Tables comparing pre/post parameters (e.g., "Pre-repair: \(\xi = -0.2\) (unphysical); Post: \(\xi = 0.15\)"), plots of model vs. market IV, and simple Monte Carlo simulations of paths to visualize hedging stability (e.g., reduced variance in delta-hedge P&L).

5. **Supporting Features for Uniqueness and Evaluation**:
   - **Command-Line Interface (CLI)**: A "surface linter" tool (e.g., `python repair_surface.py --input=spy_options.csv --output=clean_surface.csv --calibrate_heston`) that outputs pass/fail badges, violation summaries, and repaired files.
   - **Visual and Metric Reports**: Heatmaps for adjustments, density plots (pre/post negative probabilities fixed), tail risk metrics (e.g., implied skewness/kurtosis from repaired surface), and a research-style summary (e.g., "Repairs reduced Heston RMSE by 25% on average across 30 days").
   - **Unit Testing and Proof Integration**: Built-in tests asserting theorems (e.g., convexity holds on grid), linked to a mathematical document proving the constraints.
   - **Scalability Options**: Starts with one underlying (SPY) and 4 maturities; expandable to multi-asset if time allows.

These features make the project "out-of-the-box" by emphasizing interpretability (heatmaps, feasibility reports), comparison (pre/post metrics), and extension to stochastic models, going beyond library calls to show your engineering judgment.

### What You Want to Achieve and Why

**Primary Goals**:

- **Achieve a Robust Data Cleaning Pipeline**: Create an automated system that transforms noisy option data into an arbitrage-free IV surface, ensuring it's suitable for real quant applications like risk management or algorithmic trading.
- **Demonstrate Stochastic Model Integration**: By calibrating Heston pre/post, show how data quality directly impacts model reliability—e.g., fixing surfaces leads to better fits and physically sensible parameters, enabling accurate pricing of exotics or volatility derivatives.
- **Quantify Tradeoffs and Insights**: Produce metrics and visuals that reveal market realities (e.g., more violations in volatile regimes) and engineering choices (e.g., bid-ask constraints increase repair minimalism by 15%).
- **Build a Portfolio Artifact**: Generate a 10-15 page report/PDF with proofs, code snippets, results, and discussions, plus a GitHub repo for demos, to strengthen your MFE/MoF apps.

**Why This Project?**

- **Addresses Your Application Weaknesses**: As a CS student, math (especially stochastic calculus) is your gap—this forces you to derive and prove SDE-based concepts (e.g., Heston's affine structure, Dupire links if extended), signaling rigor to adcoms without requiring original research.
- **Real-World Relevance**: Quant desks at firms like Jane Street or Citadel deal with this daily; showing you understand "data quality > fancy models" demonstrates maturity. Heston adds uniqueness, as most student projects don't bridge to dynamics, making yours feel like a mini-thesis.
- **Impresses Committees**: It's scoped (finishable in 4-8 weeks), yet deep—proofs show theorem ownership, optimization shows CS skills, evaluations show analytical thinking. Uniqueness via Heston differentiates it from "fit SVI and plot" demos, while touching core curriculum topics (no-arbitrage, SDEs, calibration).
- **Personal Growth**: You'll gain hands-on with finance math, preparing for grad courses, and the "detective/fixer" aspect makes it engaging to build/research.

In essence, you achieve a tool that not only cleans data but validates it through a stochastic lens, proving why quality matters—ultimately, to avoid "garbage in, garbage out" in quant finance.

### High-Level Plan

This is a phased, iterative plan assuming 4-8 weeks (adjust based on your timeline to Dec 20, 2025—plenty of time post-apps if needed). Focus on real SPY data; use synthetic for quick tests.

1. **Preparation (Week 1)**:

   - Gather data: Download historical SPY option chains (e.g., 30 days from CBOE/Yahoo).
   - Research/Outline Math: Study no-arbitrage theorems, Heston SDE/pricing, and calibration methods. Draft proofs document.
   - Setup Environment: Install Python libs (pandas, scipy, cvxpy, QuantLib for Heston helpers).

2. **Core Build: Detection and Smoothing (Weeks 1-2)**:

   - Implement data loading and violation detection with bid-ask logic.
   - Add smoothing (SVI fit) and basic visuals (IV plots, densities).
   - Test on sample data: Ensure detects ~20-50 violations per chain.

3. **Repair Optimization (Weeks 2-3)**:

   - Build convex optimizer with constraints; integrate bid-ask penalties.
   - Add heatmaps and reports; verify zero violations post-repair.
   - Evaluate basics: Before/after metrics, feasibility analysis.

4. **Heston Extension (Weeks 3-4)**:

   - Implement calibration routine (optimize params to minimize IV RMSE).
   - Run pre/post comparisons; add metrics (RMSE, param checks, MC sims).
   - Integrate with stress testing: Segment data by vol regimes, analyze patterns.

5. **Polish and Evaluation (Weeks 4-5+)**:
   - Build CLI and unit tests tied to theorems.
   - Generate full reports: Visuals, tables, discussions on tradeoffs (e.g., "Heston stability improves in crashes").
   - Expand if time: Add density/tails from blueprint #5 for extra math.
   - Document: Write proofs PDF, repo README with usage/examples.

Throughout: Iterate with real data; document decisions (e.g., "Chose CVXPY for convexity guarantee"). If tight, minimal version: Detection + Repair + Basic Heston on one day.

### Suggested Implementation

# Project Roadmap: Arbitrage-Free IV Surface & Heston Calibration

**Overview:** We will build this project in sequential phases, each focusing on a key component. The plan emphasizes doing everything **individually** (no external collaborators), incorporating essential math to demonstrate understanding (without getting bogged down in theory), and using industry-relevant techniques for practicality. By splitting into clear phases, we can implement and test incrementally to minimize bugs and ensure each part works before moving to the next.

## Phase 1: Data Acquisition and Preparation

- **Gather Real Market Data:** Obtain historical option chain data for a liquid underlying (e.g., SPY). This includes call and put prices across strikes and expiries, along with their bid-ask quotes. You might use sources like CBOE CSV files or Yahoo Finance APIs.
- **Data Parsing:** Write a parser to load the data into Python (using pandas DataFrames). Ensure it captures key fields: underlying price, strike, expiration date, call/put flag, bid, ask, last price (or mid).
- **Initial Data Cleaning:** Remove or flag any obviously bad data (e.g., negative prices, missing fields). Calculate mid-prices (average of bid and ask) for convenience.
- **Structure Data:** Organize options by expiry and type. It might help to separate calls and puts and sort by strike. This will ease checking monotonicity and other conditions.
- **Verify Basic Assumptions:** Do a quick sanity check (e.g., no mid-price below intrinsic value or above underlying price + present value of strike) to catch glaring issues early. These are not the main focus, but ensure your dataset is reasonable.

_Output of Phase 1:_ A cleaned dataset ready for analysis, and code utilities to query option data by strike/maturity. At this point you should be comfortable with the dataset format and have eliminated trivial errors that could propagate bugs later.

## Phase 2: Arbitrage Violation Detection Module

This phase builds **"The Detective"** - a module to scan the data for no-arbitrage rule violations:

- **No-Arbitrage Conditions:** Implement checks for:
- **Monotonicity (Calls):** For each expiry, call prices should decrease as strike increases[\[1\]](https://quant.stackexchange.com/questions/36703/how-to-understand-the-no-call-or-put-spread-arbitrage-condition#:~:text=1,partial%20K%7D%5Cgeq%200). In practice: for strikes . (Similarly, put prices should increase with strike, but you can focus on calls given put-call parity.)
- **Convexity (Calls):** The call price curve vs. strike should be convex[\[1\]](https://quant.stackexchange.com/questions/36703/how-to-understand-the-no-call-or-put-spread-arbitrage-condition#:~:text=1,partial%20K%7D%5Cgeq%200). Discretely, for any three ascending strikes , check . This ensures the implied risk-neutral density is non-negative.
- **Calendar Spread:** Longer maturities should be no cheaper than shorter ones for the same strike (after adjusting for carry). Ensure for (European options)[\[1\]](https://quant.stackexchange.com/questions/36703/how-to-understand-the-no-call-or-put-spread-arbitrage-condition#:~:text=1,partial%20K%7D%5Cgeq%200). In practice, compare forward prices or carry-adjusted options to account for interest/dividends.
- **Bid-Ask Aware Analysis:** Use the bid-ask spreads to make the checks robust:
- For each potential violation, see if it's _definitively_ an arbitrage or just noise. For example, if a high-strike call has a higher _bid_ than a low-strike call's _ask_, that's a clear monotonicity violation even at best prices.
- Only flag a violation if no price within the lower's ask and higher's bid can satisfy the no-arbitrage inequality. This prevents over-reporting due to bid-ask bounce.
- **Diagnostics and Reporting:** For each expiry, count how many strikes violate monotonicity or convexity. Summarize per maturity and overall (e.g., "Maturity X: 5 monotonicity and 2 convexity breaches"). Also check across maturities for calendar spread issues.
- Create a **heatmap or table** of violations: one axis strike, another expiry, marking where violations occur. This visualization will guide the repair phase by highlighting trouble areas.
- Log examples of violations (e.g., "Strike 400 call price (\$5.10) > strike 395 call (\$5.00) at T=1m").
- **Testing:** Create a small synthetic example to verify the detector. For instance, deliberately input a price array that violates convexity and see if it's caught. Ensuring this module works correctly is crucial before moving on.

_Output of Phase 2:_ A functioning detection tool that identifies all static arbitrage violations in the input data (monotonicity, convexity, calendar). You should have quantitative results (counts of issues) and qualitative insight (which areas of the surface are problematic) before attempting any fixes.

## Phase 3: Implied Volatility Smoothing Module

Now build **"The Smoother"** to fit a continuous IV curve through the noisy data points:

- **Purpose of Smoothing:** Market quotes are discrete and can be noisy. Fitting a smooth implied volatility (IV) curve for each expiry helps interpolate missing strikes and provides a sanity-check shape for the smile. A well-chosen parametrization can also enforce no-arbitrage conditions by design[\[2\]](https://arxiv.org/abs/1204.0646#:~:text=,using%20recent%20SPX%20options%20data).
- **Choose a Parametric Fit:** A popular industry choice is the **Stochastic Volatility Inspired (SVI)** parametrization (by Gatheral). SVI expresses implied variance as a simple curve in strike space that can capture smiles. Crucially, one can calibrate SVI to ensure the surface has no arbitrage (no butterfly or calendar arbitrage) if done properly[\[2\]](https://arxiv.org/abs/1204.0646#:~:text=,using%20recent%20SPX%20options%20data). Alternatively, a piecewise spline with smoothing can be used, but ensure it's sufficiently smooth and convex.
- **Implementation Steps:**
- Convert option prices to implied volatilities for each strike and expiry (use mid-prices and an implied vol solver via Black-Scholes formula).
- For each expiry, fit the SVI curve (which has 5 parameters) to the implied vol smile. This is an optimization problem (minimize error between model IV and market IV). If using SVI, you might leverage known calibrations or use least squares.
- Ensure the fit respects shape constraints (SVI has conditions to avoid arbitrage; if using splines, add a penalty for concave sections or adjust knots to enforce convexity).
- Once fitted, use the curve to interpolate IV at a fine grid of strikes (and possibly extrapolate a bit beyond the range to cover tails cautiously).
- **Outputs and Visualizations:**
- Plot each maturity's raw implied vols (scatter) against the fitted SVI/spline smile (smooth line). This shows how well the smoother captures the market and highlights outliers.
- Derive the **risk-neutral density** for each expiry from the second derivative of the call price (you can get it from the second derivative of implied vol via Breeden-Litzenberger relationship). Plot these densities to check for any negative values. Ideally, a well-fitted, arbitrage-free smile gives non-negative densities.
- If any negative density regions appear in the smooth fit, it indicates arbitrage (or a poor fit) and needs attention.
- **Refinement:** Iterate on the fitting if needed - e.g., weight points by inverse bid-ask spread (so liquid points count more in error minimization), or exclude obvious bad points temporarily for a stable fit.
- **Intermediate Validation:** The smoothed surface itself should respect monotonicity and convexity by construction if possible. This provides a benchmark "ideal" surface to compare the raw data against.

_Output of Phase 3:_ A set of smooth implied volatility curves for each expiry (and an implied volatility surface) that closely fits the market data but is more regular. This will be used both for visualization and as a guide in the repair process. You should also have insight into the overall shape of the volatility surface and where the market data deviates from a clean shape.

## Phase 4: Arbitrage Repair Module

This phase develops **"The Fixer"** - a module to adjust the original prices minimally to remove arbitrage:

- **Formulate the Optimization:** We will cast the price adjustment as a constrained optimization problem. Based on research, a **linear programming (LP)** or convex optimization approach is effective[\[3\]](https://arxiv.org/pdf/2008.09454#:~:text=changes,that%20the%20proposed%20arbitrage%20repair):
- **Variables:** The adjusted option prices for each quoted contract (or equivalently adjusted implied vols).
- **Objective:** Minimize the total adjustment from the original prices. For example, minimize the sum of squared differences or absolute differences. This keeps changes as small as possible, maintaining data integrity[\[3\]](https://arxiv.org/pdf/2008.09454#:~:text=changes,that%20the%20proposed%20arbitrage%20repair).
- **Constraints:** Impose all no-arbitrage conditions on the adjusted prices:
  - Monotonicity: for all strikes[\[1\]](https://quant.stackexchange.com/questions/36703/how-to-understand-the-no-call-or-put-spread-arbitrage-condition#:~:text=1,partial%20K%7D%5Cgeq%200).
  - Convexity (Butterfly): for all adjacent strikes[\[1\]](https://quant.stackexchange.com/questions/36703/how-to-understand-the-no-call-or-put-spread-arbitrage-condition#:~:text=1,partial%20K%7D%5Cgeq%200).
  - Calendar: for all per strike[\[1\]](https://quant.stackexchange.com/questions/36703/how-to-understand-the-no-call-or-put-spread-arbitrage-condition#:~:text=1,partial%20K%7D%5Cgeq%200).
  - **Bid-Ask bounds:** For each option, constrain . This ensures you don't move the price outside the quoted spread - crucial for respecting market viability[\[3\]](https://arxiv.org/pdf/2008.09454#:~:text=changes,that%20the%20proposed%20arbitrage%20repair).
- This is a convex optimization (linear constraints; quadratic or linear objective). By solving it, we get the smallest modifications needed to eliminate arbitrage.
- **Use Appropriate Tools:** Implement the optimization using Python libraries like **CVXPy** (for a high-level convex problem definition) or **CVXOPT**. These can handle the size of typical option chains efficiently. If the problem becomes large (many strikes and expiries), the solver should still cope since the constraints are mostly linear inequalities.
- **Feasibility and Adjustments:** It's possible some sets of quotes are so inconsistent that even within bid-ask bounds no static-arbitrage solution exists. In such cases:
- Identify the problematic points (e.g., maybe an option's bid is too high relative to another's ask, making it impossible to satisfy monotonicity).
- You might need to relax constraints slightly (allow a tiny violation) or widen the bounds (perhaps treat deep out-of-money illiquid quotes with a bit more flexibility).
- However, in most cases a feasible solution exists if you include the bid-ask range.
- **Solution and Minimality:** The result will adjust only the quotes that are necessary[\[4\]](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3197284#:~:text=European%20option%20quotes,leave%20many%20quotes%20unchanged%2C%20as). Many liquid, already-consistent prices should remain unchanged (the optimization naturally prefers not to move points unless needed)[\[4\]](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3197284#:~:text=European%20option%20quotes,leave%20many%20quotes%20unchanged%2C%20as). This is good for preserving the integrity of data and is aligned with research findings that an optimal "arbitrage filter" leaves most quotes untouched[\[4\]](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3197284#:~:text=European%20option%20quotes,leave%20many%20quotes%20unchanged%2C%20as).
- **Outputs and Validation:**
- Generate a **before vs. after** report: e.g., "Monotonicity violations: 23 -> 0, Convexity violations: 10 -> 0, Calendar: 5 -> 0 after repair."
- Provide a table of the adjustments made: list options that changed price, by how much (absolute and percentage of bid-ask spread). This shows that changes were minimal. If changes cluster in certain strikes or maturities (often in illiquid far OTM options), highlight that observation.
- Visualize the **adjusted IV surface** (overlay with original). Show that the new surface is smooth and free of arbitrage. Perhaps plot the difference in implied vol for each point (heatmap of adjustments).
- Re-run the detection module on the repaired prices to double-check that all arbitrage conditions are indeed satisfied now.
- **Testing on Subsets:** Before applying to the whole dataset, test the optimization on a smaller subset (e.g., one expiry at a time or a simplified scenario) to ensure the formulation is correct. For example, create a mini option chain with a known violation and see that the solver fixes it as expected.
- **Iterate if Needed:** If the solver adjustments seem too large in some area, reconsider the weighting (e.g., maybe you weighted mid-price errors equally, but a very illiquid option might swing wildly - you could weight adjustments by inverse spread to penalize moving liquid quotes more).

_Output of Phase 4:_ A cleaned, arbitrage-free set of option prices (and implied vols). This dataset respects all static no-arbitrage criteria while remaining as close as possible to the original market quotes (within their bid-ask limits). You should gain confidence that the surface is now logical and can be used for modeling without producing nonsense (like negative densities or exploitable arbitrage).

## Phase 5: Heston Model Calibration Module

With a reliable volatility surface, we integrate the **Heston stochastic volatility model** to quantify improvements:

- **Model Setup:** The Heston model (with parameters ) has a known closed-form solution for European option prices via Fourier transform. You can use the formula from Heston (1993) or an implementation from QuantLib/PyVal (to avoid coding the complex integration from scratch, unless you want to). Ensure you understand the parameters:
- : initial variance, : long-run variance, : mean reversion rate, (sigma): vol-of-vol, : correlation between asset and variance.
- **Calibration Method:** Set up a calibration routine that takes a volatility surface and finds the Heston parameters that best reproduce it. Typically, this is done by minimizing the sum of squared differences between market option prices (or implied vols) and model prices[\[5\]](https://en.wikipedia.org/wiki/Heston_model#:~:text=):
- Define an error function over all strikes/maturities in the dataset.
- Use a numerical optimizer (e.g., least-squares algorithms like Levenberg-Marquardt or stochastic methods) to find the parameter vector that minimizes .
- Because this is a non-convex problem, try multiple initial seeds or use a global search technique to avoid local minima.
- Impose reasonable bounds (e.g., ; also a soft constraint to satisfy the **Feller condition** for realism[\[6\]](https://en.wikipedia.org/wiki/Heston_model#:~:text=If%20the%20parameters%20obey%20the,2)).
- **Pre- vs Post-Repair Calibration:** Perform the calibration twice:
- **Using the Raw Surface:** Input the original (potentially arbitrage-tainted) market IVs. Record the best-fit Heston parameters and the calibration error (e.g., RMSE).
- **Using the Repaired Surface:** Input the cleaned, arbitrage-free IVs and record the results.
- **Compare Results:**
- **Fit Error:** Expect the post-repair surface to fit much better. Quantify the improvement (e.g., RMSE reduced from 2% to 1% implied vol error).
- **Parameter Stability:** Check if the calibrated parameters changed drastically. Often, arbitrage or noise can lead to weird fits (e.g., extremely high vol-of-vol or correlation hitting -1 or 1 bounds). The raw data calibration might result in edge-case parameters (trying to "explain" arbitrage noise), whereas the cleaned data should yield more **sensible values**[\[7\]](https://arxiv.org/pdf/2008.09454#:~:text=calibration%20more%20robust%2C%20as%20supported,First%2C%20there%20are)[\[8\]](https://arxiv.org/pdf/2008.09454#:~:text=of%20model%20calibration%20by%20the,benefit%20of%20repairing%20data%20by). For example, maybe with raw data the optimizer drives to 0 or to an extreme to fit inconsistent smiles, or violates Feller condition, whereas with clean data you get moderate within bounds and Feller holds.
- **Physical Plausibility:** Verify conditions like Feller on both sets. If the raw-fit violates it (common if data had arbitrage leading to negative implied densities), note that. The post-fit should ideally satisfy [\[6\]](https://en.wikipedia.org/wiki/Heston_model#:~:text=If%20the%20parameters%20obey%20the,2) (no zero variance issues).
- **Robustness:** If time permits, do a small **bootstrap** or resampling: slightly perturb the input vol surface and recalibrate to see variation in parameters. Likely, the arbitrage-free surface gives more stable parameter estimates (less sensitive to tiny changes) than the raw surface. This demonstrates improved robustness.
- **Additional Analysis:**
- Simulate price paths or **hedging PnL:** Using the two sets of parameters, simulate the Heston model to see how option pricing or delta-hedging might differ. This could illustrate that a bad calibration might misprice exotic options or lead to poor hedges, whereas a stable calibration does better. (This is an optional deep-dive if you want to showcase dynamic implications.)
- Highlight any parameter that was outright non-physical in the raw fit (e.g., negative or outside \[-1,1\] if that happened) and how the repair fixed it.
- **Reference to Literature:** Note that empirical studies find removing arbitrage significantly improves model calibration. For instance, one study showed that cleaning data reduced Heston calibration error by 70-95% in noisy scenarios[\[8\]](https://arxiv.org/pdf/2008.09454#:~:text=of%20model%20calibration%20by%20the,benefit%20of%20repairing%20data%20by). Your results can echo this - they add credibility that your cleaning process has a meaningful impact.

_Output of Phase 5:_ A clear comparison of Heston model calibration on dirty vs. clean data. Deliverables include a table of calibrated parameters (pre/post repair), the calibration error metrics, and a brief discussion of how the cleaning impacted the model fit. This demonstrates to a reader (or admissions committee) that you not only cleaned the data but also tied it to better performance in an advanced quantitative model.

## Phase 6: Integration, Testing, and Documentation

With all core components ready, focus on **integration** and final touches:

- **Pipeline Integration:** Combine everything into a single pipeline or script. For example, create a master script or Jupyter notebook that:
- Reads in the raw data (Phase 1).
- Runs the detection module and prints/saves the diagnostic report (Phase 2).
- Runs the smoothing module and plots the smooth curves (Phase 3).
- Executes the repair optimization and outputs the adjusted prices/IVs (Phase 4).
- Runs the Heston calibration on both sets and compiles the comparison results (Phase 5).
- Allow for command-line arguments or config flags (e.g., --skip-heston if one only wants to clean data). This makes the tool flexible.
- **Thorough Testing:** Go through several test cycles:
- Unit tests for the math constraints: e.g., after repair, automatically check for all i, etc., to be 100% sure the solver honored them.
- Test the edge cases: data with no arbitrage at all (your repair should then change nothing significant), or extremely noisy data (does the solver still find a solution?).
- Ensure the pipeline handles real-world quirks (like if some strikes are missing bids or asks, or if there are American options in data - though you'd likely stick to European style for theoretical consistency).
- **Performance Tuning:** If the dataset is large, measure runtime of the LP solver and the Heston calibration. Optimize if needed (e.g., limit strikes to a reasonable range, or reduce the density of interpolation if it's too slow).
- **Documentation & Math Proofs:** Start writing the project report alongside coding:
- Prepare a 10-15 page write-up explaining the project's purpose, methodology, and results. Use the structure of this roadmap as a basis for sections.
- Include the **mathematical foundations**: formal statements of the no-arbitrage theorems and a sketch of proofs (as in the LaTeX snippet). This demonstrates your grasp of the theory behind the code. For example, prove that decreasing and convex implies a valid density[\[9\]](https://quant.stackexchange.com/questions/36703/how-to-understand-the-no-call-or-put-spread-arbitrage-condition#:~:text=probability%20measure%20associated%20to%20the,partial%20K%7D%28K%2CT%29%20%5Cleq%200)[\[10\]](https://quant.stackexchange.com/questions/36703/how-to-understand-the-no-call-or-put-spread-arbitrage-condition#:~:text=Similarly%2C%20%24%24%20%5Cfrac%7B%5Cpartial,2%7D%28K%2CT%29%20%5Cgeq%200). Include the Feller condition derivation to show knowledge of stochastic calc.
- The LaTeX document given in the overview can be expanded with references and made an appendix to the report. This will strongly reinforce the "math first" impression, without requiring new theoretical contributions - just solid understanding.
- **Industry Relevance Emphasis:** In the documentation and possibly README, highlight how each component is used in practice:
- Mention that trading firms do perform such "arbitrage filtering" on quotes before using models[\[4\]](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3197284#:~:text=European%20option%20quotes,leave%20many%20quotes%20unchanged%2C%20as), so your tool mirrors a real quant workflow.
- Note that SVI is a standard in banks for smoothing volatility surfaces[\[2\]](https://arxiv.org/abs/1204.0646#:~:text=,using%20recent%20SPX%20options%20data).
- Heston (while an older model) is still a common textbook model for stochastic volatility - calibrating it shows you can bridge from data cleaning to model deployment.
- By doing all this in Python with clear structure, you show engineering skills too (which MFE programs appreciate in projects).
- **Final Presentation:** Ensure you have clear visuals (embedded graphs of smiles, surfaces, calibration fits). These should be included in your report or as separate figures. Possibly prepare a short slide deck if you might be asked to present the project in an interview or as part of the application.
- **Repository:** Clean up your code, add comments and a license (if open-sourcing). A well-structured GitHub repo with instructions to run the code will make a good impression. Include example data (if license permits) or synthetic data for demonstration.

_Output of Phase 6:_ A polished end-to-end project deliverable. This includes the working codebase, a comprehensive report with both implementation details and mathematical rigor, and example results demonstrating the tool's effectiveness. By this phase, you will have tested the system enough to be confident in its correctness and robustness. The project will be in a state where you could show it to someone (like an admissions committee or industry professional) and it will clearly communicate the depth and practical value of what you built.

By following this roadmap, you ensure a logical progression: **understand the data and theory → detect issues → smooth → repair → validate with a model → present findings**. Each phase builds on the previous, reducing complexity and bugs. This structured approach will not only result in a successful project implementation but also highlight your ability to tackle a complex quantitative finance problem from theory through to practice, all on your own. Good luck with the build - upon completion, you'll have a standout project that combines programming, finance theory, and mathematical modeling!



```

```
