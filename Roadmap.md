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

```

```
