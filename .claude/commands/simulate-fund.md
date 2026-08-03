---
description: Run Monte Carlo simulation of a Safer fund portfolio to model returns, risk, and LP outcomes
allowed-tools: Read, Write, Edit, Bash
---

Run the Safer Fund Monte Carlo Simulator — James Thomason's portfolio-level financial modeling engine that simulates thousands of fund iterations to model return distributions, risk metrics, and LP outcomes for Safer-based venture studio funds.

The simulator is located at `tools/safer_monte_carlo.py`. It produces detailed statistical analysis with charts showing TVPI distributions, IRR analysis, DPI velocity, return attribution, and outcome distributions.

## SETUP

First, ensure dependencies are installed:

```bash
pip install numpy pandas matplotlib --break-system-packages 2>/dev/null || true
```

## USAGE MODES

### 1. Main Simulation (Default)

Run the full Monte Carlo simulation with configurable parameters:

```bash
python tools/safer_monte_carlo.py [options]
```

### 2. Scenario Presets (Book Chapter Analysis)

Run pre-configured scenarios from the VC 2.0 book:

```bash
python tools/safer_monte_carlo.py --scenario <preset>
```

**Available presets:**

| Preset | Description |
|--------|-------------|
| `standard` | Traditional VC assumptions (high failure rate, power law returns) |
| `revenue-focused` | Traction-selected companies ($350K+ ARR, lower failure rate) |
| `high-yield` | Optimized Safer terms for maximizing revenue share returns |
| `max-revenue` | Pure yield play ($500K+ ARR, 10% revenue share) |
| `all-scenarios` | Run all four presets and generate comparison charts |

### 3. Sensitivity Analysis

Run heatmap analysis across parameter ranges:

```bash
# Basic sensitivity (3 heatmaps: structure, skill, macro)
python tools/safer_monte_carlo.py --sensitivity

# Full sensitivity (8 heatmaps: + repurchase, valuation cap, portfolio concentration, quality, growth/failure)
python tools/safer_monte_carlo.py --sensitivity-full
```

### 4. Unicorn-Free Stress Test

Compare fund performance with and without power-law outlier exits:

```bash
python tools/safer_monte_carlo.py --unicorn-free
```

This caps exit valuations at $50M to show the Safer instrument's base-hit performance through revenue participation alone, without relying on unicorn outcomes.

## PARAMETER REFERENCE

### Simulation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--iterations` / `-n` | 10,000 | Number of Monte Carlo iterations |
| `--seed` | random | Random seed for reproducibility |
| `--years` | 10 | Fund life in years |
| `--quick` | off | Quick mode with fewer iterations (1,000) |

### Fund Structure

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--portfolio-size` | 25 | Number of portfolio companies |
| `--fund-size` | $25,000,000 | Total fund size |
| `--investment` | $750,000 | Investment per company |
| `--mgmt-fee` | $500,000 | Management fee per year |
| `--investment-period` | 12 | Investment period in quarters |
| `--front-load` | off | Deploy all capital at t=0 |

### Safer Instrument Terms

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--valuation-cap` | $5,000,000 | Post-money valuation cap |
| `--revenue-share` | 0.05 (5%) | Revenue share percentage |
| `--target-multiple` | 3.0x | Target return multiple |
| `--repurchase-pct` | 0.70 (70%) | Repurchase percentage |
| `--honeymoon` | 4 quarters | Honeymoon period |

### GP/LP Waterfall

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--carry-rate` | 0.20 (20%) | Carried interest rate |
| `--hurdle-rate` | 0.08 (8%) | Preferred return hurdle |
| `--no-waterfall` | off | Report gross = net (disable waterfall) |

### Company Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--init-revenue` | $150,000 | Median initial revenue |
| `--init-revenue-sigma` | 1.0 | Initial revenue dispersion |
| `--exit-multiple` | 6.0x | Median exit multiple of revenue |
| `--exit-multiple-sigma` | 0.9 | Exit multiple dispersion |
| `--base-fail-hazard` | 0.08 | Base quarterly failure hazard |
| `--terminal-nav-multiple` | 4.0x | Revenue multiple for terminal NAV |
| `--p-high-growth` | 0.25 | Probability of high-growth regime |
| `--p-mid-growth` | 0.35 | Probability of mid-growth regime |
| `--p-low-growth` | 0.40 | Probability of low-growth regime |
| `--max-exit-value` | unlimited | Cap on exit valuations (for stress testing) |

### Output Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--no-charts` | off | Skip chart generation |
| `--output-prefix` | '' | Prefix for output filenames |
| `--quiet` / `-q` | off | Suppress assumptions report |

## KEY METRICS OUTPUT

The simulator reports:

- **TVPI (Total Value to Paid-In)**: Distribution of gross and net fund multiples
- **DPI (Distributions to Paid-In)**: Cash-on-cash returns actually distributed
- **IRR**: Internal rate of return distribution
- **Return Attribution**: Breakdown of returns from revenue share (yield) vs. equity conversion (upside) vs. downside protection
- **Time to 1x DPI**: How quickly the fund returns LP capital
- **Outcome Distribution**: Percentage of fund iterations achieving various return thresholds
- **GP Economics**: Carry, hurdle, and net LP distributions

## CHARTS GENERATED

When charts are enabled (default), the simulator produces PNG files:

1. **Summary Dashboard** — Key metrics at a glance
2. **Cash Flow Paths** — Fan chart of cumulative fund cash flows
3. **TVPI Distribution** — Histogram with percentile markers
4. **DPI vs TVPI** — Scatter showing realized vs. total returns
5. **IRR Distribution** — Histogram of fund IRRs
6. **Return Attribution** — Stacked bar showing yield vs. equity components
7. **Time to 1x DPI** — When the fund returns capital
8. **Outcome Distribution** — Probability of various return thresholds

## WORKFLOW

1. **Gather parameters** from the user (use $ARGUMENTS if provided):
   - Fund structure (size, portfolio companies, investment per company)
   - Safer terms (valuation cap, revenue share, target multiple, honeymoon)
   - Company assumptions (initial revenue, growth regime, failure rate)
   - Analysis mode (main simulation, scenario preset, sensitivity, unicorn-free)

2. **Run the simulator** with appropriate parameters

3. **Save output** (charts and terminal output) to the user's workspace

4. **Present key findings**:
   - Median and percentile TVPI/DPI/IRR
   - Return attribution (yield vs. equity)
   - Probability of achieving 1x, 2x, 3x returns
   - Comparison across scenarios if running multiple

5. **Offer follow-up analysis**:
   - Sensitivity analysis to identify which parameters matter most
   - Unicorn-free stress test to show base-hit performance
   - Scenario comparison to evaluate different portfolio strategies

## CRITICAL CONCEPTS

### Revenue Share as Yield
Unlike traditional VC where all returns come from equity exits, Safer funds generate ongoing yield through revenue participation. The simulator tracks this separately, showing how much return comes from predictable revenue share vs. speculative equity upside.

### Anti-Power-Law Architecture
The Safer instrument is designed to produce predictable returns without requiring unicorn outcomes. The `--unicorn-free` mode demonstrates this by capping exits at $50M — showing that even without power-law outliers, revenue participation generates meaningful fund-level returns.

### Market Correlation
The simulator models correlated market conditions (boom/normal/recession) that affect all portfolio companies simultaneously. This captures systemic risk that uncorrelated models miss.

### GP/LP Waterfall
Net returns are calculated using a standard PE/VC distribution waterfall: return of capital → preferred return (hurdle) → GP catchup → 80/20 split. This shows the actual economics LPs receive.

## NOTE

This is a simulation tool for analysis and strategic planning. All results are probabilistic estimates based on model assumptions. The simulator is calibrated from real-world startup economics but does not predict actual fund performance. Advise users that past patterns do not guarantee future results.
