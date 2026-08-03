---
description: Run the Safer Scenario Simulator to model investment returns, cash flows, and liquidity outcomes
allowed-tools: Read, Write, Edit, Bash
---

Run the Safer Scenario Simulator — James Thomason's Python-based financial modeling engine for the Safer (Simple Agreement for Future Equity with Repurchase) instrument.

The simulator is located at `tools/safer.py`. It produces detailed HTML reports with quarterly cash flow projections, Safer Amount evolution, IRR calculations, and return waterfall analysis with embedded charts.

## SETUP

First, ensure dependencies are installed:

```bash
pip install matplotlib numpy --break-system-packages 2>/dev/null || true
```

## USAGE MODES

### 1. Built-In Scenarios (Quick Analysis)

Run a pre-defined scenario to demonstrate Safer mechanics:

```bash
python tools/safer.py --scenario <scenario-name> --output <output-path>
```

**Available scenarios:**

| Scenario | Description |
|----------|-------------|
| `steady-growth-acquisition` | Pre-seed SaaS, steady growth to $120M acquisition in Year 7 |
| `explosive-growth-ipo` | AI startup, explosive growth to $2B IPO in Year 7 |
| `sustainable-complete-repurchase` | Seed SaaS, sustainable growth, full 3x target repaid through operations |
| `failure-low-revenue` | Company fails to achieve significant revenue, eventual shutdown |
| `modest-growth-small-exit` | Modest growth, small $25M acquisition in Year 4 |

### 2. Custom Terms with Built-In Scenario

Override Safer terms while using a pre-defined revenue/exit profile:

```bash
python tools/safer.py \
    --investment <amount> \
    --valuation-cap <cap> \
    --target-return-multiple <multiple> \
    --revenue-share <rate> \
    --repurchase-percent <percent> \
    --honeymoon-months <months> \
    --scenario <scenario-name> \
    --output <output-path>
```

### 3. Fully Custom Scenario

Model a specific company's projected revenue trajectory and exit:

```bash
python tools/safer.py \
    --investment <amount> \
    --valuation-cap <cap> \
    --target-return-multiple <multiple> \
    --revenue-share <rate> \
    --repurchase-percent <percent> \
    --honeymoon-months <months> \
    --initial-arr <starting-arr> \
    --growth-rates "<comma-separated-annual-rates>" \
    --exit-year <year> \
    --exit-valuation <valuation> \
    --description "<scenario-name>" \
    --output <output-path>
```

## PARAMETER REFERENCE

**Safer Terms:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--investment` / `-i` | $1,000,000 | Purchase Amount (the investment) |
| `--valuation-cap` / `-v` | $10,000,000 | Post-Money Valuation Cap for conversion |
| `--target-return-multiple` / `-t` | 3.0x | Target Return as multiple of investment |
| `--revenue-share` / `-r` | 5% (0.05) | Revenue Percentage paid quarterly |
| `--repurchase-percent` / `-p` | 90% (0.90) | Percentage of Purchase Amount bought back at target |
| `--honeymoon-months` / `-m` | 12 | Months before revenue share payments begin |

**Custom Scenario:**

| Parameter | Description |
|-----------|-------------|
| `--initial-arr` | Starting annual recurring revenue |
| `--growth-rates` | Comma-separated annual growth rates (e.g., `0,0.8,0.6,0.5`) |
| `--exit-year` | Year of liquidity event |
| `--exit-valuation` | Exit valuation in dollars |
| `--description` | Scenario description for the report |

**Output:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--output` / `-o` | `safer_scenario_report.html` | HTML report output path |
| `--json` | off | Also output results as JSON |

## WORKFLOW

1. **Gather parameters** from the user (use $ARGUMENTS if provided, otherwise ask interactively):
   - What are the Safer terms? (investment amount, valuation cap, target return multiple, revenue share, honeymoon period)
   - What scenario? (built-in or custom revenue profile and exit)

2. **Run the simulator** using the appropriate command

3. **Save the HTML report** to the user's workspace folder

4. **Present the key findings** from the terminal output:
   - Total return and MOIC
   - IRR
   - Disposition (Liquidity Event, Target Return Achieved, Dissolution, etc.)
   - Safer Amount at exit (if applicable)
   - Whether payout was via Cash-Out or Conversion

5. **Offer comparative analysis** — suggest running additional scenarios to compare outcomes (e.g., "What if the exit is at $50M instead of $120M?" or "What happens with a 6-month honeymoon vs 12-month?")

## SAFER AMOUNT FORMULA

The simulator implements the exact Safer Amount formula from the legal agreement:

```
Safer Amount = Purchase Amount
             - [(Cumulative Payments / Target Return) × Repurchase Amount]
             + [Target Return - Cumulative Payments]
```

At a liquidity event, the investor receives the **greater of**:
- **Cash-Out Amount:** The Safer Amount (downside protection)
- **Conversion Amount:** The Safer Amount divided by the Valuation Cap, multiplied by the exit valuation (equity upside)

This creates automatic cap table repair: as the company makes quarterly revenue payments, the investor's equity claim shrinks proportionally.

## CRITICAL SAFER MECHANICS

- **Safers do NOT convert on qualified financing rounds.** Only terminal liquidity events (acquisition, IPO, direct listing) trigger conversion.
- **Conversion price is AT the valuation cap**, not "lesser of" cap or round price.
- **Revenue payments are quarterly**, calculated as Revenue Percentage × Quarterly Gross Revenue.
- **Payments cease** when cumulative payments reach the Target Return.
- **No anti-dilution ratchets, no liquidation preferences, no board seats.**

## NOTE

The simulator is a financial modeling tool for analysis and discussion purposes. All projections are based on assumptions and should not be treated as guarantees of future performance. Advise users to consult qualified legal and financial counsel before making investment decisions.
