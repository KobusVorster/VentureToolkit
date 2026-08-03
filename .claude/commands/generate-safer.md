---
description: Generate a Safer instrument term sheet for a venture investment
allowed-tools: Read, Write, Edit, Bash
---

Generate a complete Safer (Simple Agreement for Future Equity with Repurchase) term sheet.

If `.claude/skills/capital-structurer/references/capital-frameworks.md` exists, read it first for the capital structurer framework. It is not shipped with the free VC 2.0 Essentials release — if the file is absent, skip it and proceed using the parameters and defaults defined below.

Gather the following parameters from the user (use $ARGUMENTS if provided, otherwise ask interactively):

**Required Parameters:**
- Company name
- Investment amount
- Valuation cap

**Customizable Parameters (provide defaults if not specified):**
- Revenue participation rate (default: 5% of gross monthly revenue, range 3-8%)
- Target return multiple (default: 200%, range 150-300%)
- Honeymoon period (default: 6 months, range 0-12 months)
- Revenue definition (default: gross monthly revenue)

Generate a complete Safer term sheet document with these sections:

1. **Parties** — Investor and Company identification
2. **Investment Terms** — Amount, date, valuation cap
3. **Revenue Participation**
   - Participation rate and calculation method
   - Honeymoon period duration and start trigger
   - Payment frequency and mechanics
   - Revenue reporting requirements
4. **Return Multiple and Cap**
   - Target return multiple
   - Calculation methodology
   - Cessation of revenue payments upon cap achievement
5. **Equity Conversion**
   - Conversion trigger events: ONLY terminal liquidity events (acquisition, IPO, secondary sale)
   - **CRITICAL: Safers do NOT convert on qualified financing rounds.** Subsequent equity financing rounds (regardless of size or valuation) do NOT constitute a Conversion Event. This is a deliberate departure from traditional SAFE instruments, preserving the founder's ability to raise additional capital without triggering involuntary conversion or dilution.
   - Conversion price calculation AT the valuation cap (not "lesser of" cap or round price)
   - Post-conversion equity rights
6. **Representations and Covenants**
   - Revenue reporting obligations
   - Material change notifications
   - Standard company representations
7. **VREC Alignment Provisions**
   - Founder protection clauses (no right to remove/replace founders, no board seat, no voting rights pre-conversion)
   - Contribution-based governance
   - No anti-dilution ratchets, liquidation preferences, or participating preferred structures
   - No involuntary founder removal rights

Include a financial projection showing estimated return timeline based on projected revenue growth assumptions.

## SCENARIO SIMULATION (Optional but Recommended)

After generating the term sheet, offer to run the Safer Scenario Simulator to produce a detailed financial analysis. The simulator is at `tools/safer.py`.

To run a simulation alongside the term sheet:

```bash
pip install matplotlib numpy --break-system-packages 2>/dev/null || true
python tools/safer.py \
    --investment <investment_amount> \
    --valuation-cap <valuation_cap> \
    --target-return-multiple <target_return_multiple> \
    --revenue-share <revenue_rate_as_decimal> \
    --honeymoon-months <honeymoon_months> \
    --scenario steady-growth-acquisition \
    --output <output_path.html>
```

The simulator generates an HTML report with:
- Quarterly cash flow projections showing revenue growth and repurchase payments
- Safer Amount evolution over time (the investor's shrinking equity claim)
- Liquidity event analysis (Cash-Out vs Conversion payout)
- IRR and MOIC calculations
- Embedded charts (revenue timeline, Safer Amount curve, return waterfall)

Use `--scenario custom` with `--initial-arr`, `--growth-rates`, `--exit-year`, and `--exit-valuation` for company-specific projections. See the `/simulate-safer` command for full parameter reference.

Save the term sheet as a markdown file in the user's workspace.

Note: This is a TEMPLATE for discussion purposes. Advise the user to have qualified legal counsel review before execution.
