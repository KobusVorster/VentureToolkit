---
name: studio-os
description: >
  Run the VC 2.0 10-dimension scorecard on venture opportunities. Evaluate deals across thesis alignment, problem validation, timing assessment, and execution readiness using the EPAC-enhanced (Expert-Planner-Actor-Critic) t-shirt sizing methodology from Venture Capital 2.0. Every scorecard passes through a Critic validation loop that checks for internal consistency, evidence quality, hallucination risk, and analytical rigor before returning results.
version: 2.0.0
---

# VC 2.0 Scorecard System (EPAC Enhanced)

## EPAC FRAMEWORK

This skill implements the EPAC (Expert-Planner-Actor-Critic) pattern for agentic AI, developed by James Thomason. Every scorecard follows four explicit phases:

- **Expert** — The operator (user) who provides the specification and gives final sign-off
- **Planner** — Decomposes the request into a structured analytical plan with explicit assumptions
- **Actor** — Executes the plan: scores dimensions, conducts research, generates the report
- **Critic** — Validates outputs for internal consistency, evidence quality, hallucination risk, and analytical rigor before returning results to the Expert

**Core principle:** No scorecard is returned to the Expert without passing through the Critic.

---

## THE 10-DIMENSION SCORECARD

Adapted from James Thomason's methodology. Evaluates opportunities across 10 dimensions on an 8-point t-shirt scale.

### Scoring Scale

**Standard dimensions** (larger = better): S=1, M=2, L=3, XL=5, XXL=8
**Inverted dimensions** (smaller = better): S=8, M=5, L=3, XL=2, XXL=1

**IMPORTANT — Inverted Scale Clarification:**
For inverted dimensions, the t-shirt size describes the CHARACTERISTIC being measured:
- **Resources Required [INVERTED]:** S = minimal resources needed = Score 8 (best). XXL = massive resources needed = Score 1 (worst).
- **Cost of Sales [INVERTED]:** S = very high cost of sales = Score 8 (worst outcome, highest score). XXL = very low cost of sales = Score 1 (best outcome, lowest score).
- **Time to Market [INVERTED]:** S = very long time = Score 8. XXL = very short time = Score 1.
- **Competition [INVERTED]:** S = highly fragmented = Score 8. XXL = monopoly/nascent = Score 1.

Always reference the detailed scoring tables in `references/studio-os-frameworks.md` to ensure correct scale application.

### The 10 Dimensions

#### Thesis Alignment (Domain Conviction)
**1. Market Size [STANDARD]** — Addressable market for venture-scale returns (typically $500M+ TAM to support $100M+ exit)
**2. Competition [INVERTED]** — Competitive landscape crowdedness (fewer direct competitors = higher score)

#### Problem Validation (Problem Orientation)
**3. Customer Urgency [STANDARD]** — Solving critically important problem that customers will pay for
**4. Cost of Sales [INVERTED]** — Affordable customer acquisition relative to customer lifetime value

#### Timing Assessment (Structural Insight)
**5. Market Timing [STANDARD]** — External conditions alignment (regulatory shifts, technology maturity, behavior change)
**6. Time to Market [INVERTED]** — Speed to validate core assumptions (faster = higher score)

#### Execution Readiness
**7. Founder-Market Fit [STANDARD]** — Team's unique domain advantages, network access, domain expertise
**8. Novelty [STANDARD]** — Innovation alignment with thesis (how differentiated from existing solutions)
**9. Resources Required [INVERTED]** — Capital and expertise manageability (lean = higher score)
**10. Profitability [STANDARD]** — Unit economics support sustainable business model

### Score Interpretation

- **70-80: Exceptional** — Strong potential across all key dimensions. Rare and valuable find.
- **60-69: Very Good** — More strengths than weaknesses. Solid foundation.
- **50-59: Mixed Bag** — Significant potential but notable risks. Requires refinement.
- **40-49: Weak** — More red flags than green lights. Should probably be set aside.
- **Below 40: Very Poor** — Unlikely to succeed in current form.

### Critical Pattern Rules

These override mechanical scoring:

- **Customer Urgency below 3** = failure predictor regardless of other scores. Reposition problem or pass.
- **High Market Timing (5+) compensates** for moderate scores elsewhere. Prioritize market window exploitation.
- **Founder-Market Fit below 4** = requires team development, recruitment, or partnership. Design augmentation strategy.
- **Resources Required >6 AND Time to Market >6** = dangerous capital exposure. Scope reduction or partnership required.
- **Competition <2 AND Customer Urgency >5** = potential market creation opportunity. Premium weighting.

### Scorecard as Diagnostic Tool

The scorecard reveals structural misalignments:

- **Low founder fit + high market** = Recruitment/partnership need
- **Strong problem + weak sales model** = Monetization strategy required
- **High novelty + poor timing** = Delay engagement or develop market
- **Strong fundamentals + high resources** = Capital strategy or scope modification

---

## EPAC WORKFLOW

When this skill is invoked, follow the EPAC loop strictly.

### PHASE 1: EXPERT CAPTURE

1. **Determine what is being scored** — What company/opportunity? What documents are available?
2. **Restate the specification explicitly:**
   - "Here is what I understand you want: [restate request]"
   - "Here are the parameters I will use: [list inputs, documents, assumptions]"
   - "Here is what I do NOT have and will need to research or assume: [list gaps]"
3. **Confirm with the Expert before proceeding.** If documents are provided and the request is clear, proceed with a brief restatement.

### PHASE 2: PLANNER

Produce a structured analytical plan before scoring:

1. **Document Analysis Plan** — What documents will be analyzed, in what order, for what information
2. **Research Plan** — What web research is needed to verify claims, assess competition, validate market data
3. **Dimension Mapping** — For each of the 10 dimensions, identify the primary evidence source and what data points are needed
4. **Assumption Register** — Explicitly list every assumption being made where data is unavailable

### PHASE 3: ACTOR

Execute the plan step by step:

1. **Document Analysis** — Read and extract all relevant data from provided documents
2. **Web Research** — Verify market size claims, confirm competitors exist, validate timing signals, check founder backgrounds
3. **Dimension Scoring** — Score each dimension using the t-shirt scale:
   - Cite specific evidence for each score
   - Apply correct scale (standard vs inverted — reference the detailed tables)
   - Record confidence level (High/Medium/Low) based on evidence quality
   - Flag any dimension where evidence is thin or contradictory
4. **Pattern Rule Application** — Check all critical pattern rules and document which triggered
5. **Diagnostic Analysis** — Identify structural misalignments
6. **Report Generation** — Produce the complete scorecard report

**Score conservatively when evidence is uncertain.** When in doubt, score one t-shirt size lower than the evidence might support.

### PHASE 4: CRITIC

Validate ALL Actor outputs before returning to the Expert:

#### Internal Consistency
- [ ] Do individual dimension scores sum correctly to the reported total?
- [ ] Does each score match the t-shirt size definition for its scale type (standard vs inverted)?
- [ ] Are inverted scales applied correctly? (Reference the detailed scoring tables)
- [ ] Do triggered pattern rules match the actual scores?
- [ ] Does the interpretation band match the total score?

#### Evidence Quality
- [ ] Does every dimension score have at least one specific piece of cited evidence?
- [ ] Are market size figures sourced from identifiable reports or data?
- [ ] Are competitor names real companies that actually exist?
- [ ] Are regulatory or timing claims verifiable?
- [ ] Are financial figures sourced from documents or clearly labeled as estimates?

#### Hallucination Detection
- [ ] Are any statistics cited that do not appear in documents AND were not found via web research?
- [ ] Are any company names, product names, or market figures invented?
- [ ] Are any claims presented as facts that are actually assumptions?
- [ ] Does the report cite "industry reports" or "studies" without specific attribution?

#### Analytical Rigor
- [ ] Is scoring conservative where evidence is thin?
- [ ] Are confidence levels honestly assigned?
- [ ] Are there dimensions where the justification contradicts the score?
- [ ] Does the recommendation logically follow from the total score AND pattern rules?

#### Completeness
- [ ] Are all 10 dimensions scored?
- [ ] Are all 5 pattern rules evaluated?
- [ ] Is the assumption register included?
- [ ] Are data gaps explicitly listed?
- [ ] Are strengths AND risks both identified?

### CRITIC RESOLUTION

When the Critic flags an issue:

1. **REVISE** — If fixable (math error, scale misapplication, missing citation), the Actor revises and the Critic re-checks. Up to 3 iterations.
2. **CAVEAT** — If unresolvable (insufficient data, contradictory evidence), add an explicit caveat. Do not suppress the concern.
3. **ESCALATE** — If fundamental (documents too thin to score reliably), escalate to the Expert with explanation.

### CRITIC CONFIDENCE APPENDIX

Every scorecard MUST include a Critic Confidence Appendix:

```
## Critic Validation Summary

**Validation Status:** PASSED / PASSED WITH CAVEATS / ESCALATED
**Critic Iterations:** [number of Actor-Critic loops]

### Dimension Confidence
| Dimension | Score | Confidence | Primary Evidence Source | Critic Notes |
|-----------|-------|------------|----------------------|--------------|

### Verified Claims
- [Claims independently verified via web research]

### Unverified Assumptions
- [Assumptions that could not be independently verified]

### Hallucination Check
- [Result: CLEAR or FLAGGED with explanation]

### Pattern Rule Audit
- [Each rule, whether it triggered, and whether the trigger is correct]
```

---

## DELIVERABLE STANDARDS

All scorecards must include:

1. **Expert specification restatement** — What was asked for
2. **Scorecard table** — All 10 dimensions with t-shirt size, score, and confidence
3. **Dimension analysis** — Evidence and justification for each score
4. **Pattern rules** — Which triggered and impact
5. **Structural diagnosis** — Key strengths, risks, and strategic tensions
6. **Assumption register** — Every assumption explicitly listed
7. **Data gap inventory** — What information was unavailable
8. **Critic Confidence Appendix** — Full validation results
9. **Recommendation** — Go / Conditional Go / No-Go with justification

Reference `references/studio-os-frameworks.md` for complete dimension specifications and detailed scoring criteria.

---

**VC 2.0 Essentials** — Free venture analysis tools powered by the EPAC framework.

Want the full studio operating system with thesis development, capital structuring, opportunity scouting, return architecture, VREC governance, and institutional learning? Upgrade to **VC 2.0 Premium** at johncowan.io
