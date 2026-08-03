# Copyright (c) 2026, Next Wave Publishing LLC. All rights reserved.
# Licensed under the BSD 4-Clause License. See LICENSE file for details.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys

# -----------------------------
# Utility: robust IRR solver
# -----------------------------
def xirr(cashflows, times):
    """
    Continuous-time IRR (XIRR style).
    cashflows: array of cash flows (negative then positive)
    times: array of times in years aligned to cashflows
    """
    cashflows = np.asarray(cashflows, dtype=float)
    times = np.asarray(times, dtype=float)

    # Must have at least one negative and one positive cashflow
    if not (np.any(cashflows < 0) and np.any(cashflows > 0)):
        return np.nan

    def npv(rate):
        return np.sum(cashflows / ((1.0 + rate) ** times))

    # Bracketed bisection: robust, slower but stable
    lo, hi = -0.90, 10.0
    f_lo, f_hi = npv(lo), npv(hi)

    # If not bracketed, expand hi a bit
    if f_lo * f_hi > 0:
        for _ in range(20):
            hi *= 1.5
            f_hi = npv(hi)
            if f_lo * f_hi <= 0:
                break
        else:
            return np.nan

    for _ in range(120):
        mid = 0.5 * (lo + hi)
        f_mid = npv(mid)
        if abs(f_mid) < 1e-8:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid

    return mid


# -----------------------------
# Safer + Company simulation
# -----------------------------
class SaferParams:
    def __init__(self):
        # Your fixed terms
        self.post_money_valuation_cap = 5_000_000.0
        self.investment = 750_000.0
        self.repurchase_percent = 0.70
        self.revenue_share = 0.05
        self.target_return_multiple = 3.0 
        self.honeymoon_quarters = 4  # 12 months

        # Derived
        self.target_return = self.investment * self.target_return_multiple
        self.repurchase_amount = self.investment * self.repurchase_percent

    def safer_amount(self, repurchase_paid):
        """
        Implements Safer Amount structure consistent with doc:
        Safer Amount = Purchase Amount
                      - (RepurchasePaid/TargetReturn) * RepurchaseAmount
                      + (TargetReturn - RepurchasePaid)
        """
        paid = np.clip(repurchase_paid, 0.0, self.target_return)
        return (
            self.investment
            - (paid / self.target_return) * self.repurchase_amount
            + (self.target_return - paid)
        )


# -----------------------------
# ENHANCEMENT 1: Macro/Market Conditions
# -----------------------------
class MarketConditions:
    """
    Generates correlated market conditions that affect all portfolio companies.
    Models systemic risk via yearly market states that influence:
    - Revenue growth rates
    - Exit multiples
    - Failure hazards
    """
    def __init__(self, rng: np.random.Generator, years: int = 10):
        self.rng = rng
        self.years = years
        
        # Market regime parameters
        # Each year can be: boom (0.2), normal (0.6), or recession (0.2)
        self.regime_probs = [0.20, 0.60, 0.20]  # boom, normal, recession
        
        # Multipliers for each regime
        self.growth_multipliers = [1.3, 1.0, 0.6]      # How growth rates are affected
        self.exit_multiple_adjustments = [0.3, 0.0, -0.4]  # Added to log exit multiple
        self.failure_hazard_multipliers = [0.7, 1.0, 1.5]  # How failure rates change
        
        # Generate yearly conditions
        self.yearly_regimes = self._generate_regimes()
        
    def _generate_regimes(self):
        """Generate market regime for each year with some autocorrelation."""
        regimes = []
        prev_regime = 1  # Start normal
        
        for _ in range(self.years):
            # Some persistence in regimes (mean reversion to normal)
            if prev_regime == 0:  # Was boom
                probs = [0.3, 0.5, 0.2]
            elif prev_regime == 2:  # Was recession
                probs = [0.2, 0.5, 0.3]
            else:  # Was normal
                probs = self.regime_probs
            
            regime = self.rng.choice([0, 1, 2], p=probs)
            regimes.append(regime)
            prev_regime = regime
            
        return regimes
    
    def get_quarterly_conditions(self, quarter: int):
        """
        Returns market condition multipliers for a given quarter.
        Returns: (growth_mult, exit_mult_adj, fail_mult)
        """
        year = min(quarter // 4, self.years - 1)
        regime = self.yearly_regimes[year]
        return (
            self.growth_multipliers[regime],
            self.exit_multiple_adjustments[regime],
            self.failure_hazard_multipliers[regime]
        )
    
    def get_regime_name(self, quarter: int):
        year = min(quarter // 4, self.years - 1)
        regime = self.yearly_regimes[year]
        return ["Boom", "Normal", "Recession"][regime]


class CompanyModel:
    def __init__(self, safer: SaferParams, rng: np.random.Generator, 
                 market: MarketConditions = None, start_quarter: int = 0):
        self.safer = safer
        self.rng = rng
        self.market = market
        self.start_quarter = start_quarter  # When company was funded

        # Time grid
        self.fund_years = 10
        self.quarters = self.fund_years * 4  # 40 quarters
        self.dt_years = 0.25

        # --- Revenue dynamics (quarterly) ---
        self.p_high_growth = 0.25
        self.p_mid_growth = 0.35
        self.p_low_growth = 0.40

        self.high_mu_yr, self.high_sigma = 1.00, 0.55
        self.mid_mu_yr,  self.mid_sigma  = 0.45, 0.45
        self.low_mu_yr,  self.low_sigma  = 0.10, 0.35

        self.init_rev_logn_mu = np.log(150_000)
        self.init_rev_logn_sigma = 1.0

        self.base_fail_hazard_q = 0.08
        self.base_exit_hazard_q = 0.01
        self.max_exit_hazard_q = 0.08

        self.exit_multiple_mu = np.log(6.0)
        self.exit_multiple_sigma = 0.9

        self.dilution_low = 0.35
        self.dilution_high = 0.85

        self.allow_terminal_cashout = False
        
        # Terminal NAV multiple for surviving private companies
        self.terminal_nav_multiple = 4.0  # Revenue multiple for NAV
        
        # UNICORN-FREE MODE: Cap exit valuations to show base-hit performance
        self.unicorn_free_mode = False
        self.max_exit_value = 50_000_000.0  # $50M cap when unicorn-free

    def _draw_regime(self):
        u = self.rng.random()
        if u < self.p_high_growth:
            return self.high_mu_yr, self.high_sigma
        if u < self.p_high_growth + self.p_mid_growth:
            return self.mid_mu_yr, self.mid_sigma
        return self.low_mu_yr, self.low_sigma

    def simulate(self):
        """
        Returns:
          result dict with:
            - cashflows: array by quarter
            - repurchase_cashflows: array (yield component)
            - equity_cashflows: array (exit component)
            - protection_cashflows: array (downside protection at exit - up to investment)
            - upside_cashflows: array (true equity upside - above investment)
            - terminal_nav: NAV of surviving private company (0 if exited/failed)
            - final_revenue: revenue at end (for NAV calc)
            - repurchase_paid: total repurchase payments made
            - exit_value: the exit equity value (for unicorn analysis)
            - outcome_label: str
        """
        s = self.safer
        Q = self.quarters
        
        cf = np.zeros(Q + 1)
        cf_repurchase = np.zeros(Q + 1)  # ENHANCEMENT 4: Track yield separately
        cf_equity = np.zeros(Q + 1)       # ENHANCEMENT 4: Track equity separately
        cf_protection = np.zeros(Q + 1)   # NEW: Downside protection component
        cf_upside = np.zeros(Q + 1)       # NEW: True equity upside component
        cf[0] = -s.investment
        
        exit_value = 0.0  # Track the exit valuation

        mu_yr, sigma = self._draw_regime()

        rev_yr = self.rng.lognormal(mean=self.init_rev_logn_mu, sigma=self.init_rev_logn_sigma)
        rev_yr = float(np.clip(rev_yr, 0.0, 2_000_000.0))

        repurchase_paid = 0.0
        exited = False
        failed = False
        hit_target = False
        final_revenue = rev_yr

        for q in range(1, Q + 1):
            age_years = q * self.dt_years
            
            # Get market conditions for this quarter (ENHANCEMENT 1)
            if self.market is not None:
                abs_quarter = self.start_quarter + q
                growth_mult, exit_adj, fail_mult = self.market.get_quarterly_conditions(abs_quarter)
            else:
                growth_mult, exit_adj, fail_mult = 1.0, 0.0, 1.0

            # --- failure draw (hazard falls as revenue rises) ---
            rev_scale = 2_000_000.0
            hazard_fail = self.base_fail_hazard_q * (1.0 / (1.0 + (rev_yr / rev_scale)))
            hazard_fail = float(np.clip(hazard_fail * fail_mult, 0.01, 0.35))  # Apply macro

            if self.rng.random() < hazard_fail:
                failed = True
                rev_yr = 0.0

            # --- revenue evolves if not failed ---
            if not failed:
                mu_q = mu_yr * self.dt_years * growth_mult  # Apply macro growth multiplier
                shock = self.rng.normal(loc=mu_q, scale=sigma * np.sqrt(self.dt_years))
                rev_yr *= float(np.exp(shock))
                rev_yr = float(np.clip(rev_yr, 0.0, 100_000_000.0))
                final_revenue = rev_yr

            # --- Repurchase Payments (quarterly), only after honeymoon ---
            if q > s.honeymoon_quarters and repurchase_paid < s.target_return and rev_yr > 0 and not failed:
                repurchase_payment = s.revenue_share * (rev_yr * self.dt_years)
                repurchase_payment = float(min(repurchase_payment, s.target_return - repurchase_paid))
                repurchase_paid += repurchase_payment
                cf[q] += repurchase_payment
                cf_repurchase[q] += repurchase_payment  # Track separately

                if repurchase_paid >= s.target_return - 1e-9:
                    hit_target = True

            # --- Liquidity event hazard ---
            if age_years >= 2.0 and not failed and not exited:
                arr_proxy = rev_yr
                h = self.base_exit_hazard_q + (self.max_exit_hazard_q - self.base_exit_hazard_q) * (
                    1.0 - np.exp(-arr_proxy / 3_000_000.0)
                )
                age_boost = 0.5 + 0.5 * np.tanh((age_years - 4.0) / 1.5)
                hazard_exit = float(np.clip(h * age_boost, 0.0, self.max_exit_hazard_q))

                if self.rng.random() < hazard_exit:
                    exited = True
                    safer_amt = s.safer_amount(repurchase_paid)
                    
                    # Apply macro adjustment to exit multiple
                    adjusted_exit_mu = self.exit_multiple_mu + exit_adj
                    exit_mult = self.rng.lognormal(mean=adjusted_exit_mu, sigma=self.exit_multiple_sigma)
                    exit_equity_value = float(np.clip(exit_mult * max(arr_proxy, 250_000.0), 500_000.0, 5_000_000_000.0))
                    
                    # UNICORN-FREE MODE: Cap exit valuations at max_exit_value
                    if self.unicorn_free_mode:
                        exit_equity_value = min(exit_equity_value, self.max_exit_value)
                    
                    exit_value = exit_equity_value  # Track for analysis

                    conversion_value = safer_amt * (exit_equity_value / s.post_money_valuation_cap)
                    cashout_value = safer_amt
                    payout = float(max(cashout_value, conversion_value))

                    cf[q] += payout
                    cf_equity[q] += payout  # Track as equity return
                    
                    # BETTER ATTRIBUTION: Split into protection vs upside
                    # Protection = amount up to original investment (downside protection feature)
                    # Upside = any amount above original investment (true equity participation)
                    protection_component = min(payout, s.investment)
                    upside_component = max(0.0, payout - s.investment)
                    cf_protection[q] += protection_component
                    cf_upside[q] += upside_component
                    
                    break

        # Determine outcome label
        if hit_target and exited:
            label = "Target + Liquidity Event"
        elif hit_target:
            label = "Target Return Repaid"
        elif exited:
            label = "Liquidity Event"
        elif failed:
            label = "Failure/No Liquidity"
        else:
            label = "No Liquidity by Fund End"

        # ENHANCEMENT 2: Calculate Terminal NAV for surviving private companies
        # BUG FIX #2: "Illiquid Cash" - NAV is conversion value only
        # The "Cash-Out Amount" protection only applies at actual Liquidity Events.
        # For mark-to-market NAV of private companies, we use Fair Market Value
        # (conversion value), consistent with ASC 820 accounting standards.
        terminal_nav = 0.0
        if not exited and not failed and final_revenue > 0:
            # Company is alive but private at fund end
            safer_amt = s.safer_amount(repurchase_paid)
            estimated_value = final_revenue * self.terminal_nav_multiple
            
            # THE FIX: Apply the stress test cap to unrealized private valuations
            if self.unicorn_free_mode:
                estimated_value = min(estimated_value, self.max_exit_value)
                
            # NAV = conversion value only (no cash-out floor for illiquid holdings)
            terminal_nav = safer_amt * (estimated_value / s.post_money_valuation_cap)

        return {
            'cashflows': cf,
            'repurchase_cashflows': cf_repurchase,
            'equity_cashflows': cf_equity,
            'protection_cashflows': cf_protection,
            'upside_cashflows': cf_upside,
            'terminal_nav': terminal_nav,
            'final_revenue': final_revenue,
            'repurchase_paid': repurchase_paid,
            'exit_value': exit_value,
            'outcome_label': label
        }

# -----------------------------
# ENHANCEMENT 3: GP/LP Waterfall
# -----------------------------
class WaterfallCalculator:
    """
    Implements standard PE/VC distribution waterfall for Net returns.
    """
    def __init__(self, 
                 carry_rate: float = 0.20,
                 hurdle_rate: float = 0.08,
                 gp_catchup: float = 1.0,  # 100% catchup until GP has carry% of profits
                 mgmt_fee_offset: bool = True):
        self.carry_rate = carry_rate
        self.hurdle_rate = hurdle_rate
        self.gp_catchup = gp_catchup
        self.mgmt_fee_offset = mgmt_fee_offset
    
    def calculate_net_distributions(self, gross_distributions: float, 
                                   paid_in_capital: float, 
                                   years: float) -> dict:
        """
        Calculate LP net distributions after GP carry.
        
        Returns dict with:
        - lp_distributions: What LP actually receives
        - gp_carry: What GP takes as carry
        - gross_multiple: Gross TVPI
        - net_multiple: Net TVPI to LP
        """
        # Step 1: Return of capital to LP
        if gross_distributions <= paid_in_capital:
            return {
                'lp_distributions': gross_distributions,
                'gp_carry': 0.0,
                'gross_multiple': gross_distributions / paid_in_capital if paid_in_capital > 0 else 0,
                'net_multiple': gross_distributions / paid_in_capital if paid_in_capital > 0 else 0
            }
        
        profits = gross_distributions - paid_in_capital
        
        # Step 2: Preferred return (hurdle) to LP
        # Simplified: compound hurdle over fund life
        hurdle_amount = paid_in_capital * ((1 + self.hurdle_rate) ** years - 1)
        
        if profits <= hurdle_amount:
            return {
                'lp_distributions': gross_distributions,
                'gp_carry': 0.0,
                'gross_multiple': gross_distributions / paid_in_capital,
                'net_multiple': gross_distributions / paid_in_capital
            }
        
        # Step 3: GP Catchup
        remaining_profit = profits - hurdle_amount
        
        # GP catches up to get carry_rate of all profits above hurdle
        # Catchup amount = carry_rate / (1 - carry_rate) * hurdle_amount (for 100% catchup)
        catchup_target = (self.carry_rate / (1 - self.carry_rate)) * hurdle_amount * self.gp_catchup
        gp_catchup_amount = min(remaining_profit, catchup_target)
        remaining_profit -= gp_catchup_amount
        
        # Step 4: 80/20 split of remaining
        gp_from_split = remaining_profit * self.carry_rate
        lp_from_split = remaining_profit * (1 - self.carry_rate)
        
        total_gp_carry = gp_catchup_amount + gp_from_split
        lp_distributions = paid_in_capital + hurdle_amount + lp_from_split
        
        return {
            'lp_distributions': lp_distributions,
            'gp_carry': total_gp_carry,
            'gross_multiple': gross_distributions / paid_in_capital,
            'net_multiple': lp_distributions / paid_in_capital
        }


# -----------------------------
# Fund simulation (Enhanced)
# -----------------------------
class FundMonteCarlo:
    def __init__(
        self,
        iterations=10_000,
        portfolio_size=25,
        fund_size=25_000_000.0,
        mgmt_fee_per_year=500_000.0,
        years=10,
        seed=2026,
        enable_correlation=True,  # ENHANCEMENT 1
        enable_waterfall=True,    # ENHANCEMENT 3
        carry_rate=0.20,
        hurdle_rate=0.08
    ):
        self.iterations = iterations
        self.portfolio_size = portfolio_size
        self.fund_size = fund_size
        self.mgmt_fee_per_year = mgmt_fee_per_year
        self.years = years
        self.quarters = years * 4
        self.dt = 0.25
        self.rng = np.random.default_rng(seed)

        self.safer = SaferParams()
        self.investment_period_quarters = 12
        self.front_load_investments = False
        
        # Enhancement flags
        self.enable_correlation = enable_correlation
        self.enable_waterfall = enable_waterfall
        
        if enable_waterfall:
            self.waterfall = WaterfallCalculator(carry_rate=carry_rate, hurdle_rate=hurdle_rate)
        
        # BUG FIX #3: Centralized overrides for sensitivity analysis
        # This avoids code duplication in sensitivity methods
        self.company_overrides = {}   # Overrides for CompanyModel parameters
        self.market_overrides = {}    # Overrides for MarketConditions parameters
        
        # UNICORN-FREE MODE: For stress testing without power law outliers
        self.unicorn_free_mode = False
        self.max_exit_value = 50_000_000.0  # $50M cap when unicorn-free

    def _fund_fee_schedule(self):
        fees = np.zeros(self.quarters + 1)
        fee_q = self.mgmt_fee_per_year * self.dt
        for q in range(1, self.quarters + 1):
            fees[q] = -fee_q
        return fees

    def _investment_timing(self):
        calls = np.zeros(self.quarters + 1)
        total_deploy = self.safer.investment * self.portfolio_size

        if self.front_load_investments:
            calls[0] = -total_deploy
            return calls

        per_q = total_deploy / self.investment_period_quarters
        for q in range(0, self.investment_period_quarters):
            calls[q] += -per_q
        return calls
    
    def _apply_company_overrides(self, cm: 'CompanyModel'):
        """Apply any company-level overrides from sensitivity analysis."""
        for key, value in self.company_overrides.items():
            if hasattr(cm, key):
                setattr(cm, key, value)
    
    def _apply_market_overrides(self, market: 'MarketConditions'):
        """Apply any market-level overrides from sensitivity analysis."""
        if not market:
            return
        
        corr_strength = self.market_overrides.get('correlation_strength', None)
        exit_adj = self.market_overrides.get('exit_multiple_adj', 0.0)
        
        if corr_strength is not None:
            # Blend regime effects with correlation strength
            market.growth_multipliers = [1.0 + corr_strength * (m - 1.0) for m in market.growth_multipliers]
            market.failure_hazard_multipliers = [1.0 + corr_strength * (m - 1.0) for m in market.failure_hazard_multipliers]
        
        if exit_adj != 0.0:
            market.exit_multiple_adjustments = [adj + exit_adj for adj in market.exit_multiple_adjustments]

    def run(self):
        times = np.arange(self.quarters + 1) * self.dt

        # Result arrays
        gross_tvpi = np.zeros(self.iterations)
        net_tvpi = np.zeros(self.iterations)
        dpi = np.zeros(self.iterations)
        net_dpi = np.zeros(self.iterations)
        gross_irr = np.zeros(self.iterations)
        net_irr = np.zeros(self.iterations)
        
        # ENHANCEMENT 4: Return decomposition
        total_repurchase = np.zeros(self.iterations)
        total_equity = np.zeros(self.iterations)
        total_nav = np.zeros(self.iterations)
        time_to_1x_dpi = np.zeros(self.iterations)
        
        # NEW: Enhanced attribution tracking
        total_protection = np.zeros(self.iterations)  # Downside protection component
        total_upside = np.zeros(self.iterations)      # True equity upside
        max_exit_values = []  # Track largest exit in each iteration
        
        # NEW: DPI by year tracking for velocity analysis
        dpi_by_year = np.zeros((self.iterations, self.years + 1))
        repurchase_by_year = np.zeros((self.iterations, self.years + 1))
        equity_by_year = np.zeros((self.iterations, self.years + 1))
        
        outcome_counts = []
        sample_paths = []

        for i in range(self.iterations):
            # Create market conditions for this iteration (ENHANCEMENT 1)
            if self.enable_correlation:
                market = MarketConditions(self.rng, years=self.years)
                # Apply any market overrides from sensitivity analysis
                self._apply_market_overrides(market)
            else:
                market = None
            
            # Fund-level cashflows
            fund_cf = np.zeros(self.quarters + 1)
            fund_cf_repurchase = np.zeros(self.quarters + 1)
            fund_cf_equity = np.zeros(self.quarters + 1)
            fund_cf_protection = np.zeros(self.quarters + 1)
            fund_cf_upside = np.zeros(self.quarters + 1)

            # Fees + investment calls
            fund_cf += self._fund_fee_schedule()
            fund_cf += self._investment_timing()

            # Track terminal NAV
            portfolio_terminal_nav = 0.0
            iteration_max_exit = 0.0  # Track largest exit

            # Simulate companies
            labels = []
            for _ in range(self.portfolio_size):
                # Determine investment quarter
                if self.front_load_investments:
                    invest_q = 0
                else:
                    invest_q = int(self.rng.integers(0, self.investment_period_quarters))
                
                cm = CompanyModel(self.safer, self.rng, market=market, start_quarter=invest_q)
                # Apply any company overrides from sensitivity analysis
                self._apply_company_overrides(cm)
                
                # Apply unicorn-free mode if enabled
                if self.unicorn_free_mode:
                    cm.unicorn_free_mode = True
                    cm.max_exit_value = self.max_exit_value
                
                result = cm.simulate()
                labels.append(result['outcome_label'])
                
                # Track max exit for this iteration
                if result['exit_value'] > iteration_max_exit:
                    iteration_max_exit = result['exit_value']

                company_cf = result['cashflows'].copy()
                company_cf_repurchase = result['repurchase_cashflows'].copy()
                company_cf_equity = result['equity_cashflows'].copy()
                company_cf_protection = result['protection_cashflows'].copy()
                company_cf_upside = result['upside_cashflows'].copy()
                
                # Zero out investment (already in _investment_timing)
                company_cf[0] = 0.0

                # Add to fund cashflows (shifted if paced)
                if self.front_load_investments:
                    fund_cf += company_cf
                    fund_cf_repurchase += company_cf_repurchase
                    fund_cf_equity += company_cf_equity
                    fund_cf_protection += company_cf_protection
                    fund_cf_upside += company_cf_upside
                else:
                    shifted = np.zeros_like(fund_cf)
                    shifted_rep = np.zeros_like(fund_cf)
                    shifted_eq = np.zeros_like(fund_cf)
                    shifted_prot = np.zeros_like(fund_cf)
                    shifted_up = np.zeros_like(fund_cf)
                    max_len = len(fund_cf) - invest_q
                    shifted[invest_q:invest_q + max_len] += company_cf[:max_len]
                    shifted_rep[invest_q:invest_q + max_len] += company_cf_repurchase[:max_len]
                    shifted_eq[invest_q:invest_q + max_len] += company_cf_equity[:max_len]
                    shifted_prot[invest_q:invest_q + max_len] += company_cf_protection[:max_len]
                    shifted_up[invest_q:invest_q + max_len] += company_cf_upside[:max_len]
                    fund_cf += shifted
                    fund_cf_repurchase += shifted_rep
                    fund_cf_equity += shifted_eq
                    fund_cf_protection += shifted_prot
                    fund_cf_upside += shifted_up

                # Add terminal NAV (ENHANCEMENT 2)
                portfolio_terminal_nav += result['terminal_nav']
            
            # Track max exit for this iteration
            max_exit_values.append(iteration_max_exit)

            # Compute metrics
            paid_in = -np.sum(fund_cf[fund_cf < 0])
            # FIX: Calculate realized distributions from tracked streams,
            # not from netted fund_cf where fees reduce returns in same quarter
            realized_dists = np.sum(fund_cf_repurchase) + np.sum(fund_cf_equity)
            
            # ENHANCEMENT 2: TVPI includes NAV, DPI is realized only
            total_value = realized_dists + portfolio_terminal_nav
            
            gross_tvpi[i] = total_value / paid_in if paid_in > 0 else np.nan
            dpi[i] = realized_dists / paid_in if paid_in > 0 else np.nan
            
            # BUG FIX #1: "Ghost Asset" - Add NAV to cashflow for Gross IRR
            # Previously calculated IRR on realized cash only, marking private companies to $0
            # Now includes terminal NAV as Year 10 "paper" value
            gross_cf_with_nav = fund_cf.copy()
            gross_cf_with_nav[-1] += portfolio_terminal_nav
            gross_irr[i] = xirr(gross_cf_with_nav, times)
            
            # ENHANCEMENT 3: Net returns after waterfall
            if self.enable_waterfall:
                waterfall_result = self.waterfall.calculate_net_distributions(
                    total_value, paid_in, self.years
                )
                net_tvpi[i] = waterfall_result['net_multiple']
                
                # FIXED: Apply "Net Rate" scalar to all positive cashflows
                # This preserves timing of distributions for accurate IRR
                # (instead of deducting carry as lump sum at end, which inflates IRR)
                gross_multiple = waterfall_result['gross_multiple']
                net_multiple = waterfall_result['net_multiple']
                
                scalar = 1.0
                if gross_multiple > 0:
                    scalar = net_multiple / gross_multiple
                
                # Net DPI: apply same scalar to realized distributions only
                net_dpi[i] = (realized_dists * scalar) / paid_in if paid_in > 0 else np.nan
                
                # Create Net Cashflow Stream (start from gross_cf_with_nav for consistency)
                net_cf = fund_cf.copy()
                # Apply scalar to all positive distributions (realized)
                net_cf = np.where(net_cf > 0, net_cf * scalar, net_cf)
                
                # Add the Net Terminal NAV (also scaled) to the final period
                net_terminal_nav = portfolio_terminal_nav * scalar
                net_cf[-1] += net_terminal_nav
                
                net_irr[i] = xirr(net_cf, times)
            else:
                net_tvpi[i] = gross_tvpi[i]
                net_irr[i] = gross_irr[i]
                net_dpi[i] = dpi[i]
            
            # ENHANCEMENT 4: Return decomposition
            total_repurchase[i] = np.sum(fund_cf_repurchase)
            total_equity[i] = np.sum(fund_cf_equity)
            total_nav[i] = portfolio_terminal_nav
            total_protection[i] = np.sum(fund_cf_protection)
            total_upside[i] = np.sum(fund_cf_upside)
            
            # Time to 1x DPI
            cumulative_dist = np.cumsum(np.maximum(fund_cf, 0))
            hit_1x = np.where(cumulative_dist >= paid_in)[0]
            if len(hit_1x) > 0:
                time_to_1x_dpi[i] = hit_1x[0] * self.dt
            else:
                time_to_1x_dpi[i] = np.nan
            
            # DPI velocity: track cumulative distributions by year
            for yr in range(self.years + 1):
                end_q = min((yr + 1) * 4, self.quarters + 1)
                dpi_by_year[i, yr] = np.sum(np.maximum(fund_cf[:end_q], 0)) / paid_in if paid_in > 0 else 0
                repurchase_by_year[i, yr] = np.sum(fund_cf_repurchase[:end_q]) / paid_in if paid_in > 0 else 0
                equity_by_year[i, yr] = np.sum(fund_cf_equity[:end_q]) / paid_in if paid_in > 0 else 0

            outcome_counts.append(pd.Series(labels).value_counts())

            if i < 300:
                sample_paths.append(np.cumsum(fund_cf))

        # Aggregate label stats
        label_df = pd.DataFrame(outcome_counts).fillna(0).sum().sort_values(ascending=False)

        results = {
            "gross_tvpi": gross_tvpi,
            "net_tvpi": net_tvpi,
            "dpi": dpi,
            "net_dpi": net_dpi,
            "gross_irr": gross_irr,
            "net_irr": net_irr,
            "total_repurchase": total_repurchase,
            "total_equity": total_equity,
            "total_nav": total_nav,
            "total_protection": total_protection,
            "total_upside": total_upside,
            "time_to_1x_dpi": time_to_1x_dpi,
            "sample_paths": np.array(sample_paths),
            "label_totals": label_df,
            "max_exit_values": np.array(max_exit_values),
            "dpi_by_year": dpi_by_year,
            "repurchase_by_year": repurchase_by_year,
            "equity_by_year": equity_by_year,
            # Legacy compatibility
            "tvpi": gross_tvpi,
            "irr": gross_irr
        }
        return results


# -----------------------------
# Run + summarize + visualize
# -----------------------------
def print_assumptions(mc: FundMonteCarlo):
    """Print all simulation assumptions in a clear, auditable format."""
    s = mc.safer
    
    print("="*78)
    print("█" + " "*76 + "█")
    print("█" + "SAFER FUND MONTE CARLO SIMULATION".center(76) + "█")
    print("█" + "Institutional Due Diligence Report".center(76) + "█")
    print("█" + " "*76 + "█")
    print("="*78)
    
    print("\n" + "="*78)
    print("SECTION 1: SIMULATION PARAMETERS")
    print("="*78)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ MONTE CARLO CONFIGURATION                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ Iterations:                    {mc.iterations:>10,}    (number of fund simulations)    │
│ Random Seed:                   {str(mc.rng.bit_generator.state['state']['state'])[:10]:>10}    (for reproducibility)         │
│ Time Horizon:                  {mc.years:>10} yrs (fund life)                       │
│ Time Step:                     {mc.dt:>10.2f} yrs (quarterly)                      │
│ Total Periods:                 {mc.quarters:>10}    (quarters)                       │
└─────────────────────────────────────────────────────────────────────────────┘
""")
    
    print("="*78)
    print("SECTION 2: FUND STRUCTURE")
    print("="*78)
    
    total_invested = s.investment * mc.portfolio_size
    total_fees = mc.mgmt_fee_per_year * mc.years
    total_committed = total_invested + total_fees
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ FUND ECONOMICS                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ Fund Size (Stated):            ${mc.fund_size:>13,.0f}                              │
│ Portfolio Companies:           {mc.portfolio_size:>14}                              │
│ Investment per Company:        ${s.investment:>13,.0f}                              │
│ Total Deployed Capital:        ${total_invested:>13,.0f}                              │
│ Management Fee (per year):     ${mc.mgmt_fee_per_year:>13,.0f}                              │
│ Total Fees (10 years):         ${total_fees:>13,.0f}                              │
│ Total LP Commitment:           ${total_committed:>13,.0f}                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ INVESTMENT PACING                                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ Deployment Period:             {mc.investment_period_quarters:>14} quarters ({mc.investment_period_quarters/4:.1f} years)       │
│ Front-Load Investments:        {"Yes" if mc.front_load_investments else "No":>14}                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")
    
    if mc.enable_waterfall:
        print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ GP/LP WATERFALL (Enabled)                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ Carried Interest:              {mc.waterfall.carry_rate:>13.0%}                              │
│ Hurdle Rate (Preferred Return):{mc.waterfall.hurdle_rate:>13.0%}                              │
│ GP Catch-up:                   {mc.waterfall.gp_catchup:>13.0%}                              │
│ Distribution Order:            Return Capital → Hurdle → Catch-up → Split   │
└─────────────────────────────────────────────────────────────────────────────┘
""")
    
    print("="*78)
    print("SECTION 3: SAFER INSTRUMENT TERMS")
    print("="*78)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ SAFER STRUCTURE (Simple Agreement for Future Equity with Repurchase)        │
├─────────────────────────────────────────────────────────────────────────────┤
│ Purchase Amount:               ${s.investment:>13,.0f}                              │
│ Post-Money Valuation Cap:      ${s.post_money_valuation_cap:>13,.0f}                              │
│ Implied Ownership at Cap:      {s.investment/s.post_money_valuation_cap:>13.1%}                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ REVENUE SHARE MECHANICS                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Revenue Share Percentage:      {s.revenue_share:>13.1%}    (of gross quarterly revenue) │
│ Honeymoon Period:              {s.honeymoon_quarters:>13} qtrs  ({s.honeymoon_quarters/4:.1f} years, no payments)   │
│ Target Return Multiple:        {s.target_return_multiple:>13.1f}x                              │
│ Target Return Amount:          ${s.target_return:>13,.0f}                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ REPURCHASE MECHANICS                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ Repurchase Percentage:         {s.repurchase_percent:>13.0%}                              │
│ Repurchase Amount:             ${s.repurchase_amount:>13,.0f}                              │
│ Residual Equity (at Target):   ${s.investment - s.repurchase_amount:>13,.0f}    ({(1-s.repurchase_percent):.0%} of Purchase)      │
├─────────────────────────────────────────────────────────────────────────────┤
│ SAFER AMOUNT FORMULA                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ SaferAmount = PurchaseAmount                                                │
│             - (RepurchasePaid / TargetReturn) × RepurchaseAmount            │
│             + (TargetReturn - RepurchasePaid)                               │
│                                                                             │
│ At R=0:        SaferAmount = ${s.safer_amount(0):>12,.0f}  (max equity value)          │
│ At R=Target:   SaferAmount = ${s.safer_amount(s.target_return):>12,.0f}  (residual equity)          │
└─────────────────────────────────────────────────────────────────────────────┘
""")
    
    print("="*78)
    print("SECTION 4: COMPANY MODEL ASSUMPTIONS")
    print("="*78)
    
    # Create a temporary company model and apply overrides to show actual values
    temp_cm = CompanyModel(s, mc.rng)
    
    # Apply any overrides that were set (from CLI args)
    for key, value in mc.company_overrides.items():
        if hasattr(temp_cm, key):
            setattr(temp_cm, key, value)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ REVENUE DYNAMICS                                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ Initial Revenue Distribution:  Lognormal(μ={temp_cm.init_rev_logn_mu:.2f}, σ={temp_cm.init_rev_logn_sigma:.2f})               │
│ Median Starting Revenue:       ${np.exp(temp_cm.init_rev_logn_mu):>13,.0f} /year                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ GROWTH REGIME PROBABILITIES                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ High Growth Companies:         {temp_cm.p_high_growth:>13.0%}    (μ={temp_cm.high_mu_yr:.0%}/yr, σ={temp_cm.high_sigma:.0%})     │
│ Mid Growth Companies:          {temp_cm.p_mid_growth:>13.0%}    (μ={temp_cm.mid_mu_yr:.0%}/yr, σ={temp_cm.mid_sigma:.0%})     │
│ Low Growth Companies:          {temp_cm.p_low_growth:>13.0%}    (μ={temp_cm.low_mu_yr:.0%}/yr, σ={temp_cm.low_sigma:.0%})     │
├─────────────────────────────────────────────────────────────────────────────┤
│ FAILURE MODEL                                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ Base Failure Hazard (quarterly):{temp_cm.base_fail_hazard_q:>12.0%}                              │
│ Hazard Formula:                h(rev) = base × (1 / (1 + rev/$2M))          │
│ Hazard Range:                  {0.01:.0%} to {0.25:.0%} (clamped)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ EXIT / LIQUIDITY EVENT MODEL                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ Minimum Time to Exit:          {2.0:>13.1f} years                            │
│ Base Exit Hazard (quarterly):  {temp_cm.base_exit_hazard_q:>13.1%}                              │
│ Max Exit Hazard (quarterly):   {temp_cm.max_exit_hazard_q:>13.1%}                              │
│ Exit Hazard increases with:    Revenue level + Company age                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ EXIT VALUATION MODEL                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ Exit Multiple Distribution:    Lognormal(μ={temp_cm.exit_multiple_mu:.2f}, σ={temp_cm.exit_multiple_sigma:.2f})               │
│ Median Exit Multiple:          {np.exp(temp_cm.exit_multiple_mu):>13.1f}x revenue                       │
│ Exit Value Range:              ${500_000:>10,.0f} to ${mc.max_exit_value if mc.unicorn_free_mode else 5_000_000_000:>10,.0f}{' (CAPPED)' if mc.unicorn_free_mode else ''}              │
├─────────────────────────────────────────────────────────────────────────────┤
│ TERMINAL NAV MODEL (for surviving private companies)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ Terminal Revenue Multiple:     {temp_cm.terminal_nav_multiple:>13.1f}x                              │
│ NAV = max(SaferAmount, SaferAmount × EstValue / Cap)                        │
└─────────────────────────────────────────────────────────────────────────────┘
""")
    
    if mc.enable_correlation:
        print("="*78)
        print("SECTION 5: MACRO CORRELATION MODEL")
        print("="*78)
        
        print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ MARKET REGIME MODEL (Systemic Risk)                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ Regime Probabilities:          Boom: 20%  |  Normal: 60%  |  Recession: 20% │
│ Regime Autocorrelation:        Yes (mean-reverting to Normal)               │
├─────────────────────────────────────────────────────────────────────────────┤
│ REGIME EFFECTS ON PORTFOLIO                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                   Boom      Normal    Recession             │
│ Growth Rate Multiplier:          1.30x      1.00x      0.60x                │
│ Exit Multiple Adjustment:       +0.30      +0.00      -0.40   (log scale)   │
│ Failure Hazard Multiplier:       0.70x      1.00x      1.50x                │
├─────────────────────────────────────────────────────────────────────────────┤
│ KEY INSIGHT: All portfolio companies experience the same yearly market      │
│ conditions, creating correlated outcomes (unlike independent simulations).  │
└─────────────────────────────────────────────────────────────────────────────┘
""")
    
    print("="*78)
    print("SECTION 6: LIQUIDITY EVENT PAYOUT LOGIC")
    print("="*78)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ PAYOUT CALCULATION (per SAFER legal document)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ At Liquidity Event, Investor receives GREATER of:                           │
│                                                                             │
│   (A) Cash-Out Amount = SaferAmount                                         │
│                                                                             │
│   (B) Conversion Amount = SaferAmount × (ExitValue / ValuationCap)          │
│                                                                             │
│ This provides:                                                              │
│   • Downside protection when ExitValue < Cap (Cash-Out dominates)           │
│   • Upside participation when ExitValue > Cap (Conversion dominates)        │
├─────────────────────────────────────────────────────────────────────────────┤
│ LIQUIDATION PRIORITY                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Debt and creditor claims (senior)                                        │
│ 2. SAFER Cash-Out Amount (pari passu with preferred)                        │
│ 3. Common Stock (junior)                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ DISSOLUTION EVENT                                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ Modeled as: $0 recovery (conservative - assumes no assets after creditors)  │
└─────────────────────────────────────────────────────────────────────────────┘
""")
    
    print("="*78)
    print("SECTION 7: KEY MODEL LIMITATIONS & ASSUMPTIONS")
    print("="*78)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ SIMPLIFYING ASSUMPTIONS                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. No follow-on investments or pro-rata rights modeled                      │
│ 2. No secondary sales or early liquidity modeled                            │
│ 3. Tax implications not modeled                                             │
│ 4. Currency risk not modeled (all USD)                                      │
│ 5. No recycling of returned capital                                         │
│ 6. Companies invested later in fund have truncated simulation windows       │
│ 7. Dissolution events assume zero recovery (conservative)                   │
│ 8. Revenue growth is continuous (no step-function changes)                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ PARAMETER CALIBRATION NOTES                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Growth/failure parameters calibrated for early-stage venture profile      │
│ • Exit multiples reflect typical software/SaaS company valuations           │
│ • Macro correlation strength reflects observed market cyclicality           │
│ • Users should recalibrate for specific sector/geography focus              │
└─────────────────────────────────────────────────────────────────────────────┘
""")
    
    print("\n" + "="*78)
    print("END OF ASSUMPTIONS - SIMULATION RESULTS FOLLOW")
    print("="*78 + "\n")


def create_argument_parser():
    """
    Create argument parser with all configurable parameters.
    All defaults match the original hardcoded values for backward compatibility.
    """
    parser = argparse.ArgumentParser(
        description='SAFER Fund Monte Carlo Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MODES:
  (default)           Run main simulation with specified parameters
  --scenario X        Run a preset portfolio scenario (see below)
  --sensitivity       Basic sensitivity analysis (3 heatmaps)
  --sensitivity-full  Full sensitivity analysis (8 heatmaps)
  --unicorn-free      Unicorn-free stress test comparison

SCENARIO PRESETS (for the SAFER Portfolios chapter):
  --scenario standard          Traditional VC assumptions (high fail, power law)
  --scenario revenue-focused   Traction-selected companies ($350k+ ARR)
  --scenario high-yield        Optimized SAFER terms for revenue share
  --scenario max-revenue       Pure yield play ($500k+ ARR, 10% rev share)
  --scenario all-scenarios     Run all four and generate comparison charts

EXAMPLES:
  # Run with defaults (10,000 iterations, 25 companies, $750K each)
  python safer_montecarlo17.py
  
  # Quick test run
  python safer_montecarlo17.py --iterations 1000 --seed 42
  
  # Run the revenue-focused scenario from the book
  python safer_montecarlo17.py --scenario revenue-focused
  
  # Run all scenarios for the SAFER Portfolios chapter
  python safer_montecarlo17.py --scenario all-scenarios --iterations 5000
  
  # Different fund structure
  python safer_montecarlo17.py --portfolio-size 30 --investment 500000 --fund-size 20000000
  
  # Different SAFER terms
  python safer_montecarlo17.py --valuation-cap 3000000 --revenue-share 0.03 --target-multiple 4.0
  
  # Unicorn-free analysis with custom iterations
  python safer_montecarlo17.py --unicorn-free --iterations 3000
  
  # Sensitivity analysis in quick mode
  python safer_montecarlo17.py --sensitivity --quick
"""
    )
    
    # ==========================================================================
    # MODE SELECTION
    # ==========================================================================
    mode_group = parser.add_argument_group('Mode Selection')
    mode_group.add_argument('--sensitivity', action='store_true',
                           help='Run basic sensitivity analysis (3 heatmaps)')
    mode_group.add_argument('--sensitivity-full', action='store_true',
                           help='Run full sensitivity analysis (8 heatmaps)')
    mode_group.add_argument('--unicorn-free', action='store_true',
                           help='Run unicorn-free stress test comparison')
    mode_group.add_argument('--quick', action='store_true',
                           help='Quick mode with fewer iterations (for testing)')
    mode_group.add_argument('--scenario', type=str, default=None,
                           choices=['standard', 'revenue-focused', 'high-yield', 'max-revenue', 'all-scenarios'],
                           help='Run a preset portfolio scenario. '
                                '"standard" = traditional VC assumptions, '
                                '"revenue-focused" = traction-selected companies, '
                                '"high-yield" = optimized for revenue share, '
                                '"max-revenue" = pure yield play, '
                                '"all-scenarios" = run all four and compare')
    
    # ==========================================================================
    # SIMULATION PARAMETERS
    # ==========================================================================
    sim_group = parser.add_argument_group('Simulation Parameters')
    sim_group.add_argument('--iterations', '-n', type=int, default=10_000,
                          help='Number of Monte Carlo iterations (default: 10000)')
    sim_group.add_argument('--seed', type=int, default=None,
                          help='Random seed for reproducibility (default: random)')
    sim_group.add_argument('--years', type=int, default=10,
                          help='Fund life in years (default: 10)')
    
    # ==========================================================================
    # FUND STRUCTURE
    # ==========================================================================
    fund_group = parser.add_argument_group('Fund Structure')
    fund_group.add_argument('--portfolio-size', type=int, default=25,
                           help='Number of portfolio companies (default: 25)')
    fund_group.add_argument('--fund-size', type=float, default=25_000_000.0,
                           help='Total fund size in dollars (default: 25000000)')
    fund_group.add_argument('--investment', type=float, default=750_000.0,
                           help='Investment per company in dollars (default: 750000)')
    fund_group.add_argument('--mgmt-fee', type=float, default=500_000.0,
                           help='Management fee per year in dollars (default: 500000)')
    fund_group.add_argument('--investment-period', type=int, default=12,
                           help='Investment period in quarters (default: 12)')
    fund_group.add_argument('--front-load', action='store_true',
                           help='Front-load all investments at t=0')
    
    # ==========================================================================
    # WATERFALL PARAMETERS
    # ==========================================================================
    waterfall_group = parser.add_argument_group('GP/LP Waterfall')
    waterfall_group.add_argument('--carry-rate', type=float, default=0.20,
                                help='Carried interest rate (default: 0.20)')
    waterfall_group.add_argument('--hurdle-rate', type=float, default=0.08,
                                help='Preferred return hurdle rate (default: 0.08)')
    waterfall_group.add_argument('--no-waterfall', action='store_true',
                                help='Disable waterfall (report gross = net)')
    
    # ==========================================================================
    # SAFER INSTRUMENT TERMS
    # ==========================================================================
    safer_group = parser.add_argument_group('SAFER Instrument Terms')
    safer_group.add_argument('--valuation-cap', type=float, default=5_000_000.0,
                            help='Post-money valuation cap in dollars (default: 5000000)')
    safer_group.add_argument('--revenue-share', type=float, default=0.05,
                            help='Revenue share percentage as decimal (default: 0.05)')
    safer_group.add_argument('--target-multiple', type=float, default=3.0,
                            help='Target return multiple (default: 3.0)')
    safer_group.add_argument('--repurchase-pct', type=float, default=0.70,
                            help='Repurchase percentage as decimal (default: 0.70)')
    safer_group.add_argument('--honeymoon', type=int, default=4,
                            help='Honeymoon period in quarters (default: 4)')
    
    # ==========================================================================
    # COMPANY MODEL PARAMETERS
    # ==========================================================================
    company_group = parser.add_argument_group('Company Model Parameters')
    company_group.add_argument('--init-revenue', type=float, default=150_000.0,
                              help='Median initial revenue in dollars (default: 150000)')
    company_group.add_argument('--init-revenue-sigma', type=float, default=1.0,
                              help='Initial revenue lognormal sigma (default: 1.0)')
    company_group.add_argument('--exit-multiple', type=float, default=6.0,
                              help='Median exit multiple of revenue (default: 6.0)')
    company_group.add_argument('--exit-multiple-sigma', type=float, default=0.9,
                              help='Exit multiple lognormal sigma (default: 0.9)')
    company_group.add_argument('--base-fail-hazard', type=float, default=0.08,
                              help='Base quarterly failure hazard rate (default: 0.08)')
    company_group.add_argument('--terminal-nav-multiple', type=float, default=4.0,
                              help='Revenue multiple for terminal NAV (default: 4.0)')
    
    # Growth regime probabilities (must sum to 1.0)
    company_group.add_argument('--p-high-growth', type=float, default=0.25,
                              help='Probability of high growth regime (default: 0.25)')
    company_group.add_argument('--p-mid-growth', type=float, default=0.35,
                              help='Probability of mid growth regime (default: 0.35)')
    company_group.add_argument('--p-low-growth', type=float, default=0.40,
                              help='Probability of low growth regime (default: 0.40)')
    
    # Growth rates
    company_group.add_argument('--high-growth-rate', type=float, default=1.00,
                              help='High growth annual rate (default: 1.00 = 100%%)')
    company_group.add_argument('--mid-growth-rate', type=float, default=0.45,
                              help='Mid growth annual rate (default: 0.45 = 45%%)')
    company_group.add_argument('--low-growth-rate', type=float, default=0.10,
                              help='Low growth annual rate (default: 0.10 = 10%%)')
    company_group.add_argument('--base-exit-hazard', type=float, default=0.01,
                              help='Base quarterly exit/liquidity event hazard rate (default: 0.01)')
    company_group.add_argument('--max-exit-hazard', type=float, default=0.08,
                              help='Maximum quarterly exit hazard rate (default: 0.08)')
    company_group.add_argument('--max-exit-value', type=float, default=None,
                              help='Cap on exit valuations in dollars (default: None = unlimited). '
                                   'Set to e.g. 50000000 to test "unicorn-free" scenarios.')
    
    # ==========================================================================
    # MACRO/CORRELATION PARAMETERS
    # ==========================================================================
    macro_group = parser.add_argument_group('Macro Correlation Model')
    macro_group.add_argument('--no-correlation', action='store_true',
                            help='Disable macro correlation (independent companies)')
    
    # ==========================================================================
    # OUTPUT OPTIONS
    # ==========================================================================
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--no-charts', action='store_true',
                             help='Skip chart generation')
    output_group.add_argument('--output-prefix', type=str, default='',
                             help='Prefix for output filenames')
    output_group.add_argument('--quiet', '-q', action='store_true',
                             help='Minimal output (suppress assumptions report)')
    
    return parser


def create_fund_mc_from_args(args, seed=None, max_exit_override=None):
    """
    Create a FundMonteCarlo instance from command line arguments.
    
    This is the single source of truth for setting up the simulation.
    Both run_main_simulation and run_unicorn_free_comparison use this.
    
    Args:
        args: Parsed command line arguments
        seed: Override seed (if None, uses args.seed or generates random)
        max_exit_override: Override max_exit_value (for unicorn-free comparison)
    
    Returns:
        Configured FundMonteCarlo instance
    """
    # Generate seed if not provided
    if seed is None:
        if args.seed is None:
            seed = np.random.default_rng().integers(0, 2**31)
        else:
            seed = args.seed
    
    # Create FundMonteCarlo with CLI parameters
    mc = FundMonteCarlo(
        iterations=args.iterations,
        portfolio_size=args.portfolio_size,
        fund_size=args.fund_size,
        mgmt_fee_per_year=args.mgmt_fee,
        years=args.years,
        seed=seed,
        enable_correlation=not args.no_correlation,
        enable_waterfall=not args.no_waterfall,
        carry_rate=args.carry_rate,
        hurdle_rate=args.hurdle_rate
    )
    
    # Override SAFER terms
    mc.safer.post_money_valuation_cap = args.valuation_cap
    mc.safer.investment = args.investment
    mc.safer.repurchase_percent = args.repurchase_pct
    mc.safer.revenue_share = args.revenue_share
    mc.safer.target_return_multiple = args.target_multiple
    mc.safer.honeymoon_quarters = args.honeymoon
    # Recalculate derived values
    mc.safer.target_return = mc.safer.investment * mc.safer.target_return_multiple
    mc.safer.repurchase_amount = mc.safer.investment * mc.safer.repurchase_percent
    
    # Override fund-level settings
    mc.investment_period_quarters = args.investment_period
    mc.front_load_investments = args.front_load
    
    # Set company overrides that will be applied during simulation
    mc.company_overrides = {
        'init_rev_logn_mu': np.log(args.init_revenue),
        'init_rev_logn_sigma': args.init_revenue_sigma,
        'exit_multiple_mu': np.log(args.exit_multiple),
        'exit_multiple_sigma': args.exit_multiple_sigma,
        'base_fail_hazard_q': args.base_fail_hazard,
        'terminal_nav_multiple': args.terminal_nav_multiple,
        'p_high_growth': args.p_high_growth,
        'p_mid_growth': args.p_mid_growth,
        'p_low_growth': args.p_low_growth,
        'high_mu_yr': args.high_growth_rate,
        'mid_mu_yr': args.mid_growth_rate,
        'low_mu_yr': args.low_growth_rate,
        'base_exit_hazard_q': args.base_exit_hazard,
        'max_exit_hazard_q': args.max_exit_hazard,
    }
    
    # Handle max exit value cap (unicorn-free mode)
    # max_exit_override takes precedence, then args.max_exit_value
    max_exit = max_exit_override if max_exit_override is not None else args.max_exit_value
    if max_exit is not None:
        mc.unicorn_free_mode = True
        mc.max_exit_value = max_exit
    
    return mc, seed


def run_main_simulation(args):
    """Run the main simulation with parameters from args."""
    import time
    
    mc, seed = create_fund_mc_from_args(args)
    
    # Print assumptions unless quiet mode
    if not args.quiet:
        print_assumptions(mc)
    
    print("Running simulation...")
    start_time = time.time()
    
    res = mc.run()
    
    elapsed = time.time() - start_time
    print(f"Simulation completed in {elapsed:.1f} seconds\n")

    # Summary function
    def summarize(x, name):
        x = x[np.isfinite(x)]
        if len(x) == 0:
            print(f"\n--- {name} ---")
            print("No valid data")
            return
        s = pd.Series(x).describe(percentiles=[0.10, 0.25, 0.5, 0.75, 0.90])
        print(f"\n--- {name} ---")
        print(s)

    print("="*78)
    print("SIMULATION RESULTS")
    print("="*78)
    
    summarize(res["gross_tvpi"], "Gross TVPI (incl. NAV)")
    summarize(res["net_tvpi"], "Net TVPI (after carry)")
    summarize(res["dpi"], "DPI (Realized Only)")
    summarize(res["gross_irr"], "Gross IRR")
    summarize(res["net_irr"], "Net IRR (to LP)")

    print("\n--- Outcome Totals Across All Simulated Companies ---")
    print(res["label_totals"].astype(int))
    
    # Return Attribution
    print("\n" + "="*78)
    print("RETURN ATTRIBUTION (Source of Value - Dollar Weighted)")
    print("="*78)
    
    total_rep_dollars = np.sum(res["total_repurchase"])
    total_eq_dollars = np.sum(res["total_equity"])
    total_nav_dollars = np.sum(res["total_nav"])
    grand_total = total_rep_dollars + total_eq_dollars + total_nav_dollars
    
    if grand_total > 0:
        pct_repurchase = (total_rep_dollars / grand_total) * 100
        pct_equity = (total_eq_dollars / grand_total) * 100
        pct_nav = (total_nav_dollars / grand_total) * 100
    else:
        pct_repurchase = pct_equity = pct_nav = 0.0
    
    print(f"\nSource of Returns (Dollar-Weighted):")
    print(f"  Revenue Share (Yield):     {pct_repurchase:5.1f}%  - Low risk, steady")
    print(f"  Liquidity Events (Equity): {pct_equity:5.1f}%  - High risk, lumpy")
    print(f"  Terminal NAV (Unrealized): {pct_nav:5.1f}%  - Marked to market")
    
    # Time to 1x DPI
    valid_time = res["time_to_1x_dpi"][np.isfinite(res["time_to_1x_dpi"])]
    if len(valid_time) > 0:
        print(f"\nTime to 1.0x DPI:")
        print(f"  Median: {np.median(valid_time):.1f} years")
        print(f"  25th percentile: {np.percentile(valid_time, 25):.1f} years")
        print(f"  % of funds reaching 1x DPI: {len(valid_time)/mc.iterations*100:.1f}%")

    # Generate charts unless disabled
    if not args.no_charts:
        generate_charts(res, mc, args.output_prefix)
    
    return res, mc


def generate_charts(res, mc, prefix=''):
    """Generate all charts for the simulation results."""
    PRINT_DPI = 300
    FIGURE_SIZE = (12, 8)
    FONT_SCALE = 1.2
    
    plt.rcParams.update({
        'font.size': 12 * FONT_SCALE,
        'axes.titlesize': 14 * FONT_SCALE,
        'axes.labelsize': 12 * FONT_SCALE,
        'xtick.labelsize': 10 * FONT_SCALE,
        'ytick.labelsize': 10 * FONT_SCALE,
        'legend.fontsize': 10 * FONT_SCALE,
        'figure.titlesize': 16 * FONT_SCALE
    })
    
    print("\n" + "="*78)
    print("GENERATING HIGH-RESOLUTION CHARTS FOR BOOK PRINTING")
    print(f"Resolution: {PRINT_DPI} DPI | Figure Size: {FIGURE_SIZE[0]}x{FIGURE_SIZE[1]} inches (landscape)")
    print("="*78)

    # 1) Sample cumulative fund paths
    fname = f"{prefix}chart_01_cashflow_paths.png" if prefix else "chart_01_cashflow_paths.png"
    print(f"  Saving: {fname}")
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    paths = res["sample_paths"]
    for p in paths[:250]:
        ax.plot(p, alpha=0.05, color='blue')
    ax.axhline(0, linewidth=1.5, color='black')
    ax.set_title("Sample Fund Cumulative Cashflow Paths\n(250 Monte Carlo Iterations)", fontweight='bold')
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Cumulative Cash Flow (USD)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=PRINT_DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    # 2) TVPI distribution
    fname = f"{prefix}chart_02_tvpi_distribution.png" if prefix else "chart_02_tvpi_distribution.png"
    print(f"  Saving: {fname}")
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    valid_tvpi = res["gross_tvpi"][np.isfinite(res["gross_tvpi"])]
    ax.hist(valid_tvpi, bins=60, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(np.median(valid_tvpi), color='red', linestyle='--', linewidth=2, 
               label=f'Median: {np.median(valid_tvpi):.2f}x')
    ax.axvline(1.0, color='black', linestyle='-', linewidth=1.5, label='1.0x (Capital Return)')
    ax.set_title("Distribution of Gross TVPI (Total Value to Paid-In)\n10,000 Monte Carlo Iterations", fontweight='bold')
    ax.set_xlabel("Gross TVPI Multiple")
    ax.set_ylabel("Frequency")
    ax.legend(loc='upper right')
    ax.set_xlim(0, min(20, np.percentile(valid_tvpi, 99)))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=PRINT_DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    # 3) DPI vs TVPI scatter
    fname = f"{prefix}chart_03_dpi_vs_tvpi.png" if prefix else "chart_03_dpi_vs_tvpi.png"
    print(f"  Saving: {fname}")
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    valid_mask = np.isfinite(res["dpi"]) & np.isfinite(res["gross_tvpi"])
    dpi_valid = res["dpi"][valid_mask]
    tvpi_valid = res["gross_tvpi"][valid_mask]
    ax.scatter(dpi_valid, tvpi_valid, alpha=0.15, s=10, color='steelblue')
    max_val = min(12, max(np.percentile(dpi_valid, 99), np.percentile(tvpi_valid, 99)))
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='DPI = TVPI (Fully Realized)')
    ax.fill_between([0, max_val], [0, max_val], [max_val, max_val], alpha=0.1, color='red', 
                    label='Unrealized NAV Gap')
    ax.set_title("Realized vs Total Returns\nGap Represents Unrealized NAV at Fund End", fontweight='bold')
    ax.set_xlabel("DPI (Distributions to Paid-In) - Realized Returns")
    ax.set_ylabel("TVPI (Total Value to Paid-In) - Including NAV")
    ax.legend(loc='upper left')
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=PRINT_DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    # 4) IRR distribution
    fname = f"{prefix}chart_04_irr_distribution.png" if prefix else "chart_04_irr_distribution.png"
    print(f"  Saving: {fname}")
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    valid_irr = res["gross_irr"][np.isfinite(res["gross_irr"])]
    valid_irr_clipped = np.clip(valid_irr, -0.5, 1.5)
    ax.hist(valid_irr_clipped, bins=60, alpha=0.7, color='forestgreen', edgecolor='white')
    ax.axvline(np.median(valid_irr), color='red', linestyle='--', linewidth=2, 
               label=f'Median: {np.median(valid_irr)*100:.1f}%')
    ax.axvline(0, color='black', linestyle='-', linewidth=1.5, label='0% IRR')
    ax.set_title("Distribution of Gross IRR\n10,000 Monte Carlo Iterations", fontweight='bold')
    ax.set_xlabel("Gross IRR")
    ax.set_ylabel("Frequency")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.0f}%'))
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=PRINT_DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    # 5) Return attribution pie
    fname = f"{prefix}chart_05_return_attribution.png" if prefix else "chart_05_return_attribution.png"
    print(f"  Saving: {fname}")
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    total_rep = np.sum(res["total_repurchase"])
    total_eq = np.sum(res["total_equity"])
    total_nav = np.sum(res["total_nav"])
    values = [total_rep, total_eq, total_nav]
    labels = ['Revenue Share\n(Yield)', 'Liquidity Events\n(Equity)', 'Terminal NAV\n(Unrealized)']
    colors = ['forestgreen', 'steelblue', 'gray']
    non_zero = [(v, l, c) for v, l, c in zip(values, labels, colors) if v > 0]
    if non_zero:
        values, labels, colors = zip(*non_zero)
        wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', 
                                          colors=colors, startangle=90,
                                          textprops={'fontsize': 12 * FONT_SCALE})
        for autotext in autotexts:
            autotext.set_fontweight('bold')
    ax.set_title("Source of Fund Returns\n(Dollar-Weighted Attribution)", fontweight='bold')
    plt.tight_layout()
    plt.savefig(fname, dpi=PRINT_DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    # 6) Time to 1x DPI
    fname = f"{prefix}chart_06_time_to_1x_dpi.png" if prefix else "chart_06_time_to_1x_dpi.png"
    print(f"  Saving: {fname}")
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    valid_time = res["time_to_1x_dpi"][np.isfinite(res["time_to_1x_dpi"])]
    if len(valid_time) > 0:
        ax.hist(valid_time, bins=40, alpha=0.7, color='darkorange', edgecolor='white')
        ax.axvline(np.median(valid_time), color='red', linestyle='--', linewidth=2, 
                   label=f'Median: {np.median(valid_time):.1f} years')
        ax.set_title(f"Time to Return 1.0x Capital (DPI = 1.0)\n{len(valid_time)/mc.iterations*100:.1f}% of funds achieve this", fontweight='bold')
        ax.set_xlabel("Years to 1.0x DPI")
        ax.set_ylabel("Frequency")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=PRINT_DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    # 7) Outcome distribution
    fname = f"{prefix}chart_07_outcome_distribution.png" if prefix else "chart_07_outcome_distribution.png"
    print(f"  Saving: {fname}")
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    labels_data = res["label_totals"]
    colors_map = {
        'Failure/No Liquidity': 'crimson',
        'Liquidity Event': 'steelblue',
        'Target + Liquidity Event': 'forestgreen',
        'Target Return Repaid': 'gold',
        'No Liquidity by Fund End': 'gray'
    }
    bar_colors = [colors_map.get(l, 'steelblue') for l in labels_data.index]
    bars = ax.barh(range(len(labels_data)), labels_data.values, color=bar_colors, alpha=0.8)
    ax.set_yticks(range(len(labels_data)))
    ax.set_yticklabels(labels_data.index)
    ax.set_xlabel("Number of Companies")
    ax.set_title("Company Outcomes Across All Simulations", fontweight='bold')
    for i, (bar, val) in enumerate(zip(bars, labels_data.values)):
        pct = val / labels_data.sum() * 100
        ax.text(val + labels_data.max() * 0.01, i, f'{pct:.1f}%', va='center', fontsize=10 * FONT_SCALE)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(fname, dpi=PRINT_DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    # 8) Summary dashboard
    fname = f"{prefix}chart_00_summary_dashboard.png" if prefix else "chart_00_summary_dashboard.png"
    print(f"  Saving: {fname}")
    fig = plt.figure(figsize=(24, 14))
    
    # Recreate key charts in subplots
    ax1 = fig.add_subplot(2, 3, 1)
    for p in res["sample_paths"][:100]:
        ax1.plot(p, alpha=0.08, color='blue')
    ax1.axhline(0, linewidth=1, color='black')
    ax1.set_title("Cumulative Cashflow Paths", fontweight='bold')
    ax1.set_xlabel("Quarter")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.0f}M'))
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.hist(valid_tvpi, bins=50, alpha=0.7, color='steelblue', edgecolor='white')
    ax2.axvline(np.median(valid_tvpi), color='red', linestyle='--', linewidth=2)
    ax2.axvline(1.0, color='black', linestyle='-', linewidth=1)
    ax2.set_title(f"TVPI Distribution (Median: {np.median(valid_tvpi):.2f}x)", fontweight='bold')
    ax2.set_xlabel("Gross TVPI")
    ax2.set_xlim(0, min(15, np.percentile(valid_tvpi, 98)))
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.hist(valid_irr_clipped, bins=50, alpha=0.7, color='forestgreen', edgecolor='white')
    ax3.axvline(np.median(valid_irr), color='red', linestyle='--', linewidth=2)
    ax3.axvline(0, color='black', linestyle='-', linewidth=1)
    ax3.set_title(f"IRR Distribution (Median: {np.median(valid_irr)*100:.1f}%)", fontweight='bold')
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.0f}%'))
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(dpi_valid, tvpi_valid, alpha=0.1, s=5, color='steelblue')
    ax4.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5)
    ax4.set_title("DPI vs TVPI", fontweight='bold')
    ax4.set_xlabel("DPI")
    ax4.set_ylabel("TVPI")
    ax4.set_xlim(0, max_val)
    ax4.set_ylim(0, max_val)
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(2, 3, 5)
    if len(valid_time) > 0:
        ax5.hist(valid_time, bins=30, alpha=0.7, color='darkorange', edgecolor='white')
        ax5.axvline(np.median(valid_time), color='red', linestyle='--', linewidth=2)
        ax5.set_title(f"Time to 1x DPI (Median: {np.median(valid_time):.1f} yrs)", fontweight='bold')
        ax5.set_xlabel("Years")
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(2, 3, 6)
    if non_zero:
        ax6.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax6.set_title("Return Attribution", fontweight='bold')
    
    fig.suptitle("SAFER Fund Monte Carlo Simulation Summary", fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(fname, dpi=PRINT_DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    # Legacy combined chart
    fname = f"{prefix}safer_fund_simulation.png" if prefix else "safer_fund_simulation.png"
    print(f"  Saving: {fname} (legacy combined)")
    fig = plt.figure(figsize=(24, 14))
    # Same as dashboard
    ax1 = fig.add_subplot(2, 3, 1)
    for p in res["sample_paths"][:100]:
        ax1.plot(p, alpha=0.08, color='blue')
    ax1.axhline(0, linewidth=1, color='black')
    ax1.set_title("Cumulative Cashflow Paths", fontweight='bold')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.0f}M'))
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.hist(valid_tvpi, bins=50, alpha=0.7, color='steelblue', edgecolor='white')
    ax2.axvline(np.median(valid_tvpi), color='red', linestyle='--', linewidth=2)
    ax2.set_title(f"TVPI (Median: {np.median(valid_tvpi):.2f}x)", fontweight='bold')
    ax2.set_xlim(0, min(15, np.percentile(valid_tvpi, 98)))
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.hist(valid_irr_clipped, bins=50, alpha=0.7, color='forestgreen', edgecolor='white')
    ax3.axvline(np.median(valid_irr), color='red', linestyle='--', linewidth=2)
    ax3.set_title(f"IRR (Median: {np.median(valid_irr)*100:.1f}%)", fontweight='bold')
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.0f}%'))
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(dpi_valid, tvpi_valid, alpha=0.1, s=5, color='steelblue')
    ax4.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5)
    ax4.set_title("DPI vs TVPI", fontweight='bold')
    ax4.set_xlim(0, max_val)
    ax4.set_ylim(0, max_val)
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(2, 3, 5)
    if len(valid_time) > 0:
        ax5.hist(valid_time, bins=30, alpha=0.7, color='darkorange', edgecolor='white')
        ax5.axvline(np.median(valid_time), color='red', linestyle='--', linewidth=2)
        ax5.set_title(f"Time to 1x DPI (Median: {np.median(valid_time):.1f} yrs)", fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(2, 3, 6)
    if non_zero:
        ax6.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax6.set_title("Return Attribution", fontweight='bold')
    
    fig.suptitle("SAFER Fund Monte Carlo Simulation", fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(fname, dpi=PRINT_DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"\n  ✓ Generated 9 high-resolution chart files (300 DPI, LANDSCAPE)")


# =============================================================================
# -----------------------------
# SENSITIVITY ANALYSIS SUITE
# -----------------------------
class SensitivityAnalysis:
    """
    Runs 2D grid simulations to generate heatmaps for institutional due diligence.
    Answers three critical questions:
    1. Selection Skill: Does revenue share protect downside if we pick poorly?
    2. Structure Design: What's the optimal Revenue Share % vs Target Return?
    3. Macro Risk: How does the fund perform under market stress?
    
    All sensitivity variations are relative to the baseline parameters from CLI args.
    """
    
    def __init__(self, args, base_iterations=500):
        """
        Initialize with CLI args as the baseline configuration.
        
        Args:
            args: Parsed command line arguments (baseline for all simulations)
            base_iterations: Override for iterations per scenario (sensitivity needs fewer)
        """
        self.args = args
        self.base_iterations = base_iterations
        
        # Store original iterations so we can restore if needed
        self._original_iterations = args.iterations
    
    def _create_mc(self):
        """
        Create a Monte Carlo instance using CLI args as baseline.
        Returns a fresh instance that can be modified for sensitivity variations.
        """
        # Temporarily set iterations to base_iterations for sensitivity runs
        original_iter = self.args.iterations
        self.args.iterations = self.base_iterations
        
        mc, seed = create_fund_mc_from_args(self.args)
        
        # Restore original iterations
        self.args.iterations = original_iter
        
        return mc
    
    def run_structure_sensitivity(self, 
                                   rev_shares=[0.03, 0.04, 0.05, 0.06, 0.08],
                                   target_multiples=[2.0, 2.5, 3.0, 3.5, 4.0, 5.0]):
        """
        Heatmap 1: Revenue Share % vs Target Return Multiple
        Answers: What's the optimal structure for Net IRR?
        """
        print("="*70)
        print("SENSITIVITY ANALYSIS: Structure Design")
        print("Revenue Share % vs Target Return Multiple")
        print("="*70)
        
        results_irr = np.zeros((len(target_multiples), len(rev_shares)))
        results_tvpi = np.zeros((len(target_multiples), len(rev_shares)))
        results_dpi = np.zeros((len(target_multiples), len(rev_shares)))
        
        total_runs = len(target_multiples) * len(rev_shares)
        run_count = 0
        
        for i, target_mult in enumerate(target_multiples):
            for j, rev_share in enumerate(rev_shares):
                run_count += 1
                print(f"  Running {run_count}/{total_runs}: Rev={rev_share:.0%}, Target={target_mult}x", end='\r')
                
                mc = self._create_mc()
                mc.safer.revenue_share = rev_share
                mc.safer.target_return_multiple = target_mult
                mc.safer.target_return = mc.safer.investment * target_mult
                
                res = mc.run()
                
                results_irr[i, j] = np.nanmedian(res['net_irr'])
                results_tvpi[i, j] = np.nanmedian(res['net_tvpi'])
                results_dpi[i, j] = np.nanmedian(res['dpi'])
        
        print(f"\n  Completed {total_runs} simulations.")
        
        return {
            'net_irr': results_irr,
            'net_tvpi': results_tvpi,
            'dpi': results_dpi,
            'x_values': rev_shares,
            'y_values': target_multiples,
            'x_label': 'Revenue Share %',
            'y_label': 'Target Return Multiple (x)'
        }
    
    def run_skill_sensitivity(self,
                               high_growth_probs=[0.05, 0.10, 0.15, 0.25, 0.35, 0.50],
                               fail_hazard_mults=[0.5, 0.75, 1.0, 1.5, 2.0]):
        """
        Heatmap 2: Selection Skill (High Growth Prob) vs Risk (Failure Rate)
        Answers: Does the SAFER protect downside when we pick poorly?
        """
        print("="*70)
        print("SENSITIVITY ANALYSIS: Selection Skill Test")
        print("High Growth Probability vs Failure Hazard Multiplier")
        print("="*70)
        
        results_irr = np.zeros((len(fail_hazard_mults), len(high_growth_probs)))
        results_loss_prob = np.zeros((len(fail_hazard_mults), len(high_growth_probs)))
        results_tvpi = np.zeros((len(fail_hazard_mults), len(high_growth_probs)))
        
        total_runs = len(fail_hazard_mults) * len(high_growth_probs)
        run_count = 0
        
        for i, fail_mult in enumerate(fail_hazard_mults):
            for j, growth_prob in enumerate(high_growth_probs):
                run_count += 1
                print(f"  Running {run_count}/{total_runs}: Growth={growth_prob:.0%}, FailMult={fail_mult}x", end='\r')
                
                mc = self._create_mc()
                
                # BUG FIX #3: Use unified override system instead of duplicated code
                # Start with baseline overrides, then modify for this sensitivity point
                # The baseline failure hazard comes from args (via _create_mc)
                baseline_fail_hazard = self.args.base_fail_hazard
                mc.company_overrides.update({
                    'p_high_growth': growth_prob,
                    'p_mid_growth': (1 - growth_prob) * 0.5,
                    'p_low_growth': (1 - growth_prob) * 0.5,
                    'base_fail_hazard_q': baseline_fail_hazard * fail_mult
                })
                
                # Use main run() method - no code duplication!
                res = mc.run()
                
                results_irr[i, j] = np.nanmedian(res['net_irr'])
                results_tvpi[i, j] = np.nanmedian(res['net_tvpi'])
                
                # Calculate probability of loss (TVPI < 1.0)
                valid_tvpi = res['gross_tvpi'][np.isfinite(res['gross_tvpi'])]
                if len(valid_tvpi) > 0:
                    results_loss_prob[i, j] = np.mean(valid_tvpi < 1.0)
                else:
                    results_loss_prob[i, j] = np.nan
        
        print(f"\n  Completed {total_runs} simulations.")
        
        return {
            'net_irr': results_irr,
            'loss_probability': results_loss_prob,
            'net_tvpi': results_tvpi,
            'x_values': high_growth_probs,
            'y_values': fail_hazard_mults,
            'x_label': 'Probability of High Growth (Selection Skill)',
            'y_label': 'Failure Hazard Multiplier (Risk)'
        }
    
    # NOTE: _run_with_skill_override REMOVED - now uses unified override system in FundMonteCarlo
    
    def run_macro_sensitivity(self,
                               exit_multiple_adjustments=[-0.5, -0.3, 0.0, 0.3, 0.5],
                               correlation_strengths=[0.0, 0.25, 0.5, 0.75, 1.0]):
        """
        Heatmap 3: Macro Environment Stress Test
        Exit Multiple Compression vs Correlation Strength
        Answers: How does fund perform under market stress?
        """
        print("="*70)
        print("SENSITIVITY ANALYSIS: Macro Stress Test")
        print("Exit Multiple Adjustment vs Market Correlation")
        print("="*70)
        
        results_dpi = np.zeros((len(correlation_strengths), len(exit_multiple_adjustments)))
        results_irr = np.zeros((len(correlation_strengths), len(exit_multiple_adjustments)))
        results_tvpi = np.zeros((len(correlation_strengths), len(exit_multiple_adjustments)))
        
        total_runs = len(correlation_strengths) * len(exit_multiple_adjustments)
        run_count = 0
        
        for i, corr_strength in enumerate(correlation_strengths):
            for j, exit_adj in enumerate(exit_multiple_adjustments):
                run_count += 1
                print(f"  Running {run_count}/{total_runs}: ExitAdj={exit_adj:+.1f}, Corr={corr_strength:.0%}", end='\r')
                
                mc = self._create_mc()
                
                # BUG FIX #3: Use unified override system instead of duplicated code
                mc.market_overrides = {
                    'exit_multiple_adj': exit_adj,
                    'correlation_strength': corr_strength
                }
                
                # Use main run() method - no code duplication!
                res = mc.run()
                
                results_dpi[i, j] = np.nanmedian(res['dpi'])
                results_irr[i, j] = np.nanmedian(res['net_irr'])
                results_tvpi[i, j] = np.nanmedian(res['net_tvpi'])
        
        print(f"\n  Completed {total_runs} simulations.")
        
        return {
            'dpi': results_dpi,
            'net_irr': results_irr,
            'net_tvpi': results_tvpi,
            'x_values': exit_multiple_adjustments,
            'y_values': correlation_strengths,
            'x_label': 'Exit Multiple Adjustment (log)',
            'y_label': 'Market Correlation Strength'
        }
    
    # NOTE: _run_with_macro_override REMOVED - now uses unified override system in FundMonteCarlo
    
    @staticmethod
    def plot_heatmap(data, x_values, y_values, x_label, y_label, title, 
                     fmt=".1%", cmap="RdYlGn", figsize=(14, 10), save_path=None):
        """Generate a publication-quality heatmap for book printing (300 DPI, LANDSCAPE)."""
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=figsize)  # Landscape orientation
        
        # Format tick labels
        if all(isinstance(x, float) and x < 1 for x in x_values):
            x_labels = [f"{x:.0%}" for x in x_values]
        else:
            x_labels = [f"{x:.2f}" for x in x_values]
            
        if all(isinstance(y, float) and y < 1 for y in y_values):
            y_labels = [f"{y:.0%}" for y in y_values]
        else:
            y_labels = [f"{y:.2f}" for y in y_values]
        
        # High-resolution settings for book printing
        PRINT_DPI = 300
        
        # Enhanced font sizes for print
        annot_fontsize = 11 if len(x_values) <= 6 else 9
        
        sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap,
                   xticklabels=x_labels, yticklabels=y_labels,
                   ax=ax, cbar_kws={'label': title.split(':')[-1].strip()},
                   annot_kws={'size': annot_fontsize, 'weight': 'bold'})
        
        ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.invert_yaxis()
        
        # Increase tick label sizes
        ax.tick_params(axis='both', labelsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=PRINT_DPI, bbox_inches='tight', facecolor='white')
            plt.close()
        
        return fig, ax
    
    @staticmethod
    def print_table(data, x_values, y_values, x_label, y_label, title, fmt=".1%", value_suffix=""):
        """Print a formatted ASCII table to console."""
        print(f"\n{title}")
        print("-" * 70)
        
        # Format x-axis labels
        if all(isinstance(x, float) and x < 1 for x in x_values):
            x_labels = [f"{x:.0%}" for x in x_values]
        else:
            x_labels = [f"{x:+.2f}" if x < 0 or (isinstance(x, float) and -1 < x < 1 and x != 0) else f"{x:.2f}" for x in x_values]
        
        # Calculate column width
        col_width = max(8, max(len(xl) for xl in x_labels) + 2)
        
        # Print header
        header = f"{'':>12} |"
        for xl in x_labels:
            header += f" {xl:^{col_width}} |"
        print(header)
        print("-" * len(header))
        
        # Print rows
        for i, y_val in enumerate(y_values):
            if isinstance(y_val, float) and y_val < 1 and y_val > 0:
                y_label_str = f"{y_val:.0%}"
            else:
                y_label_str = f"{y_val:.2f}"
            
            row = f"{y_label_str:>12} |"
            for j in range(len(x_values)):
                val = data[i, j]
                if np.isnan(val):
                    cell = "N/A"
                elif fmt == ".1%":
                    cell = f"{val:.1%}"
                elif fmt == ".0%":
                    cell = f"{val:.0%}"
                elif fmt == ".2f":
                    cell = f"{val:.2f}x"
                else:
                    cell = f"{val:{fmt}}"
                row += f" {cell:^{col_width}} |"
            print(row)
        
        print("-" * len(header))
        print(f"  Rows: {y_label} | Columns: {x_label}")

    # =========================================================================
    # ADDITIONAL SENSITIVITY ANALYSES
    # =========================================================================
    
    def run_repurchase_sensitivity(self,
                                    repurchase_pcts=[0.50, 0.60, 0.70, 0.80, 0.90],
                                    honeymoon_quarters=[2, 4, 6, 8]):
        """
        SAFER-Specific: Repurchase Percentage vs Honeymoon Period
        Answers: How does equity retention vs. payment timing affect returns?
        
        Higher repurchase % = more principal returned via revenue share, less residual equity
        Longer honeymoon = delayed cash flows but more company runway
        """
        print("="*70)
        print("SENSITIVITY: SAFER Repurchase Terms")
        print("Repurchase Percentage vs Honeymoon Period")
        print("="*70)
        
        results_irr = np.zeros((len(honeymoon_quarters), len(repurchase_pcts)))
        results_tvpi = np.zeros((len(honeymoon_quarters), len(repurchase_pcts)))
        results_yield_pct = np.zeros((len(honeymoon_quarters), len(repurchase_pcts)))
        
        total_runs = len(honeymoon_quarters) * len(repurchase_pcts)
        run_count = 0
        
        for i, honeymoon in enumerate(honeymoon_quarters):
            for j, repurchase_pct in enumerate(repurchase_pcts):
                run_count += 1
                print_progress_bar(run_count, total_runs,
                                 prefix='  Progress:',
                                 suffix=f'Repurch={repurchase_pct:.0%}, Honeymoon={honeymoon}q   ')
                
                mc = self._create_mc()
                mc.safer.repurchase_percent = repurchase_pct
                mc.safer.repurchase_amount = mc.safer.investment * repurchase_pct
                mc.safer.honeymoon_quarters = honeymoon
                
                res = mc.run()
                
                results_irr[i, j] = np.nanmedian(res['net_irr'])
                results_tvpi[i, j] = np.nanmedian(res['net_tvpi'])
                
                # Calculate yield percentage
                total_return = np.sum(res['total_repurchase']) + np.sum(res['total_equity']) + np.sum(res['total_nav'])
                if total_return > 0:
                    results_yield_pct[i, j] = np.sum(res['total_repurchase']) / total_return
                else:
                    results_yield_pct[i, j] = 0
        
        print()
        return {
            'net_irr': results_irr,
            'net_tvpi': results_tvpi,
            'yield_pct': results_yield_pct,
            'x_values': repurchase_pcts,
            'y_values': honeymoon_quarters,
            'x_label': 'Repurchase Percentage',
            'y_label': 'Honeymoon Period (quarters)'
        }
    
    def run_valuation_cap_sensitivity(self,
                                       valuation_caps=[2_000_000, 3_000_000, 5_000_000, 7_500_000, 10_000_000],
                                       exit_multiples_adj=[-0.4, -0.2, 0.0, 0.2, 0.4]):
        """
        SAFER-Specific: Valuation Cap vs Exit Environment
        Answers: How does cap setting interact with exit valuations?
        
        Lower cap = more ownership, but higher risk of cash-out dominance
        Higher cap = less ownership, but more upside in big exits
        """
        print("="*70)
        print("SENSITIVITY: Valuation Cap vs Exit Environment")
        print("Valuation Cap vs Exit Multiple Adjustment")
        print("="*70)
        
        results_irr = np.zeros((len(exit_multiples_adj), len(valuation_caps)))
        results_tvpi = np.zeros((len(exit_multiples_adj), len(valuation_caps)))
        results_conversion_pct = np.zeros((len(exit_multiples_adj), len(valuation_caps)))
        
        total_runs = len(exit_multiples_adj) * len(valuation_caps)
        run_count = 0
        
        for i, exit_adj in enumerate(exit_multiples_adj):
            for j, cap in enumerate(valuation_caps):
                run_count += 1
                print_progress_bar(run_count, total_runs,
                                 prefix='  Progress:',
                                 suffix=f'Cap=${cap/1e6:.1f}M, ExitAdj={exit_adj:+.1f}   ')
                
                mc = self._create_mc()
                mc.safer.post_money_valuation_cap = cap
                
                # Use unified override system
                mc.market_overrides = {
                    'exit_multiple_adj': exit_adj,
                    'correlation_strength': 0.5
                }
                
                # Use main run() method - no code duplication!
                res = mc.run()
                
                results_irr[i, j] = np.nanmedian(res['net_irr'])
                results_tvpi[i, j] = np.nanmedian(res['net_tvpi'])
        
        print()
        return {
            'net_irr': results_irr,
            'net_tvpi': results_tvpi,
            'x_values': [c/1e6 for c in valuation_caps],  # Convert to millions for display
            'y_values': exit_multiples_adj,
            'x_label': 'Valuation Cap ($M)',
            'y_label': 'Exit Multiple Adjustment'
        }
    
    def run_portfolio_concentration_sensitivity(self,
                                                  portfolio_sizes=[10, 15, 20, 25, 30, 40],
                                                  investment_amounts=[500_000, 750_000, 1_000_000, 1_500_000]):
        """
        Portfolio Construction: Size vs Investment Amount
        Answers: Concentration risk - fewer big bets vs more small bets?
        
        NOTE: Fund size and management fees scale proportionally with total deployment
        to ensure apples-to-apples comparison. Fee ratio is kept constant at 2% of 
        deployed capital per year (industry standard).
        """
        print("="*70)
        print("SENSITIVITY: Portfolio Construction")
        print("Number of Companies vs Investment per Company")
        print("(Fund size & fees scale proportionally with deployment)")
        print("="*70)
        
        results_irr = np.zeros((len(investment_amounts), len(portfolio_sizes)))
        results_tvpi = np.zeros((len(investment_amounts), len(portfolio_sizes)))
        results_loss_prob = np.zeros((len(investment_amounts), len(portfolio_sizes)))
        results_tvpi_std = np.zeros((len(investment_amounts), len(portfolio_sizes)))
        
        total_runs = len(investment_amounts) * len(portfolio_sizes)
        run_count = 0
        
        # Base fee ratio: 2% of deployed capital per year (industry standard)
        FEE_RATIO = 0.02
        
        for i, inv_amt in enumerate(investment_amounts):
            for j, port_size in enumerate(portfolio_sizes):
                run_count += 1
                
                # Calculate scaled fund parameters
                total_deployment = inv_amt * port_size
                scaled_fund_size = total_deployment * 1.25  # 20% buffer for fees
                scaled_mgmt_fee = total_deployment * FEE_RATIO  # 2% of deployed capital
                
                print_progress_bar(run_count, total_runs,
                                 prefix='  Progress:',
                                 suffix=f'N={port_size}, Inv=${inv_amt/1000:.0f}k, Fund=${scaled_fund_size/1e6:.1f}M   ')
                
                mc = self._create_mc(
                    portfolio_size=port_size,
                    fund_size=scaled_fund_size,
                    mgmt_fee_per_year=scaled_mgmt_fee
                )
                mc.safer.investment = inv_amt
                mc.safer.target_return = inv_amt * mc.safer.target_return_multiple
                mc.safer.repurchase_amount = inv_amt * mc.safer.repurchase_percent
                
                res = mc.run()
                
                results_irr[i, j] = np.nanmedian(res['net_irr'])
                results_tvpi[i, j] = np.nanmedian(res['net_tvpi'])
                
                valid_tvpi = res['gross_tvpi'][np.isfinite(res['gross_tvpi'])]
                if len(valid_tvpi) > 0:
                    results_loss_prob[i, j] = np.mean(valid_tvpi < 1.0)
                    results_tvpi_std[i, j] = np.std(valid_tvpi)
        
        print()
        return {
            'net_irr': results_irr,
            'net_tvpi': results_tvpi,
            'loss_probability': results_loss_prob,
            'tvpi_volatility': results_tvpi_std,
            'x_values': portfolio_sizes,
            'y_values': [a/1000 for a in investment_amounts],  # Convert to thousands for display
            'x_label': 'Number of Portfolio Companies',
            'y_label': 'Investment per Company ($k)'
        }
    
    def run_company_quality_sensitivity(self,
                                         initial_revenues=[50_000, 100_000, 150_000, 250_000, 500_000],
                                         exit_multiples=[3.0, 4.5, 6.0, 8.0, 10.0]):
        """
        Company Quality: Initial Revenue vs Exit Multiple Environment
        Answers: How does deal quality (revenue traction, exit potential) affect returns?
        """
        print("="*70)
        print("SENSITIVITY: Company Quality")
        print("Initial Revenue vs Exit Multiple")
        print("="*70)
        
        results_irr = np.zeros((len(exit_multiples), len(initial_revenues)))
        results_tvpi = np.zeros((len(exit_multiples), len(initial_revenues)))
        results_time_to_1x = np.zeros((len(exit_multiples), len(initial_revenues)))
        
        total_runs = len(exit_multiples) * len(initial_revenues)
        run_count = 0
        
        for i, exit_mult in enumerate(exit_multiples):
            for j, init_rev in enumerate(initial_revenues):
                run_count += 1
                print_progress_bar(run_count, total_runs,
                                 prefix='  Progress:',
                                 suffix=f'InitRev=${init_rev/1000:.0f}k, ExitMult={exit_mult}x   ')
                
                mc = self._create_mc()
                
                # Use unified override system - update baseline, don't replace
                mc.company_overrides.update({
                    'init_rev_logn_mu': np.log(init_rev),
                    'exit_multiple_mu': np.log(exit_mult)
                })
                
                # Use main run() method - no code duplication!
                res = mc.run()
                
                results_irr[i, j] = np.nanmedian(res['net_irr'])
                results_tvpi[i, j] = np.nanmedian(res['net_tvpi'])
                
                valid_time = res['time_to_1x_dpi'][np.isfinite(res['time_to_1x_dpi'])]
                results_time_to_1x[i, j] = np.nanmedian(valid_time) if len(valid_time) > 0 else np.nan
        
        print()
        return {
            'net_irr': results_irr,
            'net_tvpi': results_tvpi,
            'time_to_1x': results_time_to_1x,
            'x_values': [r/1000 for r in initial_revenues],  # Convert to thousands
            'y_values': exit_multiples,
            'x_label': 'Initial Revenue ($k/year)',
            'y_label': 'Median Exit Multiple (x Revenue)'
        }
    
    # NOTE: _run_with_company_override REMOVED - now uses unified override system in FundMonteCarlo
    
    def run_growth_failure_tradeoff(self,
                                     high_growth_probs=[0.10, 0.20, 0.30, 0.40, 0.50],
                                     high_growth_rates=[0.5, 0.75, 1.0, 1.25, 1.5]):
        """
        Company Dynamics: Growth Probability vs Growth Magnitude
        Answers: Is it better to have more companies growing or faster growth?
        """
        print("="*70)
        print("SENSITIVITY: Growth Dynamics")
        print("High Growth Probability vs High Growth Rate")
        print("="*70)
        
        results_irr = np.zeros((len(high_growth_rates), len(high_growth_probs)))
        results_tvpi = np.zeros((len(high_growth_rates), len(high_growth_probs)))
        results_target_hit_rate = np.zeros((len(high_growth_rates), len(high_growth_probs)))
        
        total_runs = len(high_growth_rates) * len(high_growth_probs)
        run_count = 0
        
        for i, growth_rate in enumerate(high_growth_rates):
            for j, growth_prob in enumerate(high_growth_probs):
                run_count += 1
                print_progress_bar(run_count, total_runs,
                                 prefix='  Progress:',
                                 suffix=f'P(high)={growth_prob:.0%}, Rate={growth_rate:.0%}/yr   ')
                
                mc = self._create_mc()
                
                # Use unified override system - update baseline, don't replace
                mc.company_overrides.update({
                    'p_high_growth': growth_prob,
                    'p_mid_growth': (1 - growth_prob) * 0.6,
                    'p_low_growth': (1 - growth_prob) * 0.4,
                    'high_mu_yr': growth_rate
                })
                
                # Use main run() method - no code duplication!
                res = mc.run()
                
                results_irr[i, j] = np.nanmedian(res['net_irr'])
                results_tvpi[i, j] = np.nanmedian(res['net_tvpi'])
                
                # Calculate target hit rate from outcomes
                total = res['label_totals'].sum()
                target_hit = res['label_totals'].get('Target Return Repaid', 0) + \
                            res['label_totals'].get('Target + Liquidity Event', 0)
                results_target_hit_rate[i, j] = target_hit / total if total > 0 else 0
        
        print()
        return {
            'net_irr': results_irr,
            'net_tvpi': results_tvpi,
            'target_hit_rate': results_target_hit_rate,
            'x_values': high_growth_probs,
            'y_values': high_growth_rates,
            'x_label': 'Probability of High Growth',
            'y_label': 'High Growth Rate (%/year)'
        }
    
    # NOTE: _run_with_growth_override REMOVED - now uses unified override system in FundMonteCarlo


def print_progress_bar(iteration, total, prefix='', suffix='', length=40, fill='█'):
    """Print a progress bar to console."""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()


def run_sensitivity_analysis(args, quick_mode=False):
    """
    Run all three sensitivity analyses and generate heatmaps.
    
    All variations are relative to the baseline parameters from args.
    
    Args:
        args: Parsed command line arguments (baseline configuration)
        quick_mode: If True, use fewer iterations for faster results (for testing)
    """
    import time
    start_time = time.time()
    
    print("\n" + "="*70)
    print("SAFER FUND SENSITIVITY ANALYSIS SUITE")
    print("Institutional Due Diligence Package")
    print("="*70)
    
    if quick_mode:
        print("\n⚡ QUICK MODE: Using reduced iterations for speed")
        base_iter = 100
    else:
        base_iter = 300
        print(f"\n📊 Full analysis mode: {base_iter} iterations per scenario")
    
    # Show baseline configuration
    print(f"\n📋 Baseline Configuration (from CLI args):")
    print(f"   Portfolio: {args.portfolio_size} companies × ${args.investment:,.0f}")
    print(f"   SAFER Terms: {args.revenue_share:.0%} rev share, {args.target_multiple}x target")
    print(f"   Company Model: {args.base_fail_hazard:.0%} base failure, {args.p_high_growth:.0%}/{args.p_mid_growth:.0%}/{args.p_low_growth:.0%} growth mix")
    
    analyzer = SensitivityAnalysis(args=args, base_iterations=base_iter)
    
    # Calculate total scenarios
    struct_scenarios = 5 * 6  # rev_shares × target_multiples
    skill_scenarios = 5 * 5   # growth_probs × fail_mults
    macro_scenarios = 5 * 5   # exit_adj × corr_strength
    total_scenarios = struct_scenarios + skill_scenarios + macro_scenarios
    
    print(f"   Total scenarios to simulate: {total_scenarios}")
    print(f"   Estimated time: {total_scenarios * base_iter * 0.015:.0f}-{total_scenarios * base_iter * 0.025:.0f} seconds\n")
    
    # =========================================================================
    # 1. Structure Sensitivity
    # =========================================================================
    print("\n" + "="*70)
    print("[1/3] STRUCTURE SENSITIVITY ANALYSIS")
    print("      Revenue Share % vs Target Return Multiple")
    print("="*70)
    
    rev_shares = [0.03, 0.04, 0.05, 0.06, 0.08]
    target_multiples = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    
    results_irr = np.zeros((len(target_multiples), len(rev_shares)))
    results_tvpi = np.zeros((len(target_multiples), len(rev_shares)))
    results_dpi = np.zeros((len(target_multiples), len(rev_shares)))
    
    total_runs = len(target_multiples) * len(rev_shares)
    run_count = 0
    
    for i, target_mult in enumerate(target_multiples):
        for j, rev_share in enumerate(rev_shares):
            run_count += 1
            print_progress_bar(run_count, total_runs, 
                             prefix=f'  Progress:', 
                             suffix=f'Rev={rev_share:.0%}, Target={target_mult}x   ')
            
            mc = analyzer._create_mc()
            mc.safer.revenue_share = rev_share
            mc.safer.target_return_multiple = target_mult
            mc.safer.target_return = mc.safer.investment * target_mult
            
            res = mc.run()
            
            results_irr[i, j] = np.nanmedian(res['net_irr'])
            results_tvpi[i, j] = np.nanmedian(res['net_tvpi'])
            results_dpi[i, j] = np.nanmedian(res['dpi'])
    
    structure_results = {
        'net_irr': results_irr,
        'net_tvpi': results_tvpi,
        'dpi': results_dpi,
        'x_values': rev_shares,
        'y_values': target_multiples,
        'x_label': 'Revenue Share %',
        'y_label': 'Target Return Multiple (x)'
    }
    
    # Print tabular results
    SensitivityAnalysis.print_table(
        results_irr, rev_shares, target_multiples,
        "Revenue Share %", "Target Multiple",
        "NET IRR by Structure Parameters", fmt=".1%"
    )
    
    SensitivityAnalysis.print_table(
        results_tvpi, rev_shares, target_multiples,
        "Revenue Share %", "Target Multiple",
        "NET TVPI by Structure Parameters", fmt=".2f"
    )
    
    # =========================================================================
    # 2. Selection Skill Sensitivity
    # =========================================================================
    print("\n" + "="*70)
    print("[2/3] SELECTION SKILL SENSITIVITY ANALYSIS")
    print("      High Growth Probability vs Failure Hazard")
    print("="*70)
    
    high_growth_probs = [0.05, 0.15, 0.25, 0.35, 0.50]
    fail_hazard_mults = [0.5, 0.75, 1.0, 1.5, 2.0]
    
    results_irr_skill = np.zeros((len(fail_hazard_mults), len(high_growth_probs)))
    results_loss_prob = np.zeros((len(fail_hazard_mults), len(high_growth_probs)))
    results_tvpi_skill = np.zeros((len(fail_hazard_mults), len(high_growth_probs)))
    
    total_runs = len(fail_hazard_mults) * len(high_growth_probs)
    run_count = 0
    
    for i, fail_mult in enumerate(fail_hazard_mults):
        for j, growth_prob in enumerate(high_growth_probs):
            run_count += 1
            print_progress_bar(run_count, total_runs,
                             prefix=f'  Progress:',
                             suffix=f'Growth={growth_prob:.0%}, Fail={fail_mult}x   ')
            
            mc = analyzer._create_mc()
            
            # Use unified override system - update baseline, don't replace
            # The baseline failure hazard comes from args
            baseline_fail_hazard = args.base_fail_hazard
            mc.company_overrides.update({
                'p_high_growth': growth_prob,
                'p_mid_growth': (1 - growth_prob) * 0.5,
                'p_low_growth': (1 - growth_prob) * 0.5,
                'base_fail_hazard_q': baseline_fail_hazard * fail_mult
            })
            
            # Use main run() method - no code duplication!
            res = mc.run()
            
            results_irr_skill[i, j] = np.nanmedian(res['net_irr'])
            results_tvpi_skill[i, j] = np.nanmedian(res['net_tvpi'])
            
            valid_tvpi = res['gross_tvpi'][np.isfinite(res['gross_tvpi'])]
            if len(valid_tvpi) > 0:
                results_loss_prob[i, j] = np.mean(valid_tvpi < 1.0)
            else:
                results_loss_prob[i, j] = np.nan
    
    skill_results = {
        'net_irr': results_irr_skill,
        'loss_probability': results_loss_prob,
        'net_tvpi': results_tvpi_skill,
        'x_values': high_growth_probs,
        'y_values': fail_hazard_mults,
        'x_label': 'Probability of High Growth (Selection Skill)',
        'y_label': 'Failure Hazard Multiplier (Risk)'
    }
    
    # Print tabular results
    SensitivityAnalysis.print_table(
        results_loss_prob, high_growth_probs, fail_hazard_mults,
        "High Growth Prob", "Fail Hazard Mult",
        "PROBABILITY OF LOSS (TVPI < 1.0x)", fmt=".0%"
    )
    
    SensitivityAnalysis.print_table(
        results_irr_skill, high_growth_probs, fail_hazard_mults,
        "High Growth Prob", "Fail Hazard Mult",
        "NET IRR by Selection Skill", fmt=".1%"
    )
    
    # =========================================================================
    # 3. Macro Stress Test
    # =========================================================================
    print("\n" + "="*70)
    print("[3/3] MACRO STRESS TEST")
    print("      Exit Multiple Adjustment vs Market Correlation")
    print("="*70)
    
    exit_multiple_adjustments = [-0.5, -0.25, 0.0, 0.25, 0.5]
    correlation_strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    results_dpi_macro = np.zeros((len(correlation_strengths), len(exit_multiple_adjustments)))
    results_irr_macro = np.zeros((len(correlation_strengths), len(exit_multiple_adjustments)))
    results_tvpi_macro = np.zeros((len(correlation_strengths), len(exit_multiple_adjustments)))
    
    total_runs = len(correlation_strengths) * len(exit_multiple_adjustments)
    run_count = 0
    
    for i, corr_strength in enumerate(correlation_strengths):
        for j, exit_adj in enumerate(exit_multiple_adjustments):
            run_count += 1
            print_progress_bar(run_count, total_runs,
                             prefix=f'  Progress:',
                             suffix=f'ExitAdj={exit_adj:+.2f}, Corr={corr_strength:.0%}   ')
            
            mc = analyzer._create_mc()
            
            # Use unified override system
            mc.market_overrides = {
                'exit_multiple_adj': exit_adj,
                'correlation_strength': corr_strength
            }
            
            # Use main run() method - no code duplication!
            res = mc.run()
            
            results_dpi_macro[i, j] = np.nanmedian(res['dpi'])
            results_irr_macro[i, j] = np.nanmedian(res['net_irr'])
            results_tvpi_macro[i, j] = np.nanmedian(res['net_tvpi'])
    
    macro_results = {
        'dpi': results_dpi_macro,
        'net_irr': results_irr_macro,
        'net_tvpi': results_tvpi_macro,
        'x_values': exit_multiple_adjustments,
        'y_values': correlation_strengths,
        'x_label': 'Exit Multiple Adjustment (log)',
        'y_label': 'Market Correlation Strength'
    }
    
    # Print tabular results
    SensitivityAnalysis.print_table(
        results_dpi_macro, exit_multiple_adjustments, correlation_strengths,
        "Exit Mult Adj", "Correlation",
        "DPI (Cash-on-Cash) by Macro Scenario", fmt=".2f"
    )
    
    SensitivityAnalysis.print_table(
        results_irr_macro, exit_multiple_adjustments, correlation_strengths,
        "Exit Mult Adj", "Correlation",
        "NET IRR by Macro Scenario", fmt=".1%"
    )
    
    # =========================================================================
    # Generate Heatmap Images
    # =========================================================================
    print("\n" + "="*70)
    print("GENERATING HEATMAP IMAGES")
    print("="*70)
    
    print("  Saving: heatmap_structure_irr.png")
    SensitivityAnalysis.plot_heatmap(
        structure_results['net_irr'],
        structure_results['x_values'],
        structure_results['y_values'],
        structure_results['x_label'],
        structure_results['y_label'],
        "Structure Sensitivity: Net IRR",
        fmt=".1%", cmap="RdYlGn",
        save_path="heatmap_structure_irr.png"
    )
    
    print("  Saving: heatmap_skill_loss.png")
    SensitivityAnalysis.plot_heatmap(
        skill_results['loss_probability'],
        skill_results['x_values'],
        skill_results['y_values'],
        skill_results['x_label'],
        skill_results['y_label'],
        "Selection Skill: Probability of Loss (TVPI < 1.0x)",
        fmt=".0%", cmap="RdYlGn_r",
        save_path="heatmap_skill_loss.png"
    )
    
    print("  Saving: heatmap_macro_dpi.png")
    SensitivityAnalysis.plot_heatmap(
        macro_results['dpi'],
        macro_results['x_values'],
        macro_results['y_values'],
        macro_results['x_label'],
        macro_results['y_label'],
        "Macro Stress Test: DPI (Cash-on-Cash)",
        fmt=".2f", cmap="RdYlGn",
        save_path="heatmap_macro_dpi.png"
    )
    
    # =========================================================================
    # Summary Insights
    # =========================================================================
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("="*70)
    
    # Find optimal structure
    best_irr_idx = np.unravel_index(np.nanargmax(structure_results['net_irr']), 
                                     structure_results['net_irr'].shape)
    best_rev = structure_results['x_values'][best_irr_idx[1]]
    best_target = structure_results['y_values'][best_irr_idx[0]]
    best_irr = structure_results['net_irr'][best_irr_idx]
    
    print(f"\n1. OPTIMAL STRUCTURE:")
    print(f"   Best Net IRR: {best_irr:.1%}")
    print(f"   At: Revenue Share = {best_rev:.0%}, Target Multiple = {best_target}x")
    
    # Downside protection analysis
    worst_skill_idx = (len(skill_results['y_values'])-1, 0)  # High fail, low growth
    worst_case_loss = skill_results['loss_probability'][worst_skill_idx]
    worst_case_irr = skill_results['net_irr'][worst_skill_idx]
    
    print(f"\n2. DOWNSIDE PROTECTION (Worst Case: Bad Picking + High Failure):")
    print(f"   Loss Probability: {worst_case_loss:.0%}")
    print(f"   Net IRR: {worst_case_irr:.1%}")
    
    # Compare to best case
    best_skill_idx = (0, -1)  # Low fail, high growth
    best_case_loss = skill_results['loss_probability'][best_skill_idx]
    best_case_irr = skill_results['net_irr'][best_skill_idx]
    print(f"\n   Best Case (Good Picking + Low Failure):")
    print(f"   Loss Probability: {best_case_loss:.0%}")
    print(f"   Net IRR: {best_case_irr:.1%}")
    
    # Macro resilience
    worst_macro_idx = (len(macro_results['y_values'])-1, 0)  # High corr, compressed multiples
    worst_macro_dpi = macro_results['dpi'][worst_macro_idx]
    base_macro_idx = (2, 2)  # Mid correlation, no adjustment
    base_macro_dpi = macro_results['dpi'][base_macro_idx]
    
    print(f"\n3. MACRO RESILIENCE:")
    print(f"   Base Case DPI: {base_macro_dpi:.2f}x")
    print(f"   Stress Case DPI (market crash): {worst_macro_dpi:.2f}x")
    print(f"   DPI Preservation: {worst_macro_dpi/base_macro_dpi:.0%} of base case")
    print(f"   (Revenue share provides liquidity even when exits dry up)")
    
    # Timing
    elapsed = time.time() - start_time
    print(f"\n" + "="*70)
    print(f"Analysis completed in {elapsed:.1f} seconds")
    print(f"Heatmaps saved to current directory")
    print("="*70)
    
    return {
        'structure': structure_results,
        'skill': skill_results,
        'macro': macro_results
    }


def run_extended_sensitivity_analysis(args, quick_mode=False):
    """
    Run EXTENDED sensitivity analysis covering all SAFER and portfolio parameters.
    
    All variations are relative to the baseline parameters from args.
    This is the comprehensive version for deep institutional due diligence.
    """
    import time
    start_time = time.time()
    
    print("\n" + "="*78)
    print("█" + " "*76 + "█")
    print("█" + "COMPREHENSIVE SAFER SENSITIVITY ANALYSIS".center(76) + "█")
    print("█" + "Extended Institutional Due Diligence Package".center(76) + "█")
    print("█" + " "*76 + "█")
    print("="*78)
    
    if quick_mode:
        print("\n⚡ QUICK MODE: Using reduced iterations for speed")
        base_iter = 75
    else:
        base_iter = 200
        print(f"\n📊 Full analysis mode: {base_iter} iterations per scenario")
    
    # Show baseline configuration
    print(f"\n📋 Baseline Configuration (from CLI args):")
    print(f"   Portfolio: {args.portfolio_size} companies × ${args.investment:,.0f}")
    print(f"   SAFER Terms: {args.revenue_share:.0%} rev share, {args.target_multiple}x target")
    print(f"   Company Model: {args.base_fail_hazard:.0%} base failure, {args.p_high_growth:.0%}/{args.p_mid_growth:.0%}/{args.p_low_growth:.0%} growth mix")
    
    analyzer = SensitivityAnalysis(args=args, base_iterations=base_iter)
    
    all_results = {}
    
    # =========================================================================
    # CATEGORY 1: SAFER INSTRUMENT PARAMETERS
    # =========================================================================
    print("\n" + "="*78)
    print("CATEGORY 1: SAFER INSTRUMENT SENSITIVITY")
    print("="*78)
    
    # 1a. Revenue Share vs Target Return (original)
    print("\n[1a] Revenue Share % vs Target Return Multiple")
    res = analyzer.run_structure_sensitivity(
        rev_shares=[0.03, 0.04, 0.05, 0.06, 0.08],
        target_multiples=[2.0, 2.5, 3.0, 4.0, 5.0]
    )
    SensitivityAnalysis.print_table(
        res['net_irr'], res['x_values'], res['y_values'],
        res['x_label'], res['y_label'],
        "NET IRR: Revenue Share vs Target Multiple", fmt=".1%"
    )
    all_results['structure'] = res
    
    # 1b. Repurchase Percentage vs Honeymoon Period (NEW)
    print("\n[1b] Repurchase Percentage vs Honeymoon Period")
    res = analyzer.run_repurchase_sensitivity(
        repurchase_pcts=[0.50, 0.60, 0.70, 0.80, 0.90],
        honeymoon_quarters=[2, 4, 6, 8]
    )
    SensitivityAnalysis.print_table(
        res['net_irr'], res['x_values'], res['y_values'],
        res['x_label'], res['y_label'],
        "NET IRR: Repurchase % vs Honeymoon", fmt=".1%"
    )
    SensitivityAnalysis.print_table(
        res['yield_pct'], res['x_values'], res['y_values'],
        res['x_label'], res['y_label'],
        "YIELD % OF RETURN: Repurchase % vs Honeymoon", fmt=".0%"
    )
    all_results['repurchase'] = res
    
    # 1c. Valuation Cap vs Exit Environment (NEW)
    print("\n[1c] Valuation Cap vs Exit Environment")
    res = analyzer.run_valuation_cap_sensitivity(
        valuation_caps=[2_000_000, 3_500_000, 5_000_000, 7_500_000, 10_000_000],
        exit_multiples_adj=[-0.4, -0.2, 0.0, 0.2, 0.4]
    )
    SensitivityAnalysis.print_table(
        res['net_irr'], res['x_values'], res['y_values'],
        res['x_label'], res['y_label'],
        "NET IRR: Valuation Cap ($M) vs Exit Adjustment", fmt=".1%"
    )
    all_results['valuation_cap'] = res
    
    # =========================================================================
    # CATEGORY 2: COMPANY/PORTFOLIO PARAMETERS
    # =========================================================================
    print("\n" + "="*78)
    print("CATEGORY 2: COMPANY & PORTFOLIO SENSITIVITY")
    print("="*78)
    
    # 2a. Selection Skill (original)
    print("\n[2a] Selection Skill: Growth Probability vs Failure Rate")
    res = analyzer.run_skill_sensitivity(
        high_growth_probs=[0.10, 0.20, 0.30, 0.40, 0.50],
        fail_hazard_mults=[0.5, 0.75, 1.0, 1.5, 2.0]
    )
    SensitivityAnalysis.print_table(
        res['loss_probability'], res['x_values'], res['y_values'],
        res['x_label'], res['y_label'],
        "LOSS PROBABILITY: Growth Prob vs Failure Mult", fmt=".0%"
    )
    SensitivityAnalysis.print_table(
        res['net_irr'], res['x_values'], res['y_values'],
        res['x_label'], res['y_label'],
        "NET IRR: Growth Prob vs Failure Mult", fmt=".1%"
    )
    all_results['skill'] = res
    
    # 2b. Portfolio Concentration (NEW)
    print("\n[2b] Portfolio Concentration: Size vs Investment Amount")
    res = analyzer.run_portfolio_concentration_sensitivity(
        portfolio_sizes=[10, 15, 20, 25, 35],
        investment_amounts=[500_000, 750_000, 1_000_000, 1_250_000]
    )
    SensitivityAnalysis.print_table(
        res['net_irr'], res['x_values'], res['y_values'],
        res['x_label'], res['y_label'],
        "NET IRR: Portfolio Size vs Investment ($k)", fmt=".1%"
    )
    SensitivityAnalysis.print_table(
        res['loss_probability'], res['x_values'], res['y_values'],
        res['x_label'], res['y_label'],
        "LOSS PROBABILITY: Portfolio Size vs Investment", fmt=".0%"
    )
    all_results['concentration'] = res
    
    # 2c. Company Quality (NEW)
    print("\n[2c] Company Quality: Initial Revenue vs Exit Multiple")
    res = analyzer.run_company_quality_sensitivity(
        initial_revenues=[50_000, 100_000, 200_000, 400_000],
        exit_multiples=[3.0, 5.0, 7.0, 10.0]
    )
    SensitivityAnalysis.print_table(
        res['net_irr'], res['x_values'], res['y_values'],
        res['x_label'], res['y_label'],
        "NET IRR: Initial Revenue ($k) vs Exit Multiple", fmt=".1%"
    )
    SensitivityAnalysis.print_table(
        res['time_to_1x'], res['x_values'], res['y_values'],
        res['x_label'], res['y_label'],
        "TIME TO 1x DPI (years): Revenue vs Exit Mult", fmt=".1f"
    )
    all_results['quality'] = res
    
    # 2d. Growth Dynamics (NEW)
    print("\n[2d] Growth Dynamics: Probability vs Rate")
    res = analyzer.run_growth_failure_tradeoff(
        high_growth_probs=[0.15, 0.25, 0.35, 0.45],
        high_growth_rates=[0.5, 0.75, 1.0, 1.5]
    )
    SensitivityAnalysis.print_table(
        res['net_irr'], res['x_values'], res['y_values'],
        res['x_label'], res['y_label'],
        "NET IRR: Growth Probability vs Growth Rate", fmt=".1%"
    )
    SensitivityAnalysis.print_table(
        res['target_hit_rate'], res['x_values'], res['y_values'],
        res['x_label'], res['y_label'],
        "TARGET HIT RATE: Growth Prob vs Rate", fmt=".0%"
    )
    all_results['growth'] = res
    
    # =========================================================================
    # CATEGORY 3: MACRO/MARKET PARAMETERS
    # =========================================================================
    print("\n" + "="*78)
    print("CATEGORY 3: MACRO & MARKET SENSITIVITY")
    print("="*78)
    
    # 3a. Macro Stress Test (original)
    print("\n[3a] Macro Stress: Exit Multiples vs Correlation")
    res = analyzer.run_macro_sensitivity(
        exit_multiple_adjustments=[-0.5, -0.25, 0.0, 0.25, 0.5],
        correlation_strengths=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    SensitivityAnalysis.print_table(
        res['dpi'], res['x_values'], res['y_values'],
        res['x_label'], res['y_label'],
        "DPI: Exit Adjustment vs Correlation", fmt=".2f"
    )
    SensitivityAnalysis.print_table(
        res['net_irr'], res['x_values'], res['y_values'],
        res['x_label'], res['y_label'],
        "NET IRR: Exit Adjustment vs Correlation", fmt=".1%"
    )
    all_results['macro'] = res
    
    # =========================================================================
    # Generate Heatmap Images
    # =========================================================================
    print("\n" + "="*78)
    print("GENERATING HEATMAP IMAGES")
    print("="*78)
    
    heatmaps = [
        ('structure', 'net_irr', "Structure: Net IRR", ".1%", "RdYlGn", "heatmap_1a_structure.png"),
        ('repurchase', 'net_irr', "Repurchase Terms: Net IRR", ".1%", "RdYlGn", "heatmap_1b_repurchase.png"),
        ('valuation_cap', 'net_irr', "Valuation Cap: Net IRR", ".1%", "RdYlGn", "heatmap_1c_valcap.png"),
        ('skill', 'loss_probability', "Selection Skill: Loss Prob", ".0%", "RdYlGn_r", "heatmap_2a_skill.png"),
        ('concentration', 'loss_probability', "Concentration: Loss Prob", ".0%", "RdYlGn_r", "heatmap_2b_concentration.png"),
        ('quality', 'net_irr', "Company Quality: Net IRR", ".1%", "RdYlGn", "heatmap_2c_quality.png"),
        ('growth', 'target_hit_rate', "Growth: Target Hit Rate", ".0%", "RdYlGn", "heatmap_2d_growth.png"),
        ('macro', 'dpi', "Macro Stress: DPI", ".2f", "RdYlGn", "heatmap_3a_macro.png"),
    ]
    
    for key, metric, title, fmt, cmap, filename in heatmaps:
        if key in all_results and metric in all_results[key]:
            print(f"  Saving: {filename}")
            SensitivityAnalysis.plot_heatmap(
                all_results[key][metric],
                all_results[key]['x_values'],
                all_results[key]['y_values'],
                all_results[key]['x_label'],
                all_results[key]['y_label'],
                title, fmt=fmt, cmap=cmap, save_path=filename
            )
    
    # =========================================================================
    # Summary Insights
    # =========================================================================
    print("\n" + "="*78)
    print("COMPREHENSIVE SENSITIVITY SUMMARY")
    print("="*78)
    
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ KEY FINDINGS                                                                │
├─────────────────────────────────────────────────────────────────────────────┤
""")
    
    # Structure optimal
    if 'structure' in all_results:
        r = all_results['structure']
        best_idx = np.unravel_index(np.nanargmax(r['net_irr']), r['net_irr'].shape)
        print(f"│ 1. SAFER STRUCTURE OPTIMIZATION                                             │")
        print(f"│    Optimal: Rev Share={r['x_values'][best_idx[1]]:.0%}, Target={r['y_values'][best_idx[0]]}x │")
        print(f"│    Peak Net IRR: {r['net_irr'][best_idx]:.1%}                                              │")
    
    # Repurchase insight
    if 'repurchase' in all_results:
        r = all_results['repurchase']
        print(f"│                                                                             │")
        print(f"│ 2. REPURCHASE TERMS INSIGHT                                                │")
        print(f"│    Higher repurchase % → More yield, less equity upside                    │")
        print(f"│    Longer honeymoon → Better for high-growth, worse for struggling cos     │")
    
    # Portfolio concentration
    if 'concentration' in all_results:
        r = all_results['concentration']
        print(f"│                                                                             │")
        print(f"│ 3. PORTFOLIO CONCENTRATION                                                 │")
        min_loss = np.nanmin(r['loss_probability'])
        min_idx = np.unravel_index(np.nanargmin(r['loss_probability']), r['loss_probability'].shape)
        print(f"│    Lowest loss prob ({min_loss:.0%}) at N={r['x_values'][min_idx[1]]}, ${r['y_values'][min_idx[0]]:.0f}k │")
    
    # Downside protection
    if 'skill' in all_results:
        r = all_results['skill']
        worst_idx = (-1, 0)  # High fail, low growth
        best_idx = (0, -1)   # Low fail, high growth
        print(f"│                                                                             │")
        print(f"│ 4. DOWNSIDE PROTECTION (SAFER vs Pure Equity)                              │")
        print(f"│    Worst case (bad picks + high failure): {r['loss_probability'][worst_idx]:.0%} loss probability     │")
        print(f"│    This is the SAFER's key value proposition                               │")
    
    print("""│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")
    
    elapsed = time.time() - start_time
    print(f"Analysis completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Generated {len(heatmaps)} heatmap images")
    print("="*78)
    
    return all_results


def run_unicorn_free_comparison(args, max_exit_cap=50_000_000.0):
    """
    UNICORN-FREE STRESS TEST
    
    The most powerful sales tool for SAFER: proving the fund works even
    without outlier exits. Traditional VC math requires unicorns to survive;
    SAFER portfolios can generate attractive returns from base hits alone.
    
    This function runs the simulation twice using the SAME parameters from args:
    1. Standard mode (full power law distribution)
    2. Unicorn-free mode (exits capped at max_exit_cap)
    
    Then compares metrics and generates publication-quality charts.
    
    Args:
        args: Parsed command line arguments (all fund/SAFER params come from here)
        max_exit_cap: Maximum exit valuation for unicorn-free mode (default $50M)
    """
    import time
    start_time = time.time()
    
    print("\n" + "="*78)
    print("█" + " "*76 + "█")
    print("█" + "UNICORN-FREE STRESS TEST".center(76) + "█")
    print("█" + "Does the Fund Survive Without Power Law Outliers?".center(76) + "█")
    print("█" + " "*76 + "█")
    print("="*78)
    
    print(f"\n📊 Running {args.iterations:,} iterations per scenario")
    print("   Standard Mode: Full exit distribution (up to $5B)")
    print(f"   Unicorn-Free:  Exits capped at ${max_exit_cap/1e6:.0f}M (base hits only)\n")
    
    # Use a fixed seed for comparability between the two runs
    if args.seed is None:
        seed = np.random.default_rng().integers(0, 2**31)
    else:
        seed = args.seed
    
    # =========================================================================
    # 1. Run Standard Simulation (no exit cap)
    # =========================================================================
    print("Running Standard Simulation...")
    # Temporarily ensure no max_exit_value for standard run
    original_max_exit = args.max_exit_value
    args.max_exit_value = None
    mc_std, _ = create_fund_mc_from_args(args, seed=seed)
    args.max_exit_value = original_max_exit  # Restore
    
    res_std = mc_std.run()
    print("  ✓ Standard simulation complete")
    
    # =========================================================================
    # 2. Run Unicorn-Free Simulation (with exit cap)
    # =========================================================================
    print(f"Running Unicorn-Free Simulation (exits capped at ${max_exit_cap/1e6:.0f}M)...")
    mc_safe, _ = create_fund_mc_from_args(args, seed=seed, max_exit_override=max_exit_cap)
    res_safe = mc_safe.run()
    print("  ✓ Unicorn-free simulation complete")
    
    # =========================================================================
    # 3. Compare Metrics
    # =========================================================================
    print("\n" + "="*78)
    print("RESULTS COMPARISON")
    print("="*78)
    
    # Calculate medians
    med_irr_std = np.nanmedian(res_std['net_irr'])
    med_irr_safe = np.nanmedian(res_safe['net_irr'])
    med_tvpi_std = np.nanmedian(res_std['net_tvpi'])
    med_tvpi_safe = np.nanmedian(res_safe['net_tvpi'])
    med_dpi_std = np.nanmedian(res_std['dpi'])
    med_dpi_safe = np.nanmedian(res_safe['dpi'])
    
    # Loss probability
    loss_prob_std = np.mean(res_std['net_tvpi'] < 1.0) * 100
    loss_prob_safe = np.mean(res_safe['net_tvpi'] < 1.0) * 100
    
    # Max exit analysis
    max_exits_std = res_std['max_exit_values']
    unicorn_count = np.sum(max_exits_std > 100_000_000)  # Exits > $100M
    
    print(f"\n{'Metric':<20} | {'Standard':<15} | {'Unicorn-Free':<15} | {'Impact':<12}")
    print("-" * 70)
    print(f"{'Net IRR (median)':<20} | {med_irr_std:>14.1%} | {med_irr_safe:>14.1%} | {med_irr_safe - med_irr_std:>+11.1%}")
    print(f"{'Net TVPI (median)':<20} | {med_tvpi_std:>14.2f}x | {med_tvpi_safe:>14.2f}x | {med_tvpi_safe - med_tvpi_std:>+11.2f}x")
    print(f"{'DPI (median)':<20} | {med_dpi_std:>14.2f}x | {med_dpi_safe:>14.2f}x | {med_dpi_safe - med_dpi_std:>+11.2f}x")
    print(f"{'Loss Probability':<20} | {loss_prob_std:>13.1f}% | {loss_prob_safe:>13.1f}% | {loss_prob_safe - loss_prob_std:>+10.1f}%")
    print("-" * 70)
    
    print(f"\n📈 Standard mode had {unicorn_count:,} iterations with exits > $100M")
    print(f"   ({unicorn_count/args.iterations*100:.1f}% of iterations benefited from a 'unicorn')")
    
    # Key insight
    print("\n" + "─"*78)
    print("KEY INSIGHT FOR LPs:")
    print("─"*78)
    if med_tvpi_safe >= 1.0:
        print(f"  ✓ Even WITHOUT any exits above ${max_exit_cap/1e6:.0f}M, the fund returns {med_tvpi_safe:.2f}x capital")
        print(f"  ✓ The Revenue Share mechanism provides a {med_tvpi_safe:.2f}x floor")
        print(f"  ✓ Unicorns add {med_tvpi_std - med_tvpi_safe:.2f}x upside on top of this base")
    else:
        print(f"  ⚠ Without large exits, TVPI drops to {med_tvpi_safe:.2f}x")
        print(f"  → Revenue Share alone may not fully protect capital in adverse scenarios")
    
    # =========================================================================
    # 4. Generate Publication-Quality Charts (unless disabled)
    # =========================================================================
    if not args.no_charts:
        print("\n" + "="*78)
        print("GENERATING UNICORN-FREE ANALYSIS CHARTS")
        print("="*78)
        
        PRINT_DPI = 300
        FIGURE_SIZE = (12, 8)  # Landscape
        
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18
        })
        
        prefix = args.output_prefix
        
        # Chart 1: Side-by-Side TVPI Distribution
        fname = f"{prefix}chart_unicorn_tvpi_comparison.png" if prefix else "chart_unicorn_tvpi_comparison.png"
        print(f"  Saving: {fname}")
        fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)
        
        # Standard histogram
        ax1 = axes[0]
        valid_std = res_std['net_tvpi'][np.isfinite(res_std['net_tvpi'])]
        ax1.hist(valid_std, bins=50, alpha=0.7, color='steelblue', edgecolor='white')
        ax1.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Capital Return (1.0x)')
        ax1.axvline(np.median(valid_std), color='green', linestyle='-', linewidth=2, label=f'Median ({np.median(valid_std):.2f}x)')
        ax1.set_title('Standard Mode\n(Full Exit Distribution)', fontweight='bold')
        ax1.set_xlabel('Net TVPI Multiple')
        ax1.set_ylabel('Frequency')
        ax1.set_xlim(0, min(15, np.percentile(valid_std, 99)))
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Unicorn-free histogram
        ax2 = axes[1]
        valid_safe = res_safe['net_tvpi'][np.isfinite(res_safe['net_tvpi'])]
        ax2.hist(valid_safe, bins=50, alpha=0.7, color='forestgreen', edgecolor='white')
        ax2.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Capital Return (1.0x)')
        ax2.axvline(np.median(valid_safe), color='darkgreen', linestyle='-', linewidth=2, label=f'Median ({np.median(valid_safe):.2f}x)')
        ax2.set_title(f'Unicorn-Free Mode\n(Exits Capped at ${max_exit_cap/1e6:.0f}M)', fontweight='bold')
        ax2.set_xlabel('Net TVPI Multiple')
        ax2.set_ylabel('Frequency')
        ax2.set_xlim(0, min(15, np.percentile(valid_safe, 99)))
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle('SAFER Fund: Does It Work Without Unicorns?', fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(fname, dpi=PRINT_DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        # Chart 2: Box Plot Comparison (The "Mic Drop" Chart)
        fname = f"{prefix}chart_unicorn_boxplot.png" if prefix else "chart_unicorn_boxplot.png"
        print(f"  Saving: {fname}")
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        data = [valid_std, valid_safe]
        bp = ax.boxplot(data, vert=False, patch_artist=True,
                        tick_labels=['Standard Mode\n(With Power Law)', f'Unicorn-Free Mode\n(Max Exit ${max_exit_cap/1e6:.0f}M)'],
                        widths=0.6)
        
        colors = ['steelblue', 'forestgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Capital Return (1.0x)')
        ax.set_title('SAFER Fund Stress Test: Does the Fund Survive Without Unicorns?', 
                     fontsize=16, fontweight='bold')
        ax.set_xlabel('Net TVPI Multiple')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, min(12, np.percentile(valid_std, 95)))
        
        # Add annotation
        ax.annotate(f'Median: {np.median(valid_std):.2f}x', 
                    xy=(np.median(valid_std), 1), xytext=(np.median(valid_std)+1, 1.3),
                    fontsize=11, ha='left',
                    arrowprops=dict(arrowstyle='->', color='steelblue'))
        ax.annotate(f'Median: {np.median(valid_safe):.2f}x', 
                    xy=(np.median(valid_safe), 2), xytext=(np.median(valid_safe)+0.5, 2.3),
                    fontsize=11, ha='left',
                    arrowprops=dict(arrowstyle='->', color='forestgreen'))
        
        plt.tight_layout()
        plt.savefig(fname, dpi=PRINT_DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        # Chart 3: DPI Velocity (Source of Returns Over Time)
        fname = f"{prefix}chart_dpi_velocity.png" if prefix else "chart_dpi_velocity.png"
        print(f"  Saving: {fname}")
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        years = np.arange(mc_std.years + 1)
        
        # Calculate median DPI by source for each year
        med_repurchase_by_year = np.median(res_std['repurchase_by_year'], axis=0)
        med_equity_by_year = np.median(res_std['equity_by_year'], axis=0)
        
        # Stacked area chart
        ax.fill_between(years, 0, med_repurchase_by_year, alpha=0.7, label='Revenue Share (Yield)', color='forestgreen')
        ax.fill_between(years, med_repurchase_by_year, med_repurchase_by_year + med_equity_by_year, 
                        alpha=0.7, label='Liquidity Events (Equity)', color='steelblue')
        
        ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Capital Return (1.0x)')
        
        ax.set_title('DPI Velocity: Source of Distributions Over Fund Life\n(Median Across All Simulations)', 
                     fontsize=16, fontweight='bold')
        ax.set_xlabel('Fund Year')
        ax.set_ylabel('Cumulative DPI (Multiple of Paid-In Capital)')
        ax.set_xlim(0, mc_std.years)
        ax.set_ylim(0, max(3.5, (med_repurchase_by_year + med_equity_by_year).max() * 1.1))
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add percentage annotations
        for yr in [3, 5, 7]:
            if yr <= mc_std.years:
                total = med_repurchase_by_year[yr] + med_equity_by_year[yr]
                if total > 0:
                    rev_pct = med_repurchase_by_year[yr] / total * 100
                    ax.annotate(f'Year {yr}: {rev_pct:.0f}% from Revenue Share', 
                               xy=(yr, med_repurchase_by_year[yr]/2), 
                               fontsize=10, ha='center', color='white', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(fname, dpi=PRINT_DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        # Chart 4: Enhanced Attribution (Protection vs Upside)
        fname = f"{prefix}chart_attribution_enhanced.png" if prefix else "chart_attribution_enhanced.png"
        print(f"  Saving: {fname}")
        fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)
        
        # Left: Dollar-weighted attribution
        ax1 = axes[0]
        total_rep = np.sum(res_std['total_repurchase'])
        total_prot = np.sum(res_std['total_protection'])
        total_ups = np.sum(res_std['total_upside'])
        total_nav = np.sum(res_std['total_nav'])
        
        values = [total_rep, total_prot, total_ups, total_nav]
        labels = ['Revenue Share\n(Yield)', 'Exit Protection\n(Downside)', 'Exit Upside\n(Equity)', 'Terminal NAV\n(Unrealized)']
        colors = ['forestgreen', 'gold', 'steelblue', 'gray']
        
        # Filter out zero values
        non_zero = [(v, l, c) for v, l, c in zip(values, labels, colors) if v > 0]
        if non_zero:
            values, labels, colors = zip(*non_zero)
            wedges, texts, autotexts = ax1.pie(values, labels=labels, autopct='%1.1f%%', 
                                               colors=colors, startangle=90)
            for autotext in autotexts:
                autotext.set_fontsize(11)
                autotext.set_fontweight('bold')
        ax1.set_title('Total Return Attribution\n(Dollar-Weighted)', fontweight='bold')
        
        # Right: Frequency of contribution
        ax2 = axes[1]
        # Count how many iterations each source was the primary contributor
        primary_yield = np.sum(res_std['total_repurchase'] > res_std['total_equity'])
        primary_equity = np.sum(res_std['total_equity'] > res_std['total_repurchase'])
        
        freq_data = [primary_yield, primary_equity]
        freq_labels = ['Revenue Share\nDominated', 'Equity Exit\nDominated']
        freq_colors = ['forestgreen', 'steelblue']
        
        bars = ax2.bar(freq_labels, freq_data, color=freq_colors, alpha=0.7, edgecolor='black')
        ax2.set_title('Primary Return Source by Iteration\n(Frequency-Weighted)', fontweight='bold')
        ax2.set_ylabel('Number of Iterations')
        
        for bar, val in zip(bars, freq_data):
            ax2.annotate(f'{val:,}\n({val/args.iterations*100:.1f}%)', 
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax2.set_ylim(0, max(freq_data) * 1.2)
        ax2.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle('Where Do SAFER Returns Come From?', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(fname, dpi=PRINT_DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"\n  ✓ Generated 4 unicorn-free analysis charts (300 DPI, LANDSCAPE)")
    
    elapsed = time.time() - start_time
    print(f"\nAnalysis completed in {elapsed:.1f} seconds")
    
    return {
        'standard': res_std,
        'unicorn_free': res_safe,
        'comparison': {
            'net_irr_std': med_irr_std,
            'net_irr_safe': med_irr_safe,
            'net_tvpi_std': med_tvpi_std,
            'net_tvpi_safe': med_tvpi_safe,
            'loss_prob_std': loss_prob_std,
            'loss_prob_safe': loss_prob_safe
        }
    }


# =============================================================================
# PORTFOLIO SCENARIO PRESETS
# =============================================================================
SCENARIO_PRESETS = {
    'standard': {
        'label': 'Standard VC (Current Assumptions)',
        'description': 'Traditional seed-stage VC assumptions with high failure rates and power-law exits.',
        'color': 'steelblue',
        'args': {
            'init_revenue': 150_000, 'init_revenue_sigma': 1.0,
            'base_fail_hazard': 0.08,
            'p_high_growth': 0.25, 'p_mid_growth': 0.35, 'p_low_growth': 0.40,
            'high_growth_rate': 1.00, 'mid_growth_rate': 0.45, 'low_growth_rate': 0.10,
            'exit_multiple': 6.0, 'exit_multiple_sigma': 0.9,
            'terminal_nav_multiple': 4.0,
            'base_exit_hazard': 0.01, 'max_exit_hazard': 0.08,
            'revenue_share': 0.05, 'target_multiple': 3.0, 'repurchase_pct': 0.70,
        }
    },
    'revenue-focused': {
        'label': 'Revenue-Focused (Traction-Selected)',
        'description': 'Companies selected for revenue traction ($350k+ ARR). '
                       'Lower failure, moderate growth, fewer but more reliable exits.',
        'color': 'forestgreen',
        'args': {
            'init_revenue': 350_000, 'init_revenue_sigma': 0.6,
            'base_fail_hazard': 0.025,
            'p_high_growth': 0.10, 'p_mid_growth': 0.45, 'p_low_growth': 0.45,
            'high_growth_rate': 0.60, 'mid_growth_rate': 0.30, 'low_growth_rate': 0.08,
            'exit_multiple': 4.0, 'exit_multiple_sigma': 0.7,
            'terminal_nav_multiple': 3.0,
            'base_exit_hazard': 0.005, 'max_exit_hazard': 0.03,
            'revenue_share': 0.05, 'target_multiple': 3.0, 'repurchase_pct': 0.70,
        }
    },
    'high-yield': {
        'label': 'High Yield (Revenue Share Optimized)',
        'description': 'Same traction-selected companies, but SAFER terms tuned for '
                       'higher revenue share (8%) and lower target (2x).',
        'color': 'darkorange',
        'args': {
            'init_revenue': 350_000, 'init_revenue_sigma': 0.6,
            'base_fail_hazard': 0.025,
            'p_high_growth': 0.10, 'p_mid_growth': 0.45, 'p_low_growth': 0.45,
            'high_growth_rate': 0.60, 'mid_growth_rate': 0.30, 'low_growth_rate': 0.08,
            'exit_multiple': 4.0, 'exit_multiple_sigma': 0.7,
            'terminal_nav_multiple': 3.0,
            'base_exit_hazard': 0.005, 'max_exit_hazard': 0.03,
            'revenue_share': 0.08, 'target_multiple': 2.0, 'repurchase_pct': 0.70,
        }
    },
    'max-revenue': {
        'label': 'Max Revenue (Pure Yield Play)',
        'description': 'Higher-traction companies ($1M+ ARR) with aggressive revenue share (10%), '
                       'low target (2x), high repurchase (85%). Approaches private credit returns.',
        'color': 'crimson',
        'args': {
            'init_revenue': 1_000_000, 'init_revenue_sigma': 0.5,
            'base_fail_hazard': 0.02,
            'p_high_growth': 0.10, 'p_mid_growth': 0.45, 'p_low_growth': 0.45,
            'high_growth_rate': 0.50, 'mid_growth_rate': 0.25, 'low_growth_rate': 0.05,
            'exit_multiple': 3.5, 'exit_multiple_sigma': 0.6,
            'terminal_nav_multiple': 2.5,
            'base_exit_hazard': 0.003, 'max_exit_hazard': 0.02,
            'revenue_share': 0.10, 'target_multiple': 2.0, 'repurchase_pct': 0.85,
        }
    },
}


def apply_scenario_preset(args, scenario_name):
    """Apply a scenario preset to the args namespace, overriding defaults."""
    preset = SCENARIO_PRESETS[scenario_name]
    for key, value in preset['args'].items():
        setattr(args, key, value)
    return args


def run_all_scenarios(args):
    """
    Run all four portfolio scenarios and generate comparison charts + table.
    
    This is the "SAFER Portfolios" chapter analysis: showing the full spectrum
    from traditional VC assumptions to pure revenue-share plays.
    """
    import time
    start_time = time.time()
    
    scenario_names = ['standard', 'revenue-focused', 'high-yield', 'max-revenue']
    
    print("\n" + "=" * 78)
    print("█" + " " * 76 + "█")
    print("█" + "SAFER PORTFOLIO SCENARIO COMPARISON".center(76) + "█")
    print("█" + "From Equity-Dominated to Revenue-Share-Dominated".center(76) + "█")
    print("█" + " " * 76 + "█")
    print("=" * 78)
    
    all_results = []
    all_configs = []
    
    for scenario_name in scenario_names:
        preset = SCENARIO_PRESETS[scenario_name]
        print(f"\n{'─' * 78}")
        print(f"Running: {preset['label']}")
        print(f"  {preset['description']}")
        print(f"{'─' * 78}")
        
        # Create a fresh copy of args and apply preset
        import copy
        scenario_args = copy.deepcopy(args)
        apply_scenario_preset(scenario_args, scenario_name)
        
        mc, seed = create_fund_mc_from_args(scenario_args, seed=args.seed or 2026)
        
        if not args.quiet:
            # Print key parameters
            s = mc.safer
            print(f"  SAFER: {s.revenue_share:.0%} rev share, {s.target_return_multiple}x target, "
                  f"{s.repurchase_percent:.0%} repurchase")
            print(f"  Companies: ${np.exp(mc.company_overrides.get('init_rev_logn_mu', np.log(150000))):,.0f} "
                  f"median revenue, {mc.company_overrides.get('base_fail_hazard_q', 0.08):.0%} fail hazard")
        
        res = mc.run()
        all_results.append(res)
        all_configs.append(preset)
        print(f"  ✓ Complete ({mc.iterations:,} iterations)")
    
    # =====================================================================
    # Summary Table
    # =====================================================================
    total_cos = args.iterations * args.portfolio_size
    
    print("\n\n" + "=" * 115)
    print("FOUR-SCENARIO COMPARISON: SAFER Fund Performance Spectrum")
    print("=" * 115)
    
    col_labels = [SCENARIO_PRESETS[s]['label'].split('(')[0].strip() for s in scenario_names]
    
    print(f"\n{'Metric':<28} | {col_labels[0]:>16} | {col_labels[1]:>16} | {col_labels[2]:>16} | {col_labels[3]:>16}")
    print("─" * 115)
    
    def row(label, fn):
        vals = [fn(r) for r in all_results]
        print(f"{label:<28} | {vals[0]:>16} | {vals[1]:>16} | {vals[2]:>16} | {vals[3]:>16}")
    
    row("Net IRR (median)",
        lambda r: f"{np.nanmedian(r['net_irr']):.1%}")
    row("Net TVPI (median)",
        lambda r: f"{np.nanmedian(r['net_tvpi']):.2f}x")
    row("Net DPI (median)",
        lambda r: f"{np.nanmedian(r['net_dpi']):.2f}x")
    row("Loss Probability",
        lambda r: f"{np.mean(r['net_tvpi'] < 1.0):.1%}")
    row("P10 TVPI (worst decile)",
        lambda r: f"{np.percentile(r['net_tvpi'], 10):.2f}x")
    
    print("─" * 115)
    
    row("Revenue Share % of Returns",
        lambda r: f"{np.sum(r['total_repurchase'])/(np.sum(r['total_repurchase'])+np.sum(r['total_equity'])+np.sum(r['total_nav']))*100:.0f}%")
    row("Equity Exit % of Returns",
        lambda r: f"{np.sum(r['total_equity'])/(np.sum(r['total_repurchase'])+np.sum(r['total_equity'])+np.sum(r['total_nav']))*100:.0f}%")
    row("Terminal NAV % of Returns",
        lambda r: f"{np.sum(r['total_nav'])/(np.sum(r['total_repurchase'])+np.sum(r['total_equity'])+np.sum(r['total_nav']))*100:.0f}%")
    
    print("─" * 115)
    
    row("Company Failure Rate",
        lambda r: f"{r['label_totals'].get('Failure/No Liquidity', 0)/total_cos:.0%}")
    row("Target Hit Rate (rev share)",
        lambda r: f"{(r['label_totals'].get('Target Return Repaid',0)+r['label_totals'].get('Target + Liquidity Event',0))/total_cos:.0%}")
    
    valid_times = [r['time_to_1x_dpi'][np.isfinite(r['time_to_1x_dpi'])] for r in all_results]
    row("Time to 1x DPI (median yrs)",
        lambda r: f"{np.median(r['time_to_1x_dpi'][np.isfinite(r['time_to_1x_dpi'])]):.1f}" 
        if np.any(np.isfinite(r['time_to_1x_dpi'])) else "N/A")
    row("Funds Reaching 1x DPI",
        lambda r: f"{np.sum(np.isfinite(r['time_to_1x_dpi']))/args.iterations:.0%}")
    
    print("─" * 115)
    
    # =====================================================================
    # Key Insights
    # =====================================================================
    
    # =====================================================================
    # Generate Charts (unless disabled)
    # =====================================================================
    if not args.no_charts:
        _generate_scenario_charts(all_results, all_configs, scenario_names, args)
    
    elapsed = time.time() - start_time
    print(f"\nAnalysis completed in {elapsed:.1f} seconds")
    
    return all_results


def _generate_scenario_charts(all_results, all_configs, scenario_names, args):
    """Generate publication-quality charts for the scenario comparison."""
    PRINT_DPI = 300
    
    plt.rcParams.update({
        'font.size': 13, 'axes.titlesize': 16, 'axes.labelsize': 13,
        'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11
    })
    
    total_cos = args.iterations * args.portfolio_size
    short_names = [SCENARIO_PRESETS[s]['label'].split('(')[0].strip() for s in scenario_names]
    colors = [SCENARIO_PRESETS[s]['color'] for s in scenario_names]
    
    prefix = args.output_prefix
    
    print("\n" + "=" * 78)
    print("GENERATING SCENARIO COMPARISON CHARTS")
    print("=" * 78)
    
    # ---- CHART 1: Return Attribution Stacked Bars ----
    fname = f"{prefix}scenario_attribution.png" if prefix else "scenario_attribution.png"
    print(f"  Saving: {fname}")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_pos = np.arange(len(all_configs))
    bar_width = 0.6
    rev_pcts, eq_pcts, nav_pcts = [], [], []
    
    for res in all_results:
        total_rep = np.sum(res['total_repurchase'])
        total_eq = np.sum(res['total_equity'])
        total_nav = np.sum(res['total_nav'])
        grand = total_rep + total_eq + total_nav
        rev_pcts.append(total_rep / grand * 100 if grand > 0 else 0)
        eq_pcts.append(total_eq / grand * 100 if grand > 0 else 0)
        nav_pcts.append(total_nav / grand * 100 if grand > 0 else 0)
    
    ax.bar(x_pos, rev_pcts, bar_width, label='Revenue Share (Yield)', color='forestgreen', alpha=0.85)
    ax.bar(x_pos, eq_pcts, bar_width, bottom=rev_pcts, label='Liquidity Events (Equity)', color='steelblue', alpha=0.85)
    bottoms = [r + e for r, e in zip(rev_pcts, eq_pcts)]
    ax.bar(x_pos, nav_pcts, bar_width, bottom=bottoms, label='Terminal NAV (Unrealized)', color='gray', alpha=0.6)
    
    for i in range(len(all_configs)):
        if rev_pcts[i] > 5:
            ax.text(i, rev_pcts[i] / 2, f"{rev_pcts[i]:.0f}%", ha='center', va='center',
                    fontweight='bold', color='white', fontsize=12)
        if eq_pcts[i] > 5:
            ax.text(i, rev_pcts[i] + eq_pcts[i] / 2, f"{eq_pcts[i]:.0f}%", ha='center', va='center',
                    fontweight='bold', color='white', fontsize=12)
        if nav_pcts[i] > 5:
            ax.text(i, bottoms[i] + nav_pcts[i] / 2, f"{nav_pcts[i]:.0f}%", ha='center', va='center',
                    fontweight='bold', fontsize=12)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_names)
    ax.set_ylabel('Percentage of Total Returns')
    ax.set_title('SAFER Fund Return Attribution by Portfolio Strategy\n'
                 'From Equity-Dominated to Revenue-Share-Dominated', fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(fname, dpi=PRINT_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # ---- CHART 2: Risk-Return Scatter ----
    fname = f"{prefix}scenario_risk_return.png" if prefix else "scenario_risk_return.png"
    print(f"  Saving: {fname}")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, (res, name) in enumerate(zip(all_results, short_names)):
        med_irr = np.nanmedian(res['net_irr']) * 100
        loss_prob = np.mean(res['net_tvpi'] < 1.0) * 100
        ax.scatter(loss_prob, med_irr, s=300, c=colors[i], zorder=5,
                   edgecolors='black', linewidth=1.5)
        ax.annotate(name, (loss_prob, med_irr),
                    textcoords="offset points", xytext=(12, 8),
                    fontsize=12, fontweight='bold', color=colors[i])
    
    ax.set_xlabel('Loss Probability (TVPI < 1.0x)')
    ax.set_ylabel('Median Net IRR (%)')
    ax.set_title('SAFER Fund: Risk-Return Tradeoff Across Strategies\n'
                 'Moving Right = More Risk | Moving Up = More Return', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(8, color='gray', linestyle=':', linewidth=1, label='8% Hurdle Rate')
    ax.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=PRINT_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # ---- CHART 3: TVPI Box Plots ----
    fname = f"{prefix}scenario_tvpi_boxplot.png" if prefix else "scenario_tvpi_boxplot.png"
    print(f"  Saving: {fname}")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    data = [np.clip(res['net_tvpi'][np.isfinite(res['net_tvpi'])], 0, 15)
            for res in all_results]
    bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.6,
                    tick_labels=short_names)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Capital Return (1.0x)')
    ax.set_ylabel('Net TVPI Multiple')
    ax.set_title('Distribution of Net TVPI Across Portfolio Strategies\n'
                 'Higher Yield = Tighter Distribution, Lower Variance', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for i, d in enumerate(data):
        med = np.median(d)
        ax.annotate(f'{med:.2f}x', xy=(i + 1, med), xytext=(i + 1.35, med),
                    fontsize=11, fontweight='bold', color=colors[i])
    plt.tight_layout()
    plt.savefig(fname, dpi=PRINT_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # ---- CHART 4: Outcome Bars ----
    fname = f"{prefix}scenario_outcomes.png" if prefix else "scenario_outcomes.png"
    print(f"  Saving: {fname}")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    fail_rates = [res['label_totals'].get('Failure/No Liquidity', 0) / total_cos * 100
                  for res in all_results]
    target_rates = [(res['label_totals'].get('Target Return Repaid', 0) +
                     res['label_totals'].get('Target + Liquidity Event', 0)) / total_cos * 100
                    for res in all_results]
    
    bars = axes[0].bar(short_names, fail_rates, color=colors, alpha=0.8)
    for bar, val in zip(bars, fail_rates):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{val:.0f}%', ha='center', fontweight='bold', fontsize=11)
    axes[0].set_ylabel('Company Failure Rate (%)')
    axes[0].set_title('Portfolio Failure Rate', fontweight='bold')
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=15)
    
    bars = axes[1].bar(short_names, target_rates, color=colors, alpha=0.8)
    for bar, val in zip(bars, target_rates):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11)
    axes[1].set_ylabel('Target Repayment Rate (%)')
    axes[1].set_title('Revenue Share Target Hit Rate', fontweight='bold')
    axes[1].set_ylim(0, max(target_rates) * 1.3)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=15)
    
    fig.suptitle('Company Outcomes: From VC-Style to Revenue-Focused',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(fname, dpi=PRINT_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n  ✓ Generated 4 scenario comparison charts (300 DPI)")


# Add to main execution
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate growth probabilities sum to 1
    growth_sum = args.p_high_growth + args.p_mid_growth + args.p_low_growth
    if abs(growth_sum - 1.0) > 0.01:
        print(f"WARNING: Growth probabilities sum to {growth_sum:.2f}, not 1.0")
        print("         Normalizing probabilities...")
        args.p_high_growth /= growth_sum
        args.p_mid_growth /= growth_sum
        args.p_low_growth /= growth_sum
    
    # Handle quick mode - reduce iterations for all modes
    if args.quick:
        if args.iterations == 10_000:  # Only override if user didn't specify
            args.iterations = 1000
    
    # Route to appropriate mode
    if args.scenario == 'all-scenarios':
        run_all_scenarios(args)
    elif args.scenario is not None:
        apply_scenario_preset(args, args.scenario)
        print(f"\n📋 Using preset: {SCENARIO_PRESETS[args.scenario]['label']}")
        print(f"   {SCENARIO_PRESETS[args.scenario]['description']}\n")
        run_main_simulation(args)
    elif args.sensitivity:
        run_sensitivity_analysis(args, quick_mode=args.quick)
    elif args.sensitivity_full:
        run_extended_sensitivity_analysis(args, quick_mode=args.quick)
    elif args.unicorn_free:
        # Unicorn-free comparison uses all the same args, just runs twice
        # Default to 5000 iterations for comparison (can override with --iterations)
        if args.iterations == 10_000 and not args.quick:
            args.iterations = 5000
        run_unicorn_free_comparison(args)
    else:
        # Run main simulation
        run_main_simulation(args)
