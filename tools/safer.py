#!/usr/bin/env python3
"""

# Copyright (c) 2026, Next Wave Publishing LLC. All rights reserved.
# Licensed under the BSD 4-Clause License. See LICENSE file for details.

Safer Scenario Simulator
========================

A comprehensive simulator for the Safer (Simple Agreement for Future Equity with Repurchase)
instrument. Generates detailed HTML reports with cash flow analysis, scenario modeling,
and visualizations suitable for inclusion in book chapters.

This simulator faithfully implements the Safer mechanics as defined in the legal agreement:
- Post-Money Valuation Cap
- Repurchase Percentage  
- Revenue Percentage (revenue share)
- Target Return Percentage / Target Return
- Honeymoon Period
- Safer Amount calculation
- Liquidity Event conversion mechanics

Usage:
    python safer_scenario_simulator.py [options]

Example:
    python safer_scenario_simulator.py \
        --investment 1000000 \
        --valuation-cap 10000000 \
        --target-return-multiple 3.0 \
        --revenue-share 0.05 \
        --repurchase-percent 0.90 \
        --honeymoon-months 12 \
        --scenario steady-growth-acquisition \
        --output report.html
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Optional, Tuple
import base64
import io

# Try to import optional dependencies
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# =============================================================================
# SAFER LEGAL INSTRUMENT MODEL
# =============================================================================

@dataclass
class SaferTerms:
    """
    Represents the core terms of a Safer instrument as defined in the legal agreement.
    
    From the Safer Agreement:
    - Purchase Amount: The investment amount
    - Post-Money Valuation Cap: Cap for conversion calculations
    - Repurchase Percentage: Percentage of Purchase Amount that can be "bought back"
    - Revenue Percentage: Percentage of quarterly gross revenue paid as Repurchase Payments
    - Target Return Percentage: Multiple of Purchase Amount as total target return
    - Target Return: Absolute dollar amount of target return
    - Honeymoon Period: Period before Repurchase Payments begin
    """
    purchase_amount: float  # Investment amount
    post_money_valuation_cap: float
    repurchase_percentage: float  # e.g., 0.90 for 90%
    revenue_percentage: float  # e.g., 0.05 for 5%
    target_return_multiple: float  # e.g., 3.0 for 3.0x
    honeymoon_months: int  # e.g., 12 for 12 months
    
    @property
    def target_return(self) -> float:
        """Target Return = Purchase Amount × Target Return Percentage"""
        return self.purchase_amount * self.target_return_multiple
    
    @property
    def repurchase_amount(self) -> float:
        """Repurchase Amount = Purchase Amount × Repurchase Percentage"""
        return self.purchase_amount * self.repurchase_percentage
    
    @property
    def honeymoon_quarters(self) -> int:
        """Convert honeymoon months to quarters"""
        return self.honeymoon_months // 3
    
    def calculate_safer_amount(self, aggregate_repurchase_payments: float) -> float:
        """
        Calculate the Safer Amount per the legal agreement.
        
        From the Safer Agreement, "Safer Amount" is calculated as:
        
        (1) the Purchase Amount;
        
        (2) minus the product of:
            (i) the quotient of:
                (1) the aggregate Repurchase Payments actually paid under this Safer,
                divided by
                (2) the Target Return,
            multiplied by
            (ii) the Repurchase Amount;
        
        (3) plus the difference between:
            (i) the Target Return and
            (ii) the aggregate Repurchase Payments actually paid under this Safer.
        
        Formula: Safer Amount = PA - [(RP_paid / TR) × RA] + [TR - RP_paid]
        
        Where:
            PA = Purchase Amount
            RP_paid = Aggregate Repurchase Payments paid
            TR = Target Return
            RA = Repurchase Amount (= PA × Repurchase Percentage)
        """
        # Clip to target return (can't pay more than target)
        paid = min(aggregate_repurchase_payments, self.target_return)
        
        # Component 1: Start with Purchase Amount
        component_1 = self.purchase_amount
        
        # Component 2: Subtract the buyback portion
        # (paid / target_return) × repurchase_amount
        buyback_fraction = paid / self.target_return if self.target_return > 0 else 0
        component_2 = buyback_fraction * self.repurchase_amount
        
        # Component 3: Add remaining obligation
        # (target_return - paid)
        component_3 = self.target_return - paid
        
        safer_amount = component_1 - component_2 + component_3
        
        return safer_amount
    
    def calculate_quarterly_repurchase_payment(self, quarterly_gross_revenue: float) -> float:
        """
        Calculate quarterly Repurchase Payment per the legal agreement.
        
        From the Safer Agreement:
        "Starting the first calendar quarter after the expiration of the Honeymoon Period,
        the Company will make quarterly Repurchase Payments to the Investor equal to:
        (i) the Company's top line (i.e., gross) revenue in each given quarter
        multiplied by
        (ii) the Revenue Percentage."
        """
        return quarterly_gross_revenue * self.revenue_percentage
    
    def calculate_liquidity_price(self, liquidity_capitalization: float) -> float:
        """
        Calculate Liquidity Price per the legal agreement.
        
        From the Safer Agreement:
        "Liquidity Price" is calculated as the Post-Money Valuation Cap divided by
        the Liquidity Capitalization.
        """
        if liquidity_capitalization <= 0:
            return self.post_money_valuation_cap  # Fallback
        return self.post_money_valuation_cap / liquidity_capitalization
    
    def calculate_conversion_shares(self, safer_amount: float, liquidity_price: float) -> float:
        """
        Calculate number of shares on conversion.
        
        From the Safer Agreement:
        "the amount payable on the number of shares of Common Stock equal to the
        Safer Amount divided by the Liquidity Price"
        """
        if liquidity_price <= 0:
            return 0
        return safer_amount / liquidity_price
    
    def calculate_liquidity_payout(
        self,
        aggregate_repurchase_payments: float,
        exit_valuation: float,
        liquidity_capitalization: Optional[float] = None
    ) -> Tuple[float, str]:
        """
        Calculate payout at a Liquidity Event per the legal agreement.
        
        From the Safer Agreement Section 1(a) - Liquidity Event:
        "the Investor will automatically be entitled... to receive a portion of Proceeds...
        equal to the greater of:
        (i) the Safer Amount (the "Cash-Out Amount") or
        (ii) the amount payable on the number of shares of Common Stock equal to the
            Safer Amount divided by the Liquidity Price (the "Conversion Amount")"
        
        Returns:
            (payout_amount, payout_type) where payout_type is "Cash-Out" or "Conversion"
        """
        safer_amount = self.calculate_safer_amount(aggregate_repurchase_payments)
        
        # Cash-Out Amount = Safer Amount
        cash_out_amount = safer_amount
        
        # Conversion Amount calculation
        # If no liquidity_capitalization provided, use post-money cap as proxy
        if liquidity_capitalization is None:
            # Simplified: ownership = safer_amount / post_money_cap
            # Conversion value = ownership × exit_valuation
            ownership_fraction = safer_amount / self.post_money_valuation_cap
            conversion_amount = ownership_fraction * exit_valuation
        else:
            liquidity_price = self.calculate_liquidity_price(liquidity_capitalization)
            shares = self.calculate_conversion_shares(safer_amount, liquidity_price)
            price_per_share = exit_valuation / liquidity_capitalization
            conversion_amount = shares * price_per_share
        
        # Return greater of the two
        if conversion_amount >= cash_out_amount:
            return (conversion_amount, "Conversion")
        else:
            return (cash_out_amount, "Cash-Out")


class LiquidityEventType(Enum):
    """Types of Liquidity Events per the Safer Agreement"""
    CHANGE_OF_CONTROL = "Change of Control (Acquisition)"
    INITIAL_PUBLIC_OFFERING = "Initial Public Offering"
    DIRECT_LISTING = "Direct Listing"


class DispositionType(Enum):
    """Types of termination/disposition"""
    LIQUIDITY_EVENT = "Liquidity Event"
    TARGET_RETURN_ACHIEVED = "Target Return Achieved via Repurchase"
    DISSOLUTION = "Dissolution Event"
    FUND_END_NO_LIQUIDITY = "Fund End (No Liquidity Event)"


# =============================================================================
# QUARTERLY CASH FLOW MODEL
# =============================================================================

@dataclass
class QuarterlyCashFlow:
    """Represents a single quarter's cash flow"""
    quarter: int
    year: float  # Year as decimal (e.g., 2.25 for Q1 of year 3)
    date: str  # Human-readable date
    gross_revenue: float
    repurchase_payment: float
    cumulative_repurchase: float
    safer_amount: float
    notes: str = ""


@dataclass
class LiquidityEvent:
    """Represents a liquidity event"""
    quarter: int
    year: float
    date: str
    event_type: LiquidityEventType
    exit_valuation: float
    safer_amount_at_event: float
    cash_out_amount: float
    conversion_amount: float
    payout_amount: float
    payout_type: str
    ownership_percentage: float
    notes: str = ""


@dataclass 
class ScenarioResult:
    """Complete result of a scenario simulation"""
    scenario_name: str
    scenario_description: str
    safer_terms: SaferTerms
    cash_flows: List[QuarterlyCashFlow]
    liquidity_event: Optional[LiquidityEvent]
    disposition: DispositionType
    total_investment: float
    total_repurchase_payments: float
    total_liquidity_payout: float
    total_return: float
    multiple_on_invested: float
    irr: Optional[float]
    duration_years: float
    

# =============================================================================
# SCENARIO DEFINITIONS
# =============================================================================

@dataclass
class RevenueProfile:
    """Defines a revenue growth profile for simulation"""
    initial_arr: float
    growth_rates_by_year: List[float]  # Annual growth rates for each year
    
    def get_quarterly_revenue(self, quarter: int) -> float:
        """Get quarterly revenue for a given quarter (1-indexed)"""
        year_index = (quarter - 1) // 4
        if year_index >= len(self.growth_rates_by_year):
            year_index = len(self.growth_rates_by_year) - 1
        
        # Calculate ARR at this point
        arr = self.initial_arr
        for i in range(year_index):
            arr *= (1 + self.growth_rates_by_year[i])
        
        # Within-year progression (assume even quarterly growth)
        quarter_in_year = (quarter - 1) % 4
        if year_index < len(self.growth_rates_by_year):
            quarterly_growth = (1 + self.growth_rates_by_year[year_index]) ** 0.25 - 1
            arr *= (1 + quarterly_growth) ** quarter_in_year
        
        return arr / 4  # Quarterly revenue


class ScenarioType(Enum):
    """Pre-defined scenario types"""
    STEADY_GROWTH_ACQUISITION = "steady-growth-acquisition"
    EXPLOSIVE_GROWTH_IPO = "explosive-growth-ipo"
    SUSTAINABLE_COMPLETE_REPURCHASE = "sustainable-complete-repurchase"
    FAILURE_LOW_REVENUE = "failure-low-revenue"
    MODEST_GROWTH_SMALL_EXIT = "modest-growth-small-exit"
    CUSTOM = "custom"


def create_scenario_revenue_profile(scenario_type: ScenarioType, custom_params: dict = None) -> Tuple[RevenueProfile, dict]:
    """Create a revenue profile for a given scenario type"""
    
    if scenario_type == ScenarioType.STEADY_GROWTH_ACQUISITION:
        # Example 1 from chapter: Pre-seed SaaS, steady growth to acquisition
        return RevenueProfile(
            initial_arr=500_000,  # $500K ARR by end of year 1
            growth_rates_by_year=[0.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.5]  # Y1 honeymoon, then strong growth
        ), {
            "exit_year": 7,
            "exit_valuation": 120_000_000,
            "event_type": LiquidityEventType.CHANGE_OF_CONTROL,
            "description": "Pre-Seed SaaS Company - Steady Growth to Acquisition"
        }
    
    elif scenario_type == ScenarioType.EXPLOSIVE_GROWTH_IPO:
        # Example 2 from chapter: Hot AI pre-seed, explosive growth to IPO
        return RevenueProfile(
            initial_arr=2_000_000,  # $2M ARR by end of year 1
            growth_rates_by_year=[0.0, 1.5, 1.5, 1.0, 1.0, 1.0, 0.5]
        ), {
            "exit_year": 7,
            "exit_valuation": 2_000_000_000,
            "event_type": LiquidityEventType.INITIAL_PUBLIC_OFFERING,
            "description": "Hot AI Pre-Seed - Explosive Growth to IPO"
        }
    
    elif scenario_type == ScenarioType.SUSTAINABLE_COMPLETE_REPURCHASE:
        # Example 3 from chapter: Seed SaaS, sustainable growth, complete repurchase
        return RevenueProfile(
            initial_arr=3_000_000,  # $3M ARR by end of year 2
            growth_rates_by_year=[0.0, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.35, 0.3, 0.25, 0.2, 0.2, 0.2, 0.2]
        ), {
            "exit_year": 15,
            "exit_valuation": 400_000_000,
            "event_type": LiquidityEventType.CHANGE_OF_CONTROL,
            "description": "Seed Stage SaaS - Sustainable Growth, Complete Repurchase"
        }
    
    elif scenario_type == ScenarioType.FAILURE_LOW_REVENUE:
        # Failure scenario: Company never achieves significant revenue
        return RevenueProfile(
            initial_arr=50_000,
            growth_rates_by_year=[0.0, 0.2, 0.1, -0.2, -0.5, -1.0]  # Decline and shutdown
        ), {
            "exit_year": None,  # No exit
            "exit_valuation": 0,
            "event_type": None,
            "description": "Failure - Low Revenue, No Liquidity Event"
        }
    
    elif scenario_type == ScenarioType.MODEST_GROWTH_SMALL_EXIT:
        # Modest scenario: Small acquisition before full repurchase
        return RevenueProfile(
            initial_arr=300_000,
            growth_rates_by_year=[0.0, 0.6, 0.5, 0.4]
        ), {
            "exit_year": 4,
            "exit_valuation": 25_000_000,
            "event_type": LiquidityEventType.CHANGE_OF_CONTROL,
            "description": "Modest Growth - Small Early Acquisition"
        }
    
    else:  # CUSTOM
        if custom_params is None:
            custom_params = {}
        return RevenueProfile(
            initial_arr=custom_params.get("initial_arr", 500_000),
            growth_rates_by_year=custom_params.get("growth_rates", [0.0, 0.5, 0.5, 0.4, 0.4])
        ), {
            "exit_year": custom_params.get("exit_year"),
            "exit_valuation": custom_params.get("exit_valuation", 0),
            "event_type": LiquidityEventType.CHANGE_OF_CONTROL if custom_params.get("exit_valuation", 0) > 0 else None,
            "description": custom_params.get("description", "Custom Scenario")
        }


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

def calculate_irr(cash_flows: List[float], times_years: List[float]) -> Optional[float]:
    """
    Calculate IRR using bisection method.
    
    Args:
        cash_flows: List of cash flows (negative for outflows, positive for inflows)
        times_years: List of times in years corresponding to each cash flow
    
    Returns:
        IRR as a decimal (e.g., 0.25 for 25%), or None if cannot be calculated
    """
    if not cash_flows or len(cash_flows) != len(times_years):
        return None
    
    # Must have at least one negative and one positive cash flow
    has_negative = any(cf < 0 for cf in cash_flows)
    has_positive = any(cf > 0 for cf in cash_flows)
    if not (has_negative and has_positive):
        return None
    
    def npv(rate: float) -> float:
        total = 0.0
        for cf, t in zip(cash_flows, times_years):
            if rate <= -1.0:
                return float('inf')
            total += cf / ((1.0 + rate) ** t)
        return total
    
    # Bisection method
    lo, hi = -0.99, 10.0
    
    try:
        f_lo, f_hi = npv(lo), npv(hi)
    except (OverflowError, ZeroDivisionError):
        return None
    
    # Expand bounds if not bracketed
    if f_lo * f_hi > 0:
        for _ in range(20):
            hi *= 1.5
            try:
                f_hi = npv(hi)
            except (OverflowError, ZeroDivisionError):
                break
            if f_lo * f_hi <= 0:
                break
        else:
            return None
    
    # Bisection iterations
    for _ in range(150):
        mid = (lo + hi) / 2
        try:
            f_mid = npv(mid)
        except (OverflowError, ZeroDivisionError):
            hi = mid
            continue
        
        if abs(f_mid) < 1e-9:
            return mid
        
        if f_lo * f_mid <= 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    
    return mid


def simulate_scenario(
    safer_terms: SaferTerms,
    revenue_profile: RevenueProfile,
    scenario_params: dict,
    max_quarters: int = 60,  # 15 years
    start_date: datetime = None
) -> ScenarioResult:
    """
    Simulate a complete Safer scenario.
    
    Args:
        safer_terms: The Safer instrument terms
        revenue_profile: Revenue growth profile
        scenario_params: Dict with exit_year, exit_valuation, event_type, description
        max_quarters: Maximum simulation length
        start_date: Starting date for the simulation
    
    Returns:
        ScenarioResult with complete analysis
    """
    if start_date is None:
        start_date = datetime.now()
    
    cash_flows: List[QuarterlyCashFlow] = []
    cumulative_repurchase = 0.0
    target_achieved = False
    liquidity_event = None
    
    exit_quarter = scenario_params.get("exit_year")
    if exit_quarter:
        exit_quarter = int(exit_quarter * 4)  # Convert years to quarters
    
    exit_valuation = scenario_params.get("exit_valuation", 0)
    event_type = scenario_params.get("event_type")
    
    # Simulate each quarter
    for q in range(1, max_quarters + 1):
        year = q / 4
        quarter_date = start_date + timedelta(days=q * 91)  # Approximate
        date_str = quarter_date.strftime("%Y Q%q").replace("%q", str((quarter_date.month - 1) // 3 + 1))
        date_str = f"Year {(q - 1) // 4 + 1} Q{(q - 1) % 4 + 1}"
        
        # Get quarterly revenue
        gross_revenue = revenue_profile.get_quarterly_revenue(q)
        
        # Calculate repurchase payment (only after honeymoon)
        repurchase_payment = 0.0
        if q > safer_terms.honeymoon_quarters and not target_achieved:
            max_payment = safer_terms.target_return - cumulative_repurchase
            repurchase_payment = min(
                safer_terms.calculate_quarterly_repurchase_payment(gross_revenue),
                max_payment
            )
            cumulative_repurchase += repurchase_payment
            
            if cumulative_repurchase >= safer_terms.target_return - 0.01:
                target_achieved = True
        
        # Calculate Safer Amount at this point
        safer_amount = safer_terms.calculate_safer_amount(cumulative_repurchase)
        
        # Determine notes
        notes = ""
        if q <= safer_terms.honeymoon_quarters:
            notes = "Honeymoon Period"
        elif target_achieved and repurchase_payment > 0:
            notes = "Target Return Achieved"
        elif target_achieved:
            notes = "Post-Target (No Payments Due)"
        
        cash_flow = QuarterlyCashFlow(
            quarter=q,
            year=year,
            date=date_str,
            gross_revenue=gross_revenue,
            repurchase_payment=repurchase_payment,
            cumulative_repurchase=cumulative_repurchase,
            safer_amount=safer_amount,
            notes=notes
        )
        cash_flows.append(cash_flow)
        
        # Check for liquidity event
        if exit_quarter and q == exit_quarter and event_type:
            payout_amount, payout_type = safer_terms.calculate_liquidity_payout(
                cumulative_repurchase,
                exit_valuation
            )
            
            cash_out_amount = safer_amount
            ownership_pct = safer_amount / safer_terms.post_money_valuation_cap * 100
            conversion_amount = (safer_amount / safer_terms.post_money_valuation_cap) * exit_valuation
            
            liquidity_event = LiquidityEvent(
                quarter=q,
                year=year,
                date=date_str,
                event_type=event_type,
                exit_valuation=exit_valuation,
                safer_amount_at_event=safer_amount,
                cash_out_amount=cash_out_amount,
                conversion_amount=conversion_amount,
                payout_amount=payout_amount,
                payout_type=payout_type,
                ownership_percentage=ownership_pct,
                notes=f"{event_type.value}"
            )
            break
        
        # Check for effective termination
        if target_achieved and not event_type:
            # Company keeps operating, Safer just stops collecting
            # Continue simulation but no more payments
            pass
        
        # Check for failure (revenue drops to near zero)
        if gross_revenue < 1000 and q > 8:  # After 2 years, if revenue collapsed
            break
    
    # Determine disposition
    if liquidity_event:
        disposition = DispositionType.LIQUIDITY_EVENT
    elif target_achieved:
        disposition = DispositionType.TARGET_RETURN_ACHIEVED
    elif not cash_flows or cash_flows[-1].gross_revenue < 1000:
        disposition = DispositionType.DISSOLUTION
    else:
        disposition = DispositionType.FUND_END_NO_LIQUIDITY
    
    # Calculate totals
    total_investment = safer_terms.purchase_amount
    total_repurchase = cumulative_repurchase
    total_liquidity = liquidity_event.payout_amount if liquidity_event else 0
    total_return = total_repurchase + total_liquidity
    moic = total_return / total_investment if total_investment > 0 else 0
    duration = cash_flows[-1].year if cash_flows else 0
    
    # Calculate IRR
    irr_cash_flows = [-total_investment]
    irr_times = [0.0]
    
    for cf in cash_flows:
        if cf.repurchase_payment > 0:
            irr_cash_flows.append(cf.repurchase_payment)
            irr_times.append(cf.year)
    
    if liquidity_event:
        irr_cash_flows.append(liquidity_event.payout_amount)
        irr_times.append(liquidity_event.year)
    
    irr = calculate_irr(irr_cash_flows, irr_times)
    
    return ScenarioResult(
        scenario_name=scenario_params.get("description", "Unnamed Scenario"),
        scenario_description=scenario_params.get("description", ""),
        safer_terms=safer_terms,
        cash_flows=cash_flows,
        liquidity_event=liquidity_event,
        disposition=disposition,
        total_investment=total_investment,
        total_repurchase_payments=total_repurchase,
        total_liquidity_payout=total_liquidity,
        total_return=total_return,
        multiple_on_invested=moic,
        irr=irr,
        duration_years=duration
    )


# =============================================================================
# HTML REPORT GENERATION
# =============================================================================

def format_currency(value: float) -> str:
    """Format a number as currency"""
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value / 1_000:.1f}K"
    else:
        return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """Format a decimal as percentage"""
    return f"{value * 100:.2f}%"


def generate_chart_base64(result: ScenarioResult) -> Tuple[str, str, str]:
    """Generate charts as base64-encoded PNG images"""
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        return "", "", ""
    
    # Chart 1: Revenue and Investor Cash Flows Over Time (Log Scale)
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    
    quarters = [cf.quarter for cf in result.cash_flows]
    revenues = [cf.gross_revenue * 4 for cf in result.cash_flows]  # Annualized
    repurchases = [cf.repurchase_payment for cf in result.cash_flows]
    
    # Build investor cash flows including exit payout
    investor_cashflows = repurchases.copy()
    if result.liquidity_event:
        # Add exit payout to the final quarter
        exit_quarter_idx = result.liquidity_event.quarter - 1
        if exit_quarter_idx < len(investor_cashflows):
            investor_cashflows[exit_quarter_idx] += result.liquidity_event.payout_amount
    
    # Build cumulative investor cash flows
    cumulative_investor = []
    running_total = 0
    for cf in investor_cashflows:
        running_total += cf
        cumulative_investor.append(running_total)
    
    ax1.fill_between(quarters, revenues, alpha=0.3, color='#2563eb', label='ARR')
    ax1.plot(quarters, revenues, color='#2563eb', linewidth=2)
    ax1.set_yscale('log')
    
    ax2 = ax1.twinx()
    
    # Use log scale for investor cash flows to show both small payments and large exit
    ax2.set_yscale('log')
    
    # Only plot bars where there's actual cash flow (skip zeros for log scale)
    nonzero_quarters = [q for q, cf in zip(quarters, investor_cashflows) if cf > 0]
    nonzero_cashflows = [cf for cf in investor_cashflows if cf > 0]
    
    # Only plot cumulative where it's positive
    cumulative_quarters = [q for q, cf in zip(quarters, cumulative_investor) if cf > 0]
    cumulative_values = [cf for cf in cumulative_investor if cf > 0]
    
    ax2.bar(nonzero_quarters, nonzero_cashflows, alpha=0.7, color='#059669', width=0.8, label='Investor Cash Flow')
    ax2.plot(cumulative_quarters, cumulative_values, color='#dc2626', linewidth=2, linestyle='--', label='Cumulative Return')
    
    # Mark honeymoon end
    honeymoon_q = result.safer_terms.honeymoon_quarters
    ax1.axvline(x=honeymoon_q, color='#9333ea', linestyle=':', alpha=0.7, label='Honeymoon End')
    
    # Mark target return
    target = result.safer_terms.target_return
    ax2.axhline(y=target, color='#dc2626', linestyle=':', alpha=0.5, label=f'Target Return ({format_currency(target)})')
    
    # Mark liquidity event
    if result.liquidity_event:
        ax1.axvline(x=result.liquidity_event.quarter, color='#f59e0b', linewidth=2, label='Liquidity Event')
    
    ax1.set_xlabel('Quarter', fontsize=11)
    ax1.set_ylabel('Annual Revenue Rate (ARR, log scale)', color='#2563eb', fontsize=11)
    ax2.set_ylabel('Investor Cash Flows (log scale)', color='#059669', fontsize=11)
    ax1.set_title('Revenue Growth and Investor Cash Flow Timeline', fontsize=14, fontweight='bold')
    
    # Format y-axes as currency
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format_currency(x)))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format_currency(x)))
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', dpi=240, bbox_inches='tight')
    buf1.seek(0)
    chart1_b64 = base64.b64encode(buf1.read()).decode('utf-8')
    plt.close(fig1)
    
    # Chart 2: Safer Amount Evolution
    fig2, ax3 = plt.subplots(figsize=(10, 5))
    
    safer_amounts = [cf.safer_amount for cf in result.cash_flows]
    
    ax3.plot(quarters, safer_amounts, color='#7c3aed', linewidth=2.5, label='Safer Amount')
    ax3.fill_between(quarters, safer_amounts, alpha=0.2, color='#7c3aed')
    
    # Reference lines
    ax3.axhline(y=result.safer_terms.purchase_amount, color='#6b7280', linestyle='--', 
                alpha=0.6, label=f'Purchase Amount ({format_currency(result.safer_terms.purchase_amount)})')
    ax3.axhline(y=result.safer_terms.target_return, color='#dc2626', linestyle=':', 
                alpha=0.6, label=f'Target Return ({format_currency(result.safer_terms.target_return)})')
    
    if result.liquidity_event:
        ax3.scatter([result.liquidity_event.quarter], [result.liquidity_event.safer_amount_at_event], 
                   color='#f59e0b', s=150, zorder=5, marker='*', label='At Liquidity Event')
    
    ax3.set_xlabel('Quarter', fontsize=11)
    ax3.set_ylabel('Safer Amount', fontsize=11)
    ax3.set_title('Safer Amount Evolution Over Time', fontsize=14, fontweight='bold')
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format_currency(x)))
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    fig2.tight_layout()
    
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', dpi=240, bbox_inches='tight')
    buf2.seek(0)
    chart2_b64 = base64.b64encode(buf2.read()).decode('utf-8')
    plt.close(fig2)
    
    # Chart 3: Cash Flow Waterfall
    fig3, ax4 = plt.subplots(figsize=(10, 5))
    
    # Build waterfall data
    labels = ['Investment', 'Repurchase\nPayments']
    values = [-result.total_investment, result.total_repurchase_payments]
    colors = ['#dc2626', '#059669']
    
    if result.liquidity_event:
        labels.append('Liquidity\nPayout')
        values.append(result.total_liquidity_payout)
        colors.append('#2563eb')
    
    labels.append('Total\nReturn')
    values.append(result.total_return)
    colors.append('#7c3aed')
    
    x_pos = np.arange(len(labels))
    bars = ax4.bar(x_pos, values, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax4.annotate(format_currency(abs(val)),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5 if height > 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=10, fontweight='bold')
    
    ax4.axhline(y=0, color='black', linewidth=0.8)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(labels, fontsize=10)
    ax4.set_ylabel('Amount', fontsize=11)
    ax4.set_title('Investment Return Waterfall', fontsize=14, fontweight='bold')
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format_currency(x)))
    ax4.grid(True, alpha=0.3, axis='y')
    fig3.tight_layout()
    
    buf3 = io.BytesIO()
    fig3.savefig(buf3, format='png', dpi=240, bbox_inches='tight')
    buf3.seek(0)
    chart3_b64 = base64.b64encode(buf3.read()).decode('utf-8')
    plt.close(fig3)
    
    return chart1_b64, chart2_b64, chart3_b64


def generate_html_report(result: ScenarioResult, output_path: str = None) -> str:
    """Generate a comprehensive HTML report for a scenario simulation"""
    
    # Generate charts
    chart1, chart2, chart3 = generate_chart_base64(result)
    
    # Build cash flow table rows
    cf_rows = ""
    for i, cf in enumerate(result.cash_flows):
        row_class = ""
        if cf.notes == "Honeymoon Period":
            row_class = "honeymoon"
        elif "Target Return" in cf.notes:
            row_class = "target-achieved"
        
        cf_rows += f"""
        <tr class="{row_class}">
            <td>{cf.date}</td>
            <td class="currency">{format_currency(cf.gross_revenue * 4)}</td>
            <td class="currency">{format_currency(cf.gross_revenue)}</td>
            <td class="currency">{format_currency(cf.repurchase_payment)}</td>
            <td class="currency">{format_currency(cf.cumulative_repurchase)}</td>
            <td class="currency">{format_currency(cf.safer_amount)}</td>
            <td class="note">{cf.notes}</td>
        </tr>
        """
    
    # Liquidity event details
    liquidity_section = ""
    if result.liquidity_event:
        le = result.liquidity_event
        liquidity_section = f"""
        <section class="liquidity-section">
            <h2>Liquidity Event Analysis</h2>
            <div class="liquidity-grid">
                <div class="liquidity-card">
                    <h3>Event Details</h3>
                    <table class="detail-table">
                        <tr><td>Event Type:</td><td><strong>{le.event_type.value}</strong></td></tr>
                        <tr><td>Timing:</td><td>{le.date} (Year {le.year:.2f})</td></tr>
                        <tr><td>Exit Valuation:</td><td class="currency">{format_currency(le.exit_valuation)}</td></tr>
                    </table>
                </div>
                <div class="liquidity-card">
                    <h3>Payout Calculation</h3>
                    <table class="detail-table">
                        <tr><td>Safer Amount at Event:</td><td class="currency">{format_currency(le.safer_amount_at_event)}</td></tr>
                        <tr><td>Cash-Out Amount:</td><td class="currency">{format_currency(le.cash_out_amount)}</td></tr>
                        <tr><td>Conversion Amount:</td><td class="currency highlight">{format_currency(le.conversion_amount)}</td></tr>
                        <tr><td>Ownership at Cap:</td><td>{le.ownership_percentage:.2f}%</td></tr>
                    </table>
                </div>
                <div class="liquidity-card highlight-card">
                    <h3>Final Payout</h3>
                    <div class="payout-amount">{format_currency(le.payout_amount)}</div>
                    <div class="payout-type">via {le.payout_type}</div>
                    <p class="payout-note">Investor receives the <strong>greater of</strong> Cash-Out Amount or Conversion Amount per Section 1(a)</p>
                </div>
            </div>
        </section>
        """
    
    # Safer Amount formula explanation
    final_cf = result.cash_flows[-1] if result.cash_flows else None
    safer_calc = ""
    if final_cf:
        terms = result.safer_terms
        paid = min(final_cf.cumulative_repurchase, terms.target_return)
        buyback_fraction = paid / terms.target_return if terms.target_return > 0 else 0
        buyback_portion = buyback_fraction * terms.repurchase_amount
        remaining = terms.target_return - paid
        
        safer_calc = f"""
        <section class="formula-section">
            <h2>Safer Amount Calculation (at Final Period)</h2>
            <div class="formula-box">
                <div class="formula">
                    <strong>Safer Amount</strong> = Purchase Amount − [(Cumulative Payments ÷ Target Return) × Repurchase Amount] + [Target Return − Cumulative Payments]
                </div>
                <div class="formula-breakdown">
                    <div class="formula-step">
                        <span class="step-label">Component 1:</span>
                        <span class="step-value">{format_currency(terms.purchase_amount)}</span>
                        <span class="step-desc">(Purchase Amount)</span>
                    </div>
                    <div class="formula-step">
                        <span class="step-label">Component 2:</span>
                        <span class="step-value">− {format_currency(buyback_portion)}</span>
                        <span class="step-desc">([{format_currency(paid)} ÷ {format_currency(terms.target_return)}] × {format_currency(terms.repurchase_amount)} = {buyback_fraction:.4f} × {format_currency(terms.repurchase_amount)})</span>
                    </div>
                    <div class="formula-step">
                        <span class="step-label">Component 3:</span>
                        <span class="step-value">+ {format_currency(remaining)}</span>
                        <span class="step-desc">({format_currency(terms.target_return)} − {format_currency(paid)})</span>
                    </div>
                    <div class="formula-result">
                        <span class="step-label">Safer Amount:</span>
                        <span class="step-value">{format_currency(final_cf.safer_amount)}</span>
                    </div>
                </div>
            </div>
        </section>
        """
    
    # Generate full HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Safer Scenario Analysis: {result.scenario_name}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        :root {{
            --bg-primary: #faf9f7;
            --bg-secondary: #ffffff;
            --bg-tertiary: #f4f3f1;
            --text-primary: #1a1a1a;
            --text-secondary: #4a4a4a;
            --text-muted: #7a7a7a;
            --accent-blue: #2563eb;
            --accent-green: #059669;
            --accent-purple: #7c3aed;
            --accent-amber: #d97706;
            --accent-red: #dc2626;
            --border-color: #e5e5e5;
            --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
            --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Source Serif 4', Georgia, serif;
            font-size: 16px;
            line-height: 1.7;
            color: var(--text-primary);
            background: var(--bg-primary);
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1100px;
            margin: 0 auto;
            background: var(--bg-secondary);
            border-radius: 8px;
            box-shadow: var(--shadow-md);
            overflow: hidden;
        }}
        
        header {{
            background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
            color: white;
            padding: 3rem 2.5rem;
        }}
        
        header h1 {{
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
        }}
        
        header .subtitle {{
            font-size: 1.1rem;
            opacity: 0.9;
            font-weight: 400;
        }}
        
        header .meta {{
            margin-top: 1.5rem;
            font-size: 0.9rem;
            opacity: 0.8;
        }}
        
        main {{
            padding: 2.5rem;
        }}
        
        section {{
            margin-bottom: 3rem;
        }}
        
        h2 {{
            font-size: 1.4rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--border-color);
        }}
        
        h3 {{
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 1rem;
        }}
        
        /* Summary Cards */
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2rem;
        }}
        
        .summary-card {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 1.25rem;
            text-align: center;
        }}
        
        .summary-card.highlight {{
            background: linear-gradient(135deg, var(--accent-purple) 0%, #9333ea 100%);
            color: white;
        }}
        
        .summary-card .label {{
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            opacity: 0.8;
            margin-bottom: 0.5rem;
        }}
        
        .summary-card .value {{
            font-size: 1.5rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }}
        
        .summary-card .subtext {{
            font-size: 0.85rem;
            opacity: 0.7;
            margin-top: 0.25rem;
        }}
        
        /* Terms Table */
        .terms-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
        }}
        
        .term-item {{
            display: flex;
            justify-content: space-between;
            padding: 0.75rem 1rem;
            background: var(--bg-tertiary);
            border-radius: 6px;
        }}
        
        .term-item .term-label {{
            color: var(--text-secondary);
        }}
        
        .term-item .term-value {{
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }}
        
        /* Charts */
        .chart-container {{
            margin: 2rem 0;
            text-align: center;
        }}
        
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: var(--shadow-sm);
        }}
        
        /* Cash Flow Table */
        .table-wrapper {{
            overflow-x: auto;
            margin: 1.5rem 0;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        
        th {{
            background: var(--bg-tertiary);
            padding: 0.875rem 1rem;
            text-align: left;
            font-weight: 600;
            color: var(--text-secondary);
            border-bottom: 2px solid var(--border-color);
            white-space: nowrap;
        }}
        
        td {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border-color);
        }}
        
        tr:hover {{
            background: var(--bg-tertiary);
        }}
        
        tr.honeymoon {{
            background: #fef3c7;
        }}
        
        tr.target-achieved {{
            background: #d1fae5;
        }}
        
        td.currency {{
            font-family: 'JetBrains Mono', monospace;
            text-align: right;
        }}
        
        td.note {{
            color: var(--text-muted);
            font-style: italic;
            font-size: 0.85rem;
        }}
        
        /* Liquidity Section */
        .liquidity-section {{
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            margin: 2rem -2.5rem;
            padding: 2rem 2.5rem;
        }}
        
        .liquidity-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
        }}
        
        .liquidity-card {{
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: var(--shadow-sm);
        }}
        
        .liquidity-card.highlight-card {{
            background: linear-gradient(135deg, var(--accent-blue) 0%, #3b82f6 100%);
            color: white;
            text-align: center;
        }}
        
        .liquidity-card h3 {{
            margin-bottom: 1rem;
            color: inherit;
        }}
        
        .detail-table {{
            width: 100%;
        }}
        
        .detail-table td {{
            padding: 0.5rem 0;
            border: none;
        }}
        
        .detail-table td:first-child {{
            color: var(--text-muted);
        }}
        
        .detail-table td:last-child {{
            text-align: right;
            font-family: 'JetBrains Mono', monospace;
        }}
        
        .detail-table .highlight {{
            color: var(--accent-blue);
            font-weight: 600;
        }}
        
        .payout-amount {{
            font-size: 2rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            margin: 1rem 0 0.5rem;
        }}
        
        .payout-type {{
            font-size: 1rem;
            opacity: 0.9;
        }}
        
        .payout-note {{
            font-size: 0.85rem;
            margin-top: 1rem;
            opacity: 0.85;
        }}
        
        /* Formula Section */
        .formula-section {{
            background: var(--bg-tertiary);
            margin: 2rem -2.5rem;
            padding: 2rem 2.5rem;
        }}
        
        .formula-box {{
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: var(--shadow-sm);
        }}
        
        .formula {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.95rem;
            padding: 1rem;
            background: var(--bg-tertiary);
            border-radius: 6px;
            margin-bottom: 1.5rem;
            overflow-x: auto;
        }}
        
        .formula-breakdown {{
            display: grid;
            gap: 0.75rem;
        }}
        
        .formula-step {{
            display: grid;
            grid-template-columns: 120px 150px 1fr;
            align-items: baseline;
            padding: 0.5rem 0;
            border-bottom: 1px dashed var(--border-color);
        }}
        
        .formula-result {{
            display: grid;
            grid-template-columns: 120px 150px 1fr;
            align-items: baseline;
            padding: 1rem 0 0.5rem;
            font-weight: 600;
            font-size: 1.1rem;
            border-top: 2px solid var(--accent-purple);
            margin-top: 0.5rem;
        }}
        
        .step-label {{
            color: var(--text-muted);
            font-size: 0.9rem;
        }}
        
        .step-value {{
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-primary);
        }}
        
        .step-desc {{
            font-size: 0.85rem;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }}
        
        /* Disposition Badge */
        .disposition-badge {{
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 600;
            margin-top: 0.5rem;
        }}
        
        .disposition-badge.liquidity {{
            background: #dbeafe;
            color: #1e40af;
        }}
        
        .disposition-badge.target {{
            background: #d1fae5;
            color: #065f46;
        }}
        
        .disposition-badge.failure {{
            background: #fee2e2;
            color: #991b1b;
        }}
        
        .disposition-badge.pending {{
            background: #fef3c7;
            color: #92400e;
        }}
        
        /* Footer */
        footer {{
            background: var(--bg-tertiary);
            padding: 1.5rem 2.5rem;
            font-size: 0.85rem;
            color: var(--text-muted);
            text-align: center;
        }}
        
        @media (max-width: 768px) {{
            body {{
                padding: 0.5rem;
            }}
            
            header, main {{
                padding: 1.5rem;
            }}
            
            .formula-step, .formula-result {{
                grid-template-columns: 1fr;
                gap: 0.25rem;
            }}
        }}
        
        @media print {{
            body {{
                padding: 0;
                background: white;
            }}
            
            .container {{
                box-shadow: none;
            }}
            
            header {{
                background: #1e3a5f !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Safer Scenario Analysis</h1>
            <div class="subtitle">{result.scenario_name}</div>
            <div class="meta">
                Generated: {datetime.now().strftime("%B %d, %Y at %I:%M %p")} | 
                Duration: {result.duration_years:.1f} years | 
                Quarters Simulated: {len(result.cash_flows)}
            </div>
        </header>
        
        <main>
            <section>
                <h2>Investment Summary</h2>
                <div class="summary-grid">
                    <div class="summary-card">
                        <div class="label">Total Investment</div>
                        <div class="value">{format_currency(result.total_investment)}</div>
                    </div>
                    <div class="summary-card">
                        <div class="label">Repurchase Payments</div>
                        <div class="value">{format_currency(result.total_repurchase_payments)}</div>
                        <div class="subtext">{format_percentage(result.total_repurchase_payments / result.safer_terms.target_return if result.safer_terms.target_return > 0 else 0)} of Target</div>
                    </div>
                    <div class="summary-card">
                        <div class="label">Liquidity Payout</div>
                        <div class="value">{format_currency(result.total_liquidity_payout)}</div>
                    </div>
                    <div class="summary-card highlight">
                        <div class="label">Total Return</div>
                        <div class="value">{format_currency(result.total_return)}</div>
                        <div class="subtext">{result.multiple_on_invested:.2f}x MOIC</div>
                    </div>
                    <div class="summary-card">
                        <div class="label">IRR</div>
                        <div class="value">{format_percentage(result.irr) if result.irr else 'N/A'}</div>
                    </div>
                </div>
                
                <div style="text-align: center; margin-top: 1rem;">
                    <span class="disposition-badge {'liquidity' if result.disposition == DispositionType.LIQUIDITY_EVENT else 'target' if result.disposition == DispositionType.TARGET_RETURN_ACHIEVED else 'failure' if result.disposition == DispositionType.DISSOLUTION else 'pending'}">
                        {result.disposition.value}
                    </span>
                </div>
            </section>
            
            <section>
                <h2>Safer Terms</h2>
                <div class="terms-grid">
                    <div class="term-item">
                        <span class="term-label">Purchase Amount</span>
                        <span class="term-value">{format_currency(result.safer_terms.purchase_amount)}</span>
                    </div>
                    <div class="term-item">
                        <span class="term-label">Post-Money Valuation Cap</span>
                        <span class="term-value">{format_currency(result.safer_terms.post_money_valuation_cap)}</span>
                    </div>
                    <div class="term-item">
                        <span class="term-label">Revenue Percentage</span>
                        <span class="term-value">{format_percentage(result.safer_terms.revenue_percentage)}</span>
                    </div>
                    <div class="term-item">
                        <span class="term-label">Repurchase Percentage</span>
                        <span class="term-value">{format_percentage(result.safer_terms.repurchase_percentage)}</span>
                    </div>
                    <div class="term-item">
                        <span class="term-label">Target Return Multiple</span>
                        <span class="term-value">{result.safer_terms.target_return_multiple:.1f}x</span>
                    </div>
                    <div class="term-item">
                        <span class="term-label">Target Return</span>
                        <span class="term-value">{format_currency(result.safer_terms.target_return)}</span>
                    </div>
                    <div class="term-item">
                        <span class="term-label">Repurchase Amount</span>
                        <span class="term-value">{format_currency(result.safer_terms.repurchase_amount)}</span>
                    </div>
                    <div class="term-item">
                        <span class="term-label">Honeymoon Period</span>
                        <span class="term-value">{result.safer_terms.honeymoon_months} months</span>
                    </div>
                </div>
            </section>
            
            {'<section><h2>Revenue & Repurchase Timeline</h2><div class="chart-container"><img src="data:image/png;base64,' + chart1 + '" alt="Revenue and Repurchase Timeline"></div></section>' if chart1 else ''}
            
            {'<section><h2>Safer Amount Evolution</h2><div class="chart-container"><img src="data:image/png;base64,' + chart2 + '" alt="Safer Amount Evolution"></div></section>' if chart2 else ''}
            
            {liquidity_section}
            
            {safer_calc}
            
            {'<section><h2>Return Waterfall</h2><div class="chart-container"><img src="data:image/png;base64,' + chart3 + '" alt="Return Waterfall"></div></section>' if chart3 else ''}
            
            <section>
                <h2>Quarterly Cash Flow Detail</h2>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Period</th>
                                <th>ARR</th>
                                <th>Qtr Revenue</th>
                                <th>Repurchase Payment</th>
                                <th>Cumulative Repurchase</th>
                                <th>Safer Amount</th>
                                <th>Notes</th>
                            </tr>
                        </thead>
                        <tbody>
                            {cf_rows}
                        </tbody>
                    </table>
                </div>
            </section>
        </main>
        
        <footer>
            <p>Safer Scenario Simulator | Based on the Simple Agreement for Future Equity with Repurchase</p>
            <p>This report models the Safer instrument mechanics as defined in the legal agreement.</p>
        </footer>
    </div>
</body>
</html>
"""
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
    
    return html


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Safer Scenario Simulator - Generate detailed reports for Safer instrument scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a pre-defined scenario
  python safer_scenario_simulator.py --scenario steady-growth-acquisition
  
  # Customize terms with pre-defined scenario
  python safer_scenario_simulator.py \\
      --investment 1000000 \\
      --valuation-cap 10000000 \\
      --scenario steady-growth-acquisition
  
  # Full custom scenario
  python safer_scenario_simulator.py \\
      --investment 500000 \\
      --valuation-cap 5000000 \\
      --target-return-multiple 3.0 \\
      --revenue-share 0.05 \\
      --repurchase-percent 0.90 \\
      --honeymoon-months 12 \\
      --initial-arr 200000 \\
      --growth-rates 0,0.8,0.6,0.5,0.4 \\
      --exit-year 5 \\
      --exit-valuation 50000000 \\
      --output my_scenario.html

Pre-defined Scenarios:
  steady-growth-acquisition     Pre-seed SaaS, steady growth to $120M acquisition
  explosive-growth-ipo          Hot AI company, explosive growth to $2B IPO
  sustainable-complete-repurchase  Seed SaaS, sustainable growth, complete 3x repurchase
  failure-low-revenue           Company fails to achieve significant revenue
  modest-growth-small-exit      Modest growth, small early acquisition
        """
    )
    
    # Safer Terms
    parser.add_argument('--investment', '-i', type=float, default=1_000_000,
                        help='Investment (Purchase Amount) in dollars (default: 1,000,000)')
    parser.add_argument('--valuation-cap', '-v', type=float, default=10_000_000,
                        help='Post-Money Valuation Cap in dollars (default: 10,000,000)')
    parser.add_argument('--target-return-multiple', '-t', type=float, default=3.0,
                        help='Target Return as multiple of investment (default: 3.0)')
    parser.add_argument('--revenue-share', '-r', type=float, default=0.05,
                        help='Revenue Percentage for repurchase (default: 0.05 = 5%%)')
    parser.add_argument('--repurchase-percent', '-p', type=float, default=0.90,
                        help='Repurchase Percentage of Purchase Amount (default: 0.90 = 90%%)')
    parser.add_argument('--honeymoon-months', '-m', type=int, default=12,
                        help='Honeymoon Period in months (default: 12)')
    
    # Scenario Selection
    parser.add_argument('--scenario', '-s', type=str, default='steady-growth-acquisition',
                        choices=['steady-growth-acquisition', 'explosive-growth-ipo', 
                                'sustainable-complete-repurchase', 'failure-low-revenue',
                                'modest-growth-small-exit', 'custom'],
                        help='Pre-defined scenario type (default: steady-growth-acquisition)')
    
    # Custom Scenario Parameters
    parser.add_argument('--initial-arr', type=float,
                        help='Initial ARR for custom scenario')
    parser.add_argument('--growth-rates', type=str,
                        help='Comma-separated annual growth rates (e.g., "0,0.5,0.4,0.3")')
    parser.add_argument('--exit-year', type=float,
                        help='Year of liquidity event (e.g., 7 for year 7)')
    parser.add_argument('--exit-valuation', type=float,
                        help='Exit valuation in dollars')
    parser.add_argument('--description', type=str,
                        help='Custom scenario description')
    
    # Output
    parser.add_argument('--output', '-o', type=str, default='safer_scenario_report.html',
                        help='Output HTML file path (default: safer_scenario_report.html)')
    parser.add_argument('--json', action='store_true',
                        help='Also output results as JSON')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Build Safer terms from arguments
    safer_terms = SaferTerms(
        purchase_amount=args.investment,
        post_money_valuation_cap=args.valuation_cap,
        repurchase_percentage=args.repurchase_percent,
        revenue_percentage=args.revenue_share,
        target_return_multiple=args.target_return_multiple,
        honeymoon_months=args.honeymoon_months
    )
    
    # Get scenario
    scenario_type = ScenarioType(args.scenario)
    
    # Build custom params if provided
    custom_params = None
    if scenario_type == ScenarioType.CUSTOM or args.initial_arr or args.growth_rates:
        custom_params = {}
        if args.initial_arr:
            custom_params['initial_arr'] = args.initial_arr
        if args.growth_rates:
            custom_params['growth_rates'] = [float(x) for x in args.growth_rates.split(',')]
        if args.exit_year:
            custom_params['exit_year'] = args.exit_year
        if args.exit_valuation:
            custom_params['exit_valuation'] = args.exit_valuation
        if args.description:
            custom_params['description'] = args.description
        
        # If custom params provided but not using custom scenario, force custom
        if custom_params and scenario_type != ScenarioType.CUSTOM:
            scenario_type = ScenarioType.CUSTOM
    
    # Create revenue profile and scenario params
    revenue_profile, scenario_params = create_scenario_revenue_profile(scenario_type, custom_params)
    
    # If custom params provided, merge with defaults
    if custom_params:
        for key, value in custom_params.items():
            if value is not None:
                scenario_params[key] = value
    
    print(f"Running scenario: {scenario_params.get('description', 'Unknown')}")
    print(f"Investment: {format_currency(safer_terms.purchase_amount)}")
    print(f"Valuation Cap: {format_currency(safer_terms.post_money_valuation_cap)}")
    print(f"Target Return: {format_currency(safer_terms.target_return)} ({safer_terms.target_return_multiple}x)")
    print()
    
    # Run simulation
    result = simulate_scenario(
        safer_terms=safer_terms,
        revenue_profile=revenue_profile,
        scenario_params=scenario_params,
        max_quarters=80  # 20 years max
    )
    
    # Generate HTML report
    html = generate_html_report(result, args.output)
    print(f"HTML report generated: {args.output}")
    
    # Output JSON if requested
    if args.json:
        json_output = {
            'scenario_name': result.scenario_name,
            'safer_terms': {
                'purchase_amount': result.safer_terms.purchase_amount,
                'post_money_valuation_cap': result.safer_terms.post_money_valuation_cap,
                'repurchase_percentage': result.safer_terms.repurchase_percentage,
                'revenue_percentage': result.safer_terms.revenue_percentage,
                'target_return_multiple': result.safer_terms.target_return_multiple,
                'target_return': result.safer_terms.target_return,
                'repurchase_amount': result.safer_terms.repurchase_amount,
                'honeymoon_months': result.safer_terms.honeymoon_months
            },
            'results': {
                'total_investment': result.total_investment,
                'total_repurchase_payments': result.total_repurchase_payments,
                'total_liquidity_payout': result.total_liquidity_payout,
                'total_return': result.total_return,
                'multiple_on_invested': result.multiple_on_invested,
                'irr': result.irr,
                'duration_years': result.duration_years,
                'disposition': result.disposition.value
            },
            'liquidity_event': {
                'event_type': result.liquidity_event.event_type.value if result.liquidity_event else None,
                'exit_valuation': result.liquidity_event.exit_valuation if result.liquidity_event else None,
                'payout_amount': result.liquidity_event.payout_amount if result.liquidity_event else None,
                'payout_type': result.liquidity_event.payout_type if result.liquidity_event else None
            } if result.liquidity_event else None
        }
        
        json_path = args.output.replace('.html', '.json')
        with open(json_path, 'w') as f:
            json.dump(json_output, f, indent=2)
        print(f"JSON output generated: {json_path}")
    
    # Print detailed terminal output
    print()
    print("=" * 80)
    print(f"SAFER SCENARIO ANALYSIS: {result.scenario_name}")
    print("=" * 80)
    
    print()
    print("SAFER TERMS")
    print("-" * 40)
    print(f"  Purchase Amount:          {format_currency(result.safer_terms.purchase_amount)}")
    print(f"  Post-Money Valuation Cap: {format_currency(result.safer_terms.post_money_valuation_cap)}")
    print(f"  Revenue Percentage:       {format_percentage(result.safer_terms.revenue_percentage)}")
    print(f"  Repurchase Percentage:    {format_percentage(result.safer_terms.repurchase_percentage)}")
    print(f"  Target Return Multiple:   {result.safer_terms.target_return_multiple:.1f}x")
    print(f"  Target Return:            {format_currency(result.safer_terms.target_return)}")
    print(f"  Repurchase Amount:        {format_currency(result.safer_terms.repurchase_amount)}")
    print(f"  Honeymoon Period:         {result.safer_terms.honeymoon_months} months")
    
    print()
    print("RESULTS SUMMARY")
    print("-" * 40)
    print(f"  Disposition:              {result.disposition.value}")
    print(f"  Duration:                 {result.duration_years:.1f} years")
    print(f"  Total Investment:         {format_currency(result.total_investment)}")
    print(f"  Repurchase Payments:      {format_currency(result.total_repurchase_payments)}")
    pct_of_target = result.total_repurchase_payments / result.safer_terms.target_return * 100 if result.safer_terms.target_return > 0 else 0
    print(f"    (% of Target Return):   {pct_of_target:.1f}%")
    print(f"  Liquidity Payout:         {format_currency(result.total_liquidity_payout)}")
    print(f"  Total Return:             {format_currency(result.total_return)}")
    print(f"  MOIC:                     {result.multiple_on_invested:.2f}x")
    print(f"  IRR:                      {format_percentage(result.irr) if result.irr else 'N/A'}")
    
    if result.liquidity_event:
        le = result.liquidity_event
        print()
        print("LIQUIDITY EVENT")
        print("-" * 40)
        print(f"  Event Type:               {le.event_type.value}")
        print(f"  Timing:                   {le.date} (Year {le.year:.2f})")
        print(f"  Exit Valuation:           {format_currency(le.exit_valuation)}")
        print(f"  Safer Amount at Event:    {format_currency(le.safer_amount_at_event)}")
        print(f"  Ownership at Cap:         {le.ownership_percentage:.2f}%")
        print()
        print("  Payout Calculation:")
        print(f"    Cash-Out Amount:        {format_currency(le.cash_out_amount)}")
        print(f"    Conversion Amount:      {format_currency(le.conversion_amount)}")
        print(f"    --> Investor Receives:  {format_currency(le.payout_amount)} via {le.payout_type}")
    
    # Safer Amount Formula Breakdown
    if result.cash_flows:
        final_cf = result.cash_flows[-1]
        terms = result.safer_terms
        paid = min(final_cf.cumulative_repurchase, terms.target_return)
        buyback_fraction = paid / terms.target_return if terms.target_return > 0 else 0
        buyback_portion = buyback_fraction * terms.repurchase_amount
        remaining = terms.target_return - paid
        
        print()
        print("SAFER AMOUNT CALCULATION (at final period)")
        print("-" * 40)
        print("  Formula: Safer Amount = Purchase Amount")
        print("           - [(Cumulative Payments / Target Return) × Repurchase Amount]")
        print("           + [Target Return - Cumulative Payments]")
        print()
        print(f"  Component 1 (Purchase Amount):     {format_currency(terms.purchase_amount)}")
        print(f"  Component 2 (Buyback):           - {format_currency(buyback_portion)}")
        print(f"      [{format_currency(paid)} / {format_currency(terms.target_return)}] × {format_currency(terms.repurchase_amount)}")
        print(f"      = {buyback_fraction:.4f} × {format_currency(terms.repurchase_amount)}")
        print(f"  Component 3 (Remaining):         + {format_currency(remaining)}")
        print(f"      [{format_currency(terms.target_return)} - {format_currency(paid)}]")
        print()
        print(f"  Safer Amount:                      {format_currency(final_cf.safer_amount)}")
    
    # Cash Flow Table
    print()
    print("QUARTERLY CASH FLOW DETAIL")
    print("-" * 120)
    header = f"{'Period':<12} {'ARR':>14} {'Qtr Revenue':>14} {'Repurchase':>14} {'Cumulative':>14} {'Safer Amount':>14} {'Notes':<20}"
    print(header)
    print("-" * 120)
    
    for cf in result.cash_flows:
        arr = cf.gross_revenue * 4
        row = f"{cf.date:<12} {format_currency(arr):>14} {format_currency(cf.gross_revenue):>14} {format_currency(cf.repurchase_payment):>14} {format_currency(cf.cumulative_repurchase):>14} {format_currency(cf.safer_amount):>14} {cf.notes:<20}"
        print(row)
    
    print("-" * 120)
    
    # Return Waterfall
    print()
    print("RETURN WATERFALL")
    print("-" * 40)
    print(f"  Investment:              -{format_currency(result.total_investment)}")
    print(f"  Repurchase Payments:     +{format_currency(result.total_repurchase_payments)}")
    if result.liquidity_event:
        print(f"  Liquidity Payout:        +{format_currency(result.total_liquidity_payout)}")
    print(f"  ----------------------------------------")
    print(f"  Total Return:             {format_currency(result.total_return)}")
    print(f"  MOIC:                     {result.multiple_on_invested:.2f}x")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
