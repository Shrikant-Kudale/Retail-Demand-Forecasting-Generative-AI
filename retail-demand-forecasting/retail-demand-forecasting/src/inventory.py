"""
inventory.py
Inventory and replenishment decision-support module.

CRITICAL METHODOLOGICAL NOTE:
The Walmart Weekly_Sales dataset provides monetary sales revenue ($), NOT physical unit volumes.
Therefore:
1. Physical unit inventory replenishment is provided as an illustrative DEMONSTRATION SCENARIO,
   converting monetary sales to synthetic physical units via an assumed Average Selling Price (ASP).
2. A separate MONETARY DEMAND PLANNING SCENARIO calculates working capital, buffer capital ($),
   and monetary replenishment budgets.
"""

from statistics import NormalDist
from typing import Dict, Any, Optional
import numpy as np


SERVICE_LEVEL_Z = {
    85.0: 1.04,
    90.0: 1.28,
    95.0: 1.645,
    98.0: 2.05,
    99.0: 2.33,
    99.9: 3.09
}


def get_z_score(service_level_pct: float) -> float:
    """Returns the normal distribution Z-score for the requested service level percentage."""
    if service_level_pct in SERVICE_LEVEL_Z:
        return SERVICE_LEVEL_Z[service_level_pct]
    # Any other level: exact inverse of the standard normal CDF
    p = min(max(service_level_pct / 100.0, 0.5), 0.9999)
    return float(NormalDist().inv_cdf(p))


def calculate_inventory_policy(
    forecast_weekly: float,
    demand_std: float,
    lead_time_weeks: int = 2,
    review_period_weeks: int = 1,
    service_level_pct: float = 95.0,
    current_on_hand: float = 100.0,
    incoming_orders: float = 0.0,
    backorders: float = 0.0,
    unit_cost: float = 20.0,
    is_monetary_scenario: bool = False
) -> Dict[str, Any]:
    """
    Computes rigorous periodic review (R, S) or continuous review (s, S) replenishment parameters.
    
    Equations:
    - Lead Time Demand: D_L = forecast_weekly * lead_time_weeks
    - Review + Lead Time Demand: D_{L+R} = forecast_weekly * (lead_time_weeks + review_period_weeks)
    - Safety Stock: SS = Z * demand_std * sqrt(lead_time_weeks + review_period_weeks)
    - Reorder Point (ROP): ROP = D_L + SS
    - Target Inventory Position (Order-Up-To Level S): S = D_{L+R} + SS
    - Current Inventory Position: IP = current_on_hand + incoming_orders - backorders
    - Recommended Order Quantity: Q = max(0, S - IP)
    """
    z = get_z_score(service_level_pct)
    
    # Lead time demand
    lead_time_demand = forecast_weekly * lead_time_weeks
    protection_period = lead_time_weeks + review_period_weeks
    protection_demand = forecast_weekly * protection_period
    
    # Safety stock over protection interval (L + R)
    safety_stock = z * demand_std * np.sqrt(protection_period)
    
    # Reorder point and order-up-to level
    reorder_point = lead_time_demand + safety_stock
    target_inventory = protection_demand + safety_stock
    
    # Inventory position
    inventory_position = current_on_hand + incoming_orders - backorders
    
    # Recommended order
    order_quantity = max(0.0, target_inventory - inventory_position)
    
    # Monetary evaluation
    unit_label = "$" if is_monetary_scenario else "units"
    estimated_order_value = order_quantity if is_monetary_scenario else order_quantity * unit_cost
    safety_stock_value = safety_stock if is_monetary_scenario else safety_stock * unit_cost

    return {
        "service_level_pct": service_level_pct,
        "z_score": round(z, 3),
        "lead_time_weeks": lead_time_weeks,
        "review_period_weeks": review_period_weeks,
        "forecast_weekly": round(forecast_weekly, 2),
        "demand_std": round(demand_std, 2),
        "lead_time_demand": round(lead_time_demand, 2),
        "safety_stock": round(safety_stock, 2),
        "reorder_point": round(reorder_point, 2),
        "target_inventory": round(target_inventory, 2),
        "current_inventory_position": round(inventory_position, 2),
        "order_quantity": round(order_quantity, 2),
        "estimated_order_cost": round(estimated_order_value, 2),
        "safety_stock_holding_cost": round(safety_stock_value, 2),
        "unit_label": unit_label,
        "is_monetary_scenario": is_monetary_scenario,
        "disclaimer": (
            "MONETARY DEMAND PLANNING SCENARIO" if is_monetary_scenario
            else "DEMONSTRATION SCENARIO: Physical unit quantities derived via assumed Average Selling Price."
        )
    }
