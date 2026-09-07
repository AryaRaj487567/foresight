import os
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seeds
np.random.seed(42)
random.seed(42)

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)


print(f"Generating data in {DATA_DIR}...")

# -----------------
# 1. sku_master.csv
# -----------------
print("Generating sku_master.csv...")

categories_info = {
    "Furniture": {"subcategories": ["Tables", "Chairs", "Shelves", "Sofas"], "count": 12, "cost_range": (40, 250)},
    "Decor": {"subcategories": ["Pillows", "Lighting", "Wall Art", "Vases"], "count": 15, "cost_range": (5, 40)},
    "Kitchen": {"subcategories": ["Cookware", "Storage", "Utensils"], "count": 12, "cost_range": (8, 60)},
    "Bedding": {"subcategories": ["Sheets", "Comforters", "Pillows", "Mattress Toppers"], "count": 10, "cost_range": (15, 80)},
    "Outdoor": {"subcategories": ["Seating", "Lighting", "Accessories"], "count": 11, "cost_range": (20, 150)},
}

product_names = {
    "Furniture": ["Oakwood Dining Table", "Velvet Armchair", "Industrial Bookshelf", "Linen Sofa", "Modern Coffee Table", "Leather Recliner", "Rustic Dining Chair", "Glass End Table", "Wood TV Stand", "Ergonomic Office Chair", "Minimalist Desk", "Upholstered Ottoman"],
    "Decor": ["Ceramic Vase Set", "Linen Throw Pillow", "Woven Wall Hanging", "Brass Table Lamp", "Geometric Rug", "Abstract Canvas Art", "Scented Candle Set", "Decorative Mirror", "Macrame Plant Hanger", "Faux Potted Plant", "Crystal Chandelier", "Velvet Floor Cushion", "Wooden Picture Frame", "Metal Wall Sconce", "Knit Throw Blanket"],
    "Kitchen": ["Stainless Steel Cookware Set", "Bamboo Cutting Board", "Glass Storage Jars", "Cast Iron Skillet", "Silicone Utensil Set", "Chef's Knife", "Espresso Machine", "Dutch Oven", "Spice Rack", "Mixing Bowl Set", "Toaster Oven", "Coffee Grinder"],
    "Bedding": ["Egyptian Cotton Sheet Set", "Down Alternative Comforter", "Memory Foam Pillow", "Flannel Duvet Cover", "Silk Pillowcase", "Quilted Mattress Pad", "Weighted Blanket", "Linen Bed Skirt", "Microfiber Sheet Set", "Cooling Gel Pillow"],
    "Outdoor": ["Teak Patio Set", "Solar Garden Lights", "Hammock Chair", "Wicker Lounge Chair", "Outdoor Fire Pit", "Patio Umbrella", "Rattan Sectional", "Outdoor Rug", "Hanging Planter", "Bird Feeder", "String Lights"],
}

sku_data = []
sku_counter = 1

start_launch_date = datetime(2022, 1, 1)
end_launch_date = datetime(2025, 6, 1)

for cat, info in categories_info.items():
    names = product_names[cat]
    sampled_names = random.sample(names, min(len(names), info["count"]))
    if len(sampled_names) < info["count"]:
        sampled_names += random.choices(names, k=info["count"] - len(sampled_names))
        
    for i in range(info["count"]):
        sku_id = f"SKU-{sku_counter:03d}"
        sku_counter += 1
        
        name = sampled_names[i]
        subcat = random.choice(info["subcategories"])
        
        # Random launch date
        days_diff = (end_launch_date - start_launch_date).days
        launch_date = start_launch_date + timedelta(days=random.randint(0, days_diff))
        
        unit_cost = round(random.uniform(*info["cost_range"]), 2)
        multiplier = random.uniform(1.8, 2.5)
        list_price = round(unit_cost * multiplier, 2)
        
        sku_data.append({
            "sku_id": sku_id,
            "product_name": name,
            "category": cat,
            "subcategory": subcat,
            "launch_date": launch_date.strftime("%Y-%m-%d"),
            "unit_cost": unit_cost,
            "list_price": list_price
        })

sku_df = pd.DataFrame(sku_data)

# DELIBERATE QUALITY ISSUES for sku_master
# 3 SKUs: category in lowercase
lower_cat_idx = np.random.choice(sku_df.index, 3, replace=False)
sku_df.loc[lower_cat_idx, 'category'] = sku_df.loc[lower_cat_idx, 'category'].str.lower()

# 2 SKUs: category in UPPERCASE
upper_cat_idx = np.random.choice(list(set(sku_df.index) - set(lower_cat_idx)), 2, replace=False)
sku_df.loc[upper_cat_idx, 'category'] = sku_df.loc[upper_cat_idx, 'category'].str.upper()

# 3 SKUs: subcategory in lowercase
lower_subcat_idx = np.random.choice(sku_df.index, 3, replace=False)
sku_df.loc[lower_subcat_idx, 'subcategory'] = sku_df.loc[lower_subcat_idx, 'subcategory'].str.lower()

# 1 exact duplicate row
dup_row = sku_df.sample(1, random_state=42)
sku_df = pd.concat([sku_df, dup_row], ignore_index=True)

sku_df.to_csv(os.path.join(DATA_DIR, "sku_master.csv"), index=False)
print(f"sku_master.csv generated. Shape: {sku_df.shape}")

# -----------------
# 3. calendar.csv (Do this before sales to use promo info)
# -----------------
print("Generating calendar.csv...")
start_date = datetime(2024, 1, 1)
end_date = datetime(2026, 8, 31)
date_range = pd.date_range(start_date, end_date)

calendar_data = []

def get_season(month):
    if month in [12, 1, 2]: return "Winter"
    elif month in [3, 4, 5]: return "Spring"
    elif month in [6, 7, 8]: return "Summer"
    else: return "Fall"

for d in date_range:
    is_holiday = 0
    holiday_name = ""
    promo_event = ""
    
    if d.month == 1 and d.day == 1:
        is_holiday = 1; holiday_name = "New Year's Day"
    if d.month == 1 and 1 <= d.day <= 3:
        promo_event = "New Year Sale"
        
    if d.month == 2 and 15 <= d.day <= 21 and d.weekday() == 0:
        is_holiday = 1; holiday_name = "Presidents Day"
        promo_event = "Presidents Day Sale"
    elif d.month == 2 and 14 <= d.day <= 20 and promo_event == "":
        if random.random() < 0.3: promo_event = "Presidents Day Sale"

    if d.month == 3 and 15 <= d.day <= 21:
        promo_event = "Spring Clearance"

    if d.month == 5 and d.day >= 25 and d.weekday() == 0:
        is_holiday = 1; holiday_name = "Memorial Day"
        promo_event = "Memorial Day Sale"
    elif d.month == 5 and d.day >= 23 and promo_event == "":
        if random.random() < 0.4: promo_event = "Memorial Day Sale"
    
    if d.month == 7 and d.day == 4:
        is_holiday = 1; holiday_name = "Independence Day"
    if d.month == 7 and 1 <= d.day <= 7:
        promo_event = "Summer Sale"

    if d.month == 8 and 10 <= d.day <= 20:
        promo_event = "Back to School"
        
    if d.month == 9 and d.day <= 7 and d.weekday() == 0:
        is_holiday = 1; holiday_name = "Labor Day"
        promo_event = "Labor Day Sale"
    elif d.month == 9 and d.day <= 9 and promo_event == "":
        if random.random() < 0.4: promo_event = "Labor Day Sale"

    if d.month == 10 and 10 <= d.day <= 16:
        promo_event = "Fall Home Sale"

    if d.month == 11 and 22 <= d.day <= 28 and d.weekday() == 3:
        is_holiday = 1; holiday_name = "Thanksgiving"
    if d.month == 11 and 23 <= d.day <= 30:
        promo_event = "Black Friday"

    if d.month == 12 and d.day == 25:
        is_holiday = 1; holiday_name = "Christmas Day"
    if d.month == 12 and 15 <= d.day <= 25:
        promo_event = "Holiday Sale"

    calendar_data.append({
        "date": d.strftime("%Y-%m-%d"),
        "week": d.isocalendar()[1],
        "month": d.month,
        "year": d.year,
        "day_of_week": d.strftime("%A"),
        "season": get_season(d.month),
        "is_holiday": is_holiday,
        "holiday_name": holiday_name,
        "promo_event": promo_event
    })

calendar_df = pd.DataFrame(calendar_data)
calendar_df.to_csv(os.path.join(DATA_DIR, "calendar.csv"), index=False)
print(f"calendar.csv generated. Shape: {calendar_df.shape}")

# -----------------
# 2. sales_daily.csv
# -----------------
print("Generating sales_daily.csv...")
sales_data = []

sku_sales_props = {}
for idx, row in sku_df.drop_duplicates(subset=['sku_id']).iterrows():
    base_demand = random.uniform(1, 15) if random.random() < 0.6 else random.uniform(0, 3)
    sales_prob = random.uniform(0.6, 0.8)
    
    orig_cat = [k for k,v in categories_info.items() if k.lower() == row['category'].lower()][0]
    
    sku_sales_props[row['sku_id']] = {
        "base_demand": base_demand,
        "sales_prob": sales_prob,
        "launch_date": datetime.strptime(row['launch_date'], "%Y-%m-%d"),
        "list_price": row['list_price'],
        "category": orig_cat
    }

for cal_row in calendar_data:
    d = datetime.strptime(cal_row['date'], "%Y-%m-%d")
    is_promo = 1 if cal_row['promo_event'] != "" else 0
    
    for sku_id, props in sku_sales_props.items():
        if d < props['launch_date']: continue
        if random.random() > props['sales_prob']: continue
            
        seasonality = 1.0
        cat = props['category']
        m = d.month
        
        if cat == "Outdoor":
            if m in [6, 7, 8]: seasonality = 1.5
            elif m in [12, 1, 2]: seasonality = 0.5
        elif cat == "Bedding":
            if m in [10, 11, 12, 1]: seasonality = 1.4
            elif m in [5, 6, 7]: seasonality = 0.7
        elif cat == "Furniture":
            if m in [3, 4, 9, 10]: seasonality = 1.3
        elif cat == "Decor":
            if m in [11, 12]: seasonality = 1.2
        elif cat == "Kitchen":
            if m in [11, 12]: seasonality = 1.3
            
        months_since_launch = (d.year - props['launch_date'].year) * 12 + d.month - props['launch_date'].month
        trend = 1.0 + (0.0005 * max(0, months_since_launch))
        
        promo_mult = random.uniform(1.3, 2.0) if is_promo else 1.0
        
        expected_demand = props['base_demand'] * seasonality * trend * promo_mult
        units_sold = np.random.poisson(expected_demand)
        
        if units_sold == 0:
            if random.random() > 0.1: continue
                
        unit_price = props['list_price']
        if is_promo:
            discount = random.uniform(0.10, 0.25)
            unit_price = round(unit_price * (1 - discount), 2)
            
        revenue = round(units_sold * unit_price, 2)
        
        sales_data.append({
            "date": cal_row['date'],
            "sku_id": sku_id,
            "units_sold": units_sold,
            "revenue": revenue,
            "unit_price": unit_price,
            "promo_flag": is_promo
        })

sales_df = pd.DataFrame(sales_data)

# DELIBERATE QUALITY ISSUES for sales_daily
nan_units_idx = np.random.choice(sales_df.index, min(500, len(sales_df)), replace=False)
sales_df.loc[nan_units_idx, 'units_sold'] = np.nan

nan_rev_idx = np.random.choice(sales_df.index, min(250, len(sales_df)), replace=False)
sales_df.loc[nan_rev_idx, 'revenue'] = np.nan

dup_sales = sales_df.sample(min(50, len(sales_df)), random_state=42)
sales_df = pd.concat([sales_df, dup_sales], ignore_index=True)

neg_units_idx = np.random.choice(sales_df.dropna(subset=['units_sold']).index, min(20, len(sales_df)), replace=False)
sales_df.loc[neg_units_idx, 'units_sold'] = np.random.randint(-5, 0, size=len(neg_units_idx))

bad_date_idx = np.random.choice(sales_df.index, min(10, len(sales_df)), replace=False)
sales_df.loc[bad_date_idx[:5], 'date'] = "not_available"
sales_df.loc[bad_date_idx[5:], 'date'] = "N/A"

orphan_idx = np.random.choice(sales_df.index, min(2, len(sales_df)), replace=False)
sales_df.loc[orphan_idx, 'sku_id'] = "SKU-999"

sales_df = sales_df.sample(frac=1, random_state=42).reset_index(drop=True)

sales_df.to_csv(os.path.join(DATA_DIR, "sales_daily.csv"), index=False)
print(f"sales_daily.csv generated. Shape: {sales_df.shape}")


# -----------------
# 4. inventory_snapshots.csv
# -----------------
print("Generating inventory_snapshots.csv...")

mondays = [d['date'] for d in calendar_data if datetime.strptime(d['date'], "%Y-%m-%d").weekday() == 0]

inv_data = []
avg_weekly_demand = {}
for sku_id, props in sku_sales_props.items():
    avg_weekly_demand[sku_id] = props['base_demand'] * props['sales_prob'] * 7

sku_inv_state = {}
for sku_id, props in sku_sales_props.items():
    cat = props['category']
    if cat == "Furniture": lt = random.randint(21, 45)
    elif cat == "Decor": lt = random.randint(7, 14)
    elif cat == "Kitchen": lt = random.randint(10, 21)
    elif cat == "Bedding": lt = random.randint(14, 28)
    else: lt = random.randint(14, 35)
    
    rop = int(avg_weekly_demand[sku_id] * (lt / 7) * 1.5)
    
    sku_inv_state[sku_id] = {
        "on_hand": random.randint(50, 200),
        "on_order": 0,
        "lead_time": lt,
        "rop": rop,
        "order_arrival_week": -1
    }

week_idx = 0
for mon_str in mondays:
    mon_dt = datetime.strptime(mon_str, "%Y-%m-%d")
    
    for sku_id, props in sku_sales_props.items():
        if mon_dt < props['launch_date']: continue
            
        state = sku_inv_state[sku_id]
        
        if state['order_arrival_week'] == week_idx:
            state['on_hand'] += state['on_order']
            state['on_order'] = 0
            
        depletion = np.random.poisson(avg_weekly_demand[sku_id])
        state['on_hand'] = max(0, state['on_hand'] - depletion)
        
        if state['on_hand'] < state['rop'] and state['on_order'] == 0:
            order_qty = random.randint(50, 150)
            state['on_order'] = order_qty
            weeks_to_arrive = max(1, int(state['lead_time'] / 7))
            state['order_arrival_week'] = week_idx + weeks_to_arrive
            
        inv_data.append({
            "date": mon_str,
            "sku_id": sku_id,
            "on_hand_units": state['on_hand'],
            "on_order_units": state['on_order'],
            "lead_time_days": state['lead_time'],
            "reorder_point": state['rop']
        })
        
    week_idx += 1

inv_df = pd.DataFrame(inv_data)

# DELIBERATE QUALITY ISSUES for inventory
nan_inv_count = int(0.02 * len(inv_df))
nan_inv_idx = np.random.choice(inv_df.index, nan_inv_count, replace=False)
inv_df.loc[nan_inv_idx, 'on_hand_units'] = np.nan

inv_df.to_csv(os.path.join(DATA_DIR, "inventory_snapshots.csv"), index=False)
print(f"inventory_snapshots.csv generated. Shape: {inv_df.shape}")

print("Data generation complete!")
