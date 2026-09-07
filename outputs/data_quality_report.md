======================================================================
  PROJECT FORESIGHT — Data Quality & Cleaning Report
  Client: NorthBay Living
======================================================================

--------------------------------------------------
1. DATASET DIMENSIONS (Before → After Cleaning)
--------------------------------------------------
  sku_master                            61 →       60  (Δ 1)
  sales_daily                        32687 →    32625  (Δ 62)
  calendar                             974 →      974  (Δ 0)
  inventory_snapshots                 7481 →     7481  (Δ 0)

--------------------------------------------------
2. MISSING VALUES (Before Cleaning)
--------------------------------------------------
  [sales_daily]  Total missing: 757
    - date: 5 missing
    - units_sold: 500 missing
    - revenue: 252 missing
  [sku_master]  No missing values
  [calendar]  Total missing: 1763
    - holiday_name: 956 missing
    - promo_event: 807 missing
  [inventory_snapshots]  Total missing: 149
    - on_hand_units: 149 missing

--------------------------------------------------
3. DUPLICATE RECORDS (Before Cleaning)
--------------------------------------------------
  sales_daily                     50 exact duplicate rows
  sku_master                      1 exact duplicate rows
  calendar                        0 exact duplicate rows
  inventory_snapshots             0 exact duplicate rows

--------------------------------------------------
4. INVALID VALUES (Before Cleaning)
--------------------------------------------------
  Invalid/unparseable dates (sales_daily): 10
  Negative units_sold (sales_daily):       20

--------------------------------------------------
5. INCONSISTENT LABELS (Before Cleaning)
--------------------------------------------------
  [category] Unique values found (8):
    - "Bedding"
    - "Decor"
    - "Furniture"
    - "Kitchen"
    - "OUTDOOR"
    - "Outdoor"
    - "furniture"
    - "kitchen"
  [subcategory] Unique values found (17):
    - "Accessories"
    - "Chairs"
    - "Comforters"
    - "Cookware"
    - "Lighting"
    - "Mattress Toppers"
    - "Pillows"
    - "Seating"
    - "Sheets"
    - "Shelves"
    - "Sofas"
    - "Storage"
    - "Tables"
    - "Utensils"
    - "Vases"
    - "Wall Art"
    - "seating"

--------------------------------------------------
6. SKU INTEGRITY
--------------------------------------------------
  Orphan SKUs in sales (not in sku_master): ['SKU-999']

--------------------------------------------------
7. DATA TYPE ISSUES
--------------------------------------------------
  No data type issues detected.

--------------------------------------------------
8. CLEANING DECISIONS & ACTIONS
--------------------------------------------------
  1. [sku_master] duplicates
     Action:        dropped
     Rows affected: 1
     Reason:        Exact duplicate rows

  2. [sku_master] inconsistent casing
     Action:        title case
     Rows affected: 60
     Reason:        Standardize category names

  3. [sku_master] inconsistent casing
     Action:        title case
     Rows affected: 60
     Reason:        Standardize subcategory names

  4. [sales_daily] duplicates
     Action:        dropped
     Rows affected: 50
     Reason:        Exact duplicate rows

  5. [sales_daily] invalid dates
     Action:        dropped rows
     Rows affected: 10
     Reason:        Unparseable dates

  6. [sales_daily] orphan SKUs
     Action:        dropped rows
     Rows affected: 2
     Reason:        SKUs not found in sku_master

  7. [sales_daily] missing units_sold
     Action:        filled with 0
     Rows affected: 500
     Reason:        Assume 0 sales if missing

  8. [sales_daily] negative units_sold
     Action:        clipped to 0
     Rows affected: 20
     Reason:        Invalid negative sales

  9. [sales_daily] missing revenue
     Action:        calculated
     Rows affected: 250
     Reason:        Imputed using units_sold * unit_price

  10. [calendar] missing holiday_name
     Action:        filled with empty string
     Rows affected: 956
     Reason:        Standardize missing text

  11. [calendar] missing promo_event
     Action:        filled with empty string
     Rows affected: 807
     Reason:        Standardize missing text


--------------------------------------------------
9. FINAL DATASET SUMMARY
--------------------------------------------------
  [sku_master]
    Shape:   60 rows × 7 columns
    Columns: ['sku_id', 'product_name', 'category', 'subcategory', 'launch_date', 'unit_cost', 'list_price']
    Dtypes:
      - sku_id: str
      - product_name: str
      - category: str
      - subcategory: str
      - launch_date: datetime64[us]
      - unit_cost: float64
      - list_price: float64
    Remaining nulls: 0

  [sales_daily]
    Shape:   32625 rows × 6 columns
    Columns: ['date', 'sku_id', 'units_sold', 'revenue', 'unit_price', 'promo_flag']
    Dtypes:
      - date: datetime64[us]
      - sku_id: str
      - units_sold: float64
      - revenue: float64
      - unit_price: float64
      - promo_flag: int64
    Remaining nulls: 0

  [calendar]
    Shape:   974 rows × 9 columns
    Columns: ['date', 'week', 'month', 'year', 'day_of_week', 'season', 'is_holiday', 'holiday_name', 'promo_event']
    Dtypes:
      - date: datetime64[us]
      - week: int64
      - month: int64
      - year: int64
      - day_of_week: str
      - season: str
      - is_holiday: int64
      - holiday_name: str
      - promo_event: str
    Remaining nulls: 0

  [inventory_snapshots]
    Shape:   7481 rows × 6 columns
    Columns: ['date', 'sku_id', 'on_hand_units', 'on_order_units', 'lead_time_days', 'reorder_point']
    Dtypes:
      - date: datetime64[us]
      - sku_id: str
      - on_hand_units: float64
      - on_order_units: int64
      - lead_time_days: int64
      - reorder_point: int64
    Remaining nulls: 0

======================================================================
  END OF REPORT
======================================================================