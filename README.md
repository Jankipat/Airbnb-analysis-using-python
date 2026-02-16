# Airbnb Analysis Using Python

This project recreates an **Airbnb NYC listings analysis** using:
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**

## Files
- `datasets.csv` → source dataset
- `airbnb_analysis.py` → complete EDA pipeline script
- `outputs/` → generated cleaned data and charts (created after running script)

## What the script does
1. Loads `datasets.csv`
2. Cleans and prepares the data
   - converts numeric columns
   - parses date columns
   - handles missing values
   - removes invalid price rows and duplicate IDs
3. Prints key insights:
   - row/column count
   - average and median price
   - average rating
   - top neighbourhood groups
   - room type distribution
4. Saves analysis outputs:
   - `cleaned_airbnb_data.csv`
   - `price_distribution.png`
   - `avg_price_by_neighbourhood_group.png`
   - `room_type_count.png`
   - `rating_vs_price.png`

## Run
```bash
python airbnb_analysis.py
```

If your environment does not already have dependencies, install:
```bash
pip install numpy pandas matplotlib seaborn
```
