import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sqlite3
import os
import requests
import tempfile

# Set page config
st.set_page_config(page_title="Luxembourg Price-to-Rent Heatmap", page_icon="🏠", layout="wide")

# App title
st.title("Luxembourg Price-to-Rent Ratio Heatmap")

# Function to ensure numeric values
def ensure_numeric(value):
    """Convert string numbers to float and ensure proper numeric values"""
    if isinstance(value, str):
        try:
            return float(value.replace(',', '.'))
        except (ValueError, TypeError):
            return 0.0
    return float(value) if value is not None else 0.0

# Function to download the database
@st.cache_resource
def download_database(url):
    """Download the database from the URL and return a connection"""
    try:
        with st.spinner("Downloading database from GitHub..."):
            # Create a temporary file
            temp_db_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
            temp_db_path = temp_db_file.name
            
            # Download the file
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an error for bad responses
            
            # Write the content to the temporary file
            with open(temp_db_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Connect to the database
            conn = sqlite3.connect(temp_db_path)
            return conn
    except Exception as e:
        st.error(f"Error downloading database: {e}")
        return None

# Database URL
db_url = "https://github.com/masterries/LuxOpenDataToAnalysis/raw/refs/heads/main/DataGathering/luxembourg_housing.db"

# Download database and create connection
conn = download_database(db_url)

if conn is None:
    st.error("Failed to download and connect to the database.")
    st.stop()

# Get all data
def get_all_data():
    """Get all housing data from the database"""
    query = """
    SELECT 
        year,
        municipality,
        property_type,
        transaction_type,
        avg_price,
        avg_price_sqm,
        num_listings
    FROM housing_data
    WHERE year IS NOT NULL
    """
    
    df = pd.read_sql_query(query, conn)
    
    # Convert numeric columns to proper numeric types
    numeric_cols = ['avg_price', 'avg_price_sqm', 'num_listings']
    for col in numeric_cols:
        df[col] = df[col].apply(ensure_numeric)
    
    return df

# Get all data
all_data = get_all_data()

# Get list of property types
property_types = all_data['property_type'].unique().tolist()

# Property type selector
selected_property_type = st.selectbox(
    "Select Property Type",
    options=property_types,
    index=property_types.index("apartment") if "apartment" in property_types else 0
)

# Filter data by selected property type
filtered_data = all_data[all_data['property_type'] == selected_property_type]

# Function to calculate price-to-rent ratio
def calculate_price_to_rent_ratios(df):
    """Calculate price-to-rent ratio for each municipality and year"""
    # Group by municipality, year, and transaction type, taking the mean of prices
    df_grouped = df.groupby(['municipality', 'year', 'transaction_type']).agg({
        'avg_price': 'mean',
        'num_listings': 'sum'
    }).reset_index()
    
    # Create a dictionary to store all data
    ptr_data = {}
    
    # For each year, calculate the price-to-rent ratio for each municipality
    for year in df_grouped['year'].unique():
        year_data = df_grouped[df_grouped['year'] == year]
        
        # Get sales and rental data for this year
        sales_data = year_data[year_data['transaction_type'] == 'sale']
        rental_data = year_data[year_data['transaction_type'] == 'rent']
        
        # Create dictionaries for quick lookup
        sale_prices = {row['municipality']: row['avg_price'] for _, row in sales_data.iterrows()}
        sale_listings = {row['municipality']: row['num_listings'] for _, row in sales_data.iterrows()}
        rent_prices = {row['municipality']: row['avg_price'] for _, row in rental_data.iterrows()}
        rent_listings = {row['municipality']: row['num_listings'] for _, row in rental_data.iterrows()}
        
        # Calculate price-to-rent ratio for municipalities with both sales and rental data
        for municipality in set(sale_prices.keys()).intersection(set(rent_prices.keys())):
            sale_price = ensure_numeric(sale_prices[municipality])
            rent_price = ensure_numeric(rent_prices[municipality]) * 12  # Annual rent
            sale_listing_count = sale_listings.get(municipality, 0)
            rent_listing_count = rent_listings.get(municipality, 0)
            
            if sale_price > 0 and rent_price > 0:
                ratio = sale_price / rent_price
                
                # Store all data
                if municipality not in ptr_data:
                    ptr_data[municipality] = {}
                
                ptr_data[municipality][year] = {
                    'ratio': ratio,
                    'sale_price': sale_price,
                    'annual_rent': rent_price,
                    'sale_listings': sale_listing_count,
                    'rent_listings': rent_listing_count
                }
    
    return ptr_data

# Calculate price-to-rent ratios
ptr_data = calculate_price_to_rent_ratios(filtered_data)

# Convert to DataFrame for heatmap
years = sorted(filtered_data['year'].unique())
municipalities = sorted(ptr_data.keys())

# Create DataFrames for the heatmap values and hover information
ratio_matrix = []
hover_texts = []

for municipality in municipalities:
    ratio_row = []
    hover_row = []
    
    for year in years:
        if municipality in ptr_data and year in ptr_data[municipality]:
            data = ptr_data[municipality][year]
            ratio = data['ratio']
            ratio_row.append(ratio)
            
            # Create hover text with details
            hover_text = (
                f"Municipality: {municipality}<br>"
                f"Year: {year}<br>"
                f"Price-to-Rent Ratio: {ratio:.1f}<br>"
                f"Average Sale Price: €{data['sale_price']:,.0f}<br>"
                f"Annual Rent: €{data['annual_rent']:,.0f}<br>"
                f"Sale Listings: {data['sale_listings']}<br>"
                f"Rent Listings: {data['rent_listings']}"
            )
            hover_row.append(hover_text)
        else:
            ratio_row.append(None)
            hover_row.append(None)
    
    ratio_matrix.append(ratio_row)
    hover_texts.append(hover_row)

# Create Plotly heatmap
fig = go.Figure()

# Create labels directly with the values
text_annotations = []
for i, municipality in enumerate(municipalities):
    for j, year in enumerate(years):
        if municipality in ptr_data and year in ptr_data[municipality]:
            ratio = ptr_data[municipality][year]['ratio']
            text_annotations.append(
                dict(
                    x=year,
                    y=municipality,
                    text=f"{ratio:.1f}",
                    showarrow=False,
                    font=dict(color='black', size=9)
                )
            )

# Add heatmap trace
heatmap = go.Heatmap(
    z=ratio_matrix,
    x=years,
    y=municipalities,
    colorscale=[
        [0, 'green'],      # 20 (good investment)
        [0.5, 'yellow'],   # 40 (medium)
        [1, 'red']         # 60+ (poor investment)
    ],
    zmin=20, zmax=60,
    zmid=40,
    text=hover_texts,  # Hover text
    hoverinfo='text',
    colorbar=dict(
        title='Price-to-Rent Ratio',
        title_side='right',
        ticks='outside',
        tickvals=[20, 40, 60],
        ticktext=['20 (Good)', '40 (Medium)', '60 (Poor)']
    ),
    showscale=True
)

fig.add_trace(heatmap)

# Update layout
fig.update_layout(
    title=f'Price-to-Rent Ratio Heatmap for {selected_property_type.capitalize()}s',
    xaxis_title='Year',
    yaxis_title='Municipality',
    height=max(600, len(municipalities) * 20),  # Adjust height based on number of municipalities
    width=1000,
    font=dict(size=12),
    annotations=text_annotations
)

# Display the chart
st.plotly_chart(fig, use_container_width=True)

# Add explanation
st.markdown("""
### Understanding the Price-to-Rent Ratio

The Price-to-Rent ratio is calculated by dividing the average property sale price by the annual rental income.

**Interpretation:**
- **Green (20 or below)**: Good investment potential
- **Yellow (around 40)**: Moderate investment potential 
- **Red (60 or above)**: Poor investment potential

### Hover over a cell to see:
- Average Sale Price
- Annual Rent (Monthly rent × 12)
- Number of listings in each category
""")

# Add footer with data source
st.markdown("""
---
**Data Source**: [Luxembourg Open Data Portal](https://data.public.lu/) | **DB URL**: [GitHub Repository](https://github.com/masterries/LuxOpenDataToAnalysis)
""")

# Close connection
conn.close()

# Cleanup temp file - note this will happen at the end of the session
def cleanup_temp_file():
    try:
        if 'temp_db_file' in locals() and os.path.exists(temp_db_file.name):
            os.unlink(temp_db_file.name)
    except:
        pass

# Register cleanup handler
import atexit
atexit.register(cleanup_temp_file)