import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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

# Download database function - returns the path, not the connection
@st.cache_resource
def download_database(url):
    """Download the database from the URL and return the path to it"""
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
            
            return temp_db_path
    except Exception as e:
        st.error(f"Error downloading database: {e}")
        return None

# Database URL
db_url = "https://github.com/masterries/LuxOpenDataToAnalysis/raw/refs/heads/main/DataGathering/luxembourg_housing.db"

# Download database and get the path
db_path = download_database(db_url)

if db_path is None:
    st.error("Failed to download the database.")
    st.stop()

# Create a connection - now we'll keep this open until needed
conn = sqlite3.connect(db_path)

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

# Sidebar for configuration options
st.sidebar.header("Configuration")

# Property type selector
property_types = all_data['property_type'].unique().tolist()
selected_property_type = st.sidebar.selectbox(
    "Select Property Type",
    options=property_types,
    index=property_types.index("apartment") if "apartment" in property_types else 0
)

# Price metric selector - let the user choose between total price and price per square meter
price_metric = st.sidebar.radio(
    "Price Metric",
    ["Average Price", "Price per Square Meter"],
    index=1  # Default to price per square meter
)

# Filter data by selected property type
filtered_data = all_data[all_data['property_type'] == selected_property_type]

# Function to calculate price-to-rent ratio
def calculate_price_to_rent_ratios(df, use_price_sqm=True):
    """Calculate price-to-rent ratio for each municipality and year
    
    Args:
        df: DataFrame with housing data
        use_price_sqm: If True, use avg_price_sqm, otherwise use avg_price
    """
    # Select the price column based on user choice
    price_column = 'avg_price_sqm' if use_price_sqm else 'avg_price'
    
    # Group by municipality, year, and transaction type, taking the mean of prices
    df_grouped = df.groupby(['municipality', 'year', 'transaction_type']).agg({
        price_column: 'mean',
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
        sale_prices = {row['municipality']: row[price_column] for _, row in sales_data.iterrows()}
        sale_listings = {row['municipality']: row['num_listings'] for _, row in sales_data.iterrows()}
        rent_prices = {row['municipality']: row[price_column] for _, row in rental_data.iterrows()}
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

# Determine whether to use price per square meter based on user selection
use_price_sqm = (price_metric == "Price per Square Meter")

# Calculate price-to-rent ratios based on selected metric
ptr_data = calculate_price_to_rent_ratios(filtered_data, use_price_sqm)

# Convert to DataFrame for heatmap
years = sorted(filtered_data['year'].unique())
municipalities = sorted(ptr_data.keys())

# Create DataFrames for the heatmap values and hover information
ratio_matrix = []
hover_texts = []

# Prepare price unit text based on the selected metric
price_unit = "€/m²" if use_price_sqm else "€"

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
                f"Average Sale Price: {price_unit}{data['sale_price']:,.0f}<br>"
                f"Annual Rent: {price_unit}{data['annual_rent']:,.0f}<br>"
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

# Update layout with appropriate title based on the selected metric
metric_description = "per Square Meter" if use_price_sqm else "Total"
fig.update_layout(
    title=f'Price-to-Rent Ratio Heatmap for {selected_property_type.capitalize()}s ({metric_description})',
    xaxis_title='Year',
    yaxis_title='Municipality',
    height=max(600, len(municipalities) * 20),  # Adjust height based on number of municipalities
    width=1000,
    font=dict(size=12),
    annotations=text_annotations
)

# Display the heatmap chart
st.plotly_chart(fig, use_container_width=True)

# Add simple price trend graphs below the heatmap
st.header("General Price Trends Over Time")

# Prepare data for the simple graphs
def prepare_price_trends_data(df, price_column):
    """Prepare data for simple price trend graphs"""
    # Group by year and transaction type
    yearly_data = df.groupby(['year', 'transaction_type']).agg({
        price_column: 'mean',
        'num_listings': 'sum'
    }).reset_index()
    
    # Create separate dataframes for sale and rent
    sales_data = yearly_data[yearly_data['transaction_type'] == 'sale']
    rental_data = yearly_data[yearly_data['transaction_type'] == 'rent']
    
    return sales_data, rental_data

# Determine which price column to use based on user selection
price_column = 'avg_price_sqm' if use_price_sqm else 'avg_price'

# Get the sales and rental price trends
sales_data, rental_data = prepare_price_trends_data(filtered_data, price_column)

# Create two columns for the graphs
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Average Sale Price Trend ({price_unit})")
    
    # Simple line chart for sales
    fig_sales = px.line(
        sales_data, 
        x='year', 
        y=price_column,
        markers=True,
        title=f'Average Sale Price for {selected_property_type.capitalize()}s'
    )
    
    fig_sales.update_layout(
        xaxis_title='Year',
        yaxis_title=f'Average Sale Price ({price_unit})',
        yaxis=dict(rangemode='tozero')
    )
    
    st.plotly_chart(fig_sales, use_container_width=True)
    
    # Display some key statistics
    if not sales_data.empty:
        latest_year = sales_data['year'].max()
        latest_price = sales_data[sales_data['year'] == latest_year][price_column].iloc[0]
        earliest_year = sales_data['year'].min()
        earliest_price = sales_data[sales_data['year'] == earliest_year][price_column].iloc[0]
        
        if earliest_price > 0:
            percent_change = ((latest_price - earliest_price) / earliest_price) * 100
            st.metric(
                f"Price Change ({earliest_year} to {latest_year})", 
                f"{percent_change:.1f}%",
                f"{latest_price - earliest_price:.1f} {price_unit}"
            )

with col2:
    st.subheader(f"Average Rent Price Trend ({price_unit})")
    
    # Simple line chart for rentals
    fig_rentals = px.line(
        rental_data, 
        x='year', 
        y=price_column,
        markers=True,
        title=f'Average Rent Price for {selected_property_type.capitalize()}s'
    )
    
    fig_rentals.update_layout(
        xaxis_title='Year',
        yaxis_title=f'Average Rent Price ({price_unit})',
        yaxis=dict(rangemode='tozero')
    )
    
    st.plotly_chart(fig_rentals, use_container_width=True)
    
    # Display some key statistics
    if not rental_data.empty:
        latest_year = rental_data['year'].max()
        latest_price = rental_data[rental_data['year'] == latest_year][price_column].iloc[0]
        earliest_year = rental_data['year'].min()
        earliest_price = rental_data[rental_data['year'] == earliest_year][price_column].iloc[0]
        
        if earliest_price > 0:
            percent_change = ((latest_price - earliest_price) / earliest_price) * 100
            st.metric(
                f"Price Change ({earliest_year} to {latest_year})", 
                f"{percent_change:.1f}%",
                f"{latest_price - earliest_price:.1f} {price_unit}"
            )

# Compare sale and rent prices in a single chart
st.subheader(f"Sale Price vs. Rent Price Comparison ({price_unit})")

# Create a DataFrame with both sale and rent data
comparison_df = pd.DataFrame({
    'Year': sales_data['year'],
    'Sale Price': sales_data[price_column],
    'Monthly Rent': rental_data[price_column] if not rental_data.empty else 0
})

# Create the comparison chart
fig_comparison = go.Figure()

# Add sale price line
fig_comparison.add_trace(
    go.Scatter(
        x=comparison_df['Year'],
        y=comparison_df['Sale Price'],
        mode='lines+markers',
        name=f'Sale Price ({price_unit})'
    )
)

# Add rent price line (monthly)
fig_comparison.add_trace(
    go.Scatter(
        x=comparison_df['Year'],
        y=comparison_df['Monthly Rent'],
        mode='lines+markers',
        name=f'Monthly Rent ({price_unit})'
    )
)

# Update layout
fig_comparison.update_layout(
    title=f'Sale Price vs. Monthly Rent for {selected_property_type.capitalize()}s',
    xaxis_title='Year',
    yaxis_title=f'Price ({price_unit})',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_comparison, use_container_width=True)

# Add explanation
st.markdown(f"""
### Understanding the Price-to-Rent Ratio

The Price-to-Rent ratio is calculated by dividing the average property sale price {price_unit} by the annual rental income {price_unit}.

**Current Metric:** {price_metric}

**Interpretation:**
- **Green (20 or below)**: Good investment potential - buying is more favorable than renting
- **Yellow (around 40)**: Moderate investment potential 
- **Red (60 or above)**: Poor investment potential - renting is more favorable than buying

### Hover over a cell to see:
- Average Sale Price ({price_unit})
- Annual Rent ({price_unit}, Monthly rent × 12)
- Number of listings in each category

### Additional Visualizations

Use the tabs above to explore:
1. **Time Series Trends** - See how the price-to-rent ratio has changed over time
2. **Municipality Comparison** - Compare different municipalities side by side
3. **Distribution Analysis** - Analyze the statistical distribution of ratios
""")

# Add footer with data source
st.markdown("""
---
**Data Source**: [Luxembourg Open Data Portal](https://data.public.lu/) | **DB URL**: [GitHub Repository](https://github.com/masterries/LuxOpenDataToAnalysis)
""")

# Close connection at the very end of the script
conn.close()