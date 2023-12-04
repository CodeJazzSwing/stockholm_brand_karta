#Dashboard with blue line around all datapoint.
# Bad example since it does not hug around all the points.


import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
import plotly.graph_objects as go
import matplotlib.ticker as ticker

df = pd.read_csv(r'C:\\Users\\SE1D2L\\Documents\\Python Scripts\\Toy2.csv', sep=';')


image = 'StockholmsStad_logotypeStandardA3_300ppi_svart.png'

# Convert 'Tidpunkt_' to datetime
df['Tidpunkt_'] = pd.to_datetime(df['Tidpunkt_'])

#---------SIDEBAR-------------

st.sidebar.header('Filter Here:')

# Date range filter
start_date = st.sidebar.date_input('Startdatum', df['Tidpunkt_'].min().date())
end_date = st.sidebar.date_input('Slutdatum', df['Tidpunkt_'].max().date())

#Error message for date
if start_date > end_date:
    st.sidebar.error('Misstag: Slutdatum måste vara efter startdatum.')


# Filter the dataframe based on the sidebar selection
df_selection = df[(df['Tidpunkt_'].dt.date >= start_date) & (df['Tidpunkt_'].dt.date <= end_date)]


# Create a new layout with 2 columns
col1, col2 = st.columns(2)

# Image on the top left side of map. 
col1.image(image, use_column_width=True)


#Create color dictionary för column "Typ of brand" categories
color_dict = {"Brand i byggnad" : "red", "Brand i container" : "blue", "Fordonsbrand" : "green", "Mark-/skogsbrand" : "orange", "Övrigt" : "purple"}


categories = df_selection['Typ av brand'].unique()
colors = [color_dict[cat] for cat in categories]


#Adds a new column "color" to the  DF based on "Typ of brand" column

# DO I NEED THIS LINE OF CODE?! Seems like not.
df_selection["color"] = df_selection["Typ av brand"].map(color_dict)


# Use 'Typ av brand' for color parameter in the plot and set color_discrete_sequence to your list of colors
fig = px.scatter_mapbox(df_selection,
                        lon = df_selection['lng'],
                        lat = df_selection['lat'],
                        zoom = 10,
                        width = 1000,
                        height= 1000,
                        title = 'Bränder i Stockholm stad',
                        text = df_selection['Typ av brand'],
                        hover_data = {'Typ av brand': True, 'Tidpunkt_': True, 'lng': False, 'lat': False},
                        color = df_selection["Typ av brand"],
                        color_discrete_sequence = colors
                        )


fig.update_traces(marker=dict(size = 12))

fig.update_layout(mapbox_style='open-street-map')
fig.update_layout(margin={'r':0,'t':50,'l':0,'b':10})


# Display the plot in Streamlit
st.plotly_chart(fig)


#LINE GRAPH-----------------------
import numpy as np
import pandas as pd

# Assuming 'Tidpunkt_' is the time and 'Typ av brand' is the type of fire
df_selection_grouped = df_selection.groupby(['Tidpunkt_', 'Typ av brand']).size().reset_index(name='counts')

# Get unique fire types and time points
fire_types = df_selection_grouped['Typ av brand'].unique()
time_points = df_selection_grouped['Tidpunkt_'].unique()

# Create a dataframe with all combinations of time points and fire types
df_all_combinations = pd.DataFrame(index=pd.MultiIndex.from_product([time_points, fire_types], names=['Tidpunkt_', 'Typ av brand'])).reset_index()

# Merge the original dataframe with the dataframe that has all combinations
df_merged = pd.merge(df_all_combinations, df_selection_grouped, on=['Tidpunkt_', 'Typ av brand'], how='left')

# Fill NA values with 0
df_merged['counts'] = df_merged['counts'].fillna(0)

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Width of a bar 
width = 0.1

# Generate bars for each fire type
for i, fire_type in enumerate(fire_types):
    # Filter the data for the current fire type
    df_fire_type = df_merged[df_merged['Typ av brand'] == fire_type]
    
    # Create an array for the position of each bar on the x-axis
    r = np.arange(len(time_points))
    
    # Plot the data
    ax.bar(r + i*width, df_fire_type['counts'], width=width, label=fire_type)

# Set the title and labels
ax.set_title('Bränder i Stockholm stad')
ax.set_xlabel('Tidpunkt')
ax.set_ylabel('Antal bränder')

# Add xticks on the middle of the group bars
ax.set_xticks(r + width/2, time_points)

# Rotate x-axis labels
plt.xticks(rotation=45)

# Add a legend
ax.legend()

# Use MaxNLocator for y-axis
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Display the plot
st.pyplot(fig)

#-----------Create a line around the map--------------

# Create the scatter mapbox for the points
fig = go.Figure(go.Scattermapbox(
    lon = df_selection['lng'],
    lat = df_selection['lat'],
    mode = 'markers',
    marker = go.scattermapbox.Marker(
        size = 12,
        color = df_selection["Typ av brand"].map(color_dict),
        colorscale = colors
    ),
    text = df_selection['Typ av brand'],
))

# Calculate the Convex Hull of the points
points = df_selection[['lat', 'lng']].values
hull = ConvexHull(points)

# Get the coordinates of the Convex Hull vertices
hull_points = points[hull.vertices]

# Create a new trace for the Convex Hull boundary
hull_trace = go.Scattermapbox(
    lon = np.append(hull_points[:, 1], hull_points[0, 1]),  # Append the first point to the end to close the polygon
    lat = np.append(hull_points[:, 0], hull_points[0, 0]),
    mode = 'lines',
    line = dict(width = 2, color = 'blue'),
    name = 'Convex Hull'
)

# Add the new trace to the figure
fig.add_trace(hull_trace)

# Set the layout
fig.update_layout(
    autosize=True,
    hovermode='closest',
    mapbox=dict(
        accesstoken='pk.eyJ1Ijoibm9lbHN3ZWNvIiwiYSI6ImNscDZ4ZzI0dDF5Z2syaHF1dnR6eGxmMWwifQ.fCT2l0c2VFiJmOFoAU-xMA',
        bearing=0,
        center=dict(
            lat=df_selection['lat'].mean(),
            lon=df_selection['lng'].mean()
        ),
        pitch=0,
        zoom=10,
        style='basic'  # or any other Mapbox style
    ),
)

# Display the plot in Streamlit
st.plotly_chart(fig)
