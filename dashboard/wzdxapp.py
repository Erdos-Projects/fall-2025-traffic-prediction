"""
Streamlit dashboard showing AZ511 work zones on a map
Run: streamlit run dashboard/wzdxapp.py
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

# Set page config to wide mode
st.set_page_config(layout="wide")

CITY_COORDS = {
    'Phoenix': {'lat': 33.4484, 'lon': -112.0740, 'zoom': 11},
    'Tempe': {'lat': 33.4255, 'lon': -111.9400, 'zoom': 12},
    'Gilbert': {'lat': 33.3528, 'lon': -111.7890, 'zoom': 12},
    'Scottsdale': {'lat': 33.4942, 'lon': -111.9261, 'zoom': 11},
    'Mesa': {'lat': 33.4152, 'lon': -111.8315, 'zoom': 11},
    'Chandler': {'lat': 33.3062, 'lon': -111.8413, 'zoom': 12}
}

class WorkZoneDB:
    def __init__(self, db_path="workzones.db"):
        self.db_path = Path(__file__).parent.parent / "database" / db_path
        
    def get_active_workzones(self, start_date, end_date, city=None):
        """Get work zones active during the selected date range and city area"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            query = """
                SELECT id, event_type, description, update_date, 
                       latitude, longitude, road_names, data_source_id,
                       start_date, end_date
                FROM work_zones 
                WHERE update_date IS NOT NULL
                AND datetime(substr(start_date, 1, 19)) <= datetime(substr(?, 1, 19))
                AND datetime(substr(end_date, 1, 19)) >= datetime(substr(?, 1, 19))
            """
            params = [end_date, start_date]
            
            if city and city in CITY_COORDS:
                # Add 0.1 degree radius around city center (roughly 11km)
                center = CITY_COORDS[city]
                query += """
                    AND latitude BETWEEN ? AND ?
                    AND longitude BETWEEN ? AND ?
                """
                params.extend([
                    center['lat'] - 0.1, center['lat'] + 0.1,
                    center['lon'] - 0.1, center['lon'] + 0.1
                ])
            
            query += " ORDER BY datetime(substr(update_date, 1, 19)) DESC"
            return conn.execute(query, params).fetchall()
            
    def get_accidents(self, start_date, end_date, city=None):
        """Get accidents active during the selected date range and city area"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            query = """
                SELECT id, description, update_date, 
                       latitude, longitude, start_date, end_date
                FROM accidents 
                WHERE update_date IS NOT NULL
                AND datetime(substr(start_date, 1, 19)) <= datetime(substr(?, 1, 19))
                AND datetime(substr(end_date, 1, 19)) >= datetime(substr(?, 1, 19))
            """
            params = [end_date, start_date]
            
            if city and city in CITY_COORDS:
                # Add 0.1 degree radius around city center (roughly 11km)
                center = CITY_COORDS[city]
                query += """
                    AND latitude BETWEEN ? AND ?
                    AND longitude BETWEEN ? AND ?
                """
                params.extend([
                    center['lat'] - 0.1, center['lat'] + 0.1,
                    center['lon'] - 0.1, center['lon'] + 0.1
                ])
            
            query += " ORDER BY datetime(substr(update_date, 1, 19)) DESC"
            return conn.execute(query, params).fetchall()
        
    def get_daily_counts(self, start_date, end_date):
        """Get daily count of active work zones"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            # Generate a series of dates and count work zones active on each date
            query = """
                WITH RECURSIVE dates(date) AS (
                    SELECT date(?)
                    UNION ALL
                    SELECT date(date, '+1 day')
                    FROM dates
                    WHERE date < date(?)
                )
                SELECT 
                    dates.date,
                    COUNT(DISTINCT work_zones.id) as event_count
                FROM dates
                LEFT JOIN work_zones ON 
                    dates.date >= date(substr(work_zones.start_date, 1, 10)) AND
                    dates.date <= date(substr(work_zones.end_date, 1, 10))
                GROUP BY dates.date
                ORDER BY dates.date
            """
            return conn.execute(query, (start_date, end_date)).fetchall()

def main():
    st.title("Arizona Work Zones Map")
    
    # Initialize database connection
    db = WorkZoneDB()
    
    # Add city selector and date range in sidebar
    st.sidebar.subheader("Filter Options")
    
    selected_city = st.sidebar.selectbox(
        "Select City",
        ["All Cities"] + list(CITY_COORDS.keys()),
        key="city_selector"
    )
    
    st.sidebar.subheader("Select Date Range")
    
    # Get default date range (1 month before and 1 month after today)
    today = datetime.now().date()
    default_start = today - timedelta(days=30)
    default_end = today + timedelta(days=30)
    
    start_date = st.sidebar.date_input(
        "Start date",
        value=default_start,
        key="start_date"
    )
    
    end_date = st.sidebar.date_input(
        "End date",
        value=default_end,
        key="end_date",
        min_value=start_date
    )
    
    # Convert dates to ISO format strings
    start_str = datetime.combine(start_date, datetime.min.time()).isoformat()
    end_str = datetime.combine(end_date, datetime.max.time()).isoformat()
    
    # Get active work zones with city filter
    city_filter = None if selected_city == "All Cities" else selected_city
    workzones = db.get_active_workzones(start_str, end_str, city_filter)
    accidents = db.get_accidents(start_str, end_str, city_filter)
    
    # Print accident count
    print(f"Found {len(accidents) if accidents else 0} accidents in the selected date range")
    
    if workzones:
        # Convert to DataFrame for plotting
        df = pd.DataFrame([dict(row) for row in workzones])
        df_accidents = pd.DataFrame([dict(row) for row in accidents]) if accidents else pd.DataFrame()
        
        # Get map center and zoom based on selected city
        if city_filter:
            center = CITY_COORDS[city_filter]
            map_center = {"lat": center['lat'], "lon": center['lon']}
            map_zoom = center['zoom']
        else:
            map_center = {"lat": 33.4484, "lon": -112.0740}  # Phoenix as default
            map_zoom = 9
        
        # Define custom color mapping
        color_map = {
            'RADS': "#0022FF",  # Blue
            'ERS': "#FF00D9",    # Magenta
            'Accidents': "#FF0000"  # Red for accidents
        }
        
        # Create the map
        fig = px.scatter_map(
            df,
            lat='latitude',
            lon='longitude',
            color='data_source_id',
            hover_name='description',
            custom_data=['event_type', 'road_names', 'update_date', 'start_date', 'end_date', 'data_source_id'],
            color_discrete_map=color_map,
            zoom=map_zoom,
            center=map_center,
            title=f"Active Work Zones and Accidents - {selected_city} ({start_date} to {end_date})"
        )

        # Update hover template for work zones
        fig.update_traces(
            hovertemplate="<b>%{hovertext}</b><br>" +
            "Type: %{customdata[0]}<br>" +
            "Roads: %{customdata[1]}<br>" +
            "Updated: %{customdata[2]}<br>" +
            "Active: %{customdata[3]} to %{customdata[4]}<br>" +
            "Source: %{customdata[5]}<extra></extra>"
        )
        
        # Add accidents as red crosses if there are any
        if not df_accidents.empty:
            accident_trace = go.Scattermap(
                lat=df_accidents['latitude'],
                lon=df_accidents['longitude'],
                mode='markers',
                marker=dict(
                    size=20,
                    color='red',
                    opacity=0.8
                ),
                name='Accidents',
                text=df_accidents['description'],
                customdata=df_accidents[['update_date', 'start_date', 'end_date']].values,
                hovertext=df_accidents['description'],
                hovertemplate=(
                    "<b>Accident</b><br>" +
                    "Description: %{hovertext}<br>" +
                    "Updated: %{customdata[0]}<br>" +
                    "Active: %{customdata[1]} to %{customdata[2]}<extra></extra>"
                ),
                showlegend=True
            )
        fig.add_trace(accident_trace)
        # Update layout
        fig.update_layout(
            map_style="open-street-map",
            margin={"r":30,"t":30,"l":30,"b":50},
            legend_title_text='Data Source'
        )
        
        # Display the map
        st.plotly_chart(fig, use_container_width=True)
        
        # Add event count timeline
        st.subheader("Daily Active Work Zones")
        
        # Get daily counts for the timeline
        daily_counts = db.get_daily_counts(
            start_date.isoformat(),
            end_date.isoformat()
        )
        
        if daily_counts:
            df_counts = pd.DataFrame(daily_counts, columns=['date', 'event_count'])
            df_counts['date'] = pd.to_datetime(df_counts['date'])
            
            # Create timeline
            fig_timeline = px.line(
                df_counts,
                x='date',
                y='event_count',
                title="Number of Active Work Zones Over Time",
                labels={
                    'date': 'Date',
                    'event_count': 'Active Work Zones'
                }
            )
            
            # Add today's line with annotation about past/future
            today = pd.Timestamp.now()
            max_count = df_counts['event_count'].max()
            
            # Add vertical line for today
            fig_timeline.add_shape(
                type="line",
                x0=today,
                x1=today,
                y0=0,
                y1=max_count,
                line=dict(
                    color="red",
                    width=2,
                    dash="dash",
                )
            )
            
            # Add "Today" annotation
            fig_timeline.add_annotation(
                x=today,
                y=max_count,
                text="Today",
                showarrow=False,
                yshift=10
            )
            
            # Add past/future annotations
            past_date = today - pd.Timedelta(days=15)
            future_date = today + pd.Timedelta(days=15)
            
            fig_timeline.add_annotation(
                x=past_date,
                y=max_count,
                text="Past Events",
                showarrow=False,
                yshift=10
            )
            
            fig_timeline.add_annotation(
                x=future_date,
                y=max_count,
                text="Future Events",
                showarrow=False,
                yshift=10
            )
            
            # Update layout
            fig_timeline.update_layout(
                xaxis_title="Date",
                yaxis_title="Number of Active Work Zones",
                hovermode='x unified'
            )
            
            # Add range slider
            fig_timeline.update_xaxes(rangeslider_visible=True)
            
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Add some statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Active Events", f"{df_counts['event_count'].mean():.1f}")
            with col2:
                st.metric("Maximum Active Events", df_counts['event_count'].max())
            with col3:
                st.metric("Current Active Events", 
                         df_counts[df_counts['date'].dt.date == datetime.now().date()]['event_count'].iloc[0] 
                         if not df_counts[df_counts['date'].dt.date == datetime.now().date()].empty 
                         else "N/A")
        
        # Display work zone count and details
        num_events = len(workzones)
        st.subheader(f"Work Zone Details ({num_events} active events)")
        
        # Add summary metrics
        col1, col2 = st.columns(2)
        with col1:
            rads_count = len([z for z in workzones if z['data_source_id'] == 'RADS'])
            st.metric("RADS Events", rads_count)
        with col2:
            ers_count = len([z for z in workzones if z['data_source_id'] == 'ERS'])
            st.metric("ERS Events", ers_count)

        # Add visualizations
        st.subheader("Work Zone Analytics")
                # 1. Pie chart of work zones by cities
        # Calculate city for each work zone based on closest city center
        def get_nearest_city(lat, lon):
            min_dist = float('inf')
            nearest = "Other"
            for city, coords in CITY_COORDS.items():
                dist = ((lat - coords['lat'])**2 + (lon - coords['lon'])**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest = city
            return nearest
            
        df['nearest_city'] = df.apply(lambda x: get_nearest_city(x['latitude'], x['longitude']), axis=1)
        city_counts = df['nearest_city'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig_pie = px.pie(
                values=city_counts.values,
                names=city_counts.index,
                title="Work Zones by City"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Duration distribution
            df['start_datetime'] = pd.to_datetime(df['start_date'])
            # Filter out events with end dates beyond pandas datetime limits
            valid_dates_mask = df['end_date'].apply(lambda x: x[:4]).astype(int) < 2263
            df_valid = df[valid_dates_mask].copy()
            
            if not df_valid.empty:
                df_valid['end_datetime'] = pd.to_datetime(df_valid['end_date'])
                df_valid['duration_days'] = (df_valid['end_datetime'] - df_valid['start_datetime']).dt.total_seconds() / 3600 / 24
                
                # Cap duration at 90 days and count longer events by source
                rads_long = len(df_valid[(df_valid['duration_days'] > 90) & (df_valid['data_source_id'] == 'RADS')])
                ers_long = len(df_valid[(df_valid['duration_days'] > 90) & (df_valid['data_source_id'] == 'ERS')])
                df_valid.loc[df_valid['duration_days'] > 90, 'duration_days'] = 90
                
                fig_duration = px.histogram(
                    df_valid,
                    x='duration_days',
                    color='data_source_id',  # Separate by data source
                    barmode='overlay',  # Overlay the histograms
                    opacity=0.7,  # Make bars semi-transparent
                    color_discrete_map=color_map,  # Use same colors as map
                    title=f"Work Zone Duration Distribution\n(RADS: {rads_long}, ERS: {ers_long} events >90 days)",
                    labels={'duration_days': 'Duration (days)', 'data_source_id': 'Data Source'},
                    nbins=30,
                    range_x=[0, 90]
                )
                
                # Add a vertical line at 90 days
                fig_duration.add_vline(
                    x=90, 
                    line_dash="dash", 
                    line_color="gray",
                    annotation_text=f"{rads_long + ers_long} events >90 days"
                )
                
                fig_duration.update_layout(
                    xaxis_title="Duration (days, capped at 90)",
                    yaxis_title="Number of Events",
                    legend_title_text="Data Source",
                    bargap=0.1  # Reduce gap between bars
                )
                st.plotly_chart(fig_duration, use_container_width=True)
            else:
                st.warning("No events with valid duration found")
        

        col1, col2 = st.columns(2)
        with col1:
            # Update date vs Start date scatter
            df['update_datetime'] = pd.to_datetime(df['update_date'])
            fig_dates = px.scatter(
                df,
                x='update_datetime',
                y='start_datetime',
                color='data_source_id',
                title="Update Date vs Start Date",
                labels={
                    'start_datetime': 'Start Date',
                    'update_datetime': 'Update Date',
                    'data_source_id': 'Data Source'
                },
                color_discrete_map=color_map
            )
            
            # Add diagonal line where start_date = update_date
            fig_dates.add_scatter(
                x=[df['start_datetime'].min(), df['start_datetime'].max()],
                y=[df['start_datetime'].min(), df['start_datetime'].max()],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='Start = Update',
                showlegend=True
            )
            
            fig_dates.update_layout(
                xaxis_title="Update Date",
                yaxis_title="Start Date",
                legend_title_text="Data Source"
            )
            st.plotly_chart(fig_dates, use_container_width=True)

        
        # Display individual work zone details
        if accidents:
            st.subheader(f"Accident Details ({len(accidents)} accidents)")
            for accident in accidents:
                with st.expander(f"Accident - {accident['description'][:50]}..."):
                    st.write(f"Description: {accident['description']}")
                    st.write(f"Active: {accident['start_date']} to {accident['end_date']}")
                    st.write(f"Location: ({accident['latitude']}, {accident['longitude']})")
                    st.write(f"Last Updated: {accident['update_date']}")
    else:
        st.warning("No active work zones found for the selected date range.")
        st.code("""
        # Run the data fetcher:
        python inrix/dashboard/wzdx.py
        """)

if __name__ == "__main__":
    main()