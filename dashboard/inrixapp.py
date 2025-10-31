"""
INRIX Traffic Speed Data Dashboard
Streamlit dashboard showing traffic speed data for road segments
Run: streamlit run dashboard/inrixapp.py --server.address=0.0.0.0 --server.port=8501
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from pathlib import Path
import calendar
import json
import time
from datetime import datetime, timezone, timedelta, date
import math

# Set page config to wide mode
st.set_page_config(layout="wide")

class InrixData:
    def __init__(self, inrix_file_name="SR60-1year"):
        """Initialize the INRIX data class"""
        self.inrix_file_name = inrix_file_name
        self.data_dir = Path(__file__).parent.parent / "database/inrix-traffic-speed" / inrix_file_name
        self.large_file_path = self.data_dir / f"{inrix_file_name}.csv"
        self.chunks_dir = self.data_dir / "chunks"
        self.target_chunk_size_mb = 100  # ~100MB for each chunk
        self.df_1y = None  # To store aggregated yearly data
        self.tmc_locations = None
        # Add cache for map data to avoid recomputation
        self._daily_map_cache = {}
        
        # Load TMC locations on initialization
        self.load_tmc_locations()
    
    def load_tmc_locations(self):
        """Load TMC location data"""
        tmc_file = self.data_dir / "TMC_Identification.csv"
        if tmc_file.exists():
            self.tmc_locations = pd.read_csv(tmc_file)
            return True
        else:
            return False
    
    def file_exists(self):
        """Check if the source file exists"""
        return self.large_file_path.exists()
    
    def get_file_stats(self):
        """Get basic stats about the file"""
        if not self.file_exists():
            return None
        
        file_size_mb = os.path.getsize(self.large_file_path) / (1024 * 1024)
        num_chunks = math.ceil(file_size_mb / self.target_chunk_size_mb)
        
        return {
            "file_size_mb": file_size_mb,
            "estimated_chunks": num_chunks
        }
    
    def get_sample_data(self, rows=5):
        """Get a sample of the data"""
        if not self.file_exists():
            return None
        
        try:
            df_sample = pd.read_csv(self.large_file_path, nrows=rows)
            return df_sample
        except Exception as e:
            print(f"Error reading file: {e}")
            return None

    def split_csv_file(self, chunksize=10000):
        """Split the large CSV file into smaller chunks"""
        if not self.file_exists():
            return False
        
        # Create directory if it doesn't exist
        os.makedirs(self.chunks_dir, exist_ok=True)

        chunk_files = []
        
        # Convert paths to strings for pandas
        file_path_str = str(self.large_file_path)
        output_dir_str = str(self.chunks_dir)
        
        # Get the base filename without extension
        base_filename = os.path.splitext(os.path.basename(file_path_str))[0]
        
        # Check if chunks already exist
        existing_chunks = glob.glob(str(self.chunks_dir / f"{base_filename}_chunk_*.csv"))
        if existing_chunks:
            return sorted(existing_chunks)
        
        # Read the CSV in chunks
        reader = pd.read_csv(file_path_str, chunksize=chunksize)
        
        # Initialize variables for tracking
        chunk_idx = 1
        current_chunk_size = 0
        current_chunk_rows = []
        
        # Get total rows (approximate)
        if os.path.exists(file_path_str):
            # Count lines quickly
            with open(file_path_str, 'r') as f:
                total_rows = sum(1 for _ in f) - 1  # Subtract header
            progress_text = st.empty()
            progress_text.text(f"Processing file with {total_rows} rows")
            progress_bar = st.progress(0)
        else:
            total_rows = 0
        
        # Process each chunk
        rows_processed = 0
        for i, chunk in enumerate(reader):
            # Add the current chunk rows
            current_chunk_rows.append(chunk)
            
            # Track progress
            rows_processed += len(chunk)
            if (i + 1) % 10 == 0 or i == 0:  # Print status every 10 chunks
                percent_done = (rows_processed / total_rows) * 100 if total_rows > 0 else 0
                progress_text.text(f"Processed {rows_processed} rows ({percent_done:.1f}%)")
                progress_bar.progress(min(percent_done / 100, 1.0))
            
            # Calculate the size of the current chunk in memory
            chunk_size_mb = sum(chunk.memory_usage(deep=True)) / (1024 * 1024)
            current_chunk_size += chunk_size_mb
            
            # Apply a correction factor to better match target file size
            # Memory size vs file size ratio is typically around 2.5-3x
            adjusted_size = current_chunk_size * 0.4  # Adjust this factor based on testing
            
            # If we've reached the target size or it's the last chunk
            if adjusted_size >= self.target_chunk_size_mb:
                # Combine all accumulated chunks
                combined_chunk = pd.concat(current_chunk_rows, ignore_index=True)
                
                # Save the combined chunk
                chunk_filename = f"{base_filename}_chunk_{chunk_idx:03d}.csv"
                chunk_path = os.path.join(output_dir_str, chunk_filename)
                combined_chunk.to_csv(chunk_path, index=False)
                
                # Add to list of chunk files
                chunk_files.append(chunk_path)
                
                # Log information
                actual_file_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
                progress_text.text(f"Saved chunk {chunk_idx}: {len(combined_chunk)} rows, {actual_file_size_mb:.2f} MB")
                
                # Reset for the next chunk
                chunk_idx += 1
                current_chunk_size = 0
                current_chunk_rows = []
        
        # Save any remaining rows
        if current_chunk_rows:
            combined_chunk = pd.concat(current_chunk_rows, ignore_index=True)
            chunk_filename = f"{base_filename}_chunk_{chunk_idx:03d}.csv"
            chunk_path = os.path.join(output_dir_str, chunk_filename)
            combined_chunk.to_csv(chunk_path, index=False)
            chunk_files.append(chunk_path)
            
            actual_file_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
            progress_text.text(f"Saved final chunk {chunk_idx}: {len(combined_chunk)} rows, {actual_file_size_mb:.2f} MB")
    
        
        return chunk_files
    
    def get_chunk_files(self):
        """Get list of chunk files"""
        chunk_files = sorted(glob.glob(str(self.chunks_dir / f"{self.inrix_file_name}_chunk_*.csv")))
        return chunk_files
    
    def process_chunk_file(self, file_path):
        """Process a chunk file to aggregate data by hour"""
        # Read the chunk file
        chunk_df = pd.read_csv(file_path)
        
        # Convert timestamp to datetime
        chunk_df['measurement_tstamp'] = pd.to_datetime(chunk_df['measurement_tstamp'])
        
        # Extract necessary columns (tmc_code, time, speed)
        chunk_df = chunk_df[['tmc_code', 'measurement_tstamp', 'speed']]
        
        # Round timestamp to the nearest hour for aggregation
        chunk_df['hour'] = chunk_df['measurement_tstamp'].dt.floor('h')
        
        # Aggregate data by tmc_code and hour
        agg_df = chunk_df.groupby(['tmc_code', 'hour'], observed=True).agg(
            speed_mean=('speed', 'mean'),
            speed_median=('speed', 'median'),
            speed_min=('speed', 'min'),
            speed_max=('speed', 'max'),
            count=('speed', 'count')
        ).reset_index()
        
        return agg_df
    
    def aggregate_data(self, force_reprocess=False):
        """Process all chunks and aggregate data into hourly intervals over a year"""
        # Check if aggregated data file already exists
        aggregated_file_path = self.data_dir / f"{self.inrix_file_name}_aggregated.csv"
        
        # Try to load saved aggregated data if not forcing reprocessing
        if not force_reprocess and aggregated_file_path.exists():
            st.info(f"Loading pre-aggregated data from {aggregated_file_path.name}...")
            try:
                df_1y = pd.read_csv(aggregated_file_path)
                # Convert timestamp columns back to datetime
                df_1y['hour'] = pd.to_datetime(df_1y['hour'])
                if 'date' in df_1y.columns:
                    df_1y['date'] = pd.to_datetime(df_1y['date'])
                st.success(f"Successfully loaded aggregated data: {len(df_1y)} rows")
                self.df_1y = df_1y
                return df_1y
            except Exception as e:
                st.error(f"Error loading aggregated data: {e}")
                st.info("Will reprocess from chunk files...")
        
        # If no saved file or forced reprocessing, aggregate from chunks
        chunk_files = self.get_chunk_files()
        
        if not chunk_files:
            st.error("No chunk files found. Please split the file first.")
            return None
        
        # Process all chunk files and combine into a single dataframe
        st.text("Processing chunk files and aggregating data...")
        progress_bar = st.progress(0)
        df_chunks = []

        # Use simple progress tracking
        total_files = len(chunk_files)
        for i, file_path in enumerate(chunk_files):
            progress_text = f"Processing file {i+1}/{total_files} ({(i+1)/total_files*100:.1f}%)"
            st.text(progress_text)
            progress_bar.progress((i+1)/total_files)
            df_chunks.append(self.process_chunk_file(file_path))

        # Combine all processed chunks
        df_1y = pd.concat(df_chunks, ignore_index=True)

        # If there are duplicate entries for tmc_code and hour, aggregate again
        df_1y = df_1y.groupby(['tmc_code', 'hour'], observed=True).agg(
            speed_mean=('speed_mean', 'mean'),
            speed_median=('speed_median', 'mean'),
            speed_min=('speed_min', 'min'),
            speed_max=('speed_max', 'max'),
            count=('count', 'sum')
        ).reset_index()

        # Extract additional time features for analysis
        df_1y['date'] = df_1y['hour'].dt.date
        df_1y['month'] = df_1y['hour'].dt.month
        df_1y['month_name'] = df_1y['hour'].dt.month_name()
        df_1y['day'] = df_1y['hour'].dt.day
        df_1y['day_of_week'] = df_1y['hour'].dt.day_name()
        df_1y['hour_of_day'] = df_1y['hour'].dt.hour
        
        # Extract week information
        df_1y['week'] = df_1y['hour'].dt.isocalendar().week
        df_1y['year'] = df_1y['hour'].dt.isocalendar().year
        df_1y['year_week'] = df_1y['year'].astype(str) + '-' + df_1y['week'].astype(str).str.zfill(2)
        
        # Add rush hour flags
        df_1y['is_morning_rush'] = df_1y['hour_of_day'].between(7, 9)
        df_1y['is_evening_rush'] = df_1y['hour_of_day'].between(16, 18)
        df_1y['is_weekend'] = df_1y['day_of_week'].isin(['Saturday', 'Sunday'])
        
        # Add season column
        season_map = {
            1: 'Winter',  # January
            2: 'Winter',  # February
            3: 'Spring',  # March
            4: 'Spring',  # April
            5: 'Spring',  # May
            6: 'Summer',  # June
            7: 'Summer',  # July
            8: 'Summer',  # August
            9: 'Fall',    # September
            10: 'Fall',   # October
            11: 'Fall',   # November
            12: 'Winter'  # December
        }
        df_1y['season'] = df_1y['month'].map(season_map)

        st.success(f"Created aggregated dataframe with {len(df_1y)} rows")
        
        # Merge with TMC locations
        if self.tmc_locations is not None:
            df_1y = pd.merge(
                df_1y,
                self.tmc_locations[['tmc', 'road', 'direction', 'intersection', 
                                   'start_latitude', 'start_longitude', 'end_latitude', 'end_longitude']],
                left_on='tmc_code',
                right_on='tmc',
                how='left'
            )
        
        # Save the aggregated data to file
        aggregated_file_path = self.data_dir / f"{self.inrix_file_name}_aggregated.csv"
        try:
            df_1y.to_csv(aggregated_file_path, index=False)
            st.success(f"Saved aggregated data to {aggregated_file_path.name}")
        except Exception as e:
            st.warning(f"Could not save aggregated data: {e}")
        
        self.df_1y = df_1y
        return df_1y
    
    def get_weekly_pivots(self):
        """Create weekly pivot tables of speed data"""
        if self.df_1y is None:
            return None
            
        # Get unique year-weeks
        unique_year_weeks = sorted(self.df_1y['year_week'].unique())
        
        # Find global min and max for consistent colorbar
        global_min = self.df_1y['speed_mean'].min()
        global_max = self.df_1y['speed_mean'].max()
        
        # Define day order
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Create pivot tables for each week
        weekly_pivots = {}
        for year_week in unique_year_weeks:
            week_data = self.df_1y[self.df_1y['year_week'] == year_week]
            if len(week_data) > 0:
                pivot_data = week_data.pivot_table(
                    index='day_of_week', columns='hour_of_day', values='speed_mean', aggfunc='mean'
                )
                # Reindex to ensure all days are in correct order
                pivot_data = pivot_data.reindex(day_order)
                weekly_pivots[year_week] = pivot_data
        
        return {
            'weekly_pivots': weekly_pivots,
            'unique_year_weeks': unique_year_weeks,
            'global_min': global_min,
            'global_max': global_max,
            'day_order': day_order
        }
    
    def get_daily_speed_trend(self):
        """Get daily average speed trend over the year"""
        if self.df_1y is None:
            return None
            
        # Aggregate data by date
        daily_avg = self.df_1y.groupby('date', observed=True)['speed_mean'].mean().reset_index()
        daily_avg['date'] = pd.to_datetime(daily_avg['date'])
        
        return daily_avg
    
    def get_monthly_speed_stats(self):
        """Get monthly speed statistics"""
        if self.df_1y is None:
            return None
            
        # Set the order of months
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December']
        
        # Calculate monthly statistics
        monthly_stats = self.df_1y.groupby('month_name', observed=True)['speed_mean'].agg(['mean', 'median', 'std']).reindex(month_order)
        
        return monthly_stats
    
    def get_seasonal_stats(self):
        """Get seasonal speed statistics"""
        if self.df_1y is None:
            return None
            
        # Calculate seasonal statistics
        seasonal_stats = self.df_1y.groupby('season', observed=True)['speed_mean'].agg(['mean', 'median', 'std']).reindex(['Winter', 'Spring', 'Summer', 'Fall'])
        
        # Get rush hour patterns by season
        morning_rush = self.df_1y[self.df_1y['is_morning_rush']].groupby('season', observed=True)['speed_mean'].mean().reindex(['Winter', 'Spring', 'Summer', 'Fall'])
        evening_rush = self.df_1y[self.df_1y['is_evening_rush']].groupby('season', observed=True)['speed_mean'].mean().reindex(['Winter', 'Spring', 'Summer', 'Fall'])
        non_rush = self.df_1y[~(self.df_1y['is_morning_rush'] | self.df_1y['is_evening_rush'])].groupby('season', observed=True)['speed_mean'].mean().reindex(['Winter', 'Spring', 'Summer', 'Fall'])
        
        # Get weekday vs weekend patterns by season
        weekday = self.df_1y[~self.df_1y['is_weekend']].groupby('season', observed=True)['speed_mean'].mean().reindex(['Winter', 'Spring', 'Summer', 'Fall'])
        weekend = self.df_1y[self.df_1y['is_weekend']].groupby('season', observed=True)['speed_mean'].mean().reindex(['Winter', 'Spring', 'Summer', 'Fall'])
        
        return {
            'seasonal_stats': seasonal_stats,
            'morning_rush': morning_rush,
            'evening_rush': evening_rush,
            'non_rush': non_rush,
            'weekday': weekday,
            'weekend': weekend
        }
    
    def analyze_spatial_patterns(self):
        """Analyze spatial patterns in the data"""
        if self.df_1y is None or self.tmc_locations is None:
            return None
        
        # Calculate average speeds by road
        road_speeds = self.df_1y.groupby('road', observed=True)['speed_mean'].agg(['mean', 'std', 'count']).reset_index()
        road_speeds = road_speeds.sort_values(by='mean', ascending=False)
        
        # Calculate average speeds by direction
        direction_speeds = self.df_1y.groupby('direction', observed=True)['speed_mean'].agg(['mean', 'std', 'count']).reset_index()
        direction_speeds = direction_speeds.sort_values(by='mean', ascending=False)
        
        # Calculate peak vs. off-peak difference by TMC
        self.df_1y['is_peak'] = self.df_1y['is_morning_rush'] | self.df_1y['is_evening_rush']
        
        # Group by TMC, calculate peak vs. off-peak difference
        # Use include_groups=False to address FutureWarning about apply operating on grouping columns
        tmc_peak_diff = self.df_1y.groupby(['tmc', 'road', 'direction'], group_keys=False)\
            .apply(lambda x: pd.Series({
                'peak_speed_drop': x[x['is_peak']]['speed_mean'].mean() - x[~x['is_peak']]['speed_mean'].mean()
            }), include_groups=False)\
            .reset_index()
        
        # Sort by largest speed drop during peak hours
        tmc_peak_diff = tmc_peak_diff.sort_values('peak_speed_drop')
        
        return {
            'road_speeds': road_speeds,
            'direction_speeds': direction_speeds,
            'tmc_peak_diff': tmc_peak_diff
        }
    
    def _get_tmc_map_data(self, year_week=None, selected_date=None):
        """Get TMC map data for visualization"""
        if self.df_1y is None or self.tmc_locations is None:
            return None
        
        # If year_week specified, filter data (for backward compatibility)
        if year_week:
            df_map = self.df_1y[self.df_1y['year_week'] == year_week].copy()
        # If selected_date specified, filter data by specific date
        elif selected_date:
            df_map = self.df_1y[self.df_1y['date'] == pd.to_datetime(selected_date)].copy()
        else:
            df_map = self.df_1y.copy()
        
        # Group by TMC to get average speed
        # Use observed=True to address FutureWarning about operating on grouping columns
        tmc_avg = df_map.groupby(['tmc', 'road', 'direction', 'start_latitude', 'start_longitude', 
                                 'end_latitude', 'end_longitude'], observed=True)['speed_mean'].mean().reset_index()
        
        return tmc_avg
    
    def get_tmc_map_data_optimized(self, selected_date=None):
        """Optimized version that caches daily map data and uses views instead of copies"""
        if self.df_1y is None or self.tmc_locations is None:
            return None
        
        # Use a string representation of date for caching
        cache_key = str(selected_date) if selected_date else "all_data"
        
        # Check if we already have this date cached
        if cache_key in self._daily_map_cache:
            return self._daily_map_cache[cache_key]
        
        # Filter data efficiently using query (faster than boolean indexing)
        if selected_date:
            # Use query for better performance on large datasets
            df_filtered = self.df_1y.query('date == @selected_date')
        else:
            return None

        # Group by TMC efficiently - only select needed columns first
        required_cols = ['tmc', 'road', 'direction', 'start_latitude', 'start_longitude', 
                        'end_latitude', 'end_longitude', 'speed_mean']
        
        # Use observed=True to avoid categorical warnings and improve performance
        tmc_avg = (df_filtered[required_cols]
                  .groupby('tmc', observed=True)
                  .agg({
                      'speed_mean': 'mean',
                      'road': 'first',
                      'direction': 'first', 
                      'start_latitude': 'first',
                      'start_longitude': 'first',
                      'end_latitude': 'first',
                      'end_longitude': 'first'
                  })
                  .reset_index())

        # Add speed category efficiently using pd.cut
        tmc_avg['speed_category'] = pd.cut(
            tmc_avg['speed_mean'],
            bins=[0, 20, 30, 40, 50, 60, 100],
            labels=['0-20', '20-30', '30-40', '40-50', '50-60', '60+']
        )

        # Cache the result to avoid recomputation
        self._daily_map_cache[cache_key] = tmc_avg
        
        # Limit cache size to prevent memory issues (keep last 30 days)
        if len(self._daily_map_cache) > 30:
            # Remove oldest entries
            oldest_keys = list(self._daily_map_cache.keys())[:-30]
            for key in oldest_keys:
                del self._daily_map_cache[key]
        
        return tmc_avg
    
    def prepare_map_line_data(self, map_data):
        """Prepare line data for map visualization without creating unnecessary copies"""
        if map_data is None or map_data.empty:
            return pd.DataFrame()
        # Pre-calculate speed categories using numpy for vectorized operations
        speeds = map_data['speed_mean'].values
        speed_categories = np.select(
            [speeds > 60, speeds > 50, speeds > 40, speeds > 30, speeds > 20],
            ['60+', '50-60', '40-50', '30-40', '20-30'],
            default='0-20'
        )
        
        # Create segment labels vectorized
        segment_labels = map_data['road'].astype(str) + ' ' + map_data['direction'].astype(str) + ' - ' + speeds.round(1).astype(str) + ' mph'
        
        # Pre-allocate arrays for better performance
        n_segments = len(map_data)
        line_data = []
        
        # Use itertuples for fastest iteration
        for i, row in enumerate(map_data.itertuples()):
            speed_cat = speed_categories[i]
            segment_label = segment_labels.iloc[i]
            
            # Add both start and end points
            line_data.extend([
                {
                    'latitude': row.start_latitude,
                    'longitude': row.start_longitude,
                    'road': row.road,
                    'direction': row.direction,
                    'speed': row.speed_mean,
                    'speed_category': speed_cat,
                    'segment_label': segment_label,
                    'segment_id': row.tmc
                },
                {
                    'latitude': row.end_latitude,
                    'longitude': row.end_longitude,
                    'road': row.road,
                    'direction': row.direction,
                    'speed': row.speed_mean,
                    'speed_category': speed_cat,
                    'segment_label': segment_label,
                    'segment_id': row.tmc
                }
            ])
        
        result_df = pd.DataFrame(line_data)
        return result_df
    
    def get_available_dates(self):
        """Get list of unique dates available in the data"""
        if self.df_1y is None:
            return None
        
        # Get unique dates and sort them
        unique_dates = sorted(self.df_1y['date'].dropna().unique())
        # Convert to pandas datetime if they're not already
        unique_dates = pd.to_datetime(unique_dates).date
        
        return unique_dates
    
    def format_year_week_simple(self, year_week: str) -> str:
        """
        Convert a 'YYYY-WW' string (ISO week) to a label with its Monday–Sunday date range.
        Example: '2024-39' -> 'Week 39 (Sep 23 - Sep 29, 2024)'
        """
        try:
            year_str, week_str = year_week.split('-')
            year = int(year_str)
            week = int(week_str)
            # ISO: Monday=1 ... Sunday=7
            monday = datetime.fromisocalendar(year, week, 1)
            sunday = monday + timedelta(days=6)
            return f"Week {week} ({monday.strftime('%b %d')} - {sunday.strftime('%b %d, %Y')})"
        except Exception:
            return f"Week {year_week}"
    
    def get_major_holidays(self, start_date=None, end_date=None):
        """Return list of (date, name) for major US holidays within the data range.
        Holidays included: New Year's Day, MLK Day, Memorial Day, Independence Day,
        Labor Day, Thanksgiving, Christmas. Handles observed days if holiday falls on weekend.
        """
        if self.df_1y is None or 'date' not in self.df_1y.columns:
            return []
        # Determine range
        data_min = pd.to_datetime(self.df_1y['date'].min()).date()
        data_max = pd.to_datetime(self.df_1y['date'].max()).date()
        if start_date:
            start_date = pd.to_datetime(start_date).date()
            data_min = max(data_min, start_date)
        if end_date:
            end_date = pd.to_datetime(end_date).date()
            data_max = min(data_max, end_date)
        years = range(data_min.year, data_max.year + 1)
        holidays = []
        for year in years:
            # New Year's Day (Jan 1)
            ny = date(year, 1, 1)
            # Observed adjustment
            if ny.weekday() == 5:  # Saturday
                ny_obs = ny - timedelta(days=1)
            elif ny.weekday() == 6:  # Sunday
                ny_obs = ny + timedelta(days=1)
            else:
                ny_obs = ny
            holidays.append((ny_obs, "New Year's Day"))
            # MLK Day: 3rd Monday of Jan
            mlk = self._nth_weekday_of_month(year, 1, 0, 3)
            holidays.append((mlk, "MLK Day"))
            # Memorial Day: last Monday of May
            memorial = self._last_weekday_of_month(year, 5, 0)
            holidays.append((memorial, "Memorial Day"))
            # Independence Day: July 4 (observed logic)
            july4 = date(year, 7, 4)
            if july4.weekday() == 5:
                july4_obs = july4 - timedelta(days=1)
            elif july4.weekday() == 6:
                july4_obs = july4 + timedelta(days=1)
            else:
                july4_obs = july4
            holidays.append((july4_obs, "Independence Day"))
            # Labor Day: 1st Monday of Sep
            labor = self._nth_weekday_of_month(year, 9, 0, 1)
            holidays.append((labor, "Labor Day"))
            # Thanksgiving: 4th Thursday Nov
            thanksgiving = self._nth_weekday_of_month(year, 11, 3, 4)
            holidays.append((thanksgiving, "Thanksgiving"))
            # Christmas: Dec 25 (observed adjustment)
            xmas = date(year, 12, 25)
            if xmas.weekday() == 5:
                xmas_obs = xmas - timedelta(days=1)
            elif xmas.weekday() == 6:
                xmas_obs = xmas + timedelta(days=1)
            else:
                xmas_obs = xmas
            holidays.append((xmas_obs, "Christmas"))
        # Filter to data range
        holidays = [h for h in holidays if data_min <= h[0] <= data_max]
        # Deduplicate (e.g., observed overlapping with actual next year)
        seen = set()
        unique = []
        for d, n in holidays:
            if d not in seen:
                unique.append((d, n))
                seen.add(d)
        return unique

    def _nth_weekday_of_month(self, year, month, weekday, n):
        """Return date of the nth weekday (0=Mon) in given month/year."""
        first = date(year, month, 1)
        shift = (weekday - first.weekday()) % 7
        return first + timedelta(days=shift + (n - 1) * 7)

    def _last_weekday_of_month(self, year, month, weekday):
        """Return date of last weekday (0=Mon) in month/year."""
        if month == 12:
            first_next = date(year + 1, 1, 1)
        else:
            first_next = date(year, month + 1, 1)
        last = first_next - timedelta(days=1)
        # walk backwards to weekday
        while last.weekday() != weekday:
            last -= timedelta(days=1)
        return last


def main():
    st.title("INRIX Traffic Speed Data Dashboard")
    
    # Initialize INRIX data
    inrix_data = InrixData()
    
    # Add tabs for different sections
    # Added new "Map Animation" tab for hourly animation over the year (lightweight)
    tab_preprocessing, tab_temporal, tab_weekly_animation, tab_map_animation, tab_spatial, tab_map = st.tabs([
        "Data Preprocessing", 
        "Temporal Analysis", 
        "Weekly Animation",
        "Map Animation",
        "Spatial Analysis", 
        "Interactive Map"
    ])
    
    with tab_preprocessing:
        st.header("Data Preprocessing")
        
        # File information
        if inrix_data.file_exists():
            stats = inrix_data.get_file_stats()
            st.info(f"Found INRIX data file: {inrix_data.large_file_path.name}")
            st.metric("File Size", f"{stats['file_size_mb']:.2f} MB")
            st.metric("Estimated Chunks", f"{stats['estimated_chunks']}")
            
            # Show data sample
            st.subheader("Data Sample")
            sample = inrix_data.get_sample_data()
            if sample is not None:
                st.dataframe(sample)
                st.dataframe(sample.describe())
            
            # Check if chunks already exist
            chunk_files = inrix_data.get_chunk_files()
            if chunk_files:
                st.success(f"Found {len(chunk_files)} chunk files. Ready for analysis.")
                
                if st.button("Re-split the file", help="This will overwrite existing chunks"):
                    chunk_files = inrix_data.split_csv_file()
                    if chunk_files:
                        st.success(f"File split into {len(chunk_files)} chunks.")
            else:
                if st.button("Split the file into smaller chunks"):
                    chunk_files = inrix_data.split_csv_file()
                    if chunk_files:
                        st.success(f"File split into {len(chunk_files)} chunks.")
                        
            # Aggregate data
            st.subheader("Data Aggregation")
            if chunk_files:
                # Check if aggregated file already exists
                aggregated_file_path = inrix_data.data_dir / f"{inrix_data.inrix_file_name}_aggregated.csv"
                
                if aggregated_file_path.exists():
                    st.info(f"Aggregated data file exists: {aggregated_file_path.name}")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Load Aggregated Data"):
                            with st.spinner("Loading aggregated data..."):
                                df_1y = inrix_data.aggregate_data(force_reprocess=False)
                                if df_1y is not None:
                                    st.success(f"Data loaded: {len(df_1y)} rows")
                                    st.write("Preview of aggregated data:")
                                    st.dataframe(df_1y.head())
                    with col2:
                        if st.button("Regenerate Aggregated Data"):
                            with st.spinner("Re-aggregating data..."):
                                df_1y = inrix_data.aggregate_data(force_reprocess=True)
                                if df_1y is not None:
                                    st.success(f"Data regenerated: {len(df_1y)} rows")
                                    st.write("Preview of aggregated data:")
                                    st.dataframe(df_1y.head())
                else:
                    if st.button("Aggregate Data"):
                        with st.spinner("Aggregating data..."):
                            df_1y = inrix_data.aggregate_data()
                            if df_1y is not None:
                                st.success(f"Data aggregated: {len(df_1y)} rows")
                                st.write("Preview of aggregated data:")
                                st.dataframe(df_1y.head())
        else:
            st.error(f"INRIX data file not found at {inrix_data.large_file_path}")
            
    with tab_temporal:
        st.header("Temporal Analysis")
        
        # Check if data is aggregated
        if inrix_data.df_1y is None:
            # Try to load saved aggregated data first
            aggregated_file_path = inrix_data.data_dir / f"{inrix_data.inrix_file_name}_aggregated.csv"
            if aggregated_file_path.exists():
                with st.spinner("Loading aggregated data..."):
                    inrix_data.aggregate_data(force_reprocess=False)
            else:
                # Try to aggregate if chunks exist
                chunk_files = inrix_data.get_chunk_files()
                if chunk_files:
                    if st.button("Aggregate Data for Temporal Analysis"):
                        with st.spinner("Aggregating data..."):
                            inrix_data.aggregate_data()
            
            if inrix_data.df_1y is None:
                st.warning("No aggregated data available. Please go to Data Preprocessing tab and aggregate the data first.")
                st.stop()
        
        # Daily trend line
        st.subheader("Daily Speed Trend")
        daily_avg = inrix_data.get_daily_speed_trend()
        
        fig1 = px.line(
            daily_avg, 
            x='date', 
            y='speed_mean', 
            title='Daily Average Speed Over One Year',
            labels={'speed_mean': 'Average Speed (mph)', 'date': 'Date'}
        )
        # Add major holidays
        try:
            holidays = inrix_data.get_major_holidays()
            if holidays:
                daily_dates = pd.to_datetime(daily_avg['date']).dt.date
                y_series = daily_avg['speed_mean']
                y_min = float(y_series.min())
                y_max = float(y_series.max())
                y_span = y_max - y_min if y_max > y_min else 1.0
                used_x = set()
                hol_x = []
                hol_y = []
                hol_hover = []
                for h_date, h_name in holidays:
                    if h_date in daily_dates.values:
                        mask = (daily_dates == h_date)
                        y_val = float(y_series[mask].iloc[0]) if mask.any() else (y_min + 0.1 * y_span)
                        hol_x.append(h_date)
                        hol_y.append(y_val)
                        hol_hover.append(f"{h_name}<br>{y_val:.2f} mph")
                        fig1.add_vline(x=h_date, line_width=1, line_dash='dot', line_color='#aa3366', opacity=0.5)
                        if h_date not in used_x:
                            offset = 0.02 + (0.04 * (len(used_x) % 3))
                            fig1.add_annotation(
                                x=h_date,
                                y=y_val + offset * y_span,
                                text=h_name,
                                showarrow=False,
                                font=dict(size=10, color='#aa3366'),
                                yanchor='bottom',
                                textangle=-45
                            )
                            used_x.add(h_date)
                if hol_x:
                    fig1.add_scatter(
                        x=hol_x,
                        y=hol_y,
                        mode='markers',
                        marker=dict(color='#aa3366', size=8, symbol='diamond'),
                        name='Holidays',
                        hovertext=hol_hover,
                        hovertemplate='%{hovertext}<extra></extra>'
                    )
                fig1.add_annotation(
                    xref='paper', yref='paper', x=0, y=1.07, showarrow=False,
                    text='Diamonds mark major US holidays',
                    font=dict(size=11, color='#aa3366')
                )
        except Exception as e:
            print(f"Holiday annotation error: {e}")
        fig1.update_layout(height=500)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Monthly boxplots
        st.subheader("Monthly Speed Distribution")
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December']
        
        fig2 = px.box(
            inrix_data.df_1y, 
            x='month_name', 
            y='speed_mean', 
            category_orders={'month_name': month_order},
            title='Speed Distribution by Month',
            labels={'speed_mean': 'Average Speed (mph)', 'month_name': 'Month'}
        )
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Seasonal analysis
        st.subheader("Seasonal Analysis")
        
        seasonal_data = inrix_data.get_seasonal_stats()
        if seasonal_data:
            col1, col2 = st.columns(2)
            
            with col1:
                # Seasonal boxplot
                fig4 = px.box(
                    inrix_data.df_1y,
                    x='season',
                    y='speed_mean',
                    category_orders={'season': ['Winter', 'Spring', 'Summer', 'Fall']},
                    title='Speed Distribution by Season',
                    labels={'speed_mean': 'Average Speed (mph)', 'season': 'Season'}
                )
                st.plotly_chart(fig4, use_container_width=True)
                
                # Rush hour comparison
                morning_rush = seasonal_data['morning_rush']
                evening_rush = seasonal_data['evening_rush']
                non_rush = seasonal_data['non_rush']
                
                fig5 = go.Figure()
                fig5.add_trace(go.Scatter(
                    x=['Winter', 'Spring', 'Summer', 'Fall'],
                    y=morning_rush.values,
                    mode='lines+markers',
                    name='Morning Rush (7-9 AM)'
                ))
                fig5.add_trace(go.Scatter(
                    x=['Winter', 'Spring', 'Summer', 'Fall'],
                    y=evening_rush.values,
                    mode='lines+markers',
                    name='Evening Rush (4-6 PM)'
                ))
                fig5.add_trace(go.Scatter(
                    x=['Winter', 'Spring', 'Summer', 'Fall'],
                    y=non_rush.values,
                    mode='lines+markers',
                    name='Non-Rush Hours'
                ))
                
                fig5.update_layout(
                    title='Average Speed During Different Time Periods by Season',
                    xaxis_title='Season',
                    yaxis_title='Average Speed (mph)'
                )
                st.plotly_chart(fig5, use_container_width=True)
            
            with col2:
                # Display seasonal stats
                st.dataframe(seasonal_data['seasonal_stats'])
                
                # Weekday vs Weekend comparison
                weekday = seasonal_data['weekday']
                weekend = seasonal_data['weekend']
                
                fig6 = go.Figure()
                fig6.add_trace(go.Scatter(
                    x=['Winter', 'Spring', 'Summer', 'Fall'],
                    y=weekday.values,
                    mode='lines+markers',
                    name='Weekdays'
                ))
                fig6.add_trace(go.Scatter(
                    x=['Winter', 'Spring', 'Summer', 'Fall'],
                    y=weekend.values,
                    mode='lines+markers',
                    name='Weekends'
                ))
                
                fig6.update_layout(
                    title='Weekday vs Weekend Average Speed by Season',
                    xaxis_title='Season',
                    yaxis_title='Average Speed (mph)'
                )
                st.plotly_chart(fig6, use_container_width=True)
                
                # Calculate speed difference between weekday and weekend
                speed_diff = (weekend - weekday).to_frame(name='speed_diff')
                speed_diff['percent_diff'] = (weekend - weekday) / weekday * 100
                
                st.write("Weekend vs Weekday Speed Difference:")
                st.dataframe(speed_diff)
    
    with tab_weekly_animation:
        st.header("Weekly Animation")
        st.markdown("Animated weekly day/hour speed heatmap. Loops through all available weeks with a fixed color scale.")
        weekly_data = inrix_data.get_weekly_pivots()
        if weekly_data:
            unique_year_weeks = weekly_data['unique_year_weeks']
            weekly_pivots = weekly_data['weekly_pivots']
            global_min = weekly_data['global_min']
            global_max = weekly_data['global_max']

            if len(weekly_pivots) == 0:
                st.info("No weekly pivot data available.")
            else:
                # Cache figure construction to avoid rebuilding on every rerun
                if 'weekly_animation_fig' not in st.session_state:
                    # Build frames
                    frames = []
                    first_week = unique_year_weeks[0]
                    first_pivot = weekly_pivots[first_week]

                    # Create initial trace (first frame)
                    # Reverse day order (display Monday at top -> Sunday bottom) by reversing list and flipping z accordingly
                    day_labels = list(first_pivot.index)
                    # reversed_days = day_labels[::-1]
                    # def pivot_to_reversed_matrix(pivot_df):
                    #     return pivot_df.loc[reversed_days].values

                    heatmap_trace = go.Heatmap(
                        z=first_pivot.values,
                        x=first_pivot.columns,
                        y=day_labels,
                        colorscale='Viridis',
                        zmin=global_min,
                        zmax=global_max,
                        colorbar=dict(
                            title='Speed (mph)',
                            ticks='outside'
                        )
                    )

                    for yw in unique_year_weeks:
                        pivot_data = weekly_pivots[yw]
                        week_df = inrix_data.df_1y[inrix_data.df_1y['year_week'] == yw]
                        start_date = week_df['date'].min()
                        end_date = week_df['date'].max()
                        if start_date is not None and end_date is not None:
                            date_range = f"{pd.to_datetime(start_date).strftime('%b %d, %Y')} - {pd.to_datetime(end_date).strftime('%b %d, %Y')}"
                        else:
                            date_range = yw
                        frame_title = f"Avg Speed by Day/Hour - {inrix_data.format_year_week_simple(yw)}<br><sup>{date_range}</sup>"
                        frames.append(
                            go.Frame(
                                data=[go.Heatmap(
                                    z=pivot_data.values,
                                    x=pivot_data.columns,
                                    y=day_labels,
                                    colorscale='Viridis',
                                    zmin=global_min,
                                    zmax=global_max,
                                    showscale=True
                                )],
                                name=yw,
                                layout=go.Layout(title=frame_title)
                            )
                        )

                    # Slider steps
                    slider_steps = []
                    for yw in unique_year_weeks:
                        slider_steps.append({
                            'args': [[yw], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}],
                            'label': yw,
                            'method': 'animate'
                        })

                    # Figure layout with play/pause buttons
                    initial_week_df = inrix_data.df_1y[inrix_data.df_1y['year_week'] == first_week]
                    start_date0 = initial_week_df['date'].min()
                    end_date0 = initial_week_df['date'].max()
                    if start_date0 is not None and end_date0 is not None:
                        date_range0 = f"{pd.to_datetime(start_date0).strftime('%b %d, %Y')} - {pd.to_datetime(end_date0).strftime('%b %d, %Y')}"
                    else:
                        date_range0 = first_week
                    initial_title = f"Avg Speed by Day/Hour - {inrix_data.format_year_week_simple(first_week)}<br><sup>{date_range0}</sup>"

                    fig_anim = go.Figure(data=[heatmap_trace], frames=frames)
                    fig_anim.update_layout(
                        title=initial_title,
                        xaxis_title='Hour',
                        yaxis_title='Day',
                        yaxis=dict(autorange='reversed'),
                        height=600,
                        margin=dict(t=80, l=60, r=30, b=60),
                        updatemenus=[{
                            'type': 'buttons',
                            'showactive': False,
                            'x': 1.05,
                            'y': 1.15,
                            'buttons': [
                                {
                                    'label': 'Play',
                                    'method': 'animate',
                                    'args': [None, {
                                        'frame': {'duration': 600, 'redraw': True},
                                        'fromcurrent': True,
                                        'transition': {'duration': 300, 'easing': 'linear'}
                                    }]
                                },
                                {
                                    'label': 'Pause',
                                    'method': 'animate',
                                    'args': [[None], {
                                        'frame': {'duration': 0, 'redraw': False},
                                        'mode': 'immediate',
                                        'transition': {'duration': 0}
                                    }]
                                }
                            ]
                        }],
                        sliders=[{
                            'active': 0,
                            'y': -0.07,
                            'x': 0.05,
                            'len': 0.9,
                            'pad': {'b': 10, 't': 50},
                            'currentvalue': {'prefix': 'Week: '},
                            'steps': slider_steps
                        }]
                    )

                    st.session_state.weekly_animation_fig = fig_anim

                st.plotly_chart(st.session_state.weekly_animation_fig, use_container_width=True)
                st.caption("Use the play button to animate through all weeks. Color scale fixed (40 mph to yearly max).")
        else:
            st.info("Aggregate data to enable weekly animation view.")

    with tab_map_animation:
        st.header("Map Animation (Hourly Year)")
        st.markdown("""
        Lightweight animation of hourly average speeds across segments over the year. 
        To preserve performance we (a) use midpoint markers instead of full line geometry, (b) allow frame down-sampling, and (c) cache preprocessed frame data.
        """)
        # NOTE: Performance strategy for this animation:
        # - Convert segment geometry to single midpoint markers (halves points vs. start+end) to reduce plotly payload
        # - Allow user to select frame_step (hour stride) and maximum frames to cap size
        # - Use @st.cache_data for preprocessing the hourly midpoint table; invalidated only when filters change
        # - Discrete color bins avoid per-frame colorscale recalculation
        # - Frames built once and cached in session_state keyed by control parameters
        # - Optionally filter roads to further reduce point count per frame

        # Ensure aggregated data loaded
        if inrix_data.df_1y is None:
            aggregated_file_path = inrix_data.data_dir / f"{inrix_data.inrix_file_name}_aggregated.csv"
            if aggregated_file_path.exists():
                with st.spinner("Loading aggregated data..."):
                    inrix_data.aggregate_data(force_reprocess=False)
            else:
                st.warning("No aggregated data available. Please aggregate in 'Data Preprocessing' tab first.")
                st.stop()

        # Guard if still none
        if inrix_data.df_1y is None or inrix_data.tmc_locations is None:
            st.info("Need both aggregated speed data and TMC location data.")
            st.stop()

        # Controls
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1,1,1])
        with col_ctrl1:
            frame_step = st.select_slider(
                "Frame step (hours)",
                options=[1,2,3,6,12,24],
                value=6,
                help="Use larger step for fewer frames / faster playback"
            )
        with col_ctrl2:
            max_frames = st.number_input("Max frames", min_value=100, max_value=2000, value=600, step=100,
                                         help="Hard cap to avoid huge in-browser payload")
        with col_ctrl3:
            play_ms = st.slider("Frame duration (ms)", min_value=100, max_value=1000, value=300, step=50,
                                help="Animation playback speed per frame")

        # Optional road filter (helps reduce points)
        available_roads = sorted([r for r in inrix_data.df_1y['road'].dropna().unique() if r])[:200]
        selected_roads = st.multiselect("Filter roads (optional)", options=available_roads, default=[],
                                        help="Leave empty to show all. Filtering reduces per-frame points.")

        # Cached preprocessing function
        @st.cache_data(show_spinner=False)
        def _prepare_hourly_points(df, roads_filter):
            # Minimal subset
            cols_needed = ['hour','tmc','speed_mean','start_latitude','start_longitude','end_latitude','end_longitude','road','direction']
            df2 = df[cols_needed].copy()
            if roads_filter:
                df2 = df2[df2['road'].isin(roads_filter)]
            # Compute midpoints (vectorized)
            df2['lat'] = (df2['start_latitude'] + df2['end_latitude']) / 2.0
            df2['lon'] = (df2['start_longitude'] + df2['end_longitude']) / 2.0
            # Downcast for memory
            df2['speed_mean'] = pd.to_numeric(df2['speed_mean'], downcast='float')
            # Ensure hour is datetime
            df2['hour'] = pd.to_datetime(df2['hour'])
            # Sort once
            df2 = df2.sort_values('hour')
            # Global min/max for color scaling
            gmin = float(df2['speed_mean'].min())
            gmax = float(df2['speed_mean'].max())
            # Create categorical speed bins for discrete color legend (faster than continuous scale re-draw)
            bins = [0,20,30,40,50,60,1000]
            labels = ['0-20','20-30','30-40','40-50','50-60','60+']
            df2['speed_bin'] = pd.cut(df2['speed_mean'], bins=bins, labels=labels, include_lowest=True)
            return df2, gmin, gmax, labels

        df_points, global_min_speed, global_max_speed, speed_labels = _prepare_hourly_points(inrix_data.df_1y, selected_roads)

        # Build frame index (unique hours) with step & cap
        unique_hours = df_points['hour'].drop_duplicates().sort_values()
        # Apply step
        unique_hours = unique_hours[::frame_step]
        total_possible = len(unique_hours)
        if len(unique_hours) > max_frames:
            unique_hours = unique_hours[:max_frames]
        st.caption(f"Building animation with {len(unique_hours)} frames (from {total_possible} possible after step {frame_step}).")

        # Color mapping consistent
        speed_color_map = {
            '0-20': '#ff0000',
            '20-30': '#ff4500',
            '30-40': '#ff8c00',
            '40-50': '#ffff00',
            '50-60': '#7fff00',
            '60+': '#027a02'
        }

        # Cache constructed figure per parameter combination
        cache_key = (frame_step, max_frames, play_ms, tuple(sorted(selected_roads)))
        if 'map_animation_cache' not in st.session_state:
            st.session_state.map_animation_cache = {}

        if cache_key not in st.session_state.map_animation_cache:
            with st.spinner("Constructing animation frames (one-time, cached)..."):
                # Derive rough center from data once
                center_lat = float(df_points['lat'].mean()) if not df_points.empty else 0.0
                center_lon = float(df_points['lon'].mean()) if not df_points.empty else 0.0

                # --- Line segment animation (start->end) grouped by speed bin ---
                speed_bin_order = ['60+','50-60','40-50','30-40','20-30','0-20']

                def build_bin_traces(frame_df):
                    traces = []
                    for bin_label in speed_bin_order:
                        sub = frame_df[frame_df['speed_bin'] == bin_label]
                        if sub.empty:
                            # Empty placeholder to preserve trace ordering in frames
                            traces.append(go.Scattermap(lat=[], lon=[], mode='lines', name=bin_label,
                                                        line=dict(color=speed_color_map[bin_label], width=3),
                                                        hoverinfo='skip'))
                            continue
                        # Build coordinate arrays with None separators
                        lats = []
                        lons = []
                        hover_texts = []
                        for row in sub.itertuples():
                            lats.extend([row.start_latitude, row.end_latitude, None])
                            lons.extend([row.start_longitude, row.end_longitude, None])
                            hover_texts.append(f"{row.road} {row.direction} | {row.speed_mean:.1f} mph")
                        traces.append(go.Scattermap(
                            lat=lats,
                            lon=lons,
                            mode='lines',
                            name=bin_label,
                            line=dict(color=speed_color_map[bin_label], width=3),
                            hovertext=hover_texts,
                            hoverinfo='text'
                        ))
                    return traces

                frames = []
                first_hour = unique_hours.iloc[0]
                first_df = df_points[df_points['hour'] == first_hour]
                base_traces = build_bin_traces(first_df)

                for ts in unique_hours:
                    sub_df = df_points[df_points['hour'] == ts]
                    frame_traces = build_bin_traces(sub_df)
                    frames.append(go.Frame(data=frame_traces, name=str(ts)))

                slider_steps = [
                    {
                        'args': [[str(ts)], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}],
                        'label': ts.strftime('%Y-%m-%d %H:%M'),
                        'method': 'animate'
                    }
                    for ts in unique_hours
                ]

                title_base = "Hourly Segment Speeds (Line Segments)"
                fig_map_anim = go.Figure(data=base_traces, frames=frames)
                # Switch to MapLibre API: use layout.map instead of layout.mapbox
                fig_map_anim.update_layout(
                    map=dict(style='open-street-map', zoom=9, center={'lat': center_lat, 'lon': center_lon}),
                    margin=dict(t=70, l=10, r=10, b=10)
                )
                fig_map_anim.update_layout(
                    title=f"{title_base}<br><sup>{unique_hours.min().strftime('%Y-%m-%d')} to {unique_hours.max().strftime('%Y-%m-%d')}</sup>",
                    updatemenus=[{
                        'type': 'buttons', 'showactive': False, 'x': 1.05, 'y': 1.15,
                        'buttons': [
                            {
                                'label': 'Play', 'method': 'animate',
                                'args': [None, {
                                    'frame': {'duration': play_ms, 'redraw': True},
                                    'fromcurrent': True,
                                    'transition': {'duration': 0}
                                }]
                            },
                            {
                                'label': 'Pause', 'method': 'animate',
                                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]
                            }
                        ]
                    }],
                    sliders=[{
                        'active': 0,
                        'y': -0.05,
                        'x': 0.05,
                        'len': 0.9,
                        'pad': {'b': 10, 't': 40},
                        'currentvalue': {'prefix': 'Hour: '},
                        'steps': slider_steps
                    }]
                )

                # Use native legend from traces; ensure order
                fig_map_anim.update_layout(legend=dict(title='Speed Range (mph)', orientation='h', y=-0.1, x=0.5, xanchor='center'))
                st.session_state.map_animation_cache[cache_key] = fig_map_anim

        fig_cached = st.session_state.map_animation_cache[cache_key]
        st.plotly_chart(fig_cached, use_container_width=True, height=700)
        st.caption("Markers show segment midpoint colored by hourly average speed. Adjust frame step to trade detail vs performance.")
        with st.expander("Performance Notes"):
            st.write("""
            - Frames are down-sampled by the selected step to reduce payload size.
            - Midpoint markers reduce geometry compared to drawing full line segments.
            - DataFrame preprocessing is cached; changing controls invalidates only what is necessary.
            - For full 1-hour resolution (frame step = 1) consider lowering max frames or filtering roads.
            """)
    
    with tab_spatial:
        st.header("Spatial Analysis")
        
        # Check if data is aggregated
        if inrix_data.df_1y is None:
            # Try to load saved aggregated data first
            aggregated_file_path = inrix_data.data_dir / f"{inrix_data.inrix_file_name}_aggregated.csv"
            if aggregated_file_path.exists():
                with st.spinner("Loading aggregated data..."):
                    inrix_data.aggregate_data(force_reprocess=False)
            else:
                # Try to aggregate if chunks exist
                chunk_files = inrix_data.get_chunk_files()
                if chunk_files:
                    if st.button("Aggregate Data for Spatial Analysis"):
                        with st.spinner("Aggregating data..."):
                            inrix_data.aggregate_data()
            
            if inrix_data.df_1y is None:
                st.warning("No aggregated data available. Please go to Data Preprocessing tab and aggregate the data first.")
                st.stop()
        
        # Get spatial analysis data
        spatial_data = inrix_data.analyze_spatial_patterns()
        
        if spatial_data:
            col1, col2 = st.columns(2)
            
            with col1:
                # Average speed by road
                st.subheader("Average Speed by Road")
                road_data = spatial_data['road_speeds']
                road_data = road_data[road_data['count'] > 1000].sort_values(by='mean')
                
                if len(road_data) > 0:
                    fig7 = px.bar(
                        road_data,
                        y='road',
                        x='mean',
                        error_x='std',
                        title='Average Speed by Road',
                        labels={'mean': 'Average Speed (mph)', 'road': 'Road', 'std': 'Standard Deviation'},
                        orientation='h'
                    )
                    fig7.update_layout(height=500)
                    st.plotly_chart(fig7, use_container_width=True)
                else:
                    st.info("Not enough data for road speed analysis")
                
                # Direction speeds
                st.subheader("Average Speed by Direction")
                st.dataframe(spatial_data['direction_speeds'])
            
            with col2:
                # Segments with largest speed drop during peak hours
                st.subheader("Congestion Analysis")
                
                tmc_peak_diff = spatial_data['tmc_peak_diff']
                # Create a copy to avoid the SettingWithCopyWarning
                worst_segments = tmc_peak_diff.head(15).copy()
                
                if len(worst_segments) > 0:
                    # Create segment labels using loc to avoid SettingWithCopyWarning
                    worst_segments.loc[:, 'segment_label'] = worst_segments.apply(
                        lambda x: f"{x['road']} {x['direction']} ({x['tmc']})", 
                        axis=1
                    )
                    
                    fig8 = px.bar(
                        worst_segments,
                        y='segment_label',
                        x=worst_segments['peak_speed_drop'].abs(),
                        title='Segments with Largest Speed Drop During Peak Hours',
                        labels={'x': 'Speed Reduction (mph)', 'y': 'Road Segment'},
                        orientation='h',
                        color=worst_segments['peak_speed_drop'].abs(),
                        color_continuous_scale='Reds'
                    )
                    fig8.update_layout(height=500)
                    st.plotly_chart(fig8, use_container_width=True)
                    
                    st.write("Top 5 most congested segments:")
                    st.dataframe(worst_segments.head(5)[['road', 'direction', 'tmc', 'peak_speed_drop']])
                else:
                    st.info("Not enough data for congestion analysis")
    
    with tab_map:
        st.header("Interactive Map")
        
        # Initialize session state for map configuration
        if 'map_initialized' not in st.session_state:
            st.session_state.map_initialized = False
            st.session_state.map_center = None
            st.session_state.map_zoom = None
            st.session_state.map_bounds = None
            st.session_state.current_fig = None  # Cache the current figure
            st.session_state.last_selected_date = None  # Track last selected date
            st.session_state.current_map_data = None  # Cache current map data
            st.session_state.available_dates = None  # Cache available dates
        
        # Check if data is aggregated
        if inrix_data.df_1y is None:
            # Try to load saved aggregated data first
            aggregated_file_path = inrix_data.data_dir / f"{inrix_data.inrix_file_name}_aggregated.csv"
            if aggregated_file_path.exists():
                with st.spinner("Loading aggregated data..."):
                    inrix_data.aggregate_data(force_reprocess=False)
            else:
                # Try to aggregate if chunks exist
                chunk_files = inrix_data.get_chunk_files()
                if chunk_files:
                    if st.button("Aggregate Data for Map Visualization"):
                        with st.spinner("Aggregating data..."):
                            inrix_data.aggregate_data()
            
            if inrix_data.df_1y is None:
                st.warning("No aggregated data available. Please go to Data Preprocessing tab and aggregate the data first.")
                st.stop()

        # Get complete map data for calculating initial bounds (only once)
        if st.session_state.map_initialized is False:
            with st.spinner("Initializing map configuration..."):
                # Use a sample of data to calculate bounds more efficiently
                sample_data = inrix_data.df_1y.sample(min(1000, len(inrix_data.df_1y)))
                
                if not sample_data.empty:
                    # Calculate map center and bounds based on sample data
                    min_lat = min(sample_data['start_latitude'].min(), sample_data['end_latitude'].min())
                    max_lat = max(sample_data['start_latitude'].max(), sample_data['end_latitude'].max())
                    min_lon = min(sample_data['start_longitude'].min(), sample_data['end_longitude'].min())
                    max_lon = max(sample_data['start_longitude'].max(), sample_data['end_longitude'].max())
                    
                    # Calculate center
                    center_lat = (min_lat + max_lat) / 2
                    center_lon = (min_lon + max_lon) / 2
                    
                    # Calculate zoom level
                    lat_diff = max_lat - min_lat
                    lon_diff = max_lon - min_lon
                    lat_diff = max(lat_diff, 0.01)
                    lon_diff = max(lon_diff, 0.01)
                    
                    screen_ratio = 2.0
                    lat_zoom = math.log2(360 / (lat_diff * 1.1)) - 1
                    lon_zoom = math.log2(360 / (lon_diff * 1.1 * screen_ratio)) - 1
                    map_zoom = max(9, min(15, round(min(lat_zoom, lon_zoom))))
       
                    # Store values in session state
                    st.session_state.map_center = {"lat": center_lat, "lon": center_lon}
                    st.session_state.map_zoom = map_zoom
                    st.session_state.map_bounds = {
                        "south": min_lat - 0.01,
                        "west": min_lon - 0.01,
                        "north": max_lat + 0.01,
                        "east": max_lon + 0.01
                    }
                else:
                    # Default values if no data
                    st.session_state.map_center = {"lat": 33.4484, "lon": -112.0740}
                    st.session_state.map_zoom = 10
                    
                st.session_state.map_initialized = True

        # Map filters
        # Get list of available dates (cache this as well since it doesn't change)
        if 'available_dates' not in st.session_state or st.session_state.available_dates is None:
            st.session_state.available_dates = inrix_data.get_available_dates()
        
        available_dates = st.session_state.available_dates
        
        if available_dates is not None and len(available_dates) > 0:
            # Convert to datetime.date objects for the date_input widget
            min_date = min(available_dates)
            max_date = max(available_dates)
            
            # Create date picker
            t_date_input_start = time.time()
            selected_date = st.date_input(
                "Select Date",
                value=min_date,  # Default to the first available date
                min_value=min_date,
                max_value=max_date,
                help="Choose a specific date to view daily average traffic speeds"
            )
            
            # Convert selected_date to the same format as in the dataframe
            selected_date_pd = pd.to_datetime(selected_date).date()
            
            # Only update map if date has changed
            date_changed = (st.session_state.last_selected_date != selected_date_pd)
            
            # Add JavaScript timing measurement
            if date_changed or st.session_state.current_fig is None:
                
                with st.spinner("Loading map data..."):
                    # Get optimized map data
                    map_data = inrix_data.get_tmc_map_data_optimized(selected_date=selected_date_pd)
                    
                    if map_data is not None and not map_data.empty:
                        # Define color mapping once
                        speed_color_map = {
                            '0-20': '#ff0000',    # Red
                            '20-30': '#ff4500',   # Orange-red
                            '30-40': '#ff8c00',   # Orange
                            '40-50': '#ffff00',   # Yellow
                            '50-60': '#7fff00',   # Light green
                            '60+': '#027a02'      # Dark green
                        }
                        
                        # Prepare line data efficiently
                        line_df = inrix_data.prepare_map_line_data(map_data)

                        if not line_df.empty:
                            # Create the map

                            fig9 = px.line_map(
                                line_df,
                                lat="latitude",
                                lon="longitude",
                                color="speed_category",
                                color_discrete_map=speed_color_map,
                                hover_name="segment_label",
                                hover_data=["road", "direction", "speed"],
                                line_group="segment_id",
                                map_style="basic",
                                zoom=st.session_state.map_zoom,
                                center=st.session_state.map_center,
                                height=700,
                                category_orders={"speed_category": ['60+', '50-60', '40-50', '30-40', '20-30', '0-20']},
                                labels={"speed_category": "Speed Range (mph)", "speed": "Speed (mph)"}
                            )
                            
                            # Update layout
                            title_text = f"Daily Average Traffic Speed Map - {selected_date.strftime('%A, %B %d, %Y')}"
                            
                            fig9.update_layout(
                                title=title_text,
                                map=dict(
                                    style="basic",
                                    center=st.session_state.map_center,
                                    zoom=st.session_state.map_zoom,
                                    bounds=st.session_state.map_bounds
                                ),
                                height=700,
                                margin={"r":30,"t":30,"l":30,"b":30},
                                legend_title_text="Speed Range (mph)",
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=-0.1,
                                    xanchor="center",
                                    x=0.5
                                )
                            )
                            
                            # Cache the figure and data
                            st.session_state.current_fig = fig9
                            st.session_state.current_map_data = map_data
                            st.session_state.last_selected_date = selected_date_pd
                        else:
                            st.session_state.current_fig = go.Figure()
                            st.session_state.current_map_data = pd.DataFrame()
                    else:
                        st.session_state.current_fig = go.Figure()
                        st.session_state.current_map_data = pd.DataFrame()

            # Show which date is selected
            st.info(f"Showing daily average traffic speeds for: {selected_date.strftime('%A, %B %d, %Y')}")
            
            # Display the cached figure - THIS IS WHERE BROWSER RENDERING HAPPENS
            if st.session_state.current_fig is not None:
                st.plotly_chart(st.session_state.current_fig, use_container_width=True)
                # Show summary statistics using cached data
                if hasattr(st.session_state, 'current_map_data') and not st.session_state.current_map_data.empty:
                    st.subheader("Map Summary")
                    
                    map_data = st.session_state.current_map_data
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Segments Displayed", f"{len(map_data)}")
                    
                    with col2:
                        st.metric("Average Speed", f"{map_data['speed_mean'].mean():.1f} mph")
                    
                    with col3:
                        st.metric("Speed Range", f"{map_data['speed_mean'].min():.1f} - {map_data['speed_mean'].max():.1f} mph")
                    
                    # Show speed distribution
                    speed_dist = map_data['speed_category'].value_counts().sort_index()

                    fig10 = px.pie(
                        values=speed_dist.values,
                        names=speed_dist.index,
                        title="Speed Distribution",
                        color=speed_dist.index,
                        color_discrete_map={
                            '0-20': '#ff0000',
                            '20-30': '#ff4500',
                            '30-40': '#ff8c00',
                            '40-50': '#ffff00',
                            '50-60': '#7fff00',
                            '60+': '#027a02'
                        }
                    )
                    st.plotly_chart(fig10, use_container_width=True)
                    
            else:
                st.warning("No map data available for the selected date.")

        else:
            selected_date_pd = None
            st.warning("No date data available.")
        
        

    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("Clear All Caches", 
                    help="Reset all cached data to free up memory"):
            # Clear InrixData caches
            if hasattr(inrix_data, '_daily_map_cache'):
                inrix_data._daily_map_cache.clear()
            
            # Clear session state caches
            cache_keys_to_clear = [
                'map_data_all_weeks', 'line_data_all_weeks', 'weekly_data_cache',
                'map_bounds', 'map_center', 'map_zoom', 'current_fig', 
                'current_map_data', 'last_selected_date', 'available_dates',
                'map_initialized'
            ]
            
            for key in cache_keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
                    
            st.success("All caches cleared successfully!")
            st.rerun()
    
    with col2:
        with st.expander("About this dashboard"):
            st.write("""
            This dashboard visualizes INRIX traffic speed data. Features include:
            - Data preprocessing (splitting large files, aggregation)
            - Temporal analysis (daily/weekly/seasonal patterns)
            - Spatial analysis (road segment performance)
            - Interactive map visualization
            
            **Performance Optimizations:**
            - Cached map data for faster date switching
            - Optimized data processing using pandas query and itertuples
            - Smart session state management to reduce re-rendering
            - Memory-efficient data handling
            
            Created by USDOT: Work Zone Data Exchange (WZDx) Project
            """)

if __name__ == "__main__":
    main()