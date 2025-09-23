from constants import PLOTS_BASE, DATE_TIME
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import os
from datetime import datetime

class PlotManager:
    """
    Runs the full standard set of plots on the given DataFrame.

    The following plots are generated and saved:
    - Day of Week distribution
    - Bankholiday distribution by Day of Week
    - Rush Hour distribution by Day of Week
    - Duration Start_to_End interval distribution
    - Duration Start_to_Weather_Timestamp interval distribution

    Args:
        df (pd.DataFrame): Enriched DataFrame containing required columns.

    Returns:
        None
    """

    @staticmethod
    def plot_day_of_week_distribution(df: pd.DataFrame):
        order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        fig, ax = plt.subplots()
        df['Day_of_Week'].value_counts().reindex(order).plot(kind='bar', ax=ax)
        ax.set_title('Accidents per Day of Week')
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Number of Accidents')
        fig.tight_layout()
        #plt.show()
        PlotManager.save_plot(fig, 'Day_of_Week_Distribution', 'Day_of_Week_Distribution')


    @staticmethod
    def plot_bankholiday_distribution(df: pd.DataFrame):
        order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        fig, ax = plt.subplots()
        df['Day_of_Week'].value_counts().reindex(order).plot(kind='bar', ax=ax)
        ax.set_title('Accidents per Day of Week')
        ax.set_xlabel('Day of week')
        ax.set_ylabel('Number of Accidents')
        fig.tight_layout()
        #plt.show()
        PlotManager.save_plot(fig, 'Bankholiday_Distribution', 'Bankholiday_Distribution')


    @staticmethod
    def plot_rush_hour_distribution(df: pd.DataFrame):
        order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        fig, ax = plt.subplots()
        grouped = df.groupby(['Day_of_Week', 'Rush_Hour']).size().unstack(fill_value=0).reindex(order)
        grouped.plot(kind='bar', stacked=False, ax=ax, color={False: 'blue', True: 'red'})
        
        ax.set_title('Rush Hour Distribution by Day of Week')
        ax.set_xlabel('Day of week', fontsize=10)
        ax.set_ylabel('Number of Accidents')
        PlotManager.save_plot(fig, 'Rush_Hour_Distribution', 'Rush_Hour_Distribution')



    @staticmethod
    def plot_duration_distribution(df: pd.DataFrame, column_name: str, title: str, filename: str):
        """
        Plots distribution using defined interval for a duration column.

        Args:
            df (pd.DataFrame): DataFrame containing the duration column.
            column_name (str): Name of the column to plot.
            plot_title (str): Title of the plot.
            file_name (str): Name of the file to save.
        """
        # Define bins and labels depending on column
        if column_name == 'Duration_Start_to_End(min)':
            bins = [0, 20, 30, 60, 90, 120, 240, float('inf')]
            labels = ['0-20', '20-30', '30-60', '60-90','90-120', '120-240', '>240']
        elif column_name == 'Duration_Start_to_Timestamp(min)':
            bins = [0, 5, 15, 30, 60, float('inf')]
            labels = ['0-5', '5-15', '15-30', '30-60', '>60']

        # Cut into bins
        df['Duration_Interval'] = pd.cut(df[column_name], bins=bins, labels=labels, right=False, ordered=True)

        # Plot
        fig, ax = plt.subplots()
        df['Duration_Interval'].value_counts().reindex(labels).plot(kind='bar', ax=ax, title=title)
        ax.set_xlabel('Duration Interval (minutes)', fontsize=10)  # Force x-axis label to display
        ax.set_xticklabels(labels, rotation=0, ha='center')  # Set x-tick labels horizontally
        ax.set_ylabel('Number of Accidents')
        PlotManager.save_plot(fig, title, filename)

        # Cleanup
        df.drop(columns=['Duration_Interval'], inplace=True)


    @staticmethod
    def plot_datatype_summary_stacked_bar(df_datatype, plots_base, label="Initial"):
        """
        Plots a summary of empty and None occurances per column. Closely related to Explorer. 
        """            
        fig, ax = plt.subplots(figsize=(10, len(df_datatype['Header'].unique()) * 0.4))
        
        headers = df_datatype['Header'].tolist()
        green = df_datatype['Count'].tolist()
        blue = df_datatype['Count_Empty'].tolist()
        red = df_datatype['Count_None'].tolist()

        total = [g + b + r for g, b, r in zip(green, blue, red)]

        green_pct = [g / t * 100 if t > 0 else 0 for g, t in zip(green, total)]
        blue_pct = [b / t * 100 if t > 0 else 0 for b, t in zip(blue, total)]
        red_pct = [r / t * 100 if t > 0 else 0 for r, t in zip(red, total)]

        y_pos = range(len(headers))[::-1]  # first column on top!

        ax.barh(y_pos, green_pct, color='green', label='Value Present')
        ax.barh(y_pos, blue_pct, left=green_pct, color='blue', label='Empty Value')
        ax.barh(y_pos, red_pct, left=[g + b for g, b in zip(green_pct, blue_pct)], color='red', label='None')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(headers)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Fill %')
        ax.set_title('Data Type Summary — Stacked Bar (Green/Blue/Red)')
        ax.legend(loc='lower right')

        #PlotManager.save_plot(fig, 'Datatype_Stacked_Summary', 'Datatype_Stacked_Summary')

        # Save
        filename = f"Datatype_Stacked_Summary_{label}_{DATE_TIME}.png"
        save_path = os.path.join(plots_base, filename)
        plt.savefig(save_path)
        print(f"[PlotManager] Saved: {save_path}")

        plt.close()


    @staticmethod
    def load_shapefiles():
        """
        Loads needed shapefiles due to the usage of pacjage cartopyPlots. 
        """
        primary = gpd.read_file("C:/Users/avtnl/Documents/HU/shapefiles/tl_2024_us_primaryroads.shp")
        secondary_paths = [
            "C:/Users/avtnl/Documents/HU/shapefiles/tl_2024_48113_roads.shp",
            "C:/Users/avtnl/Documents/HU/shapefiles/tl_2024_48201_roads.shp",
            "C:/Users/avtnl/Documents/HU/shapefiles/tl_2024_48029_roads.shp",
            "C:/Users/avtnl/Documents/HU/shapefiles/tl_2024_48439_roads.shp",
            "C:/Users/avtnl/Documents/HU/shapefiles/tl_2024_48453_roads.shp",
        ]
        secondary = pd.concat([gpd.read_file(p) for p in secondary_paths]).reset_index(drop=True)
        return primary, secondary


    @staticmethod
    def plot_cities_basic(df: pd.DataFrame) -> None:
        """
        Plots accidents for major Texas cities and overlays primary/secondary roads using shapefiles.
        Saves each plot as a PNG in the configured PLOTS_BASE folder.
        """
        print("[PlotManager] Starting city-level plotting with roads...")
        roads, secondary_roads = PlotManager.load_shapefiles()

        print("[PlotManager] Starting city-level plotting with roads...")

        # Load shapefiles
        roads = gpd.read_file("C:/Users/avtnl/Documents/HU/shapefiles/tl_2024_us_primaryroads.shp")
        secondary_paths = [
            "C:/Users/avtnl/Documents/HU/shapefiles/tl_2024_48113_roads.shp",
            "C:/Users/avtnl/Documents/HU/shapefiles/tl_2024_48201_roads.shp",
            "C:/Users/avtnl/Documents/HU/shapefiles/tl_2024_48029_roads.shp",
            "C:/Users/avtnl/Documents/HU/shapefiles/tl_2024_48439_roads.shp",
            "C:/Users/avtnl/Documents/HU/shapefiles/tl_2024_48453_roads.shp",
        ]
        secondary_roads = pd.concat([gpd.read_file(p) for p in secondary_paths]).reset_index(drop=True)

        # Filter for valid coordinates
        df['Start_Lat'] = pd.to_numeric(df['Start_Lat'], errors='coerce')
        df['Start_Lng'] = pd.to_numeric(df['Start_Lng'], errors='coerce')
        df = df[df['Start_Lat'].between(25, 36) & df['Start_Lng'].between(-107, -93)]

        # Cities to plot
        cities_to_plot = ['Houston', 'Dallas', 'Austin', 'San Antonio', 'Fort Worth']
        output_dir = os.path.join(PLOTS_BASE, "City_Plots")
        os.makedirs(output_dir, exist_ok=True)

        for city in cities_to_plot:
            df_city = df[df['City'] == city]
            if df_city.empty:
                print(f"[PlotManager] Skipping {city}: no data.")
                continue

            print(f"[PlotManager] Plotting {len(df_city)} points for {city}...")

            lat_min, lat_max = df_city['Start_Lat'].min() - 0.3, df_city['Start_Lat'].max() + 0.3
            lng_min, lng_max = df_city['Start_Lng'].min() - 0.3, df_city['Start_Lng'].max() + 0.3

            fig = plt.figure(figsize=(10, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent([lng_min, lng_max, lat_min, lat_max], crs=ccrs.PlateCarree())

            ax.add_feature(cfeature.STATES)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.add_feature(cfeature.LAND)
            ax.add_feature(cfeature.LAKES, alpha=0.5)
            ax.gridlines(draw_labels=True)

            # Plot roads
            roads_clipped = roads.cx[lng_min:lng_max, lat_min:lat_max]
            if not roads_clipped.empty:
                roads_clipped.plot(ax=ax, linewidth=1, edgecolor='black', transform=ccrs.PlateCarree())

            secondary_clipped = secondary_roads.cx[lng_min:lng_max, lat_min:lat_max]
            if not secondary_clipped.empty:
                secondary_clipped.plot(ax=ax, linewidth=0.5, edgecolor='gray', alpha=0.5, transform=ccrs.PlateCarree())

            # Plot accident points
            ax.scatter(
                df_city['Start_Lng'],
                df_city['Start_Lat'],
                color='blue',
                s=20,
                alpha=0.6,
                transform=ccrs.PlateCarree()
            )

            plt.title(f"{city} Accident Locations with Roads")
            filename = f"Plot_{city.replace(' ', '_')}_{DATE_TIME}.png"
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"[PlotManager] Saved: {save_path}")


    @staticmethod
    def plot_cities_compared(df: pd.DataFrame) -> None:
        """
        Plots mismatched speed types between Primary_Location and Street for major Texas cities,
        using colored markers and overlaying road shapefiles.

        Needs:
        - Coordinates of accidents
        - 'Primary_Structure'
        - 'Speed_Type_Location'
        - 'Speed_Type_Street'

        """
        print("[PlotManager] Starting comparative city-level plotting...")

        roads, secondary_roads = PlotManager.load_shapefiles()

        # Filter
        df = df.copy()
        df['Start_Lat'] = pd.to_numeric(df['Start_Lat'], errors='coerce')
        df['Start_Lng'] = pd.to_numeric(df['Start_Lng'], errors='coerce')
        df = df[df['Start_Lat'].between(25, 36) & df['Start_Lng'].between(-107, -93)]
        df = df[(df['Street_in_Description'] == False) & (df['Primary_Structure'] == 'on')]

        cities_to_plot = ['Houston', 'Dallas', 'Austin', 'San Antonio', 'Fort Worth']
        output_dir = os.path.join(PLOTS_BASE, "City_Plots_Compared")
        os.makedirs(output_dir, exist_ok=True)

        for city in cities_to_plot:
            df_city = df[df['City'] == city]
            if df_city.empty:
                print(f"[PlotManager] Skipping {city}: no valid data found.")
                continue

            print(f"[PlotManager] Plotting comparisons for {city}...")

            lat_min, lat_max = df_city['Start_Lat'].min() - 0.15, df_city['Start_Lat'].max() + 0.15
            lng_min, lng_max = df_city['Start_Lng'].min() - 0.15, df_city['Start_Lng'].max() + 0.15

            fig = plt.figure(figsize=(10, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent([lng_min, lng_max, lat_min, lat_max], crs=ccrs.PlateCarree())

            ax.add_feature(cfeature.STATES)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.add_feature(cfeature.LAND)
            ax.add_feature(cfeature.LAKES, alpha=0.5)
            ax.gridlines(draw_labels=True)

            # Plot comparison points
            df_custom = df_city[df_city['Location'].notna() & (df_city['Location'].str.strip() != '')]

            cond_green = (
                (df_custom['Speed_Type_Location'] == 'Low Speed') &
                (df_custom['Speed_Type_Street'] == 'High Speed')
            )
            cond_red = (
                (df_custom['Speed_Type_Location'] == 'High Speed') &
                (df_custom['Speed_Type_Street'] == 'Low Speed')
            )

            for cond, color, label in [
                (cond_red, 'red', 'PL: High & Str: Low'),
                (cond_green, 'green', 'PL: Low & Str: High')
            ]:
                subset = df_custom[cond]
                if not subset.empty:
                    ax.scatter(
                        subset['Start_Lng'],
                        subset['Start_Lat'],
                        color=color,
                        label=label,
                        s=5,
                        alpha=0.9,
                        transform=ccrs.PlateCarree()
                    )

            # Plot roads
            roads_clipped = roads.cx[lng_min:lng_max, lat_min:lat_max]
            if not roads_clipped.empty:
                roads_clipped.plot(ax=ax, linewidth=1, edgecolor='black', transform=ccrs.PlateCarree())

            secondary_clipped = secondary_roads.cx[lng_min:lng_max, lat_min:lat_max]
            if not secondary_clipped.empty:
                secondary_clipped.plot(ax=ax, linewidth=0.5, edgecolor='gray', alpha=0.5, transform=ccrs.PlateCarree())

            ax.legend(loc='upper right', fontsize='small', title='Type Mismatch')
            plt.title(f"{city} Accidents — Mismatches")

            output_path = os.path.join(output_dir, f"Plot_{city.replace(' ', '_')}_{DATE_TIME}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"[PlotManager] Saved: {output_path}")


    @staticmethod
    def save_plot(fig, title: str, filename: str):
        """
        Saves a matplotlib figure to a PNG file in the PLOTS_BASE directory.

        Args:
            fig (matplotlib.figure.Figure): The figure to save.
            title (str): Title for potential labeling (currently unused).
            filename (str): Base filename (without extension or timestamp).
        """
        from pathlib import Path
        from datetime import datetime
        import matplotlib.pyplot as plt

        # Set these constants appropriately (or inject them if defined elsewhere)
        PLOTS_BASE = Path("plots")  # Change as needed
        DATE_TIME = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Ensure the directory exists
        PLOTS_BASE.mkdir(parents=True, exist_ok=True)

        # Compose full save path
        save_path = PLOTS_BASE / f"{filename}_{DATE_TIME}.png"

        # Save and report
        fig.savefig(save_path)
        plt.close(fig)
        print(f"[PLOT SAVED] {save_path.as_posix()}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}][PlotManager] Plot saved as: {save_path.as_posix()}")


    @staticmethod
    def plot_tasks(df: pd.DataFrame) -> None:
        PlotManager.plot_day_of_week_distribution(df)
        PlotManager.plot_bankholiday_distribution(df)
        PlotManager.plot_rush_hour_distribution(df)
        PlotManager.plot_duration_distribution(df, 'Duration_Start_to_End(min)', 'Duration Start to End', 'Duration_Start_to_End_Distribution')
        PlotManager.plot_duration_distribution(df, 'Duration_Start_to_Timestamp(min)', 'Duration to Start to Weather_Timestamp', 'Duration_Start_to_Weather_Timstamp_Distribution')
        PlotManager.plot_cities_basic(df)