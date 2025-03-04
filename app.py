
import requests
import pandas as pd
import numpy as np
import streamlit as st
import time
import emoji
from functools import lru_cache
from meteostat import Daily, Stations, Point
from datetime import datetime, timedelta
from streamlit_folium import folium_static
from geopy.distance import geodesic
from itertools import permutations
from scipy.spatial.distance import cdist
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium
import folium

# FUNCTIONS

# Function to fetch weather data for a specific day
def fetch_weather_data(lat, lon, date):
    # Create a Point object for the location
    location = Point(lat, lon)

    # Fetch daily weather data
    data = Daily(location, date, date)
    data = data.fetch()

    return data

# Function to calculate the average weather
def calculate_daily_average_weather(data):

    # Calculate average temperature, humidity, etc.
    avg_temp = data['tavg'].dropna().mean()  # Average temperature
    max_temp = data['tmax'].dropna().mean()  # Maximum temperature
    min_temp = data['tmin'].dropna().mean()  # Minimum temperature
    avg_precip = data['prcp'].dropna().mean()  # Average precipitation

    return avg_temp, max_temp, min_temp, avg_precip

# Function to define driving distances
def get_osrm_distance_matrix(locations):
    """
    Fetches the driving distance matrix (in km) between multiple locations using OSRM.

    Parameters:
        locations (list): List of dicts with 'lat' and 'lon' keys.

    Returns:
        np.ndarray: NxN matrix of driving distances in kilometers.
    """
    base_url = "http://router.project-osrm.org/table/v1/driving/"

    # Convert location coordinates to OSRM format (lon,lat)
    coordinates = ";".join(f"{loc['lon']},{loc['lat']}" for loc in locations)
    url = f"{base_url}{coordinates}?annotations=distance"

    # Send request to OSRM
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return np.array(data["distances"]) / 1000  # Convert meters to km
    else:
        print("Error fetching data:", response.text)
        return None

# Function to solve traveling salesman problem
def solve_tsp_two_options(locations, distance_matrix, rain_data, start_date, rain_threshold):
    """
    Solves the Traveling Salesman Problem (TSP) considering rain penalties and calculates the total distance.

    Parameters:
        locations (list): A list of location dictionaries.
        distance_matrix (list): A matrix of distances between locations.
        rain_data (dict): A dictionary with daily rainfall data for each location.
        start_date (str): The start date of the trip.
        rain_threshold (float): The rainfall threshold (mm) for applying penalty.

    Returns:
        tuple: The best route with rain penalty applied and the total distance of the route.
    """
    # Replace NOne with a large number to prevent selection
    processed_matrix = np.where(distance_matrix == None, 10**6, distance_matrix)

    # Set up the TSP problem using OR-Tools
    num_locations = len(processed_matrix)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(num_locations, 1, 0)

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        # Returns the distance between the two nodes.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Define search parameters.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 30  # Adjust as needed

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    def extract_route(solution):
        index = routing.Start(0)
        route = []
        total_distance = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(locations[node]["name"])
            next_index = solution.Value(routing.NextVar(index))
            total_distance += processed_matrix[node][manager.IndexToNode(next_index)]
            index = next_index
        route.append(locations[manager.IndexToNode(index)]["name"])
        return route, total_distance

    # Extract best route
    initial_route, initial_distance = extract_route(solution)

    # Solve again for the second-best route
    first_solution = routing.SolveWithParameters(search_parameters)
    if first_solution:
        first_route, first_distance = extract_route(first_solution)
        print(first_route)

    else:
      #print("no first solution found")
      None

    # Prevent the solver from picking the exact best route again
    routing.CloseModel()
    for i in range(len(first_route) - 1):
        node1 = first_route[i]
        node2 = first_route[i + 1]
        routing.solver().Add(routing.NextVar(manager.NodeToIndex(locations.index(next(item for item in locations if item["name"] == node1)))) != manager.NodeToIndex(locations.index(next(item for item in locations if item["name"] == node2))))


    # Solve again for the second route
    second_solution = routing.SolveWithParameters(search_parameters)
    if second_solution:
        second_route, second_distance = extract_route(second_solution)
        print(second_route)

    else:
      #print("no second solution found")
      None


    return first_route, first_distance, second_route, second_distance

# Function to calculate total rainfall on specific route
def calculate_total_rainfall(locations, route, start_date, rain_data):
    """
    Calculates the total rainfall for a specific route.

    Parameters:
        route (list): A list of locations with their stay days.
        start_date (date): The start date of the trip.
        rain_data (dict): A dictionary with daily rainfall data for each location.

    Returns:
        float: The total rainfall (mm) for the given route.
    """
    total_rainfall = 0
    current_date = start_date

    # Create a dictionary for quick lookup
    location_dict = {loc["name"]: loc for loc in locations}
    reordered_locations = [location_dict[name] for name in route if name in location_dict]

    for loc in reordered_locations:
        location = loc['name']
        days = loc['days']

        for i in range(days):
            day = (current_date + timedelta(days=i)).strftime("%Y-%m-%d")
            if day in rain_data.get(location, {}):
                total_rainfall += rain_data[location][day]['avg_precip']
            current_date += timedelta(days=1)

    return total_rainfall

# Function to apply penalty if rainfall is too high
def penalty_function(locations, route1, distance_1, route2, distance_2, start_date, rain_data, rain_threshold):
    """
    Applies a penalty if the rainfall of the first route is higher than the second.

    Parameters:
        route1 (list): The first route option.
        route2 (list): The second route option.
        start_date (date): The start date of the trip.
        rain_data (dict): A dictionary with daily rainfall data for each location.

    Returns:
        list: The selected route (either route1 or route2), or None if both are skipped.
    """
    # Calculate total rainfall for both routes
    route1_rainfall = calculate_total_rainfall(locations, route1, start_date, rain_data)
    route2_rainfall = calculate_total_rainfall(locations, route2, start_date, rain_data)

    #print(f"Route 1 total rainfall: {route1_rainfall:.2f} mm")
    #print(f"Route 2 total rainfall: {route2_rainfall:.2f} mm")

    penalty_note = ""

    # Only apply penalty if route1 rainfall exceeds threshold
    if route1_rainfall > rain_threshold:
        # Skip route1 if its rainfall is higher than route2
        if route1_rainfall > route2_rainfall:
            penalty_note = f" Most optimal route is expected to have {route1_rainfall:.2f} mm of rainfall. Chosing second best route to go decrease rainfall to {route2_rainfall:.2f} mm"
            return route2, distance_2, penalty_note  # Return second route as the best option

    # Otherwise, choose route1
    #print("Route 2 has more rainfall than Route 1 or route 1 does not reach threshold.")
    return route1, distance_1, penalty_note

def plot_route(locations, start_date, weather_data):
    
    # Initialize the map at the first location
    m = folium.Map(location=(locations[0]["lat"], locations[0]["lon"]), zoom_start=6)

    current_date = start_date

    # Plot each location as a marker
    for location in locations:

        # Create dates for location
        end_date = current_date + timedelta(days=location["days"] - 1)
        stay_dates = [(current_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(location["days"])]
        date_range = f"{current_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}"
        current_date = end_date

        # Fetch weather data for the location and compute averages
        if location["name"] in weather_data:
            weather_entries = [weather_data[location["name"]].get(date, {}) for date in stay_dates]

            # Extract values and compute averages, ignoring missing (NA) values
            avg_temp = np.nanmean([entry.get("avg_temp", np.nan) for entry in weather_entries])
            max_temp = np.nanmean([entry.get("max_temp", np.nan) for entry in weather_entries])
            min_temp = np.nanmean([entry.get("min_temp", np.nan) for entry in weather_entries])
            avg_precip = np.nanmean([entry.get("avg_precip", np.nan) for entry in weather_entries])

            weather_info = f"""
            Avg Temp: {avg_temp:.1f}°C<br>
            Max Temp: {max_temp:.1f}°C<br>
            Min Temp: {min_temp:.1f}°C<br>
            Avg Daily Precip: {avg_precip:.1f} mm
            """
        else:
            weather_info = "Weather data unavailable"

        # Create an HTML popup
        popup_html = f"""
        <div style="width: 250px; text-align: center;">
            <b style="font-size:14px;">{location["name"]}</b><br>
            <span style="font-size:12px;">{date_range}</span><br>
            <hr>
            {weather_info}
        </div>
        """
        
        folium.Marker(
            location=(location["lat"], location["lon"]),
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(m)
        
    # Connect the locations with a polyline
    route_coordinates = [(location["lat"], location["lon"]) for location in locations]
    folium.PolyLine(route_coordinates, color="blue", weight=2.5).add_to(m)

    return m


# Main function to optimize route
def optimize_route(locations, start_date, rain_threshold_input):

  # Define start location
  start_location = locations[0]

  # Calculate end date of trip
  date_str = start_date.strftime("%Y-%m-%d")
  start_date = datetime.strptime(date_str, "%Y-%m-%d") 
  length_of_trip = sum(location["days"] for location in locations)
  end_date = start_date + timedelta(days=sum(location["days"] for location in locations))

  # Make list of dates to consider
  selected_dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end_date - start_date).days + 1)]

  # Loop through each location and fetch daily weather data
  weather_data = {}
  for location in locations:
       name = location["name"]
       lat = location["lat"]
       lon = location["lon"]

       # Store daily weather data for each date of the trip
       location_weather = {}
       for i in range(length_of_trip):
            trip_date = start_date + timedelta(days=i)

            # Fetch weather data for the specific day for the past 5 years
            historical_data = pd.DataFrame()
            for year in [2019, 2020, 2021, 2022, 2023, 2024]:
               date_in_past = trip_date.replace(year=year)
               data = fetch_weather_data(lat, lon, date_in_past)
               historical_data = pd.concat([historical_data, data], ignore_index=True)

            # Calculate the daily average weather conditions
            avg_temp, max_temp, min_temp, avg_precip = calculate_daily_average_weather(historical_data)

            # Store the results
            location_weather[trip_date.strftime("%Y-%m-%d")] = {
                 "avg_temp": avg_temp,
                 "max_temp": max_temp,
                 "min_temp": min_temp,
                 "avg_precip": avg_precip
            }

       weather_data[name] = location_weather

  # Calculate driving distances
  dist_matrix = get_osrm_distance_matrix(locations)
  # Retrieve 2 best route options
  rain_threshold = rain_threshold_input
  first_route, first_distance, second_route, second_distance = solve_tsp_two_options(locations, dist_matrix, weather_data, start_date, rain_threshold)
  # Define best route based on historic rain
  best_route_pen, best_distance_pen, penalty_note = penalty_function(locations, first_route, first_distance, second_route, second_distance, start_date, weather_data, rain_threshold)

  # Create a dictionary for quick lookup
  location_dict = {loc["name"]: loc for loc in locations}
  # Format solution
  reordered_locations = [location_dict[name] for name in best_route_pen if name in location_dict]

  # Plot onto map
  map = plot_route(reordered_locations, start_date, weather_data)

  return best_route_pen, map, penalty_note

# DESCRIPTION
st.title("Holiday Travel Planner")
st.write(emoji.emojize(":sun_with_face: :calendar: This app helps you optimize your road trip by minimizing travel distance and considering expected weather conditions such as rainfall. No more endlessly checken traveling distances and weather forecasts, just do it in one click with your new holiday optimizer! :sun_with_face: :calendar:"))

# Add instructions before input
st.subheader("Enter Your Trip Details")
st.markdown(emoji.emojize(" :bullseye: :motorway: Please provide your start location, the locations you want to visit, number of days at each stop, and the start date of your trip. Our Traveling Salesman Algorithm will calculate the most efficient traveling route while also considering weather expectations. If the most optimal route is expected to have a lot of rainfall, another route is recommended. 	:bullseye: :motorway: "))


# USER INPUT

# User input fields
start_location = st.text_area("Enter starting location")
locations_input = st.text_area("Enter locations to visit (comma-separated):")
days_input = st.text_input("Enter days per location (comma-separated):")
start_date = st.date_input("Select the start date of your trip:")
rain_threshold_input = 10
geolocator = Nominatim(user_agent="road_trip_optimizer")
@lru_cache(maxsize=100) 

def get_coordinates(location_name):
    try:
        location = geolocator.geocode(location_name)
        time.sleep(1)
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        st.error(f"Error fetching coordinates for {location_name}: {e}")
    return None, None

if st.button("Optimize Route"):
    if locations_input and days_input and start_date:
        loc_list = locations_input.split(",")
        days_list = list(map(int, days_input.split(",")))
        loc_list.insert(0, start_location)
        days_list.insert(0, 1)

        if len(loc_list) == len(days_list):
            locations = []
            for i, loc in enumerate(loc_list):
                lat, lon = get_coordinates(loc.strip())
                if lat is not None and lon is not None:
                    locations.append({"name": loc.strip(), "lat": lat, "lon": lon, "days": days_list[i]})
                else:
                    st.error(f"Could not determine coordinates for {loc.strip()}")
                    break

            if len(locations) == len(loc_list):  # Ensure all locations were found
                optimized_route, route_map, penalty_note = optimize_route(locations, start_date, rain_threshold_input)
                st.write("Optimized Route:", optimized_route)
                st.write(penalty_note)
                folium_static(route_map)  # Display map
        else:
            st.error("Mismatch between locations and days entered!")
    else:
        st.error("Please enter all required information!")#
