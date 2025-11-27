import streamlit as st
import joblib
import pandas as pd
import requests
import time
from typing import Dict, Any, List, Optional, Tuple

# --- 1. CONFIGURATION ---

# IMPORTANT: This MUST be the path to your new model trained on ONLY the 7 features.
MODEL_PATH = "live_model_7features.joblib" 
API_KEY = "907eef2c-c869-43d0-ada8-fb17eba1b199" # Your CricAPI key
BASE_URL = "https://api.cricapi.com/v1"
REFRESH_INTERVAL_SECONDS = 20 # How often to refresh live data

#
# !!! CRITICAL !!!
# You MUST replace this placeholder with your actual venue_rating mapping.
# This map should be generated from your training notebook and saved.
# e.g., VENUE_RATING_MAP = pd.read_csv('venue_ratings.csv').set_index('venue')['rating'].to_dict()
#
VENUE_RATING_MAP = {
    'Wanderers Ground': 2,
    'Wanderers Namibia': 2,
    'GMHBA Stadium, South Geelong, Victoria': 1,
    'Sabina Park': 2,
    'Eden Gardens': 2,
    'MA Chidambaram Stadium, Chepauk, Chennai': 4,
    'Dubai International Cricket Stadium': 2,
    'Gaddafi Stadium': 1,
    'Gymkhana Club Ground, Nairobi': 3,
    'Sophia Gardens': 3,
    'Harare Sports Club': 2,
    'Providence Stadium, Guyana': 3,
    'ICC Global Cricket Academy': 2,
    'United Cricket Club Ground, Windhoek': 3,
    'ICC Academy Ground No 2': 2,
    'Al Amerat Cricket Ground Oman Cricket (Ministry Turf 1)': 3,
    'SuperSport Park, Centurion': 4,
    'Rangiri Dambulla International Stadium': 2,
    'R Premadasa Stadium': 3,
    'Entebbe Cricket Oval': 1,
    'Sheikh Zayed Stadium': 3,
    'National Cricket Stadium Grenada': 1,
    'Gahanga International Cricket Stadium. Rwanda': 4,
    'Eden Park': 2,
    'Shere Bangla National Stadium, Mirpur': 2,
    'Newlands': 3,
    'Edgbaston, Birmingham': 0,
    'Nassau County International Cricket Stadium, New York': 3,
    'The Village, Malahide': 2,
    'Zahur Ahmed Chowdhury Stadium': 2,
    'Grange Cricket Club Ground, Raeburn Place, Edinburgh': 2,
    'Mission Road Ground, Mong Kok, Hong Kong': 2,
    'Terdthai Cricket Ground': 2,
    'Achimota Senior Secondary School A Field, Accra': 2,
    'Pallekele International Cricket Stadium': 2,
    'Kinrara Academy Oval': 1,
    'Stadium Australia': 2,
    'Vidarbha Cricket Association Stadium, Jamtha': 1,
    'Melbourne Cricket Ground': 3,
    'Zayed Cricket Stadium, Abu Dhabi': 3,
    'Sir Vivian Richards Stadium, North Sound, Antigua': 3,
    'Tribhuvan University International Cricket Ground': 2,
    'ICC Academy': 1,
    'Windsor Park, Roseau': 2,
    'National Stadium Karachi': 1,
    'Edgbaston': 0,
    'Sportpark Westvliet': 2,
    'Mulpani Cricket Ground': 3,
    'Shere Bangla National Stadium': 2,
    'Kensington Oval': 2,
    'Warner Park, Basseterre, St Kitts': 4,
    'Central Broward Regional Park Stadium Turf Ground': 1,
    'Bellerive Oval, Hobart': 2,
    'Gahanga International Cricket Stadium, Rwanda': 2,
    'Castle Avenue, Dublin': 4,
    'Marrara Stadium, Darwin': 0,
    'Brisbane Cricket Ground, Woolloongabba, Brisbane': 0,
    'Kennington Oval': 2,
    'Bready Cricket Club, Magheramason, Bready': 0,
    'Punjab Cricket Association Stadium, Mohali': 4,
    'Headingley, Leeds': 0,
    'Willowmoore Park, Benoni': 4,
    'UKM-YSD Cricket Oval, Bangi': 4,
    'University of Doha for Science and Technology': 4,
    'Khan Shaheb Osman Ali Stadium': 2,
    'Beausejour Stadium, Gros Islet': 2,
    'Al Amerat Cricket Ground Oman Cricket (Ministry Turf 2)': 2,
    'County Ground': 2,
    'Hagley Oval, Christchurch': 3,
    'MA Chidambaram Stadium, Chepauk': 2,
    'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium': 0,
    'Selangor Turf Club, Kuala Lumpur': 2,
    'Mangaung Oval': 0,
    'Providence Stadium': 3,
    'Westpac Stadium': 2,
    'Bay Oval': 0,
    'ICC Academy, Dubai': 3,
    'West End Park International Cricket Stadium, Doha': 2,
    'Bay Oval, Mount Maunganui': 1,
    'Indian Association Ground': 3,
    'YSD-UKM Cricket Oval, Bangi': 4,
    'Maharashtra Cricket Association Stadium, Pune': 0,
    'Old Trafford': 3,
    'Riverside Ground, Chester-le-Street': 2,
    'Mission Road Ground, Mong Kok': 2,
    'Seddon Park': 2,
    'Sharjah Cricket Stadium': 2,
    'Kensington Oval, Bridgetown': 1,
    'Kingsmead': 2,
    'Lugogo Cricket Oval': 0,
    'Perth Stadium': 2,
    'Bellerive Oval': 1,
    'Sportpark Maarschalkerweerd, Utrecht': 2,
    'Prairie View Cricket Complex': 1,
    'Sikh Union Club Ground, Nairobi': 3,
    'Wankhede Stadium': 3,
    'Warner Park, St Kitts': 3,
    'Grand Prairie Stadium, Dallas': 4,
    'The Rose Bowl, Southampton': 0,
    'Maple Leaf North-West Ground, King City': 3,
    'Carrara Oval': 2,
    'Riverside Ground': 1,
    'Sylhet Stadium': 4,
    'Adelaide Oval': 2,
    'SuperSport Park': 1,
    'Sydney Cricket Ground': 2,
    'Mombasa Sports Club Ground': 3,
    'Barsapara Cricket Stadium, Guwahati': 2,
    'Holkar Cricket Stadium': 1,
    'Arnos Vale Ground': 0,
    'Integrated Polytechnic Regional Centre': 2,
    'Rajiv Gandhi International Stadium, Uppal': 4,
    'Queens Sports Club, Bulawayo': 2,
    'Brabourne Stadium': 4,
    'Shrimant Madhavrao Scindia Cricket Stadium, Gwalior': 4,
    'Singapore National Cricket Ground': 4,
    'Brisbane Cricket Ground, Woolloongabba': 2,
    'The Rose Bowl': 2,
    'Barabati Stadium': 2,
    'Hazelaarweg': 1,
    'Bulawayo Athletic Club': 4,
    'Wanderers Cricket Ground': 4,
    'Tafawa Balewa Square Cricket Oval, Lagos': 3,
    'Trent Bridge': 2,
    'Buffalo Park': 1,
    'TCA Oval, Blantyre': 4,
    'Sky Stadium, Wellington': 4,
    'Gymkhana Club Ground': 3,
    'Coolidge Cricket Ground, Antigua': 3,
    'Daren Sammy National Cricket Stadium, Gros Islet, St Lucia': 2,
    'Sylhet International Cricket Stadium': 2,
    'Brian Lara Stadium, Tarouba, Trinidad': 2,
    'Rajiv Gandhi International Stadium, Uppal, Hyderabad': 2,
    'Zhejiang University of Technology Cricket Field': 2,
    'The Village, Malahide, Dublin': 1,
    'Senwes Park': 2,
    'Himachal Pradesh Cricket Association Stadium, Dharamsala': 4,
    'VRA Ground': 2,
    'Tolerance Oval': 3,
    'Bayuemas Oval, Kuala Lumpur': 3,
    'Kyambogo Cricket Oval': 3,
    'Titwood, Glasgow': 2,
    'Central Broward Regional Park Stadium Turf Ground, Lauderhill': 2,
    'JSCA International Stadium Complex': 2,
    'Kingsmead, Durban': 2,
    'Bready': 4,
    'Malahide, Dublin': 2,
    'Arun Jaitley Stadium': 2,
    'Manuka Oval': 2,
    'Rawalpindi Cricket Stadium': 4,
    "Queen's Park Oval": 2,
    "Lord's": 1,
    'Saurashtra Cricket Association Stadium': 3,
    "St George's Park": 1,
    "Cazaly's Stadium, Cairns": 4,
    'Ruaraka Sports Club Ground, Nairobi': 2,
    'Barsapara Cricket Stadium': 4,
    'M Chinnaswamy Stadium': 3,
    'Warner Park, Basseterre': 4,
    'JSCA International Stadium Complex, Ranchi': 2,
    'Maharashtra Cricket Association Stadium': 2,
    'Jimmy Powell Oval, Cayman Islands': 4,
    'Narendra Modi Stadium': 3,
    'Namibia Cricket Ground, Windhoek': 4,
    'Greenfield International Stadium': 2,
    'Seddon Park, Hamilton': 0,
    'Punjab Cricket Association IS Bindra Stadium, Mohali': 2,
    'Civil Service Cricket Club, Stormont, Belfast': 1,
    'Saxton Oval, Nelson': 0,
    'Tony Ireland Stadium': 0,
    "St George's Park, Gqeberha": 4,
    'Grange Cricket Club Ground, Raeburn Place': 3,
    'Moses Mabhida Stadium': 0,
    'Grange Cricket Club, Raeburn Place': 0,
    'Vidarbha Cricket Association Stadium, Jamtha, Nagpur': 4,
    'Himachal Pradesh Cricket Association Stadium': 1,
    'Green Park': 4,
    'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow': 2,
    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam': 2,
    'University Oval': 0,
    'Sky Stadium': 4,
    'Civil Service Cricket Club, Stormont': 4,
    'Old Trafford, Manchester': 2,
    'Sir Vivian Richards Stadium, North Sound': 1,
    'Bready Cricket Club, Magheramason': 3,
    'Sheikh Abu Naser Stadium': 1,
    'University Oval, Dunedin': 3,
    'Mahinda Rajapaksa International Cricket Stadium, Sooriyawewa': 1,
    'White Hill Field, Sandys Parish': 3,
    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium': 4,
    'Arun Jaitley Stadium, Delhi': 2,
    'Barabati Stadium, Cuttack': 4,
    'Queens Sports Club': 0,
    'Kennington Oval, London': 4,
    'M.Chinnaswamy Stadium': 4,
    'Udayana Cricket Ground': 4,
    'Maple Leaf North-West Ground': 4,
    'Boland Park': 4,
    'Subrata Roy Sahara Stadium': 4,
    'Western Australia Cricket Association Ground': 2,
    'Achimota Senior Secondary School B Field, Accra': 3,
    'Windsor Park, Roseau, Dominica': 0,
    'Darren Sammy National Cricket Stadium, St Lucia': 4,
    'Saxton Oval': 0,
    'P Sara Oval': 4,
    'Sawai Mansingh Stadium, Jaipur': 4,
    'Sportpark Het Schootsveld': 0,
    'Feroz Shah Kotla': 3,
    'OUTsurance Oval': 4,
    'Shaheed Veer Narayan Singh International Stadium, Raipur': 0,
    'Arnos Vale Ground, Kingstown': 2,
    'AMI Stadium': 2,
    'Narendra Modi Stadium, Ahmedabad': 0,
    'Simonds Stadium, South Geelong': 4,
    'United Cricket Club Ground': 4,
    'Saurashtra Cricket Association Stadium, Rajkot': 0,
    'University of Lagos Cricket Oval': 4,
    'Jade Stadium': 0,
    'Hagley Oval': 2,
    'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh': 4,
    'Goldenacre, Edinburgh': 4,
    'McLean Park': 2,
    'De Beers Diamond Oval': 0,
    'Jinja Cricket Ground': 2,
    'Indian Association Ground, Singapore': 4,
    'McLean Park, Napier': 4,
    'John Davies Oval, Queenstown': 4,
    'Sardar Patel Stadium, Motera': 0,
    'Nigeria Cricket Federation Oval 1, Abuja': 4,
    'M Chinnaswamy Stadium, Bengaluru': 0,

    # --- CRITICAL FALLBACK ---
    # This 'default' key is essential. It handles any new venues
    # from the live API that were not in your training data.
    # We use '2' as it is the neutral/average rating in your 0-4 scale.
    'default': 2
}

# --- 2. MODEL & DATA LOADING (Cached) ---

# ... (Keep your imports and View 1 code the same) ...

# --- VIEW 2: LIVE PREDICTION DASHBOARD ---
else:
    if not model:
        st.error("Model could not be loaded. Dashboard cannot run.")
        if st.button("Go Back"):
            st.session_state.selected_match_id = None
            st.rerun()
    else:
        # --- Dashboard Header ---
        st.title(st.session_state.get('match_name', 'Live Win Probability'))
        if st.button("Change Match"):
            st.session_state.selected_match_id = None
            st.rerun()
            
        # --- Auto-Refreshing Dashboard ---
        placeholder = st.empty()
        
        while st.session_state.selected_match_id:
            
            # ---------------------------------------------------------
            # CORRECT FETCHING LOGIC START
            # We fetch the entire list again because 'match_score' endpoint 
            # is often restricted. We find our match inside this list.
            # ---------------------------------------------------------
            score_data = None
            try:
                # Use the SAME endpoint that worked for the dropdown
                url = f"{BASE_URL}/currentMatches"
                params = {"apikey": API_KEY, "offset": 0}
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                all_matches = resp.json().get('data', [])
                
                # Find our specific match in the list by ID
                for match in all_matches:
                    if match.get('id') == st.session_state.selected_match_id:
                        score_data = match
                        break
            except Exception as e:
                print(f"Error fetching data: {e}") # Check your terminal for this!
            # ---------------------------------------------------------
            # CORRECT FETCHING LOGIC END
            # ---------------------------------------------------------
            
            if not score_data:
                with placeholder.container():
                    st.warning("Fetching live score... (Will retry in 20s)")
                    # Debugging help:
                    st.caption("Check your terminal/console logs for error details.")
                time.sleep(REFRESH_INTERVAL_SECONDS)
                continue

            # Parse the data
            parsed_data = parse_live_data(score_data)
            status = parsed_data['status']
            stats = parsed_data['stats']

            # Use the placeholder to draw the dashboard
            with placeholder.container():
                
                # --- A. Display Live Stats ---
                st.header("Live Match State")
                st.subheader(stats.get("message", "Loading..."))
                
                if "batting_team" in stats:
                    st.text(f"Venue: {stats.get('venue')}")
                    
                    cols = st.columns(4)
                    cols[0].metric("Chasing Team", stats.get('batting_team', 'N/A'))
                    cols[1].metric("Current Score", stats.get('score_str', 'N/A'))
                    cols[2].metric("Target", stats.get('target', 'N/A'))
                    cols[3].metric("Defending Team", stats.get('bowling_team', 'N/A'))

                # --- B. Display Prediction ---
                st.divider()
                st.header("Win Probability")

                if status == "IN_PLAY":
                    model_input_df = parsed_data['model_input']
                    probabilities = model.predict_proba(model_input_df)
                    win_prob = probabilities[0][1]
                    loss_prob = probabilities[0][0]

                    prob_df = pd.DataFrame({
                        "Team": [stats['batting_team'], stats['bowling_team']],
                        "Win Probability": [win_prob, loss_prob]
                    }).set_index("Team")
                    
                    st.bar_chart(prob_df, horizontal=True)

                    st.subheader("Key Predictive Factors")
                    cols = st.columns(4)
                    cols[0].metric("Runs Required", stats.get('runs_required', 'N/A'))
                    cols[1].metric("Balls Left", stats.get('balls_left', 'N/A'))
                    cols[2].metric("Wickets Remaining", stats.get('wickets_remaining', 'N/A'))
                    cols[3].metric("Required Run Rate", stats.get('required_run_rate', 'N/A'))
                    
                elif status == "WAITING":
                    st.info(stats.get("message", "Waiting for match to enter 2nd innings."))
                
                elif status == "COMPLETE":
                    st.success(stats.get("message", "Match has ended."))
                    st.session_state.selected_match_id = None 
                
                if st.session_state.selected_match_id:
                    st.text(f"Refreshing in {REFRESH_INTERVAL_SECONDS} seconds...")
            
            time.sleep(REFRESH_INTERVAL_SECONDS)


