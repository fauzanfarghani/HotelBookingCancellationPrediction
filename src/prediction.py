import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def app():
    st.title('Hotel Reservation Cancellation Prediction')

    # Load the saved Random Forest pipeline
    try:
        model = joblib.load('rf_tuned_pipeline.pkl')
        st.write('Model loaded successfully.')
    except FileNotFoundError:
        st.error("Error: rf_tuned_pipeline.pkl not found. Make sure the file is in the same directory or provide the correct path.")
        st.stop()

    # Define the feature columns based on the training dataset
    feature_columns = [
        'no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
        'type_of_meal_plan', 'required_car_parking_space', 'room_type_reserved',
        'lead_time', 'arrival_year', 'arrival_month', 'arrival_date',
        'market_segment_type', 'repeated_guest', 'no_of_previous_cancellations',
        'no_of_previous_bookings_not_canceled', 'avg_price_per_room',
        'no_of_special_requests'
    ]

    # Create input form
    st.header('Enter Reservation Details')
    with st.form("reservation_form"):
        no_of_adults = st.number_input('Number of Adults', min_value=0, max_value=10, value=2)
        no_of_children = st.number_input('Number of Children', min_value=0, max_value=10, value=0)
        no_of_weekend_nights = st.number_input('Number of Weekend Nights', min_value=0, max_value=7, value=1)
        no_of_week_nights = st.number_input('Number of Week Nights', min_value=0, max_value=7, value=2)
        type_of_meal_plan = st.selectbox('Type of Meal Plan', ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'], index=0)
        required_car_parking_space = st.checkbox('Required Car Parking Space')
        room_type_reserved = st.selectbox('Room Type Reserved', ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'], index=0)
        lead_time = st.number_input('Lead Time (days)', min_value=0, max_value=500, value=30)
        arrival_year = st.number_input('Arrival Year', min_value=2017, max_value=2023, value=2018)
        arrival_month = st.number_input('Arrival Month', min_value=1, max_value=12, value=7)
        arrival_date = st.number_input('Arrival Date', min_value=1, max_value=31, value=15)
        market_segment_type = st.selectbox('Market Segment Type', ['Online', 'Offline', 'Corporate', 'Complementary', 'Aviation', 'TA/TO'], index=0)
        repeated_guest = st.checkbox('Repeated Guest')
        no_of_previous_cancellations = st.number_input('Number of Previous Cancellations', min_value=0, max_value=10, value=0)
        no_of_previous_bookings_not_canceled = st.number_input('Number of Previous Bookings Not Canceled', min_value=0, max_value=20, value=0)
        avg_price_per_room = st.number_input('Average Price per Room', min_value=0.0, max_value=1000.0, value=100.0)
        no_of_special_requests = st.number_input('Number of Special Requests', min_value=0, max_value=5, value=1)

        submitted = st.form_submit_button("Predict Cancellation")

    if submitted:
        # Create DataFrame from input values
        input_data = pd.DataFrame([
            {
                'no_of_adults': no_of_adults,
                'no_of_children': no_of_children,
                'no_of_weekend_nights': no_of_weekend_nights,
                'no_of_week_nights': no_of_week_nights,
                'type_of_meal_plan': type_of_meal_plan,
                'required_car_parking_space': int(required_car_parking_space),
                'room_type_reserved': room_type_reserved,
                'lead_time': lead_time,
                'arrival_year': arrival_year,
                'arrival_month': arrival_month,
                'arrival_date': arrival_date,
                'market_segment_type': market_segment_type,
                'repeated_guest': int(repeated_guest),
                'no_of_previous_cancellations': no_of_previous_cancellations,
                'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
                'avg_price_per_room': avg_price_per_room,
                'no_of_special_requests': no_of_special_requests
            }
        ])

        # Ensure all required columns are present
        input_data = input_data[feature_columns]

        # Make predictions
        predictions = model.predict(input_data)

        # Decode predictions (assuming LabelEncoder was used for 'booking_status')
        le = LabelEncoder()
        le.classes_ = np.array(['Not_Canceled', 'Canceled'])  # Based on training notebook
        decoded_predictions = le.inverse_transform(predictions)

        # Display results
        st.header('Prediction Result')
        st.write(f'The predicted booking status is: **{decoded_predictions[0]}**')