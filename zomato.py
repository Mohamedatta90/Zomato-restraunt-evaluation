import streamlit as st
import pandas as pd
import joblib

# Load the model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Set up the Streamlit app
st.set_page_config(page_title='Restaurant Evaluation')
st.header('Restaurant Evaluation')
st.image('https://img.etimg.com/thumb/msid-95349712,width-650,height-488,imgsize-45590,resizemode-75/zomato.jpg')

# Function to sort and clean input
def feature_sorting(x):
    x = x.split(',')  
    x = [item.strip() for item in x]  
    x = sorted(x) 
    return ', '.join(x)  

# Collect inputs
# Online order
online_order_options = ['Yes', 'No']
selected_online_order = st.radio("Do you support online orders?", online_order_options)

# Book table
book_table_options = ['Yes', 'No']
selected_book_table = st.radio("Do you support table bookings?", book_table_options)

# Location
locations = ['Banashankari', 'Basavanagudi', 'Mysore Road', 'Jayanagar', 'Kumaraswamy Layout', 'Rajarajeshwari Nagar', 'Vijay Nagar',
    'Uttarahalli', 'JP Nagar', 'South Bangalore', 'City Market', 'Nagarbhavi', 'Bannerghatta Road', 'BTM', 'Kanakapura Road',
    'Bommanahalli', 'CV Raman Nagar', 'Electronic City', 'HSR', 'Marathahalli', 'Sarjapur Road', 'Wilson Garden', 'Shanti Nagar',
    'Koramangala 5th Block', 'Koramangala 8th Block', 'Richmond Road', 'Koramangala 7th Block', 'Jalahalli', 'Koramangala 4th Block',
    'Bellandur', 'Whitefield', 'East Bangalore', 'Old Airport Road', 'Indiranagar', 'Koramangala 1st Block', 'Frazer Town', 'RT Nagar',
    'MG Road', 'Brigade Road', 'Lavelle Road', 'Church Street', 'Ulsoor', 'Residency Road', 'Shivajinagar', 'Infantry Road',
    'St. Marks Road', 'Cunningham Road', 'Race Course Road', 'Commercial Street', 'Vasanth Nagar', 'HBR Layout', 'Domlur',
    'Ejipura', 'Jeevan Bhima Nagar', 'Old Madras Road', 'Malleshwaram', 'Seshadripuram', 'Kammanahalli', 'Koramangala 6th Block',
    'Majestic', 'Langford Town', 'Central Bangalore', 'Sanjay Nagar', 'Brookefield', 'ITPL Main Road, Whitefield',
    'Varthur Main Road, Whitefield', 'KR Puram', 'Koramangala 2nd Block', 'Koramangala 3rd Block', 'Koramangala',
    'Hosur Road', 'Rajajinagar', 'Banaswadi', 'North Bangalore', 'Nagawara', 'Hennur', 'Kalyan Nagar', 'New BEL Road', 'Jakkur',
    'Rammurthy Nagar', 'Thippasandra', 'Kaggadasapura', 'Hebbal', 'Kengeri', 'Sankey Road', 'Sadashiv Nagar', 'Basaveshwara Nagar',
    'Yeshwantpur', 'West Bangalore', 'Magadi Road', 'Yelahanka', 'Sahakara Nagar', 'Peenya']

location = st.selectbox("Choose your nearest location", locations)

# Neighborhood
neighborhoods = ['Banashankari', 'Bannerghatta Road', 'Basavanagudi', 'Bellandur', 'Brigade Road', 'Brookefield', 'BTM', 'Church Street',
    'Electronic City', 'Frazer Town', 'HSR', 'Indiranagar', 'Jayanagar', 'JP Nagar', 'Kalyan Nagar', 'Kammanahalli',
    'Koramangala 4th Block', 'Koramangala 5th Block', 'Koramangala 6th Block', 'Koramangala 7th Block', 'Lavelle Road',
    'Malleshwaram', 'Marathahalli', 'MG Road', 'New BEL Road', 'Old Airport Road', 'Rajajinagar', 'Residency Road', 'Sarjapur Road', 'Whitefield']

selected_neighborhood = st.selectbox("Select a neighborhood:", options=neighborhoods)

# Combined Location/City
combined_location_city = f"{selected_neighborhood}/{location}" if selected_neighborhood != location else location

# Cuisine
cuisine_types = ['afghan', 'afghani', 'african', 'american', 'andhra', 'arabian', 'asian', 'assamese', 'australian', 'awadhi', 'bakery', 'bar food', 'bbq', 
                 'belgian', 'bengali', 'beverages', 'bihari', 'biryani', 'bohri', 'british', 'bubble tea', 'burger', 'burmese', 'cafe', 'cantonese', 
                 'charcoal chicken', 'chettinad', 'chinese', 'coffee', 'continental', 'desserts', 'drinks only', 'european', 'fast food', 'finger food',
                 'french', 'german', 'goan', 'greek', 'grill', 'gujarati', 'healthy food', 'hot dogs', 'hyderabadi', 'ice cream', 'indonesian', 'iranian', 
                 'italian', 'japanese', 'jewish', 'juices', 'kashmiri', 'kebab', 'kerala', 'konkan', 'korean', 'lebanese', 'lucknowi', 'maharashtrian', 
                 'malaysian', 'mangalorean', 'mediterranean', 'mexican', 'middle eastern', 'mithai', 'modern indian', 'momos', 'mongolian', 'mughlai', 
                 'naga', 'nepalese', 'north eastern', 'north indian', 'oriya', 'paan', 'pan asian', 'parsi', 'pizza', 'portuguese', 'rajasthani', 
                 'raw meats', 'roast chicken', 'rolls', 'russian', 'salad', 'sandwich', 'seafood', 'sindhi', 'singaporean', 'south american', 'south indian', 
                 'spanish', 'sri lankan', 'steak', 'street food', 'sushi', 'tamil', 'tea', 'tex-mex', 'thai', 'tibetan', 'turkish', 'vegan', 'vietnamese', 
                 'wraps']

cuisines = st.multiselect("Choose up to 4 cuisine types:", options=cuisine_types, default=[])
if len(cuisines) > 4:
    st.warning("You can only select up to 4 cuisine types.")
cuisines = feature_sorting(', '.join(cuisines))

# Restaurant Type
rest_types = ['Dining', 'Quick Service', 'Cafe & Beverage', 'Dessert & Sweets', 'Bakery', 'Bars & Pubs', 'Microbreweries & Clubs']
rest_type = st.multiselect('Choose one or two dining experiences:', options=rest_types, default=[], max_selections=2)
if len(rest_type) == 1:
    rest_type = rest_type[0]
elif len(rest_type) == 2:
    rest_type = feature_sorting(', '.join(rest_type))
else:
    rest_type = 'Please select one or two dining experiences.'

# Approximate Cost
approx_cost = st.number_input("Enter the approximate cost for two people:", min_value=0, step=1, format="%d")

# Encoding the inputs
encoding_map = {'Yes': 1, 'No': 0}
encoded_online_order = encoding_map[selected_online_order]
encoded_book_table = encoding_map[selected_book_table]

# Create DataFrame for processing
input_data = pd.DataFrame({
    'online_order': [encoded_online_order],
    'book_table': [encoded_book_table],
    'rest_type': [rest_type],
    'cuisines': [cuisines],
    'approx_cost(for two people)': [approx_cost],
    'combined_location_city': [combined_location_city]
})

# One-hot encode the DataFrame
df_encoded = pd.get_dummies(input_data[['combined_location_city', 'cuisines', 'rest_type']])
df_final = pd.concat([input_data[['online_order', 'book_table', 'approx_cost(for two people)']], df_encoded], axis=1)

# Drop original columns
df_final = df_final.reindex(columns=model.feature_names_in_, fill_value=0)

# Scale the features
user_data_scaled = scaler.transform(df_final)

# Add Predict button
if st.button('Predict'):
    # Make prediction
    prediction = model.predict(user_data_scaled)
    
    # Interpret the result
    result = "Excellent Performance" if prediction[0] == 1 else "Needs Improvement"
    
    st.write(f'The predicted outcome for the restaurant is: **{result}**')
