import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# Load the trained model
model = joblib.load('ccd_model.pkl')

img=Image.open('iconpic.jpg')
st.set_page_config(page_title='Fraud Detection',page_icon=img)

st.title("Fraud Detection App")
st.image("https://jpinfotech.org/wp-content/uploads/2023/01/JPPY2219-Credit-Card-Fraud-Detection.jpg")
st.markdown("""
### Enter the details of the transaction to predict if it's fraudulent.
""")

# Input fields
amt = st.number_input("Transaction Amount")
category = st.selectbox("Transaction type", ['grocery_net', 'shopping_net', 'misc_pos', 'grocery_pos',
       'health_fitness', 'gas_transport', 'misc_net', 'kids_pets',
       'shopping_pos', 'entertainment', 'food_dining', 'home',
       'personal_care', 'travel'])
trans_hour = st.slider("Transaction Hour", 0, 23)  # Slider for hour (0-23)
trans_day = st.selectbox("Transaction Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
age = st.slider("Age", 18, 100)
city = st.selectbox("City", ['Wales', 'Browning', 'Ravenna', 'San Jose', 'Westerville',
       'Thompson', 'Parker Dam', 'San Diego', 'Centerview', 'Vacaville',
       'La Grande', 'Port Costa', 'Meadville', 'Alva', 'Lowell',
       'Albuquerque', 'Lamy', 'Hubbell', 'Fort Washakie', 'Saint Louis',
       'Kansas City', 'Arnold', 'Mesa', 'Daly City', 'Lonetree', 'Napa',
       'Kaktovik', 'Coulee Dam', 'Aurora', 'Utica', 'Roseland', 'Newhall',
       'Fields Landing', 'June Lake', 'Eugene', 'Blairsden-Graeagle',
       'Pueblo', 'Campbell', 'Moab', 'Tomales', 'Red Cliff', 'Downey',
       'Smith River', 'Riverton', 'Conway', 'Syracuse', 'Grenada', 'Jelm',
       'Unionville', 'Scotts Mills', 'Cardwell', 'Manley', 'Lagrange',
       'American Fork', 'Rock Springs', 'Tekoa', 'Independence',
       'Sun City', 'Arvada', 'Williamsburg', 'Monitor', 'Sacramento',
       'Claremont', 'Hawthorne', 'Valentine', 'Matthews', 'Helm', 'Kent',
       'Glendale', 'Humboldt', 'Colorado Springs', 'Fiddletown', 'Omaha',
       'Yellowstone National Park', 'Phoenix', 'Nelson', 'Colton',
       'Moriarty', 'Parker', 'Wappapello', 'Westfir', 'Kissee Mills',
       'Parks', 'Greenview', 'Seattle', 'Howells', 'Pleasant Hill',
       'Meredith', 'Crownpoint', 'Paauilo', 'Brashear', 'Rocky Mount',
       'Huntington Beach', 'Fullerton', 'Powell Butte', 'Laguna Hills',
       'Wendel', 'Redford', 'Vinton', 'Sixes', 'Gardiner', 'Shedd',
       'Weeping Water', 'Carroll', 'High Rolls Mountain Park',
       'Los Angeles', 'Mendon', 'Red River', 'Oakland', 'Clarksville',
       'Athena', 'Dumont', 'Ashford', 'Bay City', 'Iliff', 'Buellton',
       'Superior', 'Norwalk', 'North Loup', 'Hatch', 'Burbank',
       'Louisiana', 'Kirk', 'Indian Wells', 'Espanola', 'Angwin',
       'Broomfield', 'Corona', 'Holstein', 'Huslia', 'Jordan Valley',
       'Freedom', 'Brainard', 'Burlington', 'Altonah', 'Issaquah',
       'Littleton', 'Sprague', 'Odessa', 'Vancouver', 'Honokaa',
       'Lake Oswego', 'Portland', 'Santa Monica', 'Sutherland',
       'Owensville', 'Manville', 'Hooper', 'Luray', 'Loving', 'Newberg',
       'Meridian', 'Mound City', 'Palmdale', 'Ruidoso', 'Blairstown',
       'Azusa', 'Kirtland', 'Laramie', 'Seligman', 'Mountain Center',
       'Llano', 'Camden', 'Paradise Valley', 'Malad City',
       'Cascade Locks', 'Spirit Lake', 'Stayton', 'Craig', 'Orient',
       'Carlotta', 'Lakeport', 'Ballwin', 'Wheaton', 'Woods Cross',
       'Kirtland Afb'])
job = st.selectbox("Job",['"Administrator, education"', 'Cytogeneticist',
       '"Solicitor, Scotland"', 'Science writer', 'Product designer',
       '"Surveyor, minerals"', 'Marketing executive', 'Comptroller',
       'Electronics engineer', 'Clinical cytogeneticist',
       '"Engineer, site"', 'Armed forces training and education officer',
       'Tourist information centre manager',
       '"Administrator, local government"', 'Systems analyst',
       'Charity officer', 'Public relations account executive',
       'Set designer', 'Information systems manager',
       'Occupational hygienist', 'Counsellor', 'Land/geomatics surveyor',
       'Landscape architect', '"Buyer, industrial"',
       'Research scientist (physical sciences)', 'Airline pilot',
       'Careers information officer', '"Pilot, airline"',
       'Industrial/product designer', '"Nurse, mental health"',
       'Health physicist', '"Scientist, audiological"',
       'Health service manager', '"Scientist, physiological"',
       'Cartographer', 'Chartered legal executive (England and Wales)',
       'Civil Service administrator', 'Further education lecturer',
       'Location manager', 'Occupational psychologist',
       'Human resources officer', 'Fine artist', 'Web designer',
       '"Lecturer, higher education"',
       '"Research officer, political party"', 'Geoscientist',
       '"Radiographer, diagnostic"', 'Public librarian',
       '"Investment banker, corporate"', '"Engineer, petroleum"',
       'Television floor manager',
       'Product/process development scientist', 'Futures trader',
       'Music therapist', 'Clothing/textile technologist',
       '"Engineer, production"', '"Designer, exhibition/display"',
       'Hotel manager', 'Glass blower/designer',
       'Medical technical officer', 'Magazine features editor',
       'Wellsite geologist', '"Editor, magazine features"',
       'Network engineer', 'Aeronautical engineer', 'Early years teacher',
       'Museum education officer', '"Accountant, chartered"',
       '"Civil engineer, contracting"',
       'Museum/gallery exhibitions officer', 'Immigration officer',
       '"Surveyor, land/geomatics"', 'Materials engineer', 'Contractor',
       'Television/film/video producer', 'Chief Marketing Officer',
       '"Scientist, marine"', '"Therapist, art"',
       'Research scientist (medical)', 'Forensic psychologist',
       '"Engineer, agricultural"', '"Geologist, engineering"',
       'Call centre manager', '"Development worker, international aid"',
       'Research scientist (maths)', 'Information officer',
       'IT consultant', '"Scientist, research (maths)"',
       'Private music teacher', 'Tax inspector',
       '"Therapist, horticultural"', '"Surveyor, mining"',
       'Freight forwarder', 'Local government officer', 'Sales executive',
       '"Teacher, adult education"', 'Investment analyst',
       'Education administrator', 'Retail merchandiser',
       '"Engineer, maintenance"', 'Radio broadcast assistant', 'Dealer',
       'Metallurgist', 'Naval architect', '"Journalist, newspaper"',
       'Barista', 'Exercise physiologist', 'Mental health nurse',
       'Video editor', 'Colour technologist', 'Community arts worker',
       'Soil scientist', 'Nature conservation officer',
       'Petroleum engineer', 'Firefighter', '"Nurse, children\'s"',
       'Musician', '"Teacher, early years/pre"', 'Learning mentor',
       'Historic buildings inspector/conservation officer',
       '"Engineer, electronics"', 'Telecommunications researcher',
       '"Engineer, civil (consulting)"', 'Economist',
       '"Pharmacist, hospital"', '"Education officer, museum"',
       'Associate Professor', 'Public house manager',
       '"Sales professional, IT"', 'Osteopath', 'Water engineer',
       'Chiropodist', 'Clinical research associate',
       'Teaching laboratory technician', 'Physiotherapist',
       'Planning and development surveyor',
       '"Engineer, building services"', 'Insurance broker',
       'Educational psychologist', '"Engineer, automotive"',
       'Town planner', 'Intelligence analyst', 'Architect',
       'Chemical engineer', 'Licensed conveyancer',
       'Agricultural consultant', 'Civil Service fast streamer',
       'TEFL teacher', '"Engineer, biomedical"', '"Therapist, music"',
       'Architectural technologist', 'Counselling psychologist',
       'Building surveyor', 'Systems developer', 'Barrister',
       'Production manager', 'Commissioning editor',
       'Advertising account planner', 'Special educational needs teacher',
       '"Therapist, occupational"', 'Podiatrist',
       '"Engineer, communications"',
       'Chartered public finance accountant'])

# Collect input data
input_data = {
    'amt': amt,
    'age': age,
    'merchant':"Wisozk and Sons",
    'trans_year':2019,
    'Amount_clipped':400,
    'trans_hour':19,
    'trans_month':'',
    'job':'', 
    'day_of_week':'',
    'trans_day':'', 
    'city':'', 
    'category':'', 
    'hour':''
}

# Prediction function
def predict_fraud(input_data):
    #st.write(input_data)
    input_df = pd.DataFrame(input_data, index=[0])
    st.write(input_df)
    # Predict using the trained model
    prediction = model.predict(input_df)
    return 'Fraudulent' if prediction[0] == 1 else 'Non-Fraudulent'

# Prediction button
if st.button("Predict Fraud"):
    result = predict_fraud(input_data)
    st.write(f"The transaction is predicted to be: **{result}**")


st.sidebar.title("About")
st.sidebar.info("This application uses a machine learning model to predict the likelihood of a transaction being fraudulent. The model was trained on historical transaction data and is designed to help identify and prevent credit card fraud.")





