import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import requests
from io import BytesIO
from config import NUTRITION_API_KEY

# Load the trained model
model = load_model('mobilenetv2_3_food_classification_model.keras')
class_names = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_tartare', 'beet_salad', 'beignet', 'bibimbap', 
    'bread_pudding', 'burrito', 'bruschetta', 'caesar_salad', 'calamari', 'cannoli', 'caprese_salad', 
    'carpaccio', 'carrot_cake', 'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 
    'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 
    'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'cup_cakes', 'deviled_eggs', 
    'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 
    'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_rice', 
    'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 
    'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 
    'lasagna', 'lobster_bisque', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 
    'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 
    'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 
    'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 
    'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'tiramisu', 'waffles', 'yogurt'
]

# Set up the Streamlit interface
st.title('Food Prediction and Nutrition Info')
st.write("""
    Welcome to the Food Prediction and Nutrition Info website! This tool allows you to upload an image of a food item, 
    and it will use a trained AI model to predict the type of food. Additionally, it provides nutritional information 
    and assesses whether the food is healthy based on specific parameters.
""")

st.subheader('How to Use:')
st.write("""
1. Click on the "Choose File" button to upload an image of the food item.
2. Click on the "Predict" button to start the prediction process.
3. The predicted food type, its confidence score, nutritional information, and health status will be displayed on the screen.
""")

st.subheader('About the Dataset:')
st.write("""
The dataset used to train the AI model is the Food-101 dataset from Kaggle, which contains 101,000 images across 101 food categories. 
Each category initially has 1,000 images. The dataset was cleaned by removing categories that could not be matched with the API 
and adjusting the labels to match the API's categories. Additionally, unclear, blurry, and incorrectly labeled images were removed, 
resulting in 500 images per category and a total of 95,000 images.
""")

st.subheader('About the AI Model:')
st.write("""
The AI model used in this application employs deep learning techniques, specifically a Convolutional Neural Network (CNN), 
to classify images into predefined food categories. The model architecture includes the following layers:
- Input layer: Takes in the image data.
- Convolutional layers: Extract features from the images.
- Max-pooling layers: Reduce the dimensionality of the feature maps.
- Fully connected layers: Map the extracted features to the output classes.
- Output layer: Produces the final classification.

The model was trained with a validation split of 20%, meaning 80% of the data was used for training and 20% for validation. 
During validation, the model's performance is evaluated on unseen data to prevent overfitting. The model achieved an accuracy 
of 48% on the validation set, indicating its ability to correctly classify nearly half of the unseen food images. 
This performance is typical for complex image classification tasks with a large number of categories.
""")


st.subheader('About Confidence:')
st.write("""
The confidence score represents the probability that the AI model's prediction is correct. 
A higher confidence score means the model is more certain about its prediction.
""")

st.subheader('Nutrition Data:')
st.write("""
The nutritional information is fetched from the API Ninjas Nutrition API. The information includes details such as total fat, 
saturated fat, sodium, potassium, cholesterol, carbohydrates, fiber, and sugar. However, there are three nutrients that are not 
displayed due to premium API restrictions: calories, serving size, and protein. Additionally, note that cholesterol, 
potassium, and sodium are measured in milligrams (mg), while the other nutrients are measured in grams (g).
""")

st.subheader('Health Parameters:')
st.write("""
The health status is determined based on guidelines from the WHO's healthy diet parameters. A food item is considered healthy 
if it meets certain thresholds for fat, saturated fat, sodium, cholesterol, and sugar. These parameters were derived from daily 
recommended intake values, divided by three to represent typical meals, and further adjusted for safety. Here are the sources 
and calculations used:
- [WHO Healthy Diet Guidelines](https://www.who.int/news-room/fact-sheets/detail/healthy-diet): Daily intake recommendations for fat, saturated fat, sodium, and sugar were divided by 3 for three meals per day, and then halved to ensure a conservative estimate per serving.
- [WebMD Calorie Chart](https://www.webmd.com/diet/calories-chart): The baseline for calorie intake is 2000 calories per day.
- [AHA Cholesterol Guidelines](https://www.ahajournals.org/doi/full/10.1161/CIR.0000000000000743): Daily cholesterol intake should be less than 300mg, divided by 3 for three meals per day, and then halved to ensure a conservative estimate per serving..

Using these guidelines, the parameters for a healthy serving are:
- Total Fat: Less than 11g
- Saturated Fat: Less than 4g
- Sodium: Less than 333mg
- Cholesterol: Less than 50mg
- Sugar: Less than 8g
""")

# Image upload
uploaded_file = st.file_uploader("Choose a food image...", type="jpg")

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = load_img(uploaded_file, target_size=(128, 128))
        st.image(image, caption='Uploaded Image.', use_column_width=True)
    
        # Image preprocessing
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)
    
        # Predict the class of the image
        prediction = model.predict(image)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        
        # Display prediction and confidence below the image
        st.write(f'Prediction: {predicted_class.replace("_", " ")}')
        st.write(f'Confidence: {confidence * 100:.2f}%')
        
    with col2:
        # Display nutrition information and health status with larger, bold text
        st.markdown(f"**<h3>Nutrition Information:</h3>**", unsafe_allow_html=True)
        
        url = f'https://api.api-ninjas.com/v1/nutrition?query={predicted_class.replace("_", " ")}'
        headers = {'X-Api-Key': NUTRITION_API_KEY}
        response = requests.get(url, headers=headers)
    
        if response.status_code == 200:
            nutrition_data = response.json()
            if nutrition_data:
                nutrition_info = nutrition_data[0]
                st.write(f"Fat Total (g): {nutrition_info.get('fat_total_g', 'N/A')}")
                st.write(f"Fat Saturated (g): {nutrition_info.get('fat_saturated_g', 'N/A')}")
                st.write(f"Sodium (mg): {nutrition_info.get('sodium_mg', 'N/A')}")
                st.write(f"Potassium (mg): {nutrition_info.get('potassium_mg', 'N/A')}")
                st.write(f"Cholesterol (mg): {nutrition_info.get('cholesterol_mg', 'N/A')}")
                st.write(f"Carbohydrates Total (g): {nutrition_info.get('carbohydrates_total_g', 'N/A')}")
                st.write(f"Fiber (g): {nutrition_info.get('fiber_g', 'N/A')}")
                st.write(f"Sugar (g): {nutrition_info.get('sugar_g', 'N/A')}")
            
                # Health status calculation
                fat_total = float(nutrition_info.get('fat_total_g', 0))
                fat_saturated = float(nutrition_info.get('fat_saturated_g', 0))
                sodium = float(nutrition_info.get('sodium_mg', 0))
                cholesterol = float(nutrition_info.get('cholesterol_mg', 0))
                sugar = float(nutrition_info.get('sugar_g', 0))
                health_status = (fat_total < 11 and fat_saturated < 4 and sodium < 333 and cholesterol < 50 and sugar < 8)
                
                # Display health status with larger, bold text
                st.markdown(f"**<h3>Health Status: {'Healthy' if health_status else 'Unhealthy'}</h3>**", unsafe_allow_html=True)
            else:
                st.markdown(f"**<h3>No nutrition information found.</h3>**", unsafe_allow_html=True)
        else:
            st.write(f"Failed to fetch nutrition info: {response.status_code}")
        st.markdown("</div>", unsafe_allow_html=True)
