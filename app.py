import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import requests
import os
from config import NUTRITION_API_KEY
from PIL import Image
import time
import streamlit.components.v1 as components

# Load the trained model
model = load_model('mobilenetv2_food_classification_finalmodels.keras')
class_names = [
    'baby_back_ribs', 'baklava', 'beef_tartare', 'beet_salad', 'beignet', 'bibimbap', 
    'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 
    'carrot_cake', 'ceviche', 'cheese_plate','cheesecake', 'chicken_curry', 
    'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'churros', 
    'clam_chowder', 'club_sandwich','creme_brulee', 'cup_cakes', 'deviled_eggs', 
    'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 
    'fish_and_chips', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_rice', 
    'garlic_bread', 'gnocchi', 'greek_salad', 'guacamole', 
    'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'ice_cream', 
    'lasagna', 'lobster_bisque', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 
    'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 
    'pho', 'pizza', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 
    'risotto', 'samosa', 'sashimi', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 
    'spring_rolls', 'strawberry_shortcake', 'sushi', 'tiramisu', 'waffles'
]

def translate_food_name(name):
    translation_dict = {
    'baby_back_ribs': 'iga panggang', 'baklava': 'baklava', 'beef_tartare': 'tartar daging sapi',
    'beet_salad': 'salad bit', 'beignet': 'roti goreng Prancis', 'bibimbap': 'bibimbap',
    'bruschetta': 'bruschetta', 'caesar_salad': 'salad caesar', 'cannoli': 'cannoli',
    'caprese_salad': 'salad caprese', 'carrot_cake': 'kue wortel', 'ceviche': 'ceviche',
    'cheese_plate': 'piring keju', 'cheesecake': 'kue keju', 'chicken_curry': 'kari ayam',
    'chicken_quesadilla': 'quesadilla ayam', 'chicken_wings': 'sayap ayam', 'chocolate_cake': 'kue coklat',
    'churros': 'churros', 'clam_chowder': 'sup krim kerang', 'club_sandwich': 'sandwich klub',
    'creme_brulee': 'creme brulee', 'cup_cakes': 'kue cangkir', 'deviled_eggs': 'telur isi',
    'donuts': 'donat', 'dumplings': 'pangsit', 'edamame': 'kedelai Jepang',
    'eggs_benedict': 'telur benedict', 'escargots': 'siput', 'falafel': 'falafel',
    'filet_mignon': 'daging sapi filet', 'fish_and_chips': 'ikan dan kentang goreng', 'french_fries': 'kentang goreng',
    'french_onion_soup': 'sup bawang Perancis', 'french_toast': 'roti panggang Perancis', 'fried_rice': 'nasi goreng',
    'garlic_bread': 'roti bawang putih', 'gnocchi': 'gnocchi', 'greek_salad': 'salad Yunani',
    'guacamole': 'guacamole', 'gyoza': 'gyoza', 'hamburger': 'hamburger',
    'hot_and_sour_soup': 'sup asam pedas', 'hot_dog': 'hot dog', 'ice_cream': 'es krim',
    'lasagna': 'lasagna', 'lobster_bisque': 'bisque lobster', 'macaroni_and_cheese': 'makaroni dan keju',
    'macarons': 'makaron', 'miso_soup': 'sup miso', 'mussels': 'kerang',
    'nachos': 'nachos', 'onion_rings': 'cincin bawang', 'oysters': 'tiram',
    'pad_thai': 'pad thai', 'paella': 'paella', 'pancakes': 'panekuk',
    'panna_cotta': 'panna cotta', 'peking_duck': 'bebek peking', 'pho': 'pho',
    'pizza': 'pizza', 'poutine': 'poutine', 'prime_rib': 'iga utama',
    'pulled_pork_sandwich': 'sandwich babi suwir', 'ramen': 'ramen', 'risotto': 'risotto',
    'samosa': 'samosa', 'sashimi': 'sashimi', 'seaweed_salad': 'salad rumput laut',
    'shrimp_and_grits': 'udang dan bubur jagung', 'spaghetti_bolognese': 'spageti bolognese', 'spring_rolls': 'lumpia',
    'strawberry_shortcake': 'kue stroberi', 'sushi': 'sushi', 'tiramisu': 'tiramisu',
    'waffles': 'wafel'
    }
    return translation_dict.get(name, name)

# Menyiapkan antarmuka Streamlit
st.title('Website Klasifikasi Makanan dan Informasi Nutrisi Berbasis Machine Learning')
st.write("""
    Selamat datang di **Website Prediksi Makanan dan Informasi Nutrisi**. Website ini dirancang untuk mengunggah gambar makanan dan 
    mendapatkan klasifikasi jenis makanannya melalui model machine learning MobileNetV2 yang telah dilatih. Selain itu, website ini 
    menyediakan informasi nutrisi serta evaluasi kesehatan makanan berdasarkan parameter gizi yang telah ditetapkan. 
    
    Unggah gambar makanan Anda dan eksplorasi informasi yang relevan tentang makanan tersebut dengan menggulir ke bagian bawah website atau menekan tombol 'Ayo Mulai Klasifikasi' untuk memulai.
""")

# JavaScript untuk scroll otomatis ke bawah
js = '''
<script>
    var body = window.parent.document.querySelector(".main");
    console.log(body);
    body.scrollTop = body.scrollHeight;
</script>
'''

# Tombol untuk scroll ke bawah
if st.button("Ayo Mulai Mengklasifikasi"):
    temp = st.empty()
    with temp:
        components.html(js, height=0)  # Sisipkan JavaScript
        time.sleep(.5)  # Memberikan waktu untuk memastikan skrip dieksekusi
    temp.empty()

st.divider()  # Atau bisa juga menggunakan st.markdown("---")

# Fungsi untuk resize gambar agar ukurannya seragam
def resize_image(image_path, size=(300, 300)):
    img = Image.open(image_path)
    img = img.resize(size, Image.LANCZOS)
    return img

# Fungsi untuk mencari semua gambar dalam folder
def find_images_in_folder(folder):
    image_paths = []
    for root, dirs, files in os.walk(folder):
        for file in sorted(files):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths

# Fungsi untuk menampilkan gambar dalam grid dengan scroll
def display_image_grid(image_paths, labels, columns=4):
    num_images = len(image_paths)
    num_rows = -(-num_images // columns)  # Ceiling division to calculate the number of rows
    
    with st.container():  # Container for scrolling
        # Create a scrollable grid
        st.write("<style>.scrollable-container { overflow: auto; }</style>", unsafe_allow_html=True)
        st.markdown('<div class="scrollable-container">', unsafe_allow_html=True)
        
        for i in range(num_rows):
            cols = st.columns(columns)
            for j in range(columns):
                index = i * columns + j
                if index < num_images:
                    with cols[j]:
                        resized_img = resize_image(image_paths[index])
                        st.image(resized_img, caption=labels[index], use_column_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Cari semua gambar di folder 'food_images'
image_folder = 'food_images'
image_paths = find_images_in_folder(image_folder)
labels = [
    'Iga Panggang (Baby Back Ribs)', 'Baklava', 'Tartar Daging Sapi (Beef Tartare)', 'Salad Bit (Beet Salad)', 
    'Roti Goreng Prancis (Beignet)', 'Bibimbap', 'Bruschetta', 'Salad Caesar (Caesar Salad)', 'Cannoli', 
    'Salad Caprese (Caprese Salad)', 'Kue Wortel (Carrot Cake)', 'Ceviche', 'Piring Keju (Cheese Plate)', 
    'Kue Keju (Cheesecake)', 'Kari Ayam (Chicken Curry)', 'Quesadilla Ayam (Chicken Quesadilla)', 
    'Sayap Ayam (Chicken Wings)', 'Kue Coklat (Chocolate Cake)', 'Churros', 'Sup Krim Kerang (Clam Chowder)', 
    'Sandwich Klub (Club Sandwich)', 'Creme Brulee', 'Kue Cangkir (Cup Cakes)', 'Telur Isi (Deviled Eggs)', 
    'Donat (Donuts)', 'Pangsit (Dumplings)', 'Kedelai Jepang (Edamame)', 'Telur Benedict (Eggs Benedict)', 
    'Siput (Escargots)', 'Falafel', 'Daging Sapi Filet (Filet Mignon)', 'Ikan Dan Kentang Goreng (Fish And Chips)', 
    'Kentang Goreng (French Fries)', 'Sup Bawang Perancis (French Onion Soup)', 'Roti Panggang Perancis (French Toast)', 
    'Nasi Goreng (Fried Rice)', 'Roti Bawang Putih (Garlic Bread)', 'Gnocchi', 'Salad Yunani (Greek Salad)', 
    'Guacamole', 'Gyoza', 'Hamburger', 'Sup Asam Pedas (Hot And Sour Soup)', 'Hot Dog', 'Es Krim (Ice Cream)', 
    'Lasagna', 'Bisque Lobster (Lobster Bisque)', 'Makaroni Dan Keju (Macaroni And Cheese)', 'Makaron (Macarons)', 
    'Sup Miso (Miso Soup)', 'Kerang (Mussels)', 'Nachos', 'Cincin Bawang (Onion Rings)', 'Tiram (Oysters)', 
    'Pad Thai', 'Paella', 'Panekuk (Pancakes)', 'Panna Cotta', 'Bebek Peking (Peking Duck)', 'Pho', 'Pizza', 
    'Poutine', 'Iga Utama (Prime Rib)', 'Sandwich Babi Suwir (Pulled Pork Sandwich)', 'Ramen', 'Risotto', 
    'Samosa', 'Sashimi', 'Salad Rumput Laut (Seaweed Salad)', 'Udang Dan Bubur Jagung (Shrimp And Grits)', 
    'Spageti Bolognese (Spaghetti Bolognese)', 'Lumpia (Spring Rolls)', 'Kue Stroberi (Strawberry Shortcake)', 
    'Sushi', 'Tiramisu', 'Wafel (Waffles)'
]

# Menampilkan grid gambar
st.title('Daftar Makanan yang Dapat Diklasifikasikan')
display_image_grid(image_paths, labels, columns=4)

# Menambahkan garis pemisah
st.divider()  # Atau bisa juga menggunakan st.markdown("---")

st.subheader('Data Nutrisi:')
st.write("""
Informasi nutrisi diambil dari API milik API Ninjas Nutrition. Informasi tersebut mencakup rincian seperti total lemak, 
lemak jenuh, natrium, kalium, kolesterol, karbohidrat, serat, dan gula. Selain itu, perlu dicatat bahwa kolesterol, 
kalium, dan natrium diukur dalam miligram (mg), sementara nutrisi lainnya diukur dalam gram (g).
""")

st.subheader('Parameter Kesehatan:')

st.write("""
Status kesehatan ditentukan berdasarkan pedoman dari parameter diet sehat WHO. Sebuah makanan dianggap sehat 
jika memenuhi ambang batas tertentu untuk lemak, lemak jenuh, natrium, kolesterol, dan gula. Parameter ini diperoleh dari nilai 
asupan harian yang disarankan, dibagi tiga untuk mewakili makanan sehari-hari, dan kemudian dibagi dua untuk memastikan perkiraan konservatif per porsi. 

Berikut adalah sumber dan perhitungan yang digunakan:
- [Pedoman Diet Kalori HaloDoc](https://www.halodoc.com/artikel/catat-ini-jumlah-minimal-kalori-yang-harus-dipenuhi-saat-diet): 
  Dapat disimpulkan bahwa manusia membutuhkan asupan kalori berupa 2000 kalori per hari.
- [Pedoman Diet Sehat WHO](https://www.who.int/news-room/fact-sheets/detail/healthy-diet): 
  - **Lemak Total**: Tidak melebihi 30% dari total kalori harian.
  - **Lemak Jenuh**: Kurang dari 10% dari total kalori harian.
  - **Natrium**: Kurang dari 2 gram per hari (setara dengan kurang dari 5 gram garam per hari).
  - **Gula**: Tidak lebih dari 10% dari total kalori harian, dengan pengurangan lebih lanjut hingga kurang dari 5% untuk manfaat kesehatan tambahan.
- [Pedoman Kolesterol AHA](https://www.ahajournals.org/doi/full/10.1161/CIR.0000000000000743): 
  Asupan kolesterol harian sebaiknya kurang dari 300mg, untuk lebih menyehatkan jantung.

Dengan menggunakan pedoman ini, parameter untuk porsi yang sehat adalah:
- **Total Lemak**: Kurang dari 11g
- **Lemak Jenuh**: Kurang dari 4g
- **Natrium**: Kurang dari 333mg
- **Kolesterol**: Kurang dari 50mg
- **Gula**: Kurang dari 8g
""")

st.divider()  # Atau bisa juga menggunakan st.markdown("---")

st.subheader('Skor Kepercayaan:')
st.write("""
Skor kepercayaan mewakili probabilitas bahwa prediksi model AI benar. 
Skor kepercayaan yang lebih tinggi berarti model lebih yakin tentang prediksinya.
""")

st.subheader('Cara Menggunakan:')
st.write("""
1. Klik tombol "Browse Files" untuk mengunggah gambar makanan.
2. Setelah gambar diunggah, model machine learning akan memprediksi makanan, mengambil data nutrisi, dan menghitung status kesehatan.
3. Jenis makanan yang diprediksi, skor kepercayaan, informasi nutrisi, dan status kesehatan akan ditampilkan di layar.
""")
# Unggah gambar
uploaded_file = st.file_uploader("Model ini hanya dapat mengklasifikasikan makanan yang ada dalam daftar di atas. Pastikan gambar yang Anda unggah sesuai dengan salah satu kategori tersebut untuk hasil yang akurat.", type=["jpg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = load_img(uploaded_file, target_size=(224, 224))
        st.image(image, caption='Gambar yang diunggah.', use_column_width=True)
    
        # Priproses gambar
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)
    
        # Prediksi kelas gambar
        prediction = model.predict(image)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        
        # Tampilkan prediksi dan kepercayaan di bawah gambar
        st.write(f'Prediksi: {translate_food_name(predicted_class).replace("_", " ")}')
        st.write(f'Skor Kepercayaan: {confidence * 100:.2f}%')
        
    with col2:
        # Tampilkan informasi nutrisi dan status kesehatan dengan teks lebih besar dan tebal
        st.markdown(f"**<h3>Informasi Nutrisi:</h3>**", unsafe_allow_html=True)
        
        url = f'https://api.api-ninjas.com/v1/nutrition?query={predicted_class.replace("_", " ")}'
        headers = {'X-Api-Key': NUTRITION_API_KEY}
        response = requests.get(url, headers=headers)
    
        if response.status_code == 200:
            nutrition_data = response.json()
            if nutrition_data:
                nutrition_info = nutrition_data[0]
                st.write(f"Lemak Total (g): {nutrition_info.get('fat_total_g', 'N/A')}")
                st.write(f"Lemak Jenuh (g): {nutrition_info.get('fat_saturated_g', 'N/A')}")
                st.write(f"Natrium (mg): {nutrition_info.get('sodium_mg', 'N/A')}")
                st.write(f"Kalium (mg): {nutrition_info.get('potassium_mg', 'N/A')}")
                st.write(f"Kolesterol (mg): {nutrition_info.get('cholesterol_mg', 'N/A')}")
                st.write(f"Karbohidrat Total (g): {nutrition_info.get('carbohydrates_total_g', 'N/A')}")
                st.write(f"Serat (g): {nutrition_info.get('fiber_g', 'N/A')}")
                st.write(f"Gula (g): {nutrition_info.get('sugar_g', 'N/A')}")
            
                # Perhitungan status kesehatan
                fat_total = float(nutrition_info.get('fat_total_g', 0))
                fat_saturated = float(nutrition_info.get('fat_saturated_g', 0))
                sodium = float(nutrition_info.get('sodium_mg', 0))
                cholesterol = float(nutrition_info.get('cholesterol_mg', 0))
                sugar = float(nutrition_info.get('sugar_g', 0))
                health_status = (fat_total < 11 and fat_saturated < 4 and sodium < 333 and cholesterol < 50 and sugar < 8)
                
                # Tampilkan status kesehatan dengan teks lebih besar dan tebal
                st.markdown(f"**<h3>Status Kesehatan: {'Sehat' if health_status else 'Tidak Sehat'}</h3>**", unsafe_allow_html=True)
            else:
                st.markdown(f"**<h3>Tidak ada informasi nutrisi yang ditemukan.</h3>**", unsafe_allow_html=True)
        else:
            st.write(f"Gagal mengambil informasi nutrisi: {response.status_code}")
        st.markdown("</div>", unsafe_allow_html=True)
