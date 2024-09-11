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
    'baby_back_ribs': 'Iga Punggung', 'baklava': 'Baklava', 'beef_tartare': 'Tartar Daging Sapi',
    'beet_salad': 'Salad Bit', 'beignet': 'Roti Goreng Prancis', 'bibimbap': 'Bibimbap',
    'bruschetta': 'Bruschetta', 'caesar_salad': 'Salad Caesar', 'cannoli': 'Cannoli',
    'caprese_salad': 'Salad Caprese', 'carrot_cake': 'Kue Wortel', 'ceviche': 'Ceviche',
    'cheese_plate': 'Piring Keju', 'cheesecake': 'Kue Keju', 'chicken_curry': 'Kari Ayam',
    'chicken_quesadilla': 'Quesadilla Ayam', 'chicken_wings': 'Sayap Ayam', 'chocolate_cake': 'Kue Coklat',
    'churros': 'Churros', 'clam_chowder': 'Sup Krim Kerang', 'club_sandwich': 'Sandwich Klub',
    'creme_brulee': 'Creme Brulee', 'cup_cakes': 'Kue Cangkir', 'deviled_eggs': 'Telur Isi',
    'donuts': 'Donat', 'dumplings': 'Pangsit', 'edamame': 'Kedelai Jepang',
    'eggs_benedict': 'Telur Benedict', 'escargots': 'Bekicot Prancis', 'falafel': 'Falafel',
    'filet_mignon': 'Daging Sapi Filet', 'fish_and_chips': 'Ikan dan Kentang Goreng', 'french_fries': 'Kentang Goreng',
    'french_onion_soup': 'Sup Bawang Prancis', 'french_toast': 'Roti Panggang Prancis', 'fried_rice': 'Nasi Goreng',
    'garlic_bread': 'Roti Bawang Putih', 'gnocchi': 'Gnocchi', 'greek_salad': 'Salad Yunani',
    'guacamole': 'Guacamole', 'gyoza': 'Gyoza', 'hamburger': 'Hamburger',
    'hot_and_sour_soup': 'Sup Asam Pedas', 'hot_dog': 'Hot Dog', 'ice_cream': 'Es Krim',
    'lasagna': 'Lasagna', 'lobster_bisque': 'Lobster Bisque', 'macaroni_and_cheese': 'Makaroni dan Keju',
    'macarons': 'Makaron', 'miso_soup': 'Sup Miso', 'mussels': 'Kerang',
    'nachos': 'Nachos', 'onion_rings': 'Cincin Bawang', 'oysters': 'Tiram',
    'pad_thai': 'Pad Thai', 'paella': 'Paella', 'pancakes': 'Panekuk',
    'panna_cotta': 'Panna Cotta', 'peking_duck': 'Bebek Peking', 'pho': 'Pho',
    'pizza': 'Pizza', 'poutine': 'Poutine', 'prime_rib': 'Iga Utama',
    'pulled_pork_sandwich': 'Sandwich Babi Suwir', 'ramen': 'Ramen', 'risotto': 'Risotto',
    'samosa': 'Samosa', 'sashimi': 'Sashimi', 'seaweed_salad': 'Salad Rumput Laut',
    'shrimp_and_grits': 'Udang dan Bubur Jagung', 'spaghetti_bolognese': 'Spageti Bolognese', 'spring_rolls': 'Lumpia',
    'strawberry_shortcake': 'Kue Stroberi', 'sushi': 'Sushi', 'tiramisu': 'Tiramisu',
    'waffles': 'Wafel'
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
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return sorted(image_paths)

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
    'Iga Punggung (Baby Back Ribs)', 'Baklava', 'Tartar Daging Sapi (Beef Tartare)', 'Salad Bit (Beet Salad)', 
    'Roti Goreng Prancis (Beignet)', 'Bibimbap', 'Bruschetta', 'Salad Caesar (Caesar Salad)', 'Cannoli', 
    'Salad Caprese (Caprese Salad)', 'Kue Wortel (Carrot Cake)', 'Ceviche', 'Piring Keju (Cheese Plate)', 
    'Kue Keju (Cheesecake)', 'Kari Ayam (Chicken Curry)', 'Quesadilla Ayam (Chicken Quesadilla)', 
    'Sayap Ayam (Chicken Wings)', 'Kue Coklat (Chocolate Cake)', 'Churros', 'Sup Krim Kerang (Clam Chowder)', 
    'Sandwich Klub (Club Sandwich)', 'Creme Brulee', 'Kue Cangkir (Cup Cakes)', 'Telur Isi (Deviled Eggs)', 
    'Donat (Donuts)', 'Pangsit (Dumplings)', 'Kedelai Jepang (Edamame)', 'Telur Benedict (Eggs Benedict)', 
    'Bekicot Prancis (Escargots)', 'Falafel', 'Daging Sapi Filet (Filet Mignon)', 'Ikan Dan Kentang Goreng (Fish And Chips)', 
    'Kentang Goreng (French Fries)', 'Sup Bawang Prancis (French Onion Soup)', 'Roti Panggang Prancis (French Toast)', 
    'Nasi Goreng (Fried Rice)', 'Roti Bawang Putih (Garlic Bread)', 'Gnocchi', 'Salad Yunani (Greek Salad)', 
    'Guacamole', 'Gyoza', 'Hamburger', 'Sup Asam Pedas (Hot And Sour Soup)', 'Hot Dog', 'Es Krim (Ice Cream)', 
    'Lasagna', 'Lobster Bisque', 'Makaroni Dan Keju (Macaroni And Cheese)', 'Makaron (Macarons)', 
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
lemak jenuh, natrium, kalium, kolesterol, karbohidrat, serat, dan gula. Selain itu, perlu dicatat bahwa data nutrisi untuk setiap item makanan diskalakan ke 100 gram. Data nutrisi seperti kolesterol, 
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
  - **Kalium**: Direkomendasikan asupan kalium tidak kurang dari 3,5 gram per hari
- [Pedoman Kolesterol AHA](https://www.ahajournals.org/doi/full/10.1161/CIR.0000000000000743): 
  Asupan kolesterol harian sebaiknya kurang dari 300 miligram, untuk lebih menyehatkan jantung.
- [Pedoman Karbohidrat AHA](https://www.heart.org/en/news/2023/08/11/confused-about-carbs-this-might-help): 
  Direkomendasikan asupan karbohidrat adalah 45% sampai 65% dari total kalori untuk setiap harinya.
- [Pedoman Serat UCSF](https://www.ucsfhealth.org/education/increasing-fiber-intake#:~:text=Although%20there%20is%20no%20dietary,day%20%E2%80%94%20coming%20from%20soluble%20fiber.): 
  Direkomendasikan asupan serat adalah 25 gram sampai 30 gram untuk setiap harinya.

Dengan menggunakan pedoman ini, parameter untuk porsi yang sehat adalah:
- **Total Lemak**: Kurang dari 11.11 gram
- **Lemak Jenuh**: Kurang dari 3.7 gram
- **Natrium**: Kurang dari 333.335 miligram
- **Kolesterol**: Kurang dari 50 miligram
- **Gula**: Kurang dari 8.335 gram
         
Dengan parameter keterangan tambahan yaitu:
- **Kalium**:  dianggap sangat cukup jika lebih dari 1166.67 miligram, cukup jika lebih dari 583.335 miligram, dan hampir cukup jika lebih dari 291.6675 miligram.
- **Karbohidrat**: dianggap sangat cukup jika lebih dari 75 gram, cukup jika lebih dari 37,5 gram, dan hampir cukup jika lebih dari 18,75 gram.
- **Serat**: dianggap sangat cukup jika lebih dari 8.33 gram, cukup jika lebih dari 4.165 gram, dan hampir cukup jika lebih dari 2.0825 gram
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
uploaded_file = st.file_uploader("Pilih gambar makanan...", type=["jpg", "png"])

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
                potassium = float(nutrition_info.get('potassium_mg', 0))
                cholesterol = float(nutrition_info.get('cholesterol_mg', 0))
                carbohydrates = float(nutrition_info.get('carbohydrates_total_g', 0))
                fiber = float(nutrition_info.get('fiber_g', 0))
                sugar = float(nutrition_info.get('sugar_g', 0))
                
                # Kondisi kesehatan
                health_status = (fat_total < 11.11 and fat_saturated < 3.7 and sodium < 333.335 and cholesterol < 50 and sugar < 8.335)
                
                # Alasan jika makanan tidak sehat
                reasons = []
                if fat_total >= 11.11:
                    reasons.append("Lemak Total")
                if fat_saturated >= 3.7:
                    reasons.append("Lemak Jenuh")
                if sodium >= 333.335:
                    reasons.append("Natrium")
                if cholesterol >= 50:
                    reasons.append("Kolesterol")
                if sugar >= 8.335:
                    reasons.append("Gula")
                    
                if health_status:
                    st.markdown(f"**<h3>Status Kesehatan: Sehat</h3>**", unsafe_allow_html=True)
                else:
                    reason_text = ", ".join(reasons)
                    st.markdown(f"**<h3>Status Kesehatan: Tidak Sehat</h3>**", unsafe_allow_html=True)
                    st.write(f"{reason_text} dalam makanan ini telah melewati batas dari parameter kesehatan, Sehingga dianggap tidak sehat.")
                
                # Keterangan tambahan
                if fiber > 8.33:
                    st.write("Serat pada makanan ini sudah sangat cukup untuk kebutuhan sehari-hari.")
                elif fiber > 4.165:
                    st.write("Serat pada makanan ini sudah cukup untuk kebutuhan sehari-hari.")
                elif fiber > 2.0825:
                    st.write("Serat pada makanan ini hampir cukup untuk kebutuhan sehari-hari.")
                    
                if carbohydrates > 75:
                    st.write("Karbohidrat pada makanan ini sudah sangat cukup untuk kebutuhan sehari-hari.")
                elif carbohydrates > 37.5:
                    st.write("Karbohidrat pada makanan ini sudah cukup untuk kebutuhan sehari-hari.")
                elif carbohydrates > 18.75:
                    st.write("Karbohidrat pada makanan ini hampir cukup untuk kebutuhan sehari-hari.")
                    
                if potassium > 1166.67:
                    st.write("Kalium pada makanan ini sudah sangat cukup untuk kebutuhan sehari-hari.")
                elif potassium > 583.335:
                    st.write("Kalium pada makanan ini sudah cukup untuk kebutuhan sehari-hari.")
                elif potassium > 291.6675:
                    st.write("Kalium pada makanan ini hampir cukup untuk kebutuhan sehari-hari.")
                
            else:
                st.markdown(f"**<h3>Tidak ada informasi nutrisi yang ditemukan.</h3>**", unsafe_allow_html=True)
        else:
            st.write(f"Gagal mengambil informasi nutrisi: {response.status_code}")
        st.markdown("</div>", unsafe_allow_html=True)

