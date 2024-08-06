import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import requests
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

def translate_food_name(name):
    translation_dict = {
        'apple_pie': 'pai apel',
        'baby_back_ribs': 'iga panggang',
        'baklava': 'baklava',
        'beef_tartare': 'tartar daging sapi',
        'beet_salad': 'salad bit',
        'beignet': 'roti goreng Prancis',
        'bibimbap': 'bibimbap',
        'bread_pudding': 'puding roti',
        'burrito': 'burrito',
        'bruschetta': 'bruschetta',
        'caesar_salad': 'salad caesar',
        'calamari': 'cumi goreng',
        'cannoli': 'cannoli',
        'caprese_salad': 'salad caprese',
        'carpaccio': 'carpaccio',
        'carrot_cake': 'kue wortel',
        'ceviche': 'ceviche',
        'cheesecake': 'kue keju',
        'cheese_plate': 'piring keju',
        'chicken_curry': 'kari ayam',
        'chicken_quesadilla': 'quesadilla ayam',
        'chicken_wings': 'sayap ayam',
        'chocolate_cake': 'kue coklat',
        'chocolate_mousse': 'mousse coklat',
        'churros': 'churros',
        'clam_chowder': 'sup krim kerang',
        'club_sandwich': 'sandwich klub',
        'crab_cakes': 'kue kepiting',
        'creme_brulee': 'creme brulee',
        'cup_cakes': 'kue cangkir',
        'deviled_eggs': 'telur isi',
        'donuts': 'donat',
        'dumplings': 'pangsit',
        'edamame': 'kedelai Jepang',
        'eggs_benedict': 'telur benedict',
        'escargots': 'siput',
        'falafel': 'falafel',
        'filet_mignon': 'daging sapi filet',
        'fish_and_chips': 'ikan dan kentang goreng',
        'foie_gras': 'foie gras',
        'french_fries': 'kentang goreng',
        'french_onion_soup': 'sup bawang Perancis',
        'french_toast': 'roti panggang Perancis',
        'fried_rice': 'nasi goreng',
        'garlic_bread': 'roti bawang putih',
        'gnocchi': 'gnocchi',
        'greek_salad': 'salad Yunani',
        'grilled_cheese_sandwich': 'sandwich keju panggang',
        'grilled_salmon': 'salmon panggang',
        'guacamole': 'guacamole',
        'gyoza': 'gyoza',
        'hamburger': 'hamburger',
        'hot_and_sour_soup': 'sup asam pedas',
        'hot_dog': 'hot dog',
        'huevos_rancheros': 'huevos rancheros',
        'hummus': 'hummus',
        'ice_cream': 'es krim',
        'lasagna': 'lasagna',
        'lobster_bisque': 'bisque lobster',
        'macaroni_and_cheese': 'makaroni dan keju',
        'macarons': 'makaron',
        'miso_soup': 'sup miso',
        'mussels': 'kerang',
        'nachos': 'nachos',
        'omelette': 'omelet',
        'onion_rings': 'cincin bawang',
        'oysters': 'tirami',
        'pad_thai': 'pad thai',
        'paella': 'paella',
        'pancakes': 'panekuk',
        'panna_cotta': 'panna cotta',
        'peking_duck': 'bebek peking',
        'pho': 'pho',
        'pizza': 'pizza',
        'pork_chop': 'daging babi',
        'poutine': 'poutine',
        'prime_rib': 'iga utama',
        'pulled_pork_sandwich': 'sandwich babi suwir',
        'ramen': 'ramen',
        'ravioli': 'ravioli',
        'risotto': 'risotto',
        'samosa': 'samosa',
        'sashimi': 'sashimi',
        'scallops': 'kerang',
        'seaweed_salad': 'salad rumput laut',
        'shrimp_and_grits': 'udang dan bubur jagung',
        'spaghetti_bolognese': 'spageti bolognese',
        'spring_rolls': 'lumpia',
        'steak': 'steak',
        'strawberry_shortcake': 'kue stroberi',
        'sushi': 'sushi',
        'tacos': 'taco',
        'tiramisu': 'tiramisu',
        'waffles': 'wafel',
        'yogurt': 'yogurt'
    }
    return translation_dict.get(name, name)

# Menyiapkan antarmuka Streamlit
st.title('Prediksi Makanan dan Informasi Nutrisi')
st.write("""
    Selamat datang di situs Prediksi Makanan dan Informasi Nutrisi! Situs ini memungkinkan Anda untuk mengunggah gambar makanan, 
    dan akan menggunakan model AI terlatih untuk memprediksi jenis makanan. Selain itu, alat ini memberikan informasi nutrisi 
    dan menilai apakah makanan tersebut sehat berdasarkan parameter tertentu.
""")

st.subheader('Cara Menggunakan:')
st.write("""
1. Klik tombol "Pilih File" untuk mengunggah gambar makanan.
2. Klik tombol "Prediksi" untuk memulai proses prediksi.
3. Jenis makanan yang diprediksi, skor kepercayaan, informasi nutrisi, dan status kesehatan akan ditampilkan di layar.
""")

st.subheader('Tentang Dataset:')
st.write("""
Dataset yang digunakan untuk melatih model AI adalah dataset Food-101 dari Kaggle, yang berisi 101.000 gambar dari 101 kategori makanan. 
Setiap kategori awalnya memiliki 1.000 gambar. Dataset ini dibersihkan dengan menghapus kategori yang tidak dapat dicocokkan dengan API dan menyesuaikan label agar sesuai dengan kategori API, 
sehingga hanya tersisa 95 kategori makanan. Selain itu, gambar yang tidak jelas dan tidak sesuai konteks dihapus, sehingga hanya menyisakan 500 gambar per kategori dan totalnya adalah 95.000 gambar.
""")

st.subheader('Tentang Model AI:')
st.write("""
Model AI yang digunakan dalam aplikasi ini menggunakan teknik pembelajaran mendalam, khususnya arsitektur MobileNetV2 yang telah 
dilatih sebelumnya pada dataset ImageNet, untuk mengklasifikasikan gambar ke dalam kategori makanan yang telah ditentukan. 
Arsitektur model mencakup lapisan-lapisan berikut:

- **MobileNetV2 Base**: Digunakan sebagai feature extractor dengan bobot yang dilatih pada dataset ImageNet. Lapisan ini 
  memiliki parameter trainable yang disetel ke False, sehingga bobotnya tidak akan diperbarui selama pelatihan.
- **Lapisan Dense Custom**: Dua lapisan fully connected (Dense) dengan 128 unit dan aktivasi ReLU ditambahkan di atas feature 
  extractor untuk memetakan fitur yang diekstrak ke kelas output.
- **Lapisan Output**: Lapisan dense terakhir dengan fungsi aktivasi softmax menghasilkan klasifikasi akhir ke dalam salah 
  satu dari beberapa kategori makanan.

Model ini dilatih menggunakan augmentasi data untuk meningkatkan keanekaragaman data pelatihan, dengan teknik-teknik seperti 
rotasi, pergeseran, shear, zoom, dan flip horizontal.

Data pelatihan dibagi menjadi 80% untuk pelatihan dan 20% untuk validasi. Selama pelatihan, callback early stopping digunakan 
untuk memantau kehilangan validasi (`val_loss`) dan menghentikan pelatihan jika tidak ada peningkatan selama 10 epoch, 
serta mengembalikan bobot terbaik dari model.

Setelah pelatihan, model mencapai akurasi yang lebih baik dibandingkan model sebelumnya, yang menunjukkan peningkatan 
kemampuan model untuk mengklasifikasikan gambar makanan yang tidak terlihat. Hasil akhir menunjukkan performa yang 
menjanjikan untuk tugas klasifikasi gambar makanan yang kompleks dengan banyak kategori.
""")

st.subheader('Tentang Skor Kepercayaan:')
st.write("""
Skor kepercayaan mewakili probabilitas bahwa prediksi model AI benar. 
Skor kepercayaan yang lebih tinggi berarti model lebih yakin tentang prediksinya.
""")

st.subheader('Data Nutrisi:')
st.write("""
Informasi nutrisi diambil dari API milik Ninjas Nutrition API. Informasi tersebut mencakup rincian seperti total lemak, 
lemak jenuh, natrium, kalium, kolesterol, karbohidrat, serat, dan gula. Namun, ada tiga nutrisi yang tidak ditampilkan 
karena batasan API premium: kalori, ukuran porsi, dan protein. Selain itu, perlu dicatat bahwa kolesterol, 
kalium, dan natrium diukur dalam miligram (mg), sementara nutrisi lainnya diukur dalam gram (g).
""")

st.subheader('Parameter Kesehatan:')
st.write("""
Status kesehatan ditentukan berdasarkan pedoman dari parameter diet sehat WHO. Sebuah makanan dianggap sehat 
jika memenuhi ambang batas tertentu untuk lemak, lemak jenuh, natrium, kolesterol, dan gula. Parameter ini diperoleh dari nilai 
asupan harian yang disarankan, dibagi tiga untuk mewakili makanan sehari-hari, dan kemudian dibagi dua untuk memastikan perkiraan konservatif per porsi. 
Berikut adalah sumber dan perhitungan yang digunakan:
- [Chart Kalori WebMD](https://www.webmd.com/diet/calories-chart): Dasar untuk asupan kalori adalah 2000 kalori per hari.
- [Pedoman Diet Sehat WHO](https://www.who.int/news-room/fact-sheets/detail/healthy-diet): Rekomendasi asupan harian untuk lemak total adalah tidak melebihi 30% dari kalori yang dikonsumsi, dan lemak jenuh harus kurang dari 10% dari kalori yang dikonsumsi untuk mencegah kenaikan berat badan yang tidak sehat. Natrium sebaiknya kurang dari 2 gram per hari untuk mencegah hipertensi, mengurangi risiko penyakit jantung, dan tekanan darah tinggi. Gula sebaiknya tidak lebih dari 10% dari kalori yang dikonsumsi untuk mencegah risiko kerusakan gigi, kenaikan berat badan yang tidak sehat, dan risiko penyakit kardiovaskular.
- [Pedoman Kolesterol AHA](https://www.ahajournals.org/doi/full/10.1161/CIR.0000000000000743): Asupan kolesterol harian sebaiknya kurang dari 300mg, untuk lebih menyehatkan jantung.

Dengan menggunakan pedoman ini, parameter untuk porsi yang sehat adalah:
- Total Lemak: Kurang dari 11g
- Lemak Jenuh: Kurang dari 4g
- Natrium: Kurang dari 333mg
- Kolesterol: Kurang dari 50mg
- Gula: Kurang dari 8g
""")

# Unggah gambar
uploaded_file = st.file_uploader("Pilih gambar makanan...", type="jpg")

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
