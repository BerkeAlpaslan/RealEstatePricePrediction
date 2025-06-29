from flask import Flask, render_template, request, jsonify
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import os
import time
import csv
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import io
import base64
from datetime import datetime

app = Flask(__name__)

# Model ve scaler global değişkenler
model = None
scaler = None
current_location = "Emlak İlanları"  # Dinamik başlık için


def update_homes(home_link=None):
    """Emlakjet'ten veri toplar"""
    global current_location

    # URL kontrolü
    if home_link and not home_link.startswith("https://www.emlakjet.com"):
        raise ValueError("Sadece emlakjet.com URL'leri kabul edilir!")

    # URL'den konum bilgisini çıkar
    if home_link:
        try:
            # URL'den konum bilgisini parse et
            parts = home_link.split('/')
            if len(parts) >= 5:
                location_parts = parts[4].split('-')
                if len(location_parts) >= 3:
                    # İlk iki kelimeyi şehir ve ilçe olarak al
                    city = location_parts[0].title()
                    district = location_parts[1].title()
                    property_type = ' '.join(location_parts[2:]).replace('-', ' ').title()
                    current_location = f"{city} {district} {property_type}"
                else:
                    current_location = "Emlak İlanları"
        except:
            current_location = "Emlak İlanları"

    # CSV dosyası varsa ve link değişmişse yeniden topla
    csv_exists = os.path.exists('homes.csv')
    should_update = False

    if csv_exists:
        # Son kullanılan URL'i kontrol et
        try:
            with open('last_url.txt', 'r') as f:
                last_url = f.read().strip()
                if last_url != home_link:
                    should_update = True
        except:
            should_update = True
    else:
        should_update = True

    # 20 dakika kontrolü veya URL değişikliği
    if should_update or (csv_exists and (time.time() - os.path.getmtime('homes.csv')) > 1200):
        try:
            options = Options()
            options.add_argument("--headless")  # Tarayıcı açılmasın
            service = Service(os.path.join(os.getcwd(), "chromedriver.exe"))
            browser = webdriver.Chrome(service=service, options=options)
            link = home_link
            home_list = []

            while True:
                browser.get(link)
                time.sleep(2)
                homes = browser.find_elements(by=By.CSS_SELECTOR, value='div[data-id]')

                for home in homes:
                    try:
                        location = home.find_element(By.CSS_SELECTOR, "span.styles_location__ieVpH").text
                        price = int(home.find_element(By.CSS_SELECTOR, "span.styles_price__8Z_OS").text.replace(".",
                                                                                                                "").replace(
                            "TL", "").strip())
                        features = home.find_element(By.CSS_SELECTOR, "div.styles_quickinfoWrapper__F5BBD").text
                        m2_match = re.search(r"(\d+)\s*m²", features)
                        m2 = int(m2_match.group(1)) if m2_match else 0

                        if m2 > 0 and price > 0:  # Sadece geçerli verileri al
                            home = {
                                "location": location,
                                "features": m2,
                                "price": price
                            }
                            print(f"Eklendi: {home}")
                            home_list.append(home)
                    except Exception as e:
                        continue

                time.sleep(1)
                try:
                    next_li = browser.find_element(By.CSS_SELECTOR, ".styles_rightArrow__RFZMm")
                    next_a = next_li.find_element(By.TAG_NAME, "a")
                    href = next_a.get_attribute("href")
                    if href:
                        link = "https://www.emlakjet.com" + href if href.startswith("/") else href
                    else:
                        break
                except:
                    break

            browser.quit()

            # CSV'ye kaydet
            with open('homes.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["location", "features", "price"])
                for home in home_list:
                    writer.writerow([home["location"], home["features"], home["price"]])

            # Son URL'i kaydet
            with open('last_url.txt', 'w') as f:
                f.write(home_link)

            print(f"Toplam {len(home_list)} ev verisi toplandı.")

        except Exception as e:
            print(f"Hata oluştu: {e}")
            home_list = []
    else:
        # CSV'den oku
        home_list = []
        with open('homes.csv', 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    home = {
                        "location": row["location"],
                        "features": int(row["features"]),
                        "price": int(row["price"])
                    }
                    home_list.append(home)
                except Exception as e:
                    continue

    return home_list


def clean_data(df):
    """Veriyi temizler ve aykırı değerleri kaldırır"""
    # Aykırı değerleri tespit et (IQR yöntemi)
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1

    # Aykırı değerleri filtrele
    df = df[(df['price'] >= Q1 - 1.5 * IQR) & (df['price'] <= Q3 + 1.5 * IQR)]

    # Metrekare için de benzer filtreleme
    Q1_m2 = df['features'].quantile(0.25)
    Q3_m2 = df['features'].quantile(0.75)
    IQR_m2 = Q3_m2 - Q1_m2

    df = df[(df['features'] >= Q1_m2 - 1.5 * IQR_m2) & (df['features'] <= Q3_m2 + 1.5 * IQR_m2)]

    # Mantıksız değerleri kaldır
    df = df[(df['features'] > 20) & (df['features'] < 1000)]  # 20-1000 m2 arası
    df = df[(df['price'] > 1000) & (df['price'] < 10000000)]  # 1000-10M TL arası

    return df


def train_model():
    """Modeli eğitir"""
    global model, scaler

    try:
        # Veriyi oku
        df = pd.read_csv('homes.csv')

        # Veriyi temizle
        df = clean_data(df)

        if len(df) < 10:
            print("Yeterli veri yok!")
            return False

        # Özellikler ve hedef değişken
        X = df[['features']].values
        y = df['price'].values

        # Veriyi ölçekle (StandardScaler kullanımı)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Veriyi böl
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Ama Random Forest daha iyi performans gösterdiği için onu kullanıyoruz
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Model performansını değerlendir
        y_pred = model.predict(X_test)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        # Hocanızın kullandığı metrikler
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"Model eğitildi:")
        print(f"  MSE: {mse:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.3f}")
        print(f"  Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")

        # Modeli kaydet
        joblib.dump(model, 'price_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')

        return True

    except Exception as e:
        print(f"Model eğitim hatası: {e}")
        return False


def load_model():
    """Kaydedilmiş modeli yükle"""
    global model, scaler
    try:
        if os.path.exists('price_model.pkl') and os.path.exists('scaler.pkl'):
            model = joblib.load('price_model.pkl')
            scaler = joblib.load('scaler.pkl')
            return True
    except:
        pass
    return False


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Model bilgilerini döndür"""
    try:
        if model is None:
            return jsonify({
                "status": "not_trained",
                "message": "Model not trained yet. Collect at least 10 properties to train."
            })

        df = pd.read_csv('homes.csv')
        df = clean_data(df)

        # Model bilgileri
        info = {
            "status": "trained",
            "algorithm": "Random Forest Regressor",
            "n_estimators": model.n_estimators,
            "training_samples": len(df),
            "model_file_date": datetime.fromtimestamp(os.path.getmtime('price_model.pkl')).strftime(
                '%Y-%m-%d %H:%M:%S') if os.path.exists('price_model.pkl') else "N/A"
        }

        return jsonify(info)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/predict', methods=['POST'])
def predict():
    """Tahmin endpoint'i"""
    try:
        data = request.json
        m2 = float(data.get('m2', 0))

        if m2 <= 0:
            return jsonify({"error": "Invalid square meter Value"}), 400

        if model is None or scaler is None:
            return jsonify({"ERROR": "the model has not yet been trained"}), 400

        # Tahmin yap
        X_pred = scaler.transform([[m2]])
        prediction = model.predict(X_pred)[0]

        # Güven aralığı hesapla (basit bir yaklaşım)
        predictions = []
        for tree in model.estimators_:
            predictions.append(tree.predict(X_pred)[0])

        std_dev = np.std(predictions)
        lower_bound = prediction - 2 * std_dev
        upper_bound = prediction + 2 * std_dev

        return jsonify({
            "prediction": int(prediction),
            "lower_bound": int(max(0, lower_bound)),
            "upper_bound": int(upper_bound),
            "per_m2": int(prediction / m2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """İstatistikleri döndür"""
    try:
        df = pd.read_csv('homes.csv')
        df = clean_data(df)

        stats = {
            "total_homes": len(df),
            "avg_price": int(df['price'].mean()),
            "avg_m2": int(df['features'].mean()),
            "avg_price_per_m2": int(df['price'].mean() / df['features'].mean()),
            "min_price": int(df['price'].min()),
            "max_price": int(df['price'].max()),
            "min_m2": int(df['features'].min()),
            "max_m2": int(df['features'].max()),
            "last_update": datetime.fromtimestamp(os.path.getmtime('homes.csv')).strftime('%Y-%m-%d %H:%M:%S')
        }

        return jsonify(stats)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/chart', methods=['GET'])
def get_chart():
    """Grafik oluştur"""
    try:
        df = pd.read_csv('homes.csv')
        df = clean_data(df)

        plt.figure(figsize=(10, 8))
        plt.scatter(df['features'], df['price'], alpha=0.6, edgecolors='black', linewidth=0.5)
        plt.xlabel('Area (m²)', fontsize=12)
        plt.ylabel('Price (TL)', fontsize=12)
        plt.title('Price-Area Relationship', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # Fiyat formatlama
        ax = plt.gca()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

        # Trend çizgisi ekle
        z = np.polyfit(df['features'], df['price'], 1)
        p = np.poly1d(z)
        plt.plot(df['features'], p(df['features']), "r--", alpha=0.8, label=f'Trend: {z[0]:.0f}x + {z[1]:.0f}')
        plt.legend()

        # Grafiği base64 olarak döndür
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        plt.close()

        img_base64 = base64.b64encode(img.getvalue()).decode()
        return jsonify({"image": f"data:image/png;base64,{img_base64}"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/', methods=['GET', 'POST'])
def index():
    show_output = False
    home_list = []
    error_message = None

    if request.method == 'POST':
        home_link = request.form.get('home_link')
        if home_link:
            try:
                # URL kontrolü
                if not home_link.startswith("https://www.emlakjet.com"):
                    error_message = "ERROR: Only emlakjet.com URLs are accepted!"
                else:
                    home_list = update_homes(home_link)
                    show_output = True

                    # Yeni veri toplandıysa modeli yeniden eğit
                    if len(home_list) > 10:
                        train_model()
            except Exception as e:
                error_message = f"Hata: {str(e)}"
    else:
        # GET request - sayfa ilk açıldığında mevcut verileri göster
        if os.path.exists('homes.csv'):
            try:
                home_list = []
                with open('homes.csv', 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            home = {
                                "location": row["location"],
                                "features": int(row["features"]),
                                "price": int(row["price"])
                            }
                            home_list.append(home)
                        except Exception as e:
                            continue

                if len(home_list) > 0:
                    show_output = True
            except Exception as e:
                print(f"CSV okuma hatası: {e}")

    return render_template("index.html",
                           title="Real Estate Price Prediction System",
                           homes=home_list,
                           show_output=show_output,
                           error_message=error_message)


# Uygulama başladığında modeli yükle
with app.app_context():
    if not load_model():
        print("Model is failed to train, new model eğitilecek")
        if os.path.exists('homes.csv'):
            train_model()

if __name__ == '__main__':
    app.run(debug=True)