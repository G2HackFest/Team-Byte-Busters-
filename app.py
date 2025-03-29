from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os

app = Flask(__name__)

# Mock dataset generation
def generate_mock_data():
    products = ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Monitor']
    data = []
    
    for product in products:
        for year in [2020, 2021, 2022, 2023]:
            for month in range(1, 13):
                # Base sales with seasonality
                base_sales = np.random.randint(50, 200)
                if month in [11, 12]: base_sales *= 1.5  # Holiday boost
                elif month in [6, 7, 8]: base_sales *= 0.8  # Summer dip
                
                # Yearly growth + randomness
                sales = int(base_sales * (1 + (year - 2020) * 0.1 * np.random.normal(1, 0.1)))
                
                data.append({
                    'product': product,
                    'date': f"{year}-{month:02d}-01",
                    'sales': sales,
                    'price': np.random.randint(300, 1500) if product == 'Laptop' else 
                            np.random.randint(200, 1000) if product == 'Smartphone' else
                            np.random.randint(100, 500) if product == 'Tablet' else
                            np.random.randint(50, 300) if product == 'Headphones' else
                            np.random.randint(150, 600),
                    'promotion': np.random.choice([0, 1], p=[0.7, 0.3])
                })
    return pd.DataFrame(data)

# Initialize data
DATA_FILE = 'sales_data.csv'
if not os.path.exists(DATA_FILE):
    df = generate_mock_data()
    df.to_csv(DATA_FILE, index=False)
else:
    df = pd.read_csv(DATA_FILE)

# Preprocess data
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter

# Train models
models = {}
for product in df['product'].unique():
    product_data = df[df['product'] == product]
    X = product_data[['year', 'month', 'quarter', 'price', 'promotion']]
    y = product_data['sales']
    model = LinearRegression()
    model.fit(X, y)
    models[product] = model

# Flask routes
@app.route('/')
def index():
    return render_template('index.html', products=sorted(df['product'].unique()))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    product = data['product']
    year = int(data['year'])
    price = float(data.get('price', df[df['product'] == product]['price'].median()))
    promotion = int(data.get('promotion', 0))
    
    predictions = []
    for month in range(1, 13):
        quarter = (month - 1) // 3 + 1
        predicted_sales = max(0, int(models[product].predict([[year, month, quarter, price, promotion]])[0]))
        predictions.append({
            'month': month,
            'predicted_sales': predicted_sales,
            'suggested_order': int(predicted_sales * 1.1)  # 10% buffer
        })
    
    return jsonify(predictions)

@app.route('/history/<product>')
def get_history(product):
    history = df[df['product'] == product].sort_values('date')
    return jsonify({
        'sales': history['sales'].tolist(),
        'dates': history['date'].astype(str).tolist(),
        'prices': history['price'].tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)