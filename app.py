from flask import Flask, request, jsonify, send_from_directory, send_file
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
import joblib
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
# Manually define the dataset based on your Excel sheet
def create_manual_dataset():
    # Create dataset directly from your Excel data
    data = {
        'crop_name': [
    'rice', 'maize', 'jowar', 'bajra', 'ragi', 'wheat', 'groundnut', 'soyabean', 'sunflower', 'sesame', 'castor', 'redgram', 'bengalgram', 'greengram', 'blackgram', 'horsegram', 'tomato', 'chilli', 'brinjal',  'okra', 'cabbage', 'cauliflower', 'gourds', 'onion', 'potato', 'carrot',  'beans', 'mango', 'cashew', 'banana', 'guava', 'papaya', 'mosambi',  'pomegranate', 'sapota', 'watermelon', 'muskmelon', 'lemon', 'turmeric',  'cotton', 'sugarcane'
    ],
        
        'ph_min': [
            5.5, 5.5, 6, 6, 4.5, 6, 6, 6, 6, 5.5, 6, 6, 6, 6.2, 6, 5.5, 6, 6, 5.5, 6, 6, 6, 6, 6, 5, 6, 5.5, 5.5, 5.5, 6, 5, 6, 6, 6, 6, 6, 6, 5.5, 5.5, 6, 6
        ],
        
        'ph_max': [
           7, 7.5, 7.5, 8.5,7.5,7.5,7.5,7.5,7.5,8,8,7.5,8,7.5,7.5,7.5,7,7,7.5,7.5,6.8,7,7.5,7.5,6.5,6.8,7,7.5,6.5,7.5,7.5,6.5,8,7.5,8,7.5,7.5,7.5,7,8,8

        ],
        
        'water_consumption': [
    '1200-1899mm', '500-800mm', '400-600mm', '250-400mm', '350-500mm','450-600mm', '500-700mm', '450-700mm', '500-600mm', '300-500mm', '500-700mm', '350-600mm', '250-400mm', '300-400mm', '350-400mm', '200-300mm', '600-800mm', '500-700mm', '500-700mm', '400-600mm','500-700mm','500-700mm', '500-800mm', '350-550mm', '500-700mm', '300-500mm', '300-500mm', '1200-1600mm', '600-1200mm', '1800-2200mm', '800-1200mm', '1500-2000mm', '900-1200mm', '500-800mm', '1000-1250mm', '400-600mm', '300-500mm', '900-1200mm', '1200-1500mm', '600-1200mm', '1500-2500mm'   
    ],
        
        'soil_types': [
            'clay,silt,alluvial','loamy','red,sandy,loam,black', 'sandy,loam,black,red','red,loam,laterite,black,sandy','loam,clay,alluvial', 'sandy,loam,red','black,loamy,sandy','loamy,sandy,black,red','sandy,loamy,red,black', 'sandy,red,black', 'loamy,red,sandy,loam,black', 'black,loamy,sandy','sandy,loam,black','sandy,loam,black', 'red,sandy,black,laterite','sandy,red,loam,black','sandy,red,loam,black','sandy,red,clay,black','sandy,red,black', 'sandy,loam,clay','sandy,loam,clay','sandy,loam,red,black', 'loamy,sandy,alluvial','loamy,sandy,alluvial','sandy,loam','sandy,loam,black','loamy,red,black,lateritic', 'red,lateritic', 'loamy,alluvial,red,black','sandy,red,alluvial,black', 'sandy,red,alluvial','loamy,sandy,alluvial', 'loamy,sandy,black', 'red,alluvial,black,lateritic', 'sandy,alluvial,red,loamy', 'sandy,loam,red,alluvial', 'sandy,loam,red,black','loamy,red,alluvial,black', 'loamy,red,alluvial,black', 'black,alluvial,sandy,red,loam'
        ],
    
        'districts': [
'Adilabad, Nizamabad, Jagtial, Nirmal, Peddapalli, Karimnagar, Rajanna Sircilla, Kamareddy, Sangareddy, Medak, Siddipet, Jangaon, Hanumakonda, Warangal, Mulugu, Bhadradri Kothagudem, Khammam, Mahabubabad, Suryapet, Nalgonda', 'Adilabad, Nizamabad, Jagtial, Nirmal, Peddapalli, Karimnagar, Rajanna Sircilla, Kamareddy, Sangareddy, Medak, Siddipet, Jangaon, Hanumakonda, Warangal, Mulugu, Bhadradri Kothagudem, Khammam, Mahabubabad, Suryapet, Nalgonda',  
'Adilabad, Nizamabad, Jagtial, Nirmal, Peddapalli, Karimnagar, Rajanna Sircilla, Kamareddy, Sangareddy, Medak, Siddipet, Jangaon, Hanumakonda, Warangal, Mulugu, Bhadradri Kothagudem, Khammam, Mahabubabad, Suryapet, Nalgonda ', 'Sangareddy, Medak, Siddipet, Jangaon, Hanumakonda, Warangal, Mulugu, Bhadradri Kothagudem, Khammam, Mahabubabad, Suryapet, Nalgonda',  'Mahabubnagar, Nalgonda, Khammam, Mulugu, Bhadradri Kothagudem, Nagarkurnool, Wanaparthy, Warangal  ','Nalgonda, Khammam, Mahabubnagar, Adilabad, Nizamabad, Karimnagar  ','Adilabad, Nizamabad, Jagtial, Nirmal, Peddapalli, Karimnagar, Rajanna Sircilla, Kamareddy, Sangareddy, Medak, Siddipet, Jangaon, Hanumakonda, Warangal, Mulugu, Bhadradri Kothagudem, Khammam, Mahabubabad, Suryapet, Nalgonda ', 'Adilabad, Nizamabad, Jagtial, Nirmal, Peddapalli, Karimnagar, Rajanna Sircilla, Kamareddy, Sangareddy, Medak, Siddipet, Jangaon, Hanumakonda, Warangal, Mulugu, Bhadradri Kothagudem, Khammam, Mahabubabad, Suryapet, Nalgonda  ','Adilabad, Nizamabad, Karimnagar, Rajanna Sircilla, Medak, Sangareddy, Nalgonda, Mahabubnagar, Wanaparthy, Nagarkurnool, Khammam, Bhadradri Kothagudem  ','Mahabubnagar, Nalgonda, Nagarkurnool, Wanaparthy, Khammam, Bhadradri Kothagudem, Medak, Sangareddy, Kamareddy, Warangal, Jangaon  ','Mahabubnagar, Nalgonda, Wanaparthy, Nagarkurnool, Khammam, Bhadradri Kothagudem, Sangareddy, Kamareddy, Medak, Warangal, Jangaon  ','Adilabad, Nizamabad, Jagtial, Nirmal, Peddapalli, Karimnagar, Rajanna Sircilla, Kamareddy, Sangareddy, Medak, Siddipet, Jangaon, Hanumakonda, Warangal, Mulugu, Bhadradri Kothagudem, Khammam, Mahabubabad, Suryapet, Nalgonda  ','Adilabad, Nizamabad, Jagtial, Nirmal, Peddapalli, Karimnagar, Rajanna Sircilla, Kamareddy, Sangareddy, Medak, Siddipet, Jangaon, Hanumakonda, Warangal, Mulugu, Bhadradri Kothagudem, Khammam, Mahabubabad, Suryapet, Nalgonda ', 'Adilabad, Nizamabad, Jagtial, Nirmal, Karimnagar, Rajanna Sircilla, Kamareddy, Sangareddy, Mahabubnagar, Nagarkurnool, Wanaparthy, Khammam, Warangal, Nalgonda, Suryapet  ','Adilabad, Nizamabad, Karimnagar, Nirmal, Rajanna Sircilla, Jangaon, Mahabubnagar, Wanaparthy, Nagarkurnool, Medak, Kamareddy, Sangareddy, Khammam, Warangal, Nalgonda, Suryapet  ','Mahabubnagar, Wanaparthy, Nagarkurnool, Nalgonda, Khammam, Bhadradri Kothagudem, Adilabad, Jogulamba Gadwal, Narayanpet  ','Nalgonda, Suryapet, Warangal, Jangaon, Hanumakonda, Karimnagar, Rajanna Sircilla, Peddapalli, Khammam, Bhadradri Kothagudem, Mahabubabad, Mulugu, Sangareddy, Kamareddy, Adilabad, Nizamabad, Jagtial, Nirmal  ','Adilabad, Nizamabad, Jagtial, Nirmal, Karimnagar, Rajanna Sircilla, Kamareddy, Sangareddy, Medak, Siddipet, Jangaon, Hanumakonda, Warangal, Mulugu, Bhadradri Kothagudem, Khammam, Mahabubabad, Nalgonda, Suryapet  ','Nalgonda, Suryapet, Warangal, Khammam, Hanumakonda, Mahabubabad, Karimnagar, Rajanna Sircilla, Medak, Sangareddy, Kamareddy, Nizamabad, Adilabad, Jagtial, Nirmal  ','Warangal, Nalgonda, Suryapet, Khammam, Hanumakonda, Karimnagar, Rajanna Sircilla, Mahabubabad, Nizamabad, Kamareddy, Medak, Sangareddy, Jagtial, Adilabad, Nirmal, Mahabubnagar  ','Ranga Reddy, Medchal Malkajgiri, Nalgonda, Suryapet, Warangal, Hanumakonda, Karimnagar, Kamareddy, Sangareddy, Medak, Khammam, Mahabubnagar, Nizamabad  ','Ranga Reddy, Medchal Malkajgiri, Nalgonda, Suryapet, Warangal, Hanumakonda, Sangareddy, Medak, Kamareddy, Khammam, Mahabubnagar, Karimnagar, Nizamabad  ','Nalgonda, Suryapet, Warangal, Khammam, Hanumakonda, Karimnagar, Rajanna Sircilla, Sangareddy, Medak, Kamareddy, Nizamabad, Adilabad, Jagtial, Nirmal, Ranga Reddy, Medchal Malkajgiri, Mahabubnagar  ','Nizamabad, Nalgonda, Suryapet, Warangal, Karimnagar, Hanumakonda, Rajanna Sircilla, Kamareddy, Sangareddy, Medak, Mahabubnagar, Khammam, Adilabad, Jagtial, Nirmal, Ranga Reddy, Medchal Malkajgiri  ','Medchal Malkajgiri, Ranga Reddy, Nalgonda, Suryapet, Warangal, Hanumakonda, Sangareddy, Kamareddy, Khammam, Karimnagar, Mahabubnagar, Nizamabad, Adilabad, Jagtial, Nirmal  ','Ranga Reddy, Medchal Malkajgiri, Nalgonda, Suryapet, Warangal, Hanumakonda, Kamareddy, Sangareddy, Medak, Khammam, Mahabubnagar, Karimnagar, Nizamabad, Jagtial, Adilabad, Nirmal  ','Ranga Reddy, Medchal Malkajgiri, Nalgonda, Suryapet, Warangal, Hanumakonda, Khammam, Sangareddy, Kamareddy, Mahabubnagar, Karimnagar, Nizamabad, Jagtial, Adilabad, Nirmal  ','Khammam, Mahabubnagar, Nalgonda, Suryapet, Warangal, Hanumakonda, Karimnagar, Rajanna Sircilla, Kamareddy, Nizamabad, Jagtial, Adilabad, Nirmal, Siddipet, Sangareddy, Vikarabad  ','Khammam, Bhadradri Kothagudem, Mahabubabad, Suryapet, Nalgonda, Warangal, Mulugu, Jayashankar Bhupalpally, Mahabubnagar, Nagarkurnool  ','Khammam, Bhadradri Kothagudem, Nalgonda, Suryapet, Warangal, Hanumakonda, Mahabubnagar, Jogulamba Gadwal, Nizamabad, Karimnagar, Rajanna Sircilla, Kamareddy, Adilabad, Jagtial, Medak, Sangareddy  ','Ranga Reddy, Mahabubnagar, Nalgonda, Suryapet, Warangal, Khammam, Karimnagar, Rajanna Sircilla, Nizamabad, Kamareddy, Sangareddy, Medak, Adilabad, Jagtial, Nirmal, Vikarabad  ','Ranga Reddy, Nalgonda, Suryapet, Warangal, Hanumakonda, Karimnagar, Medchal, Sangareddy, Nizamabad, Kamareddy, Mahabubnagar, Khammam, Adilabad, Jagtial  ','Nalgonda, Suryapet, Khammam, Bhadradri Kothagudem, Mahabubabad, Warangal, Nizamabad, Kamareddy, Karimnagar, Jagtial, Rajanna Sircilla, Sangareddy, Vikarabad, Medak, Mahabubnagar, Adilabad  ','Mahabubnagar, Wanaparthy, Nagarkurnool, Vikarabad, Sangareddy, Ranga Reddy, Nalgonda, Suryapet, Khammam, Warangal, Karimnagar, Rajanna Sircilla, Medak, Kamareddy, Nizamabad, Jagtial, Adilabad, Mancherial  ','Ranga Reddy, Vikarabad, Sangareddy, Mahabubnagar, Wanaparthy, Nagarkurnool, Nalgonda, Suryapet, Khammam, Warangal, Karimnagar, Rajanna Sircilla, Nizamabad, Kamareddy, Adilabad, Jagtial  ','Nalgonda, Suryapet, Mahabubnagar, Nagarkurnool, Wanaparthy, Vikarabad, Sangareddy, Ranga Reddy, Karimnagar, Jagtial, Khammam, Bhadradri Kothagudem, Nizamabad, Kamareddy, Adilabad, Mancherial  ','Mahabubnagar, Wanaparthy, Nagarkurnool, Nalgonda, Suryapet, Vikarabad, Sangareddy, Ranga Reddy, Medak, Karimnagar, Jagtial, Nizamabad, Kamareddy, Khammam, Adilabad  ','Nalgonda, Suryapet, Mahabubnagar, Wanaparthy, Nagarkurnool, Vikarabad, Sangareddy, Ranga Reddy, Khammam, Warangal, Karimnagar, Kamareddy, Nizamabad, Jagtial, Adilabad, Mancherial  ','Nizamabad, Nirmal, Jagtial, Karimnagar, Warangal (Rural), Mahabubabad, Bhadradri Kothagudem, Khammam, Adilabad, Mancherial, Kamareddy, Medak  ','Adilabad, Nirmal, Mancherial, Nizamabad, Kamareddy, Karimnagar, Peddapalli, Jagtial, Rajanna Sircilla, Warangal (Urban & Rural), Mahabubabad, Khammam, Nalgonda, Suryapet, Mahabubnagar, Wanaparthy, Vikarabad, Sangareddy, Medak',  'Nizamabad, Kamareddy, Nalgonda, Suryapet, Medak, Sangareddy, Karimnagar, Warangal, Khammam, Bhadradri Kothagudem, Adilabad, Mahabubnagar, Wanaparthy'
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Process water consumption into numerical values
    df['water_consumption_value'] = df['water_consumption'].apply(
        lambda x: sum(int(v) for v in x.replace('mm', '').split('-')) / 2
    )
    
    # Extract soil types into separate boolean columns
    all_soil_types = set()
    for soil_string in df['soil_types']:
        soil_list = soil_string.split(',')
        all_soil_types.update(soil_list)
    
    soil_types = list(all_soil_types)
    
    for soil_type in soil_types:
        df[f'soil_{soil_type}'] = df['soil_types'].apply(lambda x: 1 if soil_type in x.split(',') else 0)
    
    # Create district mapping
    all_districts = set()
    for district_string in df['districts']:
        districts = [d.strip() for d in district_string.split(',')]
        all_districts.update(districts)
    
    district_mapping = {district: i for i, district in enumerate(all_districts)}
    
    return df, district_mapping, soil_types

# Train the machine learning model
def train_model(df):
    # Features for training (exclude non-numeric and target columns)
    feature_cols = [col for col in df.columns if col not in ['crop_name', 'soil_types', 'districts', 'water_consumption']]
    X = df[feature_cols]
    y = df['crop_name']
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, feature_cols

# Initialize data and model
df, district_mapping, soil_types = create_manual_dataset()
model, feature_cols = train_model(df)

# Save model path
model_path = 'crop_recommendation_model.joblib'
if not os.path.exists(model_path):
    joblib.dump(model, model_path)

@app.route('/api/recommend', methods=['POST'])
def recommend_crops():
    try:
        data = request.get_json()
        
        # Extract and process input data
        soil_ph = float(data['soil_ph'])
                # Check for out-of-range pH
        if soil_ph < 4.0 or soil_ph > 10.0:
            return jsonify({
                'status': 'success',
                'recommendations': [],
                'input_params': {
                    'soil_ph': soil_ph,
                    'soil_type': data['soil_type'],
                    'district': data['district'],
                    'water_availability': data['water_availability']
                },
                'message': 'No crops grow in this soil pH range. Please adjust the pH between 4.0 and 10.0.'
            })
        soil_type = data['soil_type']
        district = data['district']
        water_availability = data['water_availability']
        print("Received data:", data)
        
        # Convert water availability to numerical value
        water_map = {'low': 300, 'medium': 600, 'high': 900}
        water_value = water_map[water_availability]
        
        # Prepare input data for prediction
        input_data = {
            'ph_min': soil_ph - 0.5,
            'ph_max': soil_ph + 0.5,
            'water_consumption_value': water_value
        }
        
        # Add soil type features
        for st in soil_types:
            input_data[f'soil_{st}'] = 1 if st in soil_type else 0
        
        # Create features DataFrame with all required columns
        input_df = pd.DataFrame({col: [input_data.get(col, 0)] for col in feature_cols})
        
        # Make prediction
        probabilities = model.predict_proba(input_df)[0]
        crop_prob_pairs = list(zip(model.classes_, probabilities))
        crop_prob_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 5 recommended crops
        top_crops = crop_prob_pairs[:5]
        
        # Create detailed recommendations
        recommendations = []
        for crop_name, probability in top_crops:
            crop_data = df[df['crop_name'] == crop_name].iloc[0]
            
            recommendations.append({
                'crop_name': crop_name,
                'suitability_score': int(probability * 100),
                'water_requirements': crop_data['water_consumption'],
                'optimal_ph': f"{crop_data['ph_min']} - {crop_data['ph_max']}",
                'suitable_soil_types': crop_data['soil_types'],
                'growing_districts': crop_data['districts']
            })
        
        return jsonify({
            'status': 'success',
            'recommendations': recommendations,
            'input_params': {
                'soil_ph': soil_ph,
                'soil_type': soil_type,
                'district': district,
                'water_availability': water_availability
            }
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/soil_types', methods=['GET'])
def get_soil_types():
    unique_soil_types = df['soil_types'].unique().tolist()
    return jsonify({
        'status': 'success',
        'soil_types': unique_soil_types
    })

@app.route('/api/districts', methods=['GET'])
def get_districts():
    all_districts = set()
    for district_string in df['districts']:
        districts = [d.strip() for d in district_string.split(',')]
        all_districts.update(districts)
    
    return jsonify({
        'status': 'success',
        'districts': sorted(list(all_districts))
    })

# Add endpoint to serve static files (HTML, CSS, JS)
@app.route('/')
def index():
    # Assume the HTML file is in a templates directory
    return send_file('input.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)
@app.route('/crop/<crop_name>')
def crop_detail(crop_name):
    path = f'crops/{crop_name.lower()}.html'
    if os.path.exists(path):
        return send_file(path)
    else:
        return "Crop info not found", 404


if __name__ == '__main__':
    app.run(debug=True)