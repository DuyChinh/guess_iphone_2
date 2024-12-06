from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import pickle
import os
from werkzeug.utils import secure_filename
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import re


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'model'  
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['CURRENT_MODEL'] = os.path.join(MODEL_FOLDER, 'stacking_model.pkl') 


for folder in [UPLOAD_FOLDER, MODEL_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    try:
        model_path = app.config.get('CURRENT_MODEL')
        if not os.path.exists(model_path):
            return None
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
@app.route('/get_models')
def get_models():
    """Lấy danh sách các model có sẵn"""
    try:
        model_files = glob.glob(os.path.join(app.config['MODEL_FOLDER'], '*.pkl'))
        models = [os.path.basename(f) for f in model_files]
        current_model = os.path.basename(app.config['CURRENT_MODEL'])
        return jsonify({
            'models': models,
            'current_model': current_model
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/select_model', methods=['POST'])
def select_model():
    """Chọn model để sử dụng"""
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        
        if not model_name:
            return jsonify({'error': 'Tên model không hợp lệ'}), 400
            
        model_path = os.path.join(app.config['MODEL_FOLDER'], model_name)
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model không tồn tại'}), 404
            
        try:
            test_model = joblib.load(model_path)
        except Exception as e:
            return jsonify({'error': f'Không thể load model: {str(e)}'}), 500
        
        app.config['CURRENT_MODEL'] = model_path
        
        return jsonify({
            'success': True, 
            'message': 'Đã chọn model thành công',
            'model_name': model_name
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info')
def model_info():
    """Lấy thông tin về model hiện tại"""
    try:
        current_model = os.path.basename(app.config['CURRENT_MODEL'])
        model_path = app.config['CURRENT_MODEL']
        model_size = os.path.getsize(model_path) 
        model_modified = os.path.getmtime(model_path) 
        
        return jsonify({
            'name': current_model,
            'size': f'{model_size}KB',
            'last_modified': model_modified,
            'path': model_path
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/predict_next_gen', methods=['GET'])
def predict_next_gen():
    try:
        next_gen_price = 45000000 
        return jsonify({
            'success': True,
            'price': next_gen_price
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

def classify_and_split_models(data):
    # Create a mapping for types
    type_mapping = {
        "Pro Max": 3,
        "Pro": 2,
        "Plus": 1,
        "Regular": 0
    }

    # Classify models into types
    def classify_model(model):
        if "Pro Max" in model:
            return "Pro Max"
        elif "Pro" in model:
            return "Pro"
        elif "Plus" in model:
            return "Plus"
        else:
            return "Regular"

    # Extract iPhone series (number part)
    def extract_series(model):
        import re
        match = re.search(r'\d+', model)
        return int(match.group()) if match else None

    # Apply classification
    data['Type'] = data['Model'].apply(classify_model)
    # Encode types into numbers
    data['Type_Code'] = data['Type'].map(type_mapping)
    # Extract series
    data['Series'] = data['Model'].apply(extract_series)

    return data


def extract_camera_specs(camera_str):
    # Extract all MP values using regex
    mp_values = re.findall(r'(\d+)MP', camera_str)
    # Convert to list of strings with 'MP' suffix
    mp_values = [f"{x}MP" for x in mp_values]
    # Pad with empty strings if less than 3 cameras
    while len(mp_values) < 3:
        mp_values.append('')
    return mp_values

def clean_price(price_str):
    # Loại bỏ ký tự ₫ và dấu chấm
    return int(price_str.replace('₫', '').replace('.', ''))


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        


        try:
            df = pd.read_excel(filepath)

            transformed_data = []   


            for _, row in df.iterrows():
                # Extract camera specifications
                camera_specs = extract_camera_specs(row['Real Camera'])

                # Create new row
                new_row = {
                    'Model': row['Model'],
                    'Dung lượng': row['Dung lượng bộ nhớ (GB)'],
                    'Launch Year': row['Launch Year'],
                    'Screen Size (inch)': row['Screen Size (inch)'],
                    'Processor': row['Processor'],
                    'RAM': row['RAM'],
                    'Battery (mAh)': row['Battery (mAh)'],
                    'Camera 1': camera_specs[0],
                    'Camera 2': camera_specs[1],
                    'Camera 3': camera_specs[2],
                    'Front Camera': row['Front Camera']
                }
                transformed_data.append(new_row)

            # Create transformed DataFrame
            transformed_df = pd.DataFrame(transformed_data)

            # transformed_df = parse_iphone_data(df)


            df = transformed_df
            df = classify_and_split_models(df)
            df['Processor'] = df['Processor'].apply(lambda x: ''.join(filter(str.isdigit, x)))  # Keeps only the numbers
            df['RAM'] = df['RAM'].apply(lambda x: ''.join(filter(str.isdigit, x)))  # Keeps only the numbers

            # Extract the numeric part from Camera columns
            df['Camera 1'] = df['Camera 1'].apply(lambda x: ''.join(filter(str.isdigit, x)) if x else 0)
            df['Camera 2'] = df['Camera 2'].apply(lambda x: ''.join(filter(str.isdigit, x)) if x else 0)
            df['Camera 3'] = df['Camera 3'].apply(lambda x: ''.join(filter(str.isdigit, x)) if x else 0)
            df['Front Camera'] = df['Front Camera'].apply(lambda x: ''.join(filter(str.isdigit, x)) if x else 0)

            df = df.drop(columns=['Model', 'Type'])


            
            X = df 

            # Chuẩn hóa dữ liệu số bằng Min-Max scaling
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled
            
            model = load_model()
            if model is None:
                return jsonify({'error': 'Không thể load model. Vui lòng kiểm tra lại model đã chọn'}), 500
            
            predictions = model.predict(df)
            predictions = predictions.round().astype(int)
            df['Predicted_Price'] = predictions
            
            result_filename = f'prediction_result_{os.path.splitext(filename)[0]}.xlsx'
            result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            df.to_excel(result_filepath, index=False)
            
            return jsonify({
                'success': True,
                'message': 'Dự đoán thành công',
                'predictions': df.to_dict(orient='records'),
                'result_file': result_filename
            })
            
        except Exception as e:
            return jsonify({'error': f'Lỗi khi xử lý: {str(e)}'}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
            
    return jsonify({'error': 'File không hợp lệ'}), 400

if __name__ == '__main__':
    app.run(debug=True)