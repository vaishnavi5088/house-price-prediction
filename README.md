# 🏡 House Price Prediction – Machine Learning Project

This project predicts **house prices** using a trained machine learning regression model.  
The project includes data preprocessing, model training, feature importance extraction, and a Flask-based web application for real-time price prediction.

## 📂 Project Structure
```
.
├── templates/
│   └── index.html          
├── app.py                  
├── project.py              
├── train.py                
├── model.pkl               
├── columns.pkl             
├── feature_importance      
├── requirements.txt        
```

## 🚀 Project Description
This is an end-to-end **machine learning project** that predicts house prices based on input features. It includes:

- Data preprocessing  
- Feature engineering  
- Model training and evaluation  
- Feature importance  
- Flask web application for predictions  

## 🧠 Key Features
- Trained ML model  
- Flask-based prediction UI  
- Feature importance insights  
- Reusable `.pkl` model files  

## 🌐 Web Application
The web interface built using `index.html` allows users to input property details and get an instant prediction.

Backend logic is handled by `app.py`.

## 🔧 How to Run the Project

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Train the model
(Optional)
```
python train.py
```

### 3. Run the Flask app
```
python app.py
```

Visit: http://127.0.0.1:5000/

## 📊 Files Explained
- `model.pkl` – saved trained ML model  
- `columns.pkl` – feature metadata  
- `feature_importance` – importance of each input feature  
- `train.py` – script to train model  
- `project.py` – helpers/utilities  
- `index.html` – UI for predictions  

## 🛠 Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-Learn  
- Flask  
- HTML (Jinja2 Templates)  
- Pickle  

## 👩‍💻 Author
**Vaishnavi Patil**  
Data Scientist • ML Enthusiast

## ⭐ Support
If you like this project, please ⭐ star this repository!
