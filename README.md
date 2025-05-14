# Real Estate Price Prediction

![Real Estate](https://img.shields.io/badge/Real%20Estate-Price%20Prediction-blue)
![Python](https://img.shields.io/badge/Python-3.x-brightgreen)
![Flask](https://img.shields.io/badge/Flask-2.x-red)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-yellow)

## Project Overview

This project simulates a real-world data science scenario where you work as a data scientist for a real estate company (like Zillow or Magic Bricks). The goal is to build a machine learning model that can predict property prices based on features such as square footage, number of bedrooms/bathrooms, location, and other relevant factors.

In addition to building the prediction model, this project includes a web application that allows users to get property price estimates through an intuitive interface.

## Features

- **Data Analysis & Processing**: Clean and prepare real estate data from Bangalore, India
- **Machine Learning Model**: Predict property prices using regression techniques
- **Interactive Web Application**: User-friendly interface for obtaining price predictions
- **RESTful API**: Backend service to handle prediction requests

## Tech Stack

### Backend
- Python
- Pandas (Data Cleaning)
- Matplotlib/Seaborn (Data Visualization)
- Scikit-Learn (Machine Learning)
- Flask (Web Server)
- Pickle (Model Serialization)

### Frontend
- HTML
- CSS
- JavaScript

## Project Architecture

1. **Data Collection**: Utilize a Bangalore real estate dataset from Kaggle
2. **Data Preprocessing**: 
   - Data cleaning
   - Feature engineering
   - Dimensionality reduction
   - Outlier removal
3. **Model Building**:
   - Train various regression models
   - Hyperparameter tuning
   - Model evaluation
   - Export the best model as a pickle file
4. **Web Service**:
   - Python Flask server that loads the trained model
   - RESTful API endpoints for price prediction
5. **User Interface**:
   - Responsive web design
   - Form for entering property details
   - Display of prediction results

## Getting Started

### Prerequisites
- Python 3.x
- pip package manager

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/real-estate-price-prediction.git
cd real-estate-price-prediction
```

2. Create and activate virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Running the Application

1. Start the Flask server:
```bash
python server/app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

## Project Structure

```
real-estate-price-prediction/
│
├── data/
│   ├── raw_data.csv              # Original dataset
│   └── processed_data.csv        # Cleaned dataset
│
├── model/
│   ├── model.pickle              # Serialized model
│   └── columns.json              # Feature information
│
├── notebooks/
│   ├── data_cleaning.ipynb       # Data preprocessing steps
│   └── model_building.ipynb      # Model creation and evaluation
│
├── server/
│   ├── app.py                    # Flask application
│   ├── util.py                   # Utility functions
│   └── artifacts/                # Saved model artifacts
│
├── static/
│   ├── css/                      # Stylesheet files
│   ├── js/                       # JavaScript files
│   └── images/                   # Image assets
│
├── templates/
│   ├── index.html                # Home page
│   └── prediction.html           # Results page
│
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

## Model Development

The machine learning pipeline implements several key data science concepts:

1. **Data Cleaning**: Handle missing values, incorrect formats, and duplicates
2. **Feature Engineering**: Create new features that help improve model performance
3. **Dimensionality Reduction**: Remove irrelevant features to improve model efficiency
4. **Outlier Removal**: Identify and handle outliers that might skew the model
5. **Model Selection**: Compare multiple regression algorithms to select the best performer
6. **Hyperparameter Tuning**: Optimize model parameters for best prediction accuracy

## API Reference

The Flask server exposes the following endpoints:

### Get Locations
```
GET /get_location_names
```
Returns all available locations in the dataset.

### Get Estimated Price
```
POST /predict_home_price
```
Request Body:
```json
{
  "location": "Example Location",
  "sqft": 1000,
  "bhk": 2,
  "bath": 2
}
```
Returns the estimated property price.

## Future Improvements

- Add user authentication
- Expand to more cities/regions
- Include more property features
- Implement advanced ML algorithms
- Add map-based visualization

## Contributors

- Your Name (@yourusername)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Kaggle for providing the dataset
- Zillow and Magic Bricks for inspiration
