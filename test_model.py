#!/usr/bin/env python3
"""
Test script to verify the model loading and prediction works correctly
"""

import os
import sys
from train_model import PropertyValuePredictor

def test_model_loading():
    """Test if the model can be loaded successfully"""
    print("🔍 Testing model loading...")
    
    predictor = PropertyValuePredictor()
    
    # Check if model file exists
    model_file = 'simple_property_model.joblib'
    if os.path.exists(model_file):
        print(f"✅ Model file '{model_file}' found")
        file_size = os.path.getsize(model_file) / (1024 * 1024)
        print(f"📦 File size: {file_size:.2f} MB")
    else:
        print(f"❌ Model file '{model_file}' not found")
        return False
    
    # Try to load the model
    try:
        predictor.load_model(model_file)
        print("✅ Model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def test_prediction():
    """Test if predictions work correctly"""
    print("\n🔮 Testing predictions...")
    
    predictor = PropertyValuePredictor()
    predictor.load_model('simple_property_model.joblib')
    
    # Sample property data
    sample_property = {
        'current_value': 500000,
        'year_built': 2010,
        'square_feet': 2000,
        'bedrooms': 3,
        'bathrooms': 2.5,
        'state': 'CA',
        'property_type': 'Single Family Home',
        'school_district_rating': 'Good'
    }
    
    try:
        result = predictor.predict_future_value(sample_property)
        print("✅ Prediction successful!")
        print(f"   Current value: ${result.get('current_value', 0):,.2f}")
        print(f"   Predicted 5-year value: ${result.get('predicted_future_value', 0):,.2f}")
        print(f"   Appreciation: {result.get('appreciation_percentage', 0):.2f}%")
        return True
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Model Testing Suite")
    print("=" * 50)
    
    # Test 1: Model loading
    loading_success = test_model_loading()
    
    if loading_success:
        # Test 2: Prediction
        prediction_success = test_prediction()
        
        if prediction_success:
            print("\n🎉 All tests passed! Model is ready for deployment.")
            return True
        else:
            print("\n❌ Prediction test failed!")
            return False
    else:
        print("\n❌ Model loading test failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
