import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Try to import the PropertyValuePredictor class
try:
    from train_model import PropertyValuePredictor
    print("✅ Successfully imported PropertyValuePredictor from train_model")
except ImportError as e:
    print(f"❌ Import error: {e}")
    # Fallback: create a simple predictor class
    class PropertyValuePredictor:
        def __init__(self):
            self.model = None
            self.scaler = None
            self.feature_columns = None
            self.label_encoders = {}
            
        def load_model(self, model_path):
            try:
                if os.path.exists(model_path):
                    model_data = joblib.load(model_path)
                    self.model = model_data['model']
                    self.scaler = model_data['scaler']
                    self.feature_columns = model_data['feature_columns']
                    self.label_encoders = model_data.get('label_encoders', {})
                    print(f"✅ Model loaded from {model_path}")
                    return True
                else:
                    print(f"❌ Model file {model_path} not found")
                    return False
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                return False
        
        def predict_future_value(self, property_data):
            if self.model is None:
                return {"error": "Model not loaded"}
            
            try:
                # Convert to DataFrame
                if isinstance(property_data, dict):
                    df = pd.DataFrame([property_data])
                else:
                    df = property_data.copy()
                
                # Feature engineering
                df['property_age'] = 2025 - df['year_built']
                df['price_per_sqft'] = df['current_value'] / df['square_feet']
                df['value_per_bedroom'] = df['current_value'] / df['bedrooms'].replace(0, 1)
                
                # Encode categorical features
                for col, encoder in self.label_encoders.items():
                    if col.replace('_encoded', '') in df.columns:
                        original_col = col.replace('_encoded', '')
                        try:
                            df[col] = encoder.transform(df[original_col].astype(str))
                        except:
                            df[col] = 0
                
                # Select features
                if self.feature_columns:
                    # Ensure all features exist
                    for col in self.feature_columns:
                        if col not in df.columns:
                            df[col] = 0
                    
                    X = df[self.feature_columns]
                else:
                    # Use available numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    X = df[numeric_cols]
                
                # Scale features
                if self.scaler:
                    X_scaled = self.scaler.transform(X)
                else:
                    X_scaled = X.values
                
                # Predict
                predicted_value = self.model.predict(X_scaled)[0]
                
                # Calculate metrics
                current_value = df['current_value'].iloc[0]
                appreciation = predicted_value - current_value
                appreciation_pct = (appreciation / current_value) * 100
                
                return {
                    'current_value': current_value,
                    'predicted_future_value': predicted_value,
                    'predicted_appreciation': appreciation,
                    'appreciation_percentage': appreciation_pct,
                    'annual_appreciation_rate': appreciation_pct / 5                }
                
            except Exception as e:
                return {"error": f"Prediction error: {str(e)}"}

class PropertyPredictionApp:
    def __init__(self):
        self.predictor = PropertyValuePredictor()
        self.model_loaded = False
        self.load_model()
        
        # State mappings
        self.states = [
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
            'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
            'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
        ]
        
        self.property_types = [
            'Single Family Home', 'Townhouse', 'Condo', 'Multi-Family', 
            'Duplex', 'Ranch', 'Colonial', 'Victorian', 'Modern', 'Split Level'        ]
        
        self.school_ratings = [
            'Excellent', 'Very Good', 'Good', 'Average', 'Below Average'
        ]
    
    def load_model(self):
        """Load the trained model with fallback options"""
        model_files = [
            'simple_property_model.joblib',
            'enhanced_property_model.joblib',
            'property_value_model.joblib'
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                try:
                    success = self.predictor.load_model(model_file)
                    if success:
                        self.model_loaded = True
                        print(f"✅ Successfully loaded model from {model_file}")
                        return
                except Exception as e:
                    print(f"❌ Failed to load {model_file}: {e}")
                    continue
        
        print("❌ No model file found. Using demo mode.")
        self.model_loaded = False
    
    def predict_property_value(self, current_value, year_built, state, property_type,
                             bedrooms, bathrooms, square_feet, lot_size, 
                             school_district_rating, monthly_rent, property_tax, 
                             hoa_monthly):
        """Predict future property value"""
        
        if not self.model_loaded:
            # Demo mode - simple calculation
            base_appreciation = 0.05  # 5% annual
            years = 5
            demo_value = current_value * (1 + base_appreciation) ** years
            
            prediction_text = (
                f"🚨 **Demo Mode** (No model loaded)\n\n"
                f"**Current Property Value:** ${current_value:,.2f}\n\n"
                f"**Estimated 5-Year Value:** ${demo_value:,.2f}\n\n"
                f"**Estimated Appreciation:** ${demo_value - current_value:,.2f} ({((demo_value - current_value) / current_value) * 100:.1f}%)\n\n"
                f"**Annual Appreciation Rate:** {((demo_value / current_value) ** (1/5) - 1) * 100:.1f}%\n\n"
                f"*Note: This is a demo calculation using 5% annual appreciation. Train and upload a model for accurate predictions.*"
            )
            
            chart = self.create_prediction_chart(current_value, None)
            return prediction_text, chart
        
        try:
            # Prepare property data
            property_data = {
                'current_value': current_value,
                'year_built': year_built,
                'state': state,
                'property_type': property_type,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'square_feet': square_feet,
                'lot_size': lot_size,
                'school_district_rating': school_district_rating,
                'monthly_rent_estimate': monthly_rent,
                'property_tax_annual': property_tax,
                'hoa_monthly': hoa_monthly
            }
            
            # Get prediction
            result = self.predictor.predict_future_value(property_data)
            
            if "error" in result:
                error_text = f"❌ **Error:** {result['error']}"
                chart = self.create_prediction_chart(current_value, None)
                return error_text, chart
            
            # Format results
            current_val = result['current_value']
            future_val = result['predicted_future_value']
            appreciation = result['predicted_appreciation']
            appreciation_pct = result['appreciation_percentage']
            annual_rate = result['annual_appreciation_rate']
            
            prediction_text = (
                f"## 🏠 Property Value Prediction Results\n\n"
                f"**Current Property Value:** ${current_val:,.2f}\n\n"
                f"**Predicted 5-Year Value:** ${future_val:,.2f}\n\n"
                f"**Total Appreciation:** ${appreciation:,.2f}\n\n"
                f"**Appreciation Percentage:** {appreciation_pct:.2f}%\n\n"
                f"**Annual Appreciation Rate:** {annual_rate:.2f}%\n\n"
                f"---\n\n"
                f"📊 **Investment Analysis:**\n"
                f"- Monthly appreciation: ${appreciation/60:,.2f}\n"
                f"- ROI over 5 years: {appreciation_pct:.1f}%\n"
                f"- Estimated monthly rent: ${monthly_rent:,.2f}\n"
                f"- Property tax burden: {(property_tax/current_val)*100:.2f}% of value"            )
            
            # Create chart with actual prediction
            chart = self.create_prediction_chart(current_val, future_val)
            
            return prediction_text, chart
            
        except Exception as e:
            error_text = f"❌ **Prediction Error:** {str(e)}"
            chart = self.create_prediction_chart(current_value, None)
            return error_text, chart
    
    def create_prediction_chart(self, current_value, predicted_value=None):
        """Create a chart showing actual prediction results"""
        years = list(range(0, 6))
        
        if predicted_value is None or not self.model_loaded:
            # Demo mode - use 5% annual appreciation
            values = [current_value * (1.05 ** year) for year in years]
            title = "Demo: Property Value Growth (5% Annual)"
            line_color = '#FFA500'  # Orange for demo
        else:
            # Use actual prediction - interpolate between current and predicted
            annual_rate = (predicted_value / current_value) ** (1/5) - 1
            total_appreciation = ((predicted_value - current_value) / current_value) * 100
            values = [current_value * (1 + annual_rate) ** year for year in years]
            title = f"AI Prediction: {total_appreciation:.1f}% Total Growth ({annual_rate*100:.1f}% Annual)"
            line_color = '#2E86AB'  # Blue for real predictions
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years,
            y=values,
            mode='lines+markers',
            name='Property Value',
            line=dict(color=line_color, width=3),
            marker=dict(size=8)
        ))
        
        # Add annotations for key points
        fig.add_annotation(
            x=0, y=values[0],
            text=f"Current: ${values[0]:,.0f}",
            showarrow=True,
            arrowhead=2,
            bgcolor="white",
            bordercolor=line_color
        )
        fig.add_annotation(
            x=5, y=values[5],
            text=f"5-Year: ${values[5]:,.0f}",
            showarrow=True,
            arrowhead=2,
            bgcolor="white",
            bordercolor=line_color
        )
        
        # Add total appreciation annotation for ML predictions
        if predicted_value is not None and self.model_loaded:
            total_appreciation = ((predicted_value - current_value) / current_value) * 100
            fig.add_annotation(
                x=2.5, y=values[2] + (values[5] - values[2]) * 0.3,
                text=f"Total Growth: +{total_appreciation:.1f}%",
                showarrow=False,
                bgcolor="rgba(46, 134, 171, 0.8)",
                bordercolor=line_color,
                font=dict(color="white", size=14, family="Arial Black")
            )
        
        fig.update_layout(
            title=title,
            xaxis_title="Years from Now",
            yaxis_title="Property Value ($)",
            template="plotly_white",
            height=400,
            yaxis=dict(tickformat='$,.0f')
        )
        
        return fig
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        with gr.Blocks(title="🏠 Property Value Predictor", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 🏠 AI Property Value Predictor
            
            Predict your property's value 5 years from now using advanced machine learning!
            
            **How it works:**
            1. Enter your property details below
            2. Click "Predict Future Value" 
            3. Get instant AI-powered predictions with detailed analysis
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Property Details")
                    
                    current_value = gr.Number(
                        label="Current Property Value ($)",
                        value=500000,
                        minimum=50000,
                        maximum=10000000,
                        step=10000
                    )
                    
                    year_built = gr.Number(
                        label="Year Built",
                        value=2010,
                        minimum=1800,
                        maximum=2025,
                        step=1
                    )
                    
                    state = gr.Dropdown(
                        choices=self.states,
                        label="State",
                        value="CA"
                    )
                    
                    property_type = gr.Dropdown(
                        choices=self.property_types,
                        label="Property Type",
                        value="Single Family Home"
                    )
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 🏡 Property Features")
                    
                    bedrooms = gr.Number(
                        label="Bedrooms",
                        value=3,
                        minimum=1,
                        maximum=10,
                        step=1
                    )
                    
                    bathrooms = gr.Number(
                        label="Bathrooms",
                        value=2.5,
                        minimum=1,
                        maximum=10,
                        step=0.5
                    )
                    
                    square_feet = gr.Number(
                        label="Square Feet",
                        value=2000,
                        minimum=300,
                        maximum=20000,
                        step=100
                    )
                    
                    lot_size = gr.Number(
                        label="Lot Size (acres)",
                        value=0.25,
                        minimum=0.01,
                        maximum=5.0,
                        step=0.01
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 💰 Financial Details")
                    
                    school_rating = gr.Dropdown(
                        choices=self.school_ratings,
                        label="School District Rating",
                        value="Good"
                    )
                    
                    monthly_rent = gr.Number(
                        label="Estimated Monthly Rent ($)",
                        value=2500,
                        minimum=500,
                        maximum=20000,
                        step=100
                    )
                    
                    property_tax = gr.Number(
                        label="Annual Property Tax ($)",
                        value=6000,
                        minimum=500,
                        maximum=100000,
                        step=500
                    )
                    
                    hoa_monthly = gr.Number(
                        label="Monthly HOA Fee ($)",
                        value=0,
                        minimum=0,
                        maximum=2000,
                        step=50
                    )
            
            with gr.Row():
                predict_btn = gr.Button(
                    "🔮 Predict Future Value", 
                    variant="primary",
                    size="lg"
                )
            
            with gr.Row():
                with gr.Column(scale=2):                    prediction_output = gr.Markdown(
                        label="Prediction Results",
                        value="Enter property details and click 'Predict Future Value' to see AI predictions!"
                    )
                
                with gr.Column(scale=1):
                    chart_output = gr.Plot(label="Value Projection Chart")
            
            # Event handlers
            predict_btn.click(
                fn=self.predict_property_value,
                inputs=[
                    current_value, year_built, state, property_type,
                    bedrooms, bathrooms, square_feet, lot_size,
                    school_rating, monthly_rent, property_tax, hoa_monthly
                ],
                outputs=[prediction_output, chart_output]
            )
            
            # Update chart when current value changes (demo mode only)
            current_value.change(
                fn=lambda val: self.create_prediction_chart(val, None),
                inputs=current_value,
                outputs=chart_output
            )
            
            # Set initial chart
            interface.load(
                fn=lambda: self.create_prediction_chart(500000, None),
                outputs=chart_output
            )
            
            gr.Markdown("""
            ---
            ### 📝 About This Tool
            
            This AI-powered property value predictor uses machine learning to analyze:
            - **Property characteristics** (age, size, type, location)
            - **Market factors** (school ratings, comparable sales)
            - **Financial metrics** (rent potential, taxes, HOA fees)
            - **Historical trends** and regional market patterns
            
            **Disclaimer:** Predictions are estimates based on historical data and should not be considered as professional real estate advice.
            """)
        
        return interface

def main():
    """Main function to create and launch the app"""
    print("🚀 Starting Property Value Predictor App...")
    
    app = PropertyPredictionApp()
    interface = app.create_interface()
    
    # Launch the interface
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )

if __name__ == "__main__":
    main()
