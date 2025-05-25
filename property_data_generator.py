import csv
import random
from datetime import datetime, timedelta
import os

def generate_property_dataset(filename='property_dataset_3gb.csv', target_size_mb=3000):
    """
    Generate a large property dataset to reach approximately 3GB (10x larger)
    """
    
    # Lists of realistic data for generation
    cities_states = [
        ('New York', 'NY'), ('Los Angeles', 'CA'), ('Chicago', 'IL'), ('Houston', 'TX'),
        ('Phoenix', 'AZ'), ('Philadelphia', 'PA'), ('San Antonio', 'TX'), ('San Diego', 'CA'),
        ('Dallas', 'TX'), ('San Jose', 'CA'), ('Austin', 'TX'), ('Jacksonville', 'FL'),
        ('Fort Worth', 'TX'), ('Columbus', 'OH'), ('Charlotte', 'NC'), ('San Francisco', 'CA'),
        ('Indianapolis', 'IN'), ('Seattle', 'WA'), ('Denver', 'CO'), ('Washington', 'DC'),
        ('Boston', 'MA'), ('El Paso', 'TX'), ('Detroit', 'MI'), ('Nashville', 'TN'),
        ('Portland', 'OR'), ('Memphis', 'TN'), ('Oklahoma City', 'OK'), ('Las Vegas', 'NV'),
        ('Louisville', 'KY'), ('Baltimore', 'MD'), ('Milwaukee', 'WI'), ('Albuquerque', 'NM'),
        ('Tucson', 'AZ'), ('Fresno', 'CA'), ('Sacramento', 'CA'), ('Long Beach', 'CA'),
        ('Kansas City', 'MO'), ('Mesa', 'AZ'), ('Virginia Beach', 'VA'), ('Atlanta', 'GA'),
        ('Colorado Springs', 'CO'), ('Omaha', 'NE'), ('Raleigh', 'NC'), ('Miami', 'FL'),
        ('Oakland', 'CA'), ('Minneapolis', 'MN'), ('Tulsa', 'OK'), ('Cleveland', 'OH'),
        ('Wichita', 'KS'), ('Arlington', 'TX'), ('New Orleans', 'LA'), ('Bakersfield', 'CA'),
        ('Tampa', 'FL'), ('Honolulu', 'HI'), ('Aurora', 'CO'), ('Anaheim', 'CA'),
        ('Santa Ana', 'CA'), ('St. Louis', 'MO'), ('Riverside', 'CA'), ('Corpus Christi', 'TX'),
        ('Lexington', 'KY'), ('Pittsburgh', 'PA'), ('Anchorage', 'AK'), ('Stockton', 'CA'),
        ('Cincinnati', 'OH'), ('St. Paul', 'MN'), ('Toledo', 'OH'), ('Greensboro', 'NC'),
        ('Newark', 'NJ'), ('Plano', 'TX'), ('Henderson', 'NV'), ('Lincoln', 'NE'),
        ('Buffalo', 'NY'), ('Jersey City', 'NJ'), ('Chula Vista', 'CA'), ('Fort Wayne', 'IN'),
        ('Orlando', 'FL'), ('St. Petersburg', 'FL'), ('Chandler', 'AZ'), ('Laredo', 'TX'),
        ('Norfolk', 'VA'), ('Durham', 'NC'), ('Madison', 'WI'), ('Lubbock', 'TX'),
        ('Irvine', 'CA'), ('Winston-Salem', 'NC'), ('Glendale', 'AZ'), ('Garland', 'TX'),
        ('Hialeah', 'FL'), ('Reno', 'NV'), ('Chesapeake', 'VA'), ('Gilbert', 'AZ'),
        ('Baton Rouge', 'LA'), ('Irving', 'TX'), ('Scottsdale', 'AZ'), ('North Las Vegas', 'NV'),
        ('Fremont', 'CA'), ('Boise', 'ID'), ('Richmond', 'VA'), ('San Bernardino', 'CA'),
        ('Birmingham', 'AL'), ('Spokane', 'WA'), ('Rochester', 'NY'), ('Des Moines', 'IA'),
        ('Modesto', 'CA'), ('Fayetteville', 'NC'), ('Tacoma', 'WA'), ('Oxnard', 'CA'),
        ('Fontana', 'CA'), ('Columbus', 'GA'), ('Montgomery', 'AL'), ('Moreno Valley', 'CA'),
        ('Shreveport', 'LA'), ('Aurora', 'IL'), ('Yonkers', 'NY'), ('Akron', 'OH'),
        ('Huntington Beach', 'CA'), ('Little Rock', 'AR'), ('Augusta', 'GA'), ('Amarillo', 'TX'),
        ('Glendale', 'CA'), ('Mobile', 'AL'), ('Grand Rapids', 'MI'), ('Salt Lake City', 'UT'),
        ('Tallahassee', 'FL'), ('Huntsville', 'AL'), ('Grand Prairie', 'TX'), ('Knoxville', 'TN'),
        ('Worcester', 'MA'), ('Newport News', 'VA'), ('Brownsville', 'TX'), ('Overland Park', 'KS'),
        ('Santa Clarita', 'CA'), ('Providence', 'RI'), ('Garden Grove', 'CA'), ('Chattanooga', 'TN'),
        ('Oceanside', 'CA'), ('Jackson', 'MS'), ('Fort Lauderdale', 'FL'), ('Santa Rosa', 'CA'),
        ('Rancho Cucamonga', 'CA'), ('Port St. Lucie', 'FL'), ('Tempe', 'AZ'), ('Ontario', 'CA'),
        ('Vancouver', 'WA'), ('Cape Coral', 'FL'), ('Sioux Falls', 'SD'), ('Springfield', 'MO'),
        ('Peoria', 'AZ'), ('Pembroke Pines', 'FL'), ('Elk Grove', 'CA'), ('Salem', 'OR'),
        ('Lancaster', 'CA'), ('Corona', 'CA'), ('Eugene', 'OR'), ('Palmdale', 'CA'),
        ('Salinas', 'CA'), ('Springfield', 'MA'), ('Pasadena', 'CA'), ('Fort Collins', 'CO'),
        ('Hayward', 'CA'), ('Pomona', 'CA'), ('Cary', 'NC'), ('Rockford', 'IL'),
        ('Alexandria', 'VA'), ('Escondido', 'CA'), ('Sunnyvale', 'CA'), ('Paterson', 'NJ'),
        ('Kansas City', 'KS'), ('Hollywood', 'FL'), ('Torrance', 'CA'), ('Bridgeport', 'CT'),
        ('Savannah', 'GA'), ('Lakewood', 'CO'), ('Bellevue', 'WA'), ('Charleston', 'SC')
    ]
    
    # Property types to add variety
    property_types = [
        'Single Family Home', 'Townhouse', 'Condo', 'Multi-Family', 'Duplex', 
        'Ranch', 'Colonial', 'Victorian', 'Modern', 'Split Level'
    ]
    
    # Property features to add more data
    features = [
        'Hardwood Floors', 'Granite Countertops', 'Stainless Steel Appliances',
        'Central Air', 'Fireplace', 'Swimming Pool', 'Garage', 'Basement',
        'Deck/Patio', 'Garden', 'Security System', 'Solar Panels'
    ]
    
    # School districts
    school_districts = [
        'Excellent', 'Very Good', 'Good', 'Average', 'Below Average'
    ]
    
    def get_base_price_by_state(state):
        """Return base price range based on state"""
        high_cost_states = ['CA', 'NY', 'MA', 'WA', 'HI', 'CT', 'NJ', 'MD', 'VA', 'DC']
        medium_cost_states = ['TX', 'FL', 'CO', 'NC', 'GA', 'AZ', 'NV', 'OR', 'IL', 'PA']
        
        if state in high_cost_states:
            return (400000, 1200000)
        elif state in medium_cost_states:
            return (250000, 700000)
        else:
            return (150000, 500000)
    
    def calculate_value_increases(base_value, state):
        """Calculate 5-year value increases with some regional variation"""
        # Base appreciation rate varies by region
        high_growth_states = ['CA', 'WA', 'TX', 'FL', 'CO', 'AZ']
        
        if state in high_growth_states:
            base_rate = random.uniform(0.04, 0.08)  # 4-8% annual growth
        else:
            base_rate = random.uniform(0.02, 0.06)  # 2-6% annual growth
        
        increases = []
        current_value = base_value
        
        for year in range(5):
            # Add some randomness to each year
            year_rate = base_rate + random.uniform(-0.015, 0.015)
            increase = int(current_value * year_rate)
            increases.append(increase)
            current_value += increase
        
        return increases
    
    # Estimate rows needed for 300MB (roughly 2000 characters per row)
    target_rows = (target_size_mb * 1024 * 1024) // 2000
    print(f"Generating approximately {target_rows:,} rows to reach {target_size_mb}MB...")
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'property_id', 'current_value', 'year_built', 'city', 'state',
            'property_type', 'bedrooms', 'bathrooms', 'square_feet', 'lot_size',
            'school_district_rating', 'property_features', 'neighborhood',
            'year_1_increase', 'year_2_increase', 'year_3_increase', 
            'year_4_increase', 'year_5_increase', 'total_5yr_appreciation',
            'monthly_rent_estimate', 'property_tax_annual', 'hoa_monthly',
            'days_on_market', 'listing_agent', 'last_sold_date', 'last_sold_price'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(1, target_rows + 1):
            city, state = random.choice(cities_states)
            year_built = random.randint(1950, 2024)
            
            # Get price range based on state
            min_price, max_price = get_base_price_by_state(state)
            
            # Adjust price based on year built (newer = more expensive)
            age_factor = 1 + (2024 - year_built) * -0.01  # Depreciate 1% per year of age
            age_factor = max(0.7, age_factor)  # Minimum 70% of base price
            
            current_value = int(random.randint(min_price, max_price) * age_factor)
            
            # Calculate increases
            increases = calculate_value_increases(current_value, state)
            total_appreciation = sum(increases)
            
            # Generate additional property details
            bedrooms = random.randint(1, 6)
            bathrooms = round(random.uniform(1, 4.5), 1)
            square_feet = random.randint(800, 4500)
            lot_size = round(random.uniform(0.1, 2.0), 2)
            
            # Property features (3-8 features per property)
            num_features = random.randint(3, 8)
            selected_features = random.sample(features, num_features)
            features_str = ', '.join(selected_features)
            
            # Generate neighborhood name
            neighborhood_types = ['Heights', 'Hills', 'Gardens', 'Park', 'Grove', 'Manor', 'Valley', 'Ridge']
            neighborhood_names = ['Oak', 'Maple', 'Pine', 'Cedar', 'Willow', 'Rose', 'Sunset', 'Green']
            neighborhood = f"{random.choice(neighborhood_names)} {random.choice(neighborhood_types)}"
            
            # Estimate monthly rent (roughly 0.5-1% of property value per month)
            monthly_rent = int(current_value * random.uniform(0.005, 0.01))
            
            # Property tax (varies by state, roughly 0.5-2.5% annually)
            property_tax = int(current_value * random.uniform(0.005, 0.025))
            
            # HOA fees (0-500 monthly)
            hoa_monthly = random.randint(0, 500) if random.random() > 0.3 else 0
            
            # Days on market
            days_on_market = random.randint(1, 180)
            
            # Generate agent name
            first_names = ['John', 'Sarah', 'Michael', 'Lisa', 'David', 'Jennifer', 'Robert', 'Emily', 'James', 'Ashley']
            last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
            agent_name = f"{random.choice(first_names)} {random.choice(last_names)}"
            
            # Last sold date (within last 5 years)
            days_ago = random.randint(0, 1825)  # 5 years
            last_sold_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
            
            # Last sold price (90-110% of current value adjusted for time)
            time_factor = 1 - (days_ago / 365) * 0.05  # 5% annual appreciation
            last_sold_price = int(current_value * time_factor * random.uniform(0.9, 1.1))
            
            row = {
                'property_id': i,
                'current_value': current_value,
                'year_built': year_built,
                'city': city,
                'state': state,
                'property_type': random.choice(property_types),
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'square_feet': square_feet,
                'lot_size': lot_size,
                'school_district_rating': random.choice(school_districts),
                'property_features': features_str,
                'neighborhood': neighborhood,
                'year_1_increase': increases[0],
                'year_2_increase': increases[1],
                'year_3_increase': increases[2],
                'year_4_increase': increases[3],
                'year_5_increase': increases[4],
                'total_5yr_appreciation': total_appreciation,
                'monthly_rent_estimate': monthly_rent,
                'property_tax_annual': property_tax,
                'hoa_monthly': hoa_monthly,
                'days_on_market': days_on_market,
                'listing_agent': agent_name,
                'last_sold_date': last_sold_date,
                'last_sold_price': last_sold_price
            }
            
            writer.writerow(row)
            
            # Progress indicator
            if i % 10000 == 0:
                current_size = os.path.getsize(filename) / (1024 * 1024)
                print(f"Generated {i:,} rows - Current file size: {current_size:.1f}MB")
    
    # Final file size
    final_size = os.path.getsize(filename) / (1024 * 1024)
    print(f"\nDataset generation complete!")
    print(f"File: {filename}")
    print(f"Total rows: {target_rows:,}")
    print(f"Final file size: {final_size:.1f}MB")
    print(f"Columns: {len(fieldnames)}")

if __name__ == "__main__":
    # Generate the dataset
    generate_property_dataset()
    
    print("\nDataset includes:")
    print("- Property ID, current value, year built, location")
    print("- Property details (type, bedrooms, bathrooms, sq ft)")
    print("- Market data (rent estimates, property tax, HOA fees)")
    print("- 5-year value projections with regional variations")
    print("- Additional features like neighborhood, agent, sale history")
    print("- Realistic data based on US real estate markets")