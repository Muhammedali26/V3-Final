from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_cors import CORS
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
import requests

app = Flask(__name__)
CORS(app)
app.secret_key = 'your-secret-key-here'  # Add this for flash messages

# API endpoint configuration
API_BASE_URL = "http://localhost:8002"  # Your FastAPI service URL

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():
    if request.method == 'POST':
        try:
            # Get form data directly (values are already 0-100)
            user_data = {
                "Name": request.form.get('name'),
                "DRYNESS": float(request.form.get('dryness', 0)),
                "DAMAGE": float(request.form.get('damage', 0)),
                "SENSITIVITY": float(request.form.get('sensitivity', 0)),
                "SEBUM_Oil": float(request.form.get('sebum_oil', 0)),
                "DRY_SCALP": float(request.form.get('dry_scalp', 0)),
                "FLAKES": float(request.form.get('flakes', 0)),
                "Goal": request.form.get('goal', ''),
                "Issue": request.form.get('issue', ''),
                "Hair_Type": request.form.get('hair_type', ''),
                "Hair_Texture": request.form.get('hair_texture', ''),
                "Hair_Behaviour": request.form.get('hair_behaviour', ''),
                "Scalp_Feeling": request.form.get('scalp_feeling', ''),
                "Wash_Hair": request.form.get('wash_hair', ''),
                "Treatments": request.form.get('treatments', ''),
                "Scalp_Flaky": request.form.get('scalp_flaky', ''),
                "Oily_Scalp": request.form.get('oily_scalp', ''),
                "Dry_Scalp": request.form.get('dry_scalp', '')
            }

            # Save user preferences
            preferences_path = os.path.join(os.path.dirname(__file__), "preferences.json")
            
            # Load existing preferences
            preferences = []
            if os.path.exists(preferences_path):
                with open(preferences_path, 'r') as f:
                    preferences = json.load(f)

            # Check if user exists and get their ID
            user_name = user_data["Name"]
            existing_user = None
            for pref in preferences:
                if pref.get('Name') == user_name:
                    existing_user = pref
                    break

            if existing_user:
                # Update existing user
                user_data['Unique_ID'] = existing_user['Unique_ID']
                user_data['timestamp'] = datetime.now().isoformat()
                # Update the existing record
                for i, pref in enumerate(preferences):
                    if pref.get('Name') == user_name:
                        preferences[i] = user_data
                        break
            else:
                # Add new user with incremented ID
                next_id = 1
                for pref in preferences:
                    if pref.get('Name', '').startswith(user_name):
                        try:
                            id_num = int(pref.get('Unique_ID', '').split('#')[1])
                            next_id = max(next_id, id_num + 1)
                        except:
                            pass
                user_data['Unique_ID'] = f"{user_name} #{next_id}"
                user_data['timestamp'] = datetime.now().isoformat()
                preferences.append(user_data)

            # Save updated preferences
            with open(preferences_path, 'w') as f:
                json.dump(preferences, f, indent=4)

            # Get recommendations from API
            try:
                response = requests.post(f"{API_BASE_URL}/recommend", json=user_data)
                api_data = response.json()
                
                # Create recommendations structure
                recommendations = {
                    'user': {
                        'base_name': user_data.get('Name'),
                        'unique_id': user_data.get('Unique_ID')
                    },
                    'condition_scores': {
                        'DRYNESS': max(float(api_data.get('condition_scores', {}).get('DRYNESS', 0)), float(user_data.get('DRYNESS', 0))),
                        'DAMAGE': max(float(api_data.get('condition_scores', {}).get('DAMAGE', 0)), float(user_data.get('DAMAGE', 0))),
                        'SENSITIVITY': max(float(api_data.get('condition_scores', {}).get('SENSITIVITY', 0)), float(user_data.get('SENSITIVITY', 0))),
                        'SEBUM_Oil': max(float(api_data.get('condition_scores', {}).get('SEBUM_Oil', 0)), float(user_data.get('SEBUM_Oil', 0))),
                        'DRY_SCALP': max(float(api_data.get('condition_scores', {}).get('DRY_SCALP', 0)), float(user_data.get('DRY_SCALP', 0))),
                        'FLAKES': max(float(api_data.get('condition_scores', {}).get('FLAKES', 0)), float(user_data.get('FLAKES', 0)))
                    },
                    'primary_condition': api_data.get('primary_condition', 'Unknown'),
                    'primary_category': api_data.get('primary_category', 'Unknown'),
                    'recommendations': {}
                }

                # Process recommendations by category
                api_recommendations = api_data.get('recommendations', {})
                for category, products in api_recommendations.items():
                    recommendations['recommendations'][category] = []
                    
                    # Convert single product to list if necessary
                    if isinstance(products, dict):
                        products = [products]
                    
                    # Process each product
                    for i, product in enumerate(products[:2]):  # Take first two products
                        # Extract product details
                        product_name = product.get('product', '')
                        if isinstance(product_name, dict):
                            product_name = product_name.get('product', '')
                        
                        processed_product = {
                            'product': {'product': product_name},
                            'final_score': product.get('final_score', 0),
                            'sentiment_scores': {
                                'likability': product.get('sentiment_scores', {}).get('likability', 0),
                                'effectiveness': product.get('sentiment_scores', {}).get('effectiveness', 0),
                                'value_for_money': product.get('sentiment_scores', {}).get('value_for_money', 0),
                                'ingredient_quality': product.get('sentiment_scores', {}).get('ingredient_quality', 0),
                                'ease_of_use': product.get('sentiment_scores', {}).get('ease_of_use', 0)
                            },
                            'usage_frequency': product.get('usage_frequency', '')
                        }
                        recommendations['recommendations'][category].append(processed_product)

                    # If we have less than 2 products, pad with empty products
                    while len(recommendations['recommendations'][category]) < 2:
                        empty_product = {
                            'product': {'product': 'No recommendation available'},
                            'final_score': 0,
                            'sentiment_scores': {
                                'likability': 0,
                                'effectiveness': 0,
                                'value_for_money': 0,
                                'ingredient_quality': 0,
                                'ease_of_use': 0
                            },
                            'usage_frequency': ''
                        }
                        recommendations['recommendations'][category].append(empty_product)

                # Update user data with recommendations
                for pref in preferences:
                    if pref.get('Unique_ID') == user_data['Unique_ID']:
                        pref.update(recommendations)
                        break
                
                # Save updated preferences with recommendations
                with open(preferences_path, 'w') as f:
                    json.dump(preferences, f, indent=4)
                
                return render_template('recommendations.html', recommendations=recommendations)
            except Exception as e:
                print(f"Error getting recommendations: {str(e)}")
                return render_template('error.html', error=str(e))

        except Exception as e:
            print(f"Error processing form: {str(e)}")
            return render_template('error.html', error=str(e))

    return render_template('form.html')

@app.route('/view-recommendations/<user_id>')
def view_recommendations(user_id):
    try:
        # Load preferences from JSON file
        preferences_path = os.path.join(os.path.dirname(__file__), "preferences.json")
        if not os.path.exists(preferences_path):
            flash('User data not found', 'error')
            return redirect(url_for('name_search'))
        
        with open(preferences_path, 'r') as f:
            preferences = json.load(f)
        
        # Find user by ID
        user_data = None
        for pref in preferences:
            if pref.get('Unique_ID') == user_id:
                user_data = pref
                break
        
        if not user_data:
            flash('User not found', 'error')
            return redirect(url_for('name_search'))

        # Try to get recommendations from API if not present
        try:
            response = requests.post(f"{API_BASE_URL}/recommend", json=user_data)
            api_data = response.json()
            
            # Create recommendations structure
            recommendations = {
                'user': {
                    'base_name': user_data.get('Name'),
                    'unique_id': user_data.get('Unique_ID')
                },
                'condition_scores': {
                    'DRYNESS': max(float(api_data.get('condition_scores', {}).get('DRYNESS', 0)), float(user_data.get('DRYNESS', 0))),
                    'DAMAGE': max(float(api_data.get('condition_scores', {}).get('DAMAGE', 0)), float(user_data.get('DAMAGE', 0))),
                    'SENSITIVITY': max(float(api_data.get('condition_scores', {}).get('SENSITIVITY', 0)), float(user_data.get('SENSITIVITY', 0))),
                    'SEBUM_Oil': max(float(api_data.get('condition_scores', {}).get('SEBUM_Oil', 0)), float(user_data.get('SEBUM_Oil', 0))),
                    'DRY_SCALP': max(float(api_data.get('condition_scores', {}).get('DRY_SCALP', 0)), float(user_data.get('DRY_SCALP', 0))),
                    'FLAKES': max(float(api_data.get('condition_scores', {}).get('FLAKES', 0)), float(user_data.get('FLAKES', 0)))
                },
                'primary_condition': api_data.get('primary_condition', 'Unknown'),
                'primary_category': api_data.get('primary_category', 'Unknown'),
                'recommendations': {}
            }

            # Process recommendations by category
            api_recommendations = api_data.get('recommendations', {})
            for category, products in api_recommendations.items():
                recommendations['recommendations'][category] = []
                for product in products:
                    processed_product = {
                        'product': {'product': product.get('product_name', '')},
                        'final_score': product.get('confidence_score', 0),
                        'sentiment_scores': product.get('sentiment_scores', {}),
                        'usage_frequency': product.get('usage_frequency', '')
                    }
                    recommendations['recommendations'][category].append(processed_product)

            # Update user data with new recommendations
            user_data.update(api_data)
            for i, pref in enumerate(preferences):
                if pref.get('Unique_ID') == user_id:
                    preferences[i] = user_data
                    break

            # Save updated preferences
            with open(preferences_path, 'w') as f:
                json.dump(preferences, f, indent=4)

            return render_template('recommendations.html', recommendations=recommendations)

        except Exception as api_error:
            print(f"API Error: {str(api_error)}")
            # If API call fails, try to use existing recommendations
            recommendations = {
                'user': {
                    'base_name': user_data.get('Name'),
                    'unique_id': user_data.get('Unique_ID')
                },
                'condition_scores': {
                    'DRYNESS': max(float(api_data.get('condition_scores', {}).get('DRYNESS', 0)), float(user_data.get('DRYNESS', 0))),
                    'DAMAGE': max(float(api_data.get('condition_scores', {}).get('DAMAGE', 0)), float(user_data.get('DAMAGE', 0))),
                    'SENSITIVITY': max(float(api_data.get('condition_scores', {}).get('SENSITIVITY', 0)), float(user_data.get('SENSITIVITY', 0))),
                    'SEBUM_Oil': max(float(api_data.get('condition_scores', {}).get('SEBUM_Oil', 0)), float(user_data.get('SEBUM_Oil', 0))),
                    'DRY_SCALP': max(float(api_data.get('condition_scores', {}).get('DRY_SCALP', 0)), float(user_data.get('DRY_SCALP', 0))),
                    'FLAKES': max(float(api_data.get('condition_scores', {}).get('FLAKES', 0)), float(user_data.get('FLAKES', 0)))
                },
                'primary_condition': api_data.get('primary_condition', 'Unknown'),
                'primary_category': api_data.get('primary_category', 'Unknown'),
                'recommendations': api_data.get('recommendations', {})
            }
            
            return render_template('recommendations.html', recommendations=recommendations)
        
    except Exception as e:
        print(f"Error in view_recommendations: {str(e)}")
        return render_template('error.html', error=str(e))

@app.route('/name-search', methods=['GET', 'POST'])
def name_search():
    try:
        if request.method == 'POST':
            search_name = request.form.get('name')
            if not search_name:
                flash('Please enter a name to search', 'warning')
                return render_template('name_search.html')
            
            # Call the FastAPI recommend_by_name endpoint
            try:
                response = requests.post(
                    f"{API_BASE_URL}/recommend_by_name",
                    json={"Name": search_name}
                )
                api_data = response.json()
                
                # Debug: Print raw API response
                print("\nRaw API Response from name search:")
                print(json.dumps(api_data, indent=2))
                
                if "recommendations" in api_data and api_data["recommendations"]:
                    # Process recommendations
                    recommendations = {
                        'user': {
                            'base_name': api_data["matched_user"]["name"],
                            'unique_id': api_data["matched_user"]["unique_id"]
                        },
                        'condition_scores': {
                            'DRYNESS': max(float(api_data.get('condition_scores', {}).get('DRYNESS', 0)), float(api_data.get('condition_scores', {}).get('DRYNESS', 0))),
                            'DAMAGE': max(float(api_data.get('condition_scores', {}).get('DAMAGE', 0)), float(api_data.get('condition_scores', {}).get('DAMAGE', 0))),
                            'SENSITIVITY': max(float(api_data.get('condition_scores', {}).get('SENSITIVITY', 0)), float(api_data.get('condition_scores', {}).get('SENSITIVITY', 0))),
                            'SEBUM_Oil': max(float(api_data.get('condition_scores', {}).get('SEBUM_Oil', 0)), float(api_data.get('condition_scores', {}).get('SEBUM_Oil', 0))),
                            'DRY_SCALP': max(float(api_data.get('condition_scores', {}).get('DRY_SCALP', 0)), float(api_data.get('condition_scores', {}).get('DRY_SCALP', 0))),
                            'FLAKES': max(float(api_data.get('condition_scores', {}).get('FLAKES', 0)), float(api_data.get('condition_scores', {}).get('FLAKES', 0)))
                        },
                        'primary_condition': api_data.get('primary_condition', 'Unknown'),
                        'primary_category': api_data.get('primary_category', 'Unknown'),
                        'recommendations': {}
                    }

                    # Print condition scores for debugging
                    print("\nProcessed condition scores:")
                    for condition, score in recommendations['condition_scores'].items():
                        print(f"{condition}: {score:.2%}")

                    # Process recommendations by category
                    api_recommendations = api_data.get('recommendations', {})
                    print("\nProcessing recommendations by category:")
                    for category, products in api_recommendations.items():
                        print(f"\nCategory: {category}")
                        print(f"Raw products: {json.dumps(products, indent=2)}")
                        
                        recommendations['recommendations'][category] = []
                        
                        # Convert single product to list if necessary
                        if isinstance(products, dict):
                            products = [products]
                        
                        # Process each product
                        for i, product in enumerate(products[:2]):  # Take first two products
                            print(f"\nProcessing product {i+1}:")
                            print(json.dumps(product, indent=2))
                            
                            # Extract product details
                            product_name = product.get('product', '')
                            if isinstance(product_name, dict):
                                product_name = product_name.get('product', '')
                            
                            processed_product = {
                                'product': {'product': product_name},
                                'final_score': product.get('final_score', 0),
                                'sentiment_scores': {
                                    'likability': product.get('sentiment_scores', {}).get('likability', 0),
                                    'effectiveness': product.get('sentiment_scores', {}).get('effectiveness', 0),
                                    'value_for_money': product.get('sentiment_scores', {}).get('value_for_money', 0),
                                    'ingredient_quality': product.get('sentiment_scores', {}).get('ingredient_quality', 0),
                                    'ease_of_use': product.get('sentiment_scores', {}).get('ease_of_use', 0)
                                },
                                'usage_frequency': product.get('usage_frequency', '')
                            }
                            recommendations['recommendations'][category].append(processed_product)
                            print(f"Processed product: {json.dumps(processed_product, indent=2)}")

                        # If we have less than 2 products, pad with empty products
                        while len(recommendations['recommendations'][category]) < 2:
                            empty_product = {
                                'product': {'product': 'No recommendation available'},
                                'final_score': 0,
                                'sentiment_scores': {
                                    'likability': 0,
                                    'effectiveness': 0,
                                    'value_for_money': 0,
                                    'ingredient_quality': 0,
                                    'ease_of_use': 0
                                },
                                'usage_frequency': ''
                            }
                            recommendations['recommendations'][category].append(empty_product)
                            print(f"Added empty product to category {category}")

                        # Debug print final structure
                        print("\nFinal recommendations structure:")
                        print(json.dumps(recommendations, indent=2))

                    # Save the recommendations to preferences.json
                    preferences_path = os.path.join(os.path.dirname(__file__), "preferences.json")
                    preferences = []
                    if os.path.exists(preferences_path):
                        with open(preferences_path, 'r') as f:
                            preferences = json.load(f)
                    
                    # Update or add the new recommendations
                    user_data = {
                        'Name': api_data["matched_user"]["name"],
                        'Unique_ID': api_data["matched_user"]["unique_id"],
                        'timestamp': datetime.now().isoformat(),
                        **recommendations  # Include processed recommendations
                    }
                    
                    # Update existing or append new
                    updated = False
                    for i, pref in enumerate(preferences):
                        if pref.get('Unique_ID') == user_data['Unique_ID']:
                            preferences[i] = user_data
                            updated = True
                            break
                    
                    if not updated:
                        preferences.append(user_data)
                    
                    # Save updated preferences
                    with open(preferences_path, 'w') as f:
                        json.dump(preferences, f, indent=4)
                    
                    # Return the recommendations page directly
                    return render_template('recommendations.html', recommendations=recommendations)
                else:
                    # No recommendations found
                    return render_template('name_search.html', 
                                        search_name=search_name,
                                        search_results=[])
                    
            except requests.RequestException as e:
                print(f"API Error: {str(e)}")
                flash('Error connecting to recommendation service', 'error')
                return render_template('name_search.html', search_name=search_name)
        
        return render_template('name_search.html')
        
    except Exception as e:
        print(f"Error in name_search: {str(e)}")
        return render_template('error.html', error=str(e))

@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/submit-preferences', methods=['POST'])
def submit_preferences():
    try:
        data = request.get_json()
        # Process the preferences data here
        return jsonify({
            "status": "success",
            "message": "Preferences saved successfully"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 