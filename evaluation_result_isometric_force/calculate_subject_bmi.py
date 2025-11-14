import json
import os
import glob

# ==================== Configuration ====================
SUBJECT_INFO_DIR = "evaluation_result_isometric_force\\subject_info"
OUTPUT_DIR = os.path.join(os.path.dirname(SUBJECT_INFO_DIR), "output")


def calculate_bmi(weight_kg, height_cm):
    """Calculate BMI given weight in kg and height in cm.
    
    BMI = weight (kg) / (height (m))^2
    """
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2)
    return bmi


def get_bmi_category(bmi):
    """Return BMI category based on WHO standards."""
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def load_subject_info(json_file):
    """Load subject information from JSON file."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        subject_info = data.get('subject_information', {})
        
        # Extract relevant information
        name = subject_info.get('name', 'Unknown')
        age = subject_info.get('age', 'N/A')
        sex = subject_info.get('sex', 'N/A')
        height_cm = subject_info.get('height_cm', None)
        weight_kg = subject_info.get('weight_kg', None)
        
        # Convert to float if available
        if height_cm is not None and height_cm != '':
            height_cm = float(height_cm)
        else:
            height_cm = None
            
        if weight_kg is not None and weight_kg != '':
            weight_kg = float(weight_kg)
        else:
            weight_kg = None
        
        return {
            'name': name,
            'age': age,
            'sex': sex,
            'height_cm': height_cm,
            'weight_kg': weight_kg
        }
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return None


if __name__ == "__main__":
    # Find all subject info JSON files
    search_pattern = os.path.join(SUBJECT_INFO_DIR, "subject_info_*.json")
    json_files = glob.glob(search_pattern)
    
    if not json_files:
        print(f"No subject info files found in '{SUBJECT_INFO_DIR}'!")
        exit(1)
    
    print(f"Found {len(json_files)} subject info files\n")
    
    # Process all subjects
    subject_bmi_list = []
    
    for json_file in sorted(json_files):
        subject_data = load_subject_info(json_file)
        
        if subject_data is None:
            continue
        
        # Calculate BMI if height and weight are available
        if subject_data['height_cm'] is not None and subject_data['weight_kg'] is not None:
            bmi = calculate_bmi(subject_data['weight_kg'], subject_data['height_cm'])
            category = get_bmi_category(bmi)
            
            subject_bmi_list.append({
                'name': subject_data['name'],
                'age': subject_data['age'],
                'sex': subject_data['sex'],
                'height_cm': subject_data['height_cm'],
                'weight_kg': subject_data['weight_kg'],
                'bmi': bmi,
                'category': category
            })
        else:
            print(f"⚠️  Warning: Missing height or weight for {subject_data['name']}")
    
    # Sort by name
    subject_bmi_list.sort(key=lambda x: x['name'].lower())
    
    # Print results
    print("="*100)
    print("SUBJECT BMI CALCULATIONS")
    print("="*100)
    print(f"{'Name':<15} {'Age':<6} {'Sex':<8} {'Height (cm)':<12} {'Weight (kg)':<12} {'BMI':<8} {'Category':<15}")
    print("-"*100)
    
    for subject in subject_bmi_list:
        print(f"{subject['name']:<15} "
              f"{subject['age']:<6} "
              f"{subject['sex']:<8} "
              f"{subject['height_cm']:<12.1f} "
              f"{subject['weight_kg']:<12.1f} "
              f"{subject['bmi']:<8.2f} "
              f"{subject['category']:<15}")
    
    print("="*100)
    
    # Calculate summary statistics
    bmi_values = [s['bmi'] for s in subject_bmi_list]
    
    print(f"\nSummary Statistics:")
    print(f"  Total subjects: {len(subject_bmi_list)}")
    print(f"  Mean BMI: {sum(bmi_values)/len(bmi_values):.2f}")
    print(f"  Min BMI: {min(bmi_values):.2f} ({[s['name'] for s in subject_bmi_list if s['bmi'] == min(bmi_values)][0]})")
    print(f"  Max BMI: {max(bmi_values):.2f} ({[s['name'] for s in subject_bmi_list if s['bmi'] == max(bmi_values)][0]})")
    
    # Count by category
    categories = {}
    for subject in subject_bmi_list:
        cat = subject['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\nBMI Categories:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} ({count/len(subject_bmi_list)*100:.1f}%)")
    
    # Save to text file
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    output_file = os.path.join(OUTPUT_DIR, "subject_bmi_summary.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("SUBJECT BMI CALCULATIONS\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"{'Name':<15} {'Age':<6} {'Sex':<8} {'Height (cm)':<12} {'Weight (kg)':<12} {'BMI':<8} {'Category':<15}\n")
        f.write("-"*100 + "\n")
        
        for subject in subject_bmi_list:
            f.write(f"{subject['name']:<15} "
                   f"{subject['age']:<6} "
                   f"{subject['sex']:<8} "
                   f"{subject['height_cm']:<12.1f} "
                   f"{subject['weight_kg']:<12.1f} "
                   f"{subject['bmi']:<8.2f} "
                   f"{subject['category']:<15}\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"Total subjects: {len(subject_bmi_list)}\n")
        f.write(f"Mean BMI: {sum(bmi_values)/len(bmi_values):.2f}\n")
        f.write(f"Min BMI: {min(bmi_values):.2f} ({[s['name'] for s in subject_bmi_list if s['bmi'] == min(bmi_values)][0]})\n")
        f.write(f"Max BMI: {max(bmi_values):.2f} ({[s['name'] for s in subject_bmi_list if s['bmi'] == max(bmi_values)][0]})\n\n")
        
        f.write("BMI Categories:\n")
        for cat, count in sorted(categories.items()):
            f.write(f"  {cat}: {count} ({count/len(subject_bmi_list)*100:.1f}%)\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("\nBMI FORMULA: BMI = weight (kg) / (height (m))^2\n")
        f.write("\nWHO BMI Categories:\n")
        f.write("  - Underweight: BMI < 18.5\n")
        f.write("  - Normal weight: 18.5 ≤ BMI < 25\n")
        f.write("  - Overweight: 25 ≤ BMI < 30\n")
        f.write("  - Obese: BMI ≥ 30\n")
        f.write("="*100 + "\n")
    
    print(f"\n✓ Saved BMI summary to: {output_file}")
    
    # Also save as CSV for easy import to Excel/analysis tools
    csv_file = os.path.join(OUTPUT_DIR, "subject_bmi_data.csv")
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("Name,Age,Sex,Height_cm,Weight_kg,BMI,Category\n")
        for subject in subject_bmi_list:
            f.write(f"{subject['name']},"
                   f"{subject['age']},"
                   f"{subject['sex']},"
                   f"{subject['height_cm']:.1f},"
                   f"{subject['weight_kg']:.1f},"
                   f"{subject['bmi']:.2f},"
                   f"{subject['category']}\n")
    
    print(f"✓ Saved BMI data (CSV) to: {csv_file}")
    
    print("\n" + "="*100)
    print("BMI calculation complete!")
    print("="*100)

