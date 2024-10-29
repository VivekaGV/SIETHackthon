import pandas as pd

def calculate_success_rate(df, drug_name):
    # Filter rows for the given drugName
    drug_rows = df[df['drugName'] == drug_name]
    
    if not drug_rows.empty:
        # Calculate total useful count and total positive useful count
        total_use_count = drug_rows['usefulCount'].sum()
        total_positive_count = drug_rows[drug_rows['sentiment'] == 'Positive']['usefulCount'].sum()
        
        # Compute success rate
        if total_use_count > 0:
            success_rate = (total_positive_count / total_use_count) * 100
        else:
            success_rate = 0
        
        return success_rate
    else:
        return None

def recommend_top_drugs(df, condition):
    # Filter rows for the given condition
    condition_rows = df[df['condition'].str.lower() == condition.lower()]
    
    if not condition_rows.empty:
        # Get unique drug names for the condition
        unique_drug_names = condition_rows['drugName'].unique()
        
        # Calculate success rate for each drug
        drug_success_rates = {}
        for drug_name in unique_drug_names:
            success_rate = calculate_success_rate(condition_rows, drug_name)
            if success_rate is not None:
                drug_success_rates[drug_name] = success_rate
        
        # Sort drugs by success rate in descending order
        sorted_drugs = sorted(drug_success_rates.items(), key=lambda x: x[1], reverse=True)
        
        # Get top three drugs
        top_three_drugs = sorted_drugs[:3]
        
        return top_three_drugs
    else:
        return None

if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv('data/output.csv')