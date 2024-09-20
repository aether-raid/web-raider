# assets/utils.py

import json
import pandas as pd

def json_to_table(json_string):
    """
    Note: Somehow, this function only works when AzureOpenAI gpt-4o is used as the model to output in `call_pro_con`
    """
    # Parse the JSON string
    data = json.loads(json_string)
    
    # Extract codebases
    codebases = data.get("codebases", [])
    
    # Prepare data for DataFrame
    rows = []
    for codebase in codebases:
        name = codebase.get("name", "N/A")
        pros = ", ".join(codebase.get("pros", []))  # Join pros into a single string
        cons = ", ".join(codebase.get("cons", []))  # Join cons into a single string
        
        rows.append({"Name": name, "Pros": pros, "Cons": cons})
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Return the formatted table as a string
    return df.to_string(index=False)

def useless_func(*_):
    return