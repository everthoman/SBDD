import pandas as pd
import argparse
import sys

def reorder_columns(input_file, output_file, column_order):
    """
    Reorders columns in a data file using pandas.
    """
    try:
        # Read the data file (assumes CSV-like format; adjust sep if needed)
        df = pd.read_csv(input_file)
        print(f"Original columns: {list(df.columns)}")
        
        # Reorder columns (must include all or specify only these; extras dropped if not in df)
        df_reordered = df[column_order]
        
        # Write to output file
        df_reordered.to_csv(output_file, index=False)
        print(f"Reordered data saved to {output_file}")
        print(f"New columns: {list(df_reordered.columns)}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorder columns in a CSV data file.")
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("output_file", help="Path to output CSV file")
    parser.add_argument("column_order", nargs="+", help="List of column names in desired order")
    
    args = parser.parse_args()
    reorder_columns(args.input_file, args.output_file, args.column_order)

