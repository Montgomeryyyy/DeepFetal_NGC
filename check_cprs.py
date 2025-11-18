import pandas as pd
import argparse
import glob
import os

def check_cprs(path):
    df = pd.read_csv(path, sep="#")
    results = {}

    pattern = r"\b\d{10}\b"   # exactly 10 digits

    for col in df.columns:
        # Convert to string to search safely
        mask = df[col].astype(str).str.contains(pattern, regex=True)
        
        if mask.any():
            results[col] = df[mask]

    return results

def main():
    parser = argparse.ArgumentParser(description='Check CPR numbers in .asc files')
    parser.add_argument('--filepath', type=str, required=True, help='Path to directory containing .asc files')
    parser.add_argument('--test', action='store_true', help='Test mode: only process first 2 files')
    args = parser.parse_args()
    
    # Find all .asc files in the directory
    asc_files = glob.glob(os.path.join(args.filepath, "*.asc"))
    
    # For testing, take only the first 2 files
    if args.test:
        asc_files = asc_files[:2]
        test_msg = " (test mode: first 2 files only)"
    else:
        test_msg = ""
    
    if not asc_files:
        print(f"No .asc files found in {args.filepath}")
        return
    
    print(f"Found {len(asc_files)} .asc file(s){test_msg}:")
    for file_path in asc_files:
        print(f"\n{'='*60}")
        print(f"Processing: {file_path}")
        print(f"{'='*60}")
        
        try:
            results = check_cprs(file_path)
            
            if results:
                for col, matches in results.items():
                    print(f"\nColumn '{col}' contains CPR numbers:")
                    print(matches)
            else:
                print("No CPR numbers found in any column.")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    main()