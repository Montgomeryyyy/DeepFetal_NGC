import pandas as pd
import argparse
import os
import re
import subprocess
from pathlib import Path

PSEUDONYMIZER = Path("/storage/archive/UCPH/DeepFetal/SDS/pseudonymizer")

def validate_cpr_format(cpr_str):
    """
    Validate CPR format according to requirements:
    - Exactly 10 alphanumeric characters
    - Letters can only appear in positions 8-9 (for temporary CPRs)
    """
    if pd.isna(cpr_str) or cpr_str is None:
        return False, "Value is NaN or None"
    
    # Convert to string and strip all non-alphanumeric characters
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', str(cpr_str))
    
    # Check length is exactly 10
    if len(cleaned) != 10:
        return False, f"Length is {len(cleaned)}, expected 10"
    
    # Validate that letters can only appear in positions 8-9 (0-indexed: 7-8)
    # CPR format (after cleaning): DDMMYYXXXX (10 alphanumeric characters)
    # - Positions 0-6: must be digits (DDMMYY + first digit of serial)
    # - Positions 7-8 (8th and 9th): can be letters or digits (for temporary CPRs)
    # - Position 9 (10th): must be a digit (last digit of serial)
    
    # Check positions 0-6 and 9 must be digits (ensures letters only in 7-8)
    digit_positions = cleaned[0:7] + cleaned[9]
    if not digit_positions.isdigit():
        invalid_pos = next((i for i, c in enumerate(cleaned) if not c.isdigit() and i not in [7, 8]), None)
        if invalid_pos is not None:
            return False, f"Position {invalid_pos + 1} must be a digit, found: '{cleaned[invalid_pos]}'"
    
    return True, None

import subprocess
from pathlib import Path

PSEUDONYMIZER = Path("/storage/archive/UCPH/DeepFetal/SDS/pseudonymizer")

def pseudonymize_cpr(cpr_value) -> str:
    # Preserve missing values
    if pd.isna(cpr_value) or cpr_value is None or str(cpr_value).strip() == "":
        return cpr_value

    # Clean and validate
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', str(cpr_value))
    is_valid, error_msg = validate_cpr_format(cleaned)
    if not is_valid:
        raise ValueError(f"Invalid CPR format: {error_msg}. Original value: {cpr_value}")

    try:
        result = subprocess.run(
            [str(PSEUDONYMIZER), cleaned],
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Pseudonymizer failed for value '{cleaned}'. "
            f"Return code={e.returncode}. STDERR={e.stderr.strip()}"
        ) from e

    return result.stdout.strip()

def anon(df, cols):
    """
    Anonymize specified columns in the dataframe.
    Returns anonymized dataframe and list of errors encountered.
    """
    df = df.copy()
    errors = []
    stop_processing = False
    
    for col in cols:
        if col not in df.columns:
            errors.append(f"Column '{col}' not found in dataframe")
            continue
        
        if stop_processing:
            errors.append(f"Skipping column '{col}' due to previous errors")
            continue
        
        print(f"Processing column '{col}'...")
        
        try:
            for idx in df.index:
                try:
                    original_value = df.loc[idx, col]
                    pseudonymized = pseudonymize_cpr(original_value)
                    df.loc[idx, col] = pseudonymized
                except ValueError as e:
                    error_msg = f"Row {idx}, Column '{col}': {str(e)}"
                    errors.append(error_msg)
                    print(f"WARNING: {error_msg}")
                    stop_processing = True
                    break
                except Exception as e:
                    error_msg = f"Row {idx}, Column '{col}': Unexpected error - {str(e)}"
                    errors.append(error_msg)
                    print(f"WARNING: {error_msg}")
                    stop_processing = True
                    break
            
            if stop_processing:
                print(f"ERROR: Stopping anonymization due to errors. Column '{col}' processing incomplete.")
                break
                    
        except Exception as e:
            error_msg = f"Error processing column '{col}': {str(e)}"
            errors.append(error_msg)
            print(f"WARNING: {error_msg}")
            stop_processing = True
            break
    
    return df, errors

def main():
    parser = argparse.ArgumentParser(description='Anonymize CPR numbers in .asc files')
    parser.add_argument('--filepath', type=str, required=True, help='Path to input .asc file')
    parser.add_argument('--cols', type=str, required=True, help='Comma-separated list of columns to anonymize')
    parser.add_argument('--output', type=str, required=True, help='Path to output file')
    parser.add_argument('--test', action='store_true', help='Test mode: only process first 5 rows')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Chunk size for processing (default: 1000)')
    args = parser.parse_args()
    
    # Parse columns
    cols_to_anonymize = [col.strip() for col in args.cols.split(',')]
    
    print(f"Reading file: {args.filepath}")
    df = pd.read_csv(args.filepath, sep="#", engine="python")
    
    original_length = len(df)
    print(f"Original file has {original_length} rows")
    
    # Test mode: only process first 5 rows
    if args.test:
        print("TEST MODE: Processing only first 5 rows")
        df = df.head(5).copy()
    
    # Check if output file exists - if so, stop with error
    if os.path.exists(args.output):
        print(f"ERROR: Output file '{args.output}' already exists. Please remove it or choose a different output path.")
        return
    
    write_mode = 'w'
    write_header = True    
    print(f"Anonymizing columns: {', '.join(cols_to_anonymize)}")
    print(f"Processing in chunks of {args.chunk_size} rows")
    total_rows = len(df)
    chunk_size = args.chunk_size
    all_errors = []
    
    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        print(f"\nProcessing chunk: rows {start_idx} to {end_idx-1}")
        
        chunk_df = df.iloc[start_idx:end_idx].copy()
        
        try:
            anonymized_chunk, errors = anon(chunk_df, cols_to_anonymize)
            all_errors.extend(errors)
            
            if errors:
                print(f"ERROR: Encountered {len(errors)} error(s). Stopping processing.")
                break
            
            # Append write
            anonymized_chunk.to_csv(
                args.output, 
                sep="#", 
                index=False, 
                mode=write_mode, 
                header=write_header
            )
            write_mode = 'a'  # After first write, always append
            write_header = False  # Don't write header after first chunk
            
            print(f"Chunk saved to {args.output}")
            
        except Exception as e:
            error_msg = f"Fatal error processing chunk [{start_idx}:{end_idx}]: {str(e)}"
            all_errors.append(error_msg)
            print(f"ERROR: {error_msg}")
            break
    
    # Summary
    print(f"\n{'='*60}")
    print("Processing Summary:")
    print(f"{'='*60}")
    if all_errors:
        print(f"ERRORS ENCOUNTERED: {len(all_errors)}")
        for error in all_errors:
            print(f"  - {error}")
        print(f"\nWARNING: Anonymization may be incomplete. Review errors above.")
    else:
        print("SUCCESS: All rows processed without errors.")
        print(f"Anonymized data saved to {args.output}")
    
    if args.test:
        print("\nNote: This was a test run with only 5 rows.")

if __name__ == "__main__":
    main()
