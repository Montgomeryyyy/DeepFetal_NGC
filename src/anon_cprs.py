#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

PSEUDONYMIZER = Path("/storage/archive/UCPH/DeepFetal/SDS/pseudonymizer")


def validate_cpr_format(cpr_str):
    """
    Validate CPR format according to requirements:
    - Exactly 9 or 10 alphanumeric characters
    - Letters can only appear in positions 8-9 (for temporary CPRs, 10-digit only)
    Returns (is_valid, error_msg, is_9_digit) tuple.
    """
    if pd.isna(cpr_str) or cpr_str is None:
        return False, "Value is NaN or None", False

    cleaned = re.sub(r"[^a-zA-Z0-9]", "", str(cpr_str))

    # Accept both 9 and 10 digit CPRs
    if len(cleaned) == 9:
        # 9-digit CPR: all must be digits
        if not cleaned.isdigit():
            return False, f"9-digit CPR must contain only digits, found: '{cleaned}'", True
        return True, None, True
    elif len(cleaned) == 10:
        # 10-digit CPR: validate format
        digit_positions = cleaned[0:7] + cleaned[9]
        if not digit_positions.isdigit():
            invalid_pos = next(
                (i for i, c in enumerate(cleaned) if not c.isdigit() and i not in [7, 8]),
                None,
            )
            if invalid_pos is not None:
                return (
                    False,
                    f"Position {invalid_pos + 1} must be a digit, found: '{cleaned[invalid_pos]}'",
                    False,
                )
        return True, None, False
    else:
        return False, f"Length is {len(cleaned)}, expected 9 or 10", False


def _pseudonymize_one(value):
    """
    Worker function for multiprocessing.
    Takes a single CPR-like value, cleans, validates, then calls the pseudonymizer binary.
    Returns (pseudonymized_value, is_invalid, is_9_digit) tuple.
    Invalid entries return (original_value, True, False) to skip them.
    """
    # Preserve missing values
    if pd.isna(value) or value is None or str(value).strip() == "":
        return (value, False, False)

    # Clean the value
    cleaned = re.sub(r"[^a-zA-Z0-9]", "", str(value))
    
    # Validate format
    is_valid, error_msg, is_9_digit = validate_cpr_format(cleaned)
    if not is_valid:
        # Invalid entry - return original to skip, but mark as invalid
        return (value, True, False)

    try:
        result = subprocess.run(
            [str(PSEUDONYMIZER), cleaned],
            capture_output=True,
            text=True,
            check=True,
        )
        pseudonymized = result.stdout.strip()
    except subprocess.CalledProcessError as e:
        # Pseudonymizer failed - skip this entry
        return (value, True, False)
    except Exception as e:
        # Any other error - skip this entry
        return (value, True, False)

    return (pseudonymized, False, is_9_digit)


def anon_parallel(df, cols, workers=None):
    """
    Anonymize specified columns in the dataframe using parallel subprocess calls.
    Strategy:
      - For each column, pseudonymize only the UNIQUE non-missing values in parallel.
      - Map results back to the column (fast if many duplicates).
    Returns anonymized dataframe, list of errors, and statistics dict.
    """
    df = df.copy()
    errors = []
    stats = {
        "invalid_entries": {},    # {col: [list of invalid values that were skipped]}
        "nine_digit_cprs": {},    # {col: [list of 9-digit CPR values]}
    }

    for col in cols:
        if col not in df.columns:
            errors.append(f"Column '{col}' not found in dataframe")
            continue

        print(f"Processing column '{col}' (parallel)...")
        stats["invalid_entries"][col] = []
        stats["nine_digit_cprs"][col] = []

        try:
            s = df[col]

            # Non-missing mask
            mask = ~s.isna() & (s.astype(str).str.strip() != "")
            if not mask.any():
                print(f"Column '{col}': no non-missing values to pseudonymize.")
                continue

            # Work only on unique values (big speedup when values repeat)
            unique_vals = s.loc[mask].astype(str).unique().tolist()

            with ProcessPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(_pseudonymize_one, unique_vals))

            # Extract pseudonymized values, track invalid entries and 9-digit CPRs
            mapping = {}
            for orig_val, (pseudo_val, is_invalid, is_9_digit) in zip(unique_vals, results):
                if is_invalid:
                    # Invalid entry - keep original value, track it
                    mapping[orig_val] = orig_val
                    stats["invalid_entries"][col].append(orig_val)
                else:
                    mapping[orig_val] = pseudo_val
                    if is_9_digit:
                        stats["nine_digit_cprs"][col].append(orig_val)

            df.loc[mask, col] = s.loc[mask].astype(str).map(mapping)

        except Exception as e:
            errors.append(f"Error processing column '{col}': {str(e)}")
            break

    return df, errors, stats


def main():
    parser = argparse.ArgumentParser(description="Anonymize CPR numbers in .asc files")
    parser.add_argument("--filepath", type=str, required=True, help="Path to input .asc file")
    parser.add_argument("--cols", type=str, required=True, help="Comma-separated list of columns to anonymize")
    parser.add_argument("--output", type=str, required=True, help="Path to output file")
    parser.add_argument("--test", action="store_true", default=False, help="Test mode: only process first 5 rows")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Chunk size for processing (default: 10000)")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: use CPU count)",
    )
    args = parser.parse_args()

    # Sanity checks
    if not PSEUDONYMIZER.exists():
        raise FileNotFoundError(f"Pseudonymizer binary not found at: {PSEUDONYMIZER}")
    if not os.access(PSEUDONYMIZER, os.X_OK):
        raise PermissionError(f"Pseudonymizer is not executable: {PSEUDONYMIZER}")

    cols_to_anonymize = [c.strip() for c in args.cols.split(",") if c.strip()]

    # Do not overwrite any existing output
    if os.path.exists(args.output):
        print(
            f"ERROR: Output file '{args.output}' already exists. "
            "Please remove it or choose a different output path."
        )
        return

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    # Optional: safer to avoid partial final output
    tmp_output = args.output + ".tmp"
    if os.path.exists(tmp_output):
        print(
            f"ERROR: Temp output file '{tmp_output}' already exists. "
            "Please remove it before running."
        )
        return

    print(f"Anonymizing columns: {', '.join(cols_to_anonymize)}")
    print(f"Processing in chunks of {args.chunk_size} rows")
    if args.workers is not None:
        print(f"Parallel workers: {args.workers}")

    # Read file in chunks instead of loading entire file
    chunk_size = args.chunk_size
    write_mode = "w"
    write_header = True
    all_errors = []
    chunk_num = 0
    total_rows_processed = 0
    all_stats = {
        "invalid_entries": {},  # Accumulate across all chunks
        "nine_digit_cprs": {},   # Accumulate across all chunks
    }

    print(f"Reading file in chunks: {args.filepath}")
    
    # Use chunk reading for memory efficiency
    chunk_reader = pd.read_csv(args.filepath, sep="¤", engine="python", chunksize=chunk_size)
    
    for chunk_df in chunk_reader:
        # Handle test mode: only process first 5 rows total
        if args.test:
            if total_rows_processed >= 5:
                break
            chunk_df = chunk_df.head(5 - total_rows_processed)
        
        rows_in_chunk = len(chunk_df)
        if rows_in_chunk == 0:
            break
            
        print(f"\nProcessing chunk {chunk_num + 1}: rows {total_rows_processed} to {total_rows_processed + rows_in_chunk - 1}")

        try:
            anonymized_chunk, errors, stats = anon_parallel(chunk_df, cols_to_anonymize, workers=args.workers)
            all_errors.extend(errors)

            # Accumulate statistics
            for col in stats["invalid_entries"]:
                if col not in all_stats["invalid_entries"]:
                    all_stats["invalid_entries"][col] = []
                all_stats["invalid_entries"][col].extend(stats["invalid_entries"][col])
            
            for col in stats["nine_digit_cprs"]:
                if col not in all_stats["nine_digit_cprs"]:
                    all_stats["nine_digit_cprs"][col] = []
                all_stats["nine_digit_cprs"][col].extend(stats["nine_digit_cprs"][col])

            if errors:
                print(f"ERROR: Encountered {len(errors)} error(s). Stopping processing.")
                break

            anonymized_chunk.to_csv(
                tmp_output,
                sep="¤",
                index=False,
                mode=write_mode,
                header=write_header,
            )
            write_mode = "a"
            write_header = False

            print(f"Chunk saved to {tmp_output}")
            total_rows_processed += rows_in_chunk
            chunk_num += 1

        except Exception as e:
            error_msg = f"Fatal error processing chunk {chunk_num + 1}: {str(e)}"
            all_errors.append(error_msg)
            print(f"ERROR: {error_msg}")
            break

    print("\n" + "=" * 60)
    print("Processing Summary:")
    print("=" * 60)
    print(f"Total rows processed: {total_rows_processed}")

    # Report 9-digit CPR statistics
    for col in all_stats["nine_digit_cprs"]:
        nine_digit_list = all_stats["nine_digit_cprs"][col]
        if nine_digit_list:
            unique_nine_digit = list(set(nine_digit_list))
            print(f"\nColumn '{col}': {len(unique_nine_digit)} unique 9-digit CPR(s) found and processed")

    # Collect and save invalid entries that were skipped
    skipped_cprs_file = args.output.replace(".asc", "_skipped_cprs.csv")
    skipped_data = []
    
    for col in all_stats["invalid_entries"]:
        invalid_list = all_stats["invalid_entries"][col]
        if invalid_list:
            unique_invalid = list(set(invalid_list))
            print(f"\nColumn '{col}': {len(unique_invalid)} unique invalid entry/entries skipped")
            
            # Add to skipped data for saving
            for val in unique_invalid:
                skipped_data.append({"column": col, "skipped_value": val})
            
            if len(unique_invalid) <= 10:  # Only show if not too many
                for val in unique_invalid[:10]:  # Show first 10
                    print(f"  - '{val}'")
                if len(unique_invalid) > 10:
                    print(f"  ... and {len(unique_invalid) - 10} more")
    
    # Save skipped CPRs to file
    if skipped_data:
        # Ensure directory exists for skipped CPRs file
        skipped_dir = os.path.dirname(skipped_cprs_file)
        if skipped_dir and not os.path.exists(skipped_dir):
            os.makedirs(skipped_dir, exist_ok=True)
        
        skipped_df = pd.DataFrame(skipped_data)
        skipped_df.to_csv(skipped_cprs_file, index=False)
        print(f"\nSkipped CPRs saved to: {skipped_cprs_file}")
    else:
        print("\nNo invalid CPRs were skipped.")

    if all_errors:
        print(f"\nERRORS ENCOUNTERED: {len(all_errors)}")
        for err in all_errors:
            print(f"  - {err}")
        print(f"\nWARNING: Output left as temp file for inspection: {tmp_output}")
        print("No original files were overwritten.")
    else:
        os.replace(tmp_output, args.output)
        print("\nSUCCESS: All rows processed without errors.")
        print(f"Anonymized data saved to {args.output}")
        print("No original files were overwritten.")

    if args.test:
        print("\nNote: This was a test run with only 5 rows.")


if __name__ == "__main__":
    main()
