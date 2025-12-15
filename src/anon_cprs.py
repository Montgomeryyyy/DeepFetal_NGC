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
    - Exactly 10 alphanumeric characters
    - Letters can only appear in positions 8-9 (for temporary CPRs)
    """
    if pd.isna(cpr_str) or cpr_str is None:
        return False, "Value is NaN or None"

    cleaned = re.sub(r"[^a-zA-Z0-9]", "", str(cpr_str))

    if len(cleaned) != 10:
        return False, f"Length is {len(cleaned)}, expected 10"

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
            )

    return True, None


def _pseudonymize_one(value):
    """
    Worker function for multiprocessing.
    Takes a single CPR-like value, cleans + validates, then calls the pseudonymizer binary.
    """
    # Preserve missing values
    if pd.isna(value) or value is None or str(value).strip() == "":
        return value

    cleaned = re.sub(r"[^a-zA-Z0-9]", "", str(value))

    is_valid, error_msg = validate_cpr_format(cleaned)
    if not is_valid:
        raise ValueError(f"Invalid CPR format: {error_msg}. Original value: {value}")

    try:
        result = subprocess.run(
            [str(PSEUDONYMIZER), cleaned],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Pseudonymizer failed for value '{cleaned}'. "
            f"Return code={e.returncode}. STDERR={e.stderr.strip()}"
        ) from e

    return result.stdout.strip()


def anon_parallel(df, cols, workers=None):
    """
    Anonymize specified columns in the dataframe using parallel subprocess calls.
    Strategy:
      - For each column, pseudonymize only the UNIQUE non-missing values in parallel.
      - Map results back to the column (fast if many duplicates).
    Returns anonymized dataframe and list of errors encountered.
    """
    df = df.copy()
    errors = []

    for col in cols:
        if col not in df.columns:
            errors.append(f"Column '{col}' not found in dataframe")
            continue

        print(f"Processing column '{col}' (parallel)...")

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
                pseudo_vals = list(ex.map(_pseudonymize_one, unique_vals))

            mapping = dict(zip(unique_vals, pseudo_vals))

            df.loc[mask, col] = s.loc[mask].astype(str).map(mapping)

        except Exception as e:
            errors.append(f"Error processing column '{col}': {str(e)}")
            break

    return df, errors


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
            anonymized_chunk, errors = anon_parallel(chunk_df, cols_to_anonymize, workers=args.workers)
            all_errors.extend(errors)

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

    if all_errors:
        print(f"ERRORS ENCOUNTERED: {len(all_errors)}")
        for err in all_errors:
            print(f"  - {err}")
        print(f"\nWARNING: Output left as temp file for inspection: {tmp_output}")
        print("No original files were overwritten.")
    else:
        os.replace(tmp_output, args.output)
        print("SUCCESS: All rows processed without errors.")
        print(f"Anonymized data saved to {args.output}")
        print("No original files were overwritten.")

    if args.test:
        print("\nNote: This was a test run with only 5 rows.")


if __name__ == "__main__":
    main()
