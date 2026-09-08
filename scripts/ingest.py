import argparse

def main():
    parser = argparse.ArgumentParser(description="Ingest data")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--module", required=True)
    args = parser.parse_args()
    
    print(f"Ingesting data from {args.data_dir} using module {args.module}...")
    print("Ingestion stub complete.")

if __name__ == "__main__":
    main()
