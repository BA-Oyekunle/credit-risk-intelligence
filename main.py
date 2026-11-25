from credit_risk.data_ingestion import (
    load_raw_data,
    save_processed_data
)

from credit_risk.data_validation import (
    validate_dataset
)

from credit_risk.config import PROCESSED_DATA_PATH


def main():

    print("=" * 50)

    print("CREDIT RISK INTELLIGENCE")

    print("DATA PIPELINE")

    print("=" * 50)


    # 1. Load raw data
    df = load_raw_data()


    # 2. Validate dataset
    validation_report = validate_dataset(df)


    # 3. Display validation report
    print("\nDATA VALIDATION REPORT")

    print("-" * 30)

    for key, value in validation_report.items():

        print(f"{key}: {value}")


    # 4. Save dataset
    save_processed_data(
        df,
        PROCESSED_DATA_PATH
    )


    print("\nPipeline completed successfully.")


if __name__ == "__main__":

    main()