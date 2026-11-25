
from sqlalchemy import inspect

from src.database.connection import Base, engine
from src.database.models import BorrowerAssessment


def initialize_database():

    print("=" * 60)
    print("CREDIT RISK INTELLIGENCE")
    print("DATABASE INITIALIZATION")
    print("=" * 60)

    try:

        # Create all tables defined in SQLAlchemy models
        Base.metadata.create_all(bind=engine)

        print("\nDatabase tables initialized successfully.")

        # Verify table creation
        inspector = inspect(engine)

        tables = inspector.get_table_names()

        print("\nAVAILABLE DATABASE TABLES:")

        for table in tables:
            print(f"- {table}")

        expected_table = "borrower_assessments"

        if expected_table not in tables:
            raise RuntimeError(
                f"Expected table '{expected_table}' "
                "was not created."
            )

        # Inspect table columns
        columns = inspector.get_columns(expected_table)

        print(f"\nCOLUMNS IN {expected_table}:")

        for column in columns:
            print(
                f"- {column['name']}: "
                f"{column['type']}"
            )

        print("\nDATABASE INITIALIZATION PASSED.")

    except Exception as error:

        print("\nDATABASE INITIALIZATION FAILED.")
        print(f"Error: {error}")

        raise

    finally:
        engine.dispose()


if __name__ == "__main__":
    initialize_database()
