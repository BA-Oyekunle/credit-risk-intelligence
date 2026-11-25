
from sqlalchemy import text

from src.database.connection import engine


def test_database_connection():

    print("=" * 60)
    print("CREDIT RISK INTELLIGENCE")
    print("POSTGRESQL CONNECTION TEST")
    print("=" * 60)

    try:

        with engine.connect() as connection:

            result = connection.execute(
                text("SELECT current_database();")
            )

            database_name = result.scalar_one()

            print("CONNECTION SUCCESSFUL")
            print(f"Database: {database_name}")

    except Exception as error:

        print("DATABASE CONNECTION FAILED")
        print(f"Error: {error}")

        raise

    finally:

        engine.dispose()


if __name__ == "__main__":
    test_database_connection()
