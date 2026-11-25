import logging
from pathlib import Path

import pandas as pd

from credit_risk.config import RAW_DATA_PATH


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def load_raw_data(
    file_path: Path = RAW_DATA_PATH
) -> pd.DataFrame:

    """
    Load raw loan data from a CSV file.

    Parameters
    ----------
    file_path : Path
        Location of the CSV dataset.

    Returns
    -------
    pd.DataFrame
        Loaded loan dataset.
    """

    # Check if the file exists
    if not file_path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    logger.info("Loading dataset from %s", file_path)

    # Load the dataset
    df = pd.read_csv(
        file_path,
        low_memory=False
    )

    logger.info("Dataset loaded successfully")

    logger.info(
        "Dataset shape: %s",
        df.shape
    )

    return df


def save_processed_data(
    df: pd.DataFrame,
    file_path: Path
) -> None:

    """
    Save dataset in Parquet format.
    """

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_parquet(
        file_path,
        index=False
    )

    logger.info(
        "Data saved to %s",
        file_path
    )