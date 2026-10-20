# Dataset

Place the original `Fraud.csv` dataset at `data/raw/Fraud.csv`. The local
repository contained this file before restructuring, with 6,362,620 rows and
the columns described in the root README and `docs/data_dictionary.txt`.

The local project files do not record a dataset download URL. If you need to
obtain another copy, use the same dataset described by those documents and
preserve the filename and column names. The dataset is excluded from Docker;
the Streamlit application does not load it at runtime.

The notebook reads `Fraud.csv` relative to its working directory. To run it
without editing the preserved notebook, set the notebook kernel's working
directory to `data/raw/` before executing its cells.
