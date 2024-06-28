import pandas as pd
import os


class DataFrameHandler:
    """
    Class for handling dataframe reading and formatting.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.extension = file_path.split(".")[-1].lower()

    def __format__(self, format_spec):
        file_name, file_extension = os.path.splitext(
            os.path.basename(self.file_path))
        if format_spec == "filename":
            return f"{file_name}"
        elif format_spec == "extension":
            return f"{file_extension}"
        elif format_spec == "file":
            return f"{file_name, file_extension}"

    def to_dataframe(self):
        if self.extension == "csv":
            return pd.read_csv(self.file_path)
        elif self.extension in ["xls", "xlsx"]:
            return pd.read_excel(self.file_path)
        elif self.extension == "json":
            return pd.read_json(self.file_path)
        else:
            raise ValueError(f"Unsupported file extension: {self.extension}")
