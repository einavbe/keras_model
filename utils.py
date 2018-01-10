import pandas as pd


def create_full_csv(reference_db_path,
                    csv_file_name=r"REFERENCE.csv",
                    arg1=0,
                    arg2=0,
                    ):
    full_path= reference_db_path+csv_file_name
    df = pd.read_csv(full_path,header=-1)
    db_length= df.shape[0]