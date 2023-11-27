from utils import paths
from utils.get_employee import split_row_unfilled, split_train_test, split_data_file

if __name__ == "__main__":
    # Split training and test sets by availability of number of employees into train/test_change_rate.csv
    # train/test_change_rate.csv: One cik(company) per row, with change rates of 2017-2022
    split_train_test(full_data_path=f"{paths.output_dir}/change_rate.csv", output_dir=paths.output_dir)
    # Split files in data_cleaned into train_data and test_data
    split_data_file(source_dir=paths.data_cleaned_dir, target_dir=paths.data_split_dir, index_dir=paths.output_dir)
    # Split the calendar year data for the companies in a row into each company one row per year
    split_row_unfilled(output_dir=paths.output_dir, prefix="train")
    split_row_unfilled(output_dir=paths.output_dir, prefix="test")
