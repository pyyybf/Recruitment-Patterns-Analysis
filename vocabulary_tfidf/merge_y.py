import pandas as pd
from tqdm import tqdm

from utils import paths


def merge_X_y(classification=True):
    if "train" in paths.base_dir:
        prefix = "train"
    else:
        prefix = "test"
    df = pd.read_csv(f"{paths.change_rates_dir}/{prefix}_change_rate_unfilled.csv", index_col=["cik", "year"])
    samples = []
    for year in range(2017, 2023):
        sampled = False
        with open(f"{paths.tfidf_dir}/{year}_TFIDF.csv", "r") as source:
            lines = source.readlines()
            with open(f"{paths.y_X_dir}/y_X{'_cate' if classification else ''}.csv", "w") as target:
                with tqdm(total=int(len(lines) / 2 - 1), unit="file", desc=f"Match y of {year}") as pbar:
                    for line in lines[2:]:
                        line = line.strip()
                        if line == "":
                            continue
                        count = 0
                        third_comma_idx = 0
                        for i in range(len(line)):
                            if line[i] == ",":
                                count += 1
                            if count == 3:
                                third_comma_idx = i
                                break
                        info = line[:third_comma_idx].split(",")  # FILE_NAME,CIK,YEAR
                        line = line[third_comma_idx + 1:]
                        try:
                            change_rate = df.loc[(int(info[1]), int(info[2])), "change_rate"]
                            if classification:
                                change_rate = 1 if change_rate > 0 else 0
                            target.write(f"{change_rate},{line}\n")
                            if not sampled:
                                sampled = True
                                samples.append(f"{change_rate},{line}")
                        except:
                            # print(int(info[1]), int(info[2]))
                            pass
                        pbar.update(1)

    with open(f"{paths.y_X_dir}/y_X{'_cate' if classification else ''}_sample.csv", "w") as target:
        target.write("\n".join(samples))


if __name__ == "__main__":
    # Merge TFIDF with change rates
    merge_X_y(classification=False)
    merge_X_y(classification=False)
    # Merge TFIDF with the classified labels which is calculated with change rates
    merge_X_y(classification=True)
    merge_X_y(classification=True)
