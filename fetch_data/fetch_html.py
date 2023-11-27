import os
import pandas as pd
import requests
from lxml import etree
from tqdm import tqdm

from utils import paths

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
}


def fetch_html(html_dir, data_original_dir, url_dir):
    if not os.path.exists(html_dir):
        os.makedirs(html_dir)

    for q_dir in os.listdir(data_original_dir):
        if q_dir == ".DS_Store":
            continue
        df_sub = pd.read_csv(f"{data_original_dir}/{q_dir}/sub.txt", sep="\t")
        # Filter 10-K from 2016 to 2022
        df_sub_10k = df_sub.loc[(df_sub["form"] == "10-K") & (df_sub["fy"] > 2015) & (df_sub["fy"] < 2023)]

        with tqdm(total=len(df_sub_10k), unit="file", desc=f"Fetch html file of {q_dir}") as pbar_fetch:
            for index, row in df_sub_10k.iterrows():
                try:
                    cik = row["cik"]
                    adsh = row["adsh"]
                    accession = row["adsh"].replace("-", "")
                    year = int(row["fy"])
                    # Url of filing detail page
                    url = f"http://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{adsh}-index.html"

                    # Fetch index html page to parse the url of html document
                    response = requests.get(url, headers=headers, timeout=20)
                    if response.status_code != 200:
                        with open(paths.error_log_file, "a") as fp:
                            fp.write(f"Error in {q_dir}, document of year {year}.\n")
                            fp.write(f"{url}\n")
                            fp.write(f"Fetch index: Status Code {response.status_code}\n")
                            fp.write(f"=======================================================\n")
                        continue

                    element = etree.HTML(response.content)  # Root of html
                    # Get <tr> list of the table `Document Format Files`
                    tr_list = element.xpath("/html/body/div[4]/div[3]/div/table[position()=1]/tr")

                    # Parse table headers to get index of file type and name
                    th_list = tr_list[0].xpath(".//th")
                    type_idx = -1
                    doc_idx = -1
                    for i, th in enumerate(th_list):
                        if th.text == "Type":
                            type_idx = i
                        elif th.text == "Document":
                            doc_idx = i

                    # Find the row with a Type of `10-K`, and get its document name
                    doc_name = None
                    for tr in tr_list[1:]:
                        td_list = tr.xpath(".//td")
                        if td_list[type_idx].text == "10-K":
                            doc_name = td_list[doc_idx].xpath(".//a")[0].text
                            break

                    # Url of the real document html
                    url_doc = f"http://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc_name}"

                    # Fetch real document html content
                    response = requests.get(url_doc, headers=headers, timeout=20)
                    if response.status_code != 200:
                        with open(paths.error_log_file, "a") as fp:
                            fp.write(f"Error in {q_dir}, document of year {year}.\n")
                            fp.write(f"{url_doc}\n")
                            fp.write(f"Fetch document: Status Code {response.status_code}\n")
                            fp.write(f"=======================================================\n")
                        continue

                    # Write into final data file as html
                    with open(f"./data_html/{year}/{cik}_{accession}_{doc_name}", "w", encoding="utf-8") as fp:
                        fp.write(response.text)
                    # Write into url list
                    with open(f"{url_dir}/{year}.txt", "a") as fp:
                        fp.write(f"{url_doc}\n")

                except Exception as e:
                    with open(paths.error_log_file, "a") as fp:
                        fp.write(f"Error in {q_dir}, document of year {year}.\n")
                        fp.write(f"http://www.sec.gov/Archives/edgar/data/{cik}/{adsh}/\n")
                        fp.write(f"{e}\n")
                        fp.write(f"=======================================================\n")

                pbar_fetch.update(1)


if __name__ == "__main__":
    fetch_html(paths.html_dir, paths.data_original_dir, paths.url_dir)
