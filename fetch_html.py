import os
import pandas as pd
import requests
from lxml import etree

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
}

if not os.path.exists(f"./data_html"):
    os.makedirs(f"./data_html")

for q_dir in os.listdir("./data_original"):
    if q_dir == ".DS_Store":
        continue
    df_sub = pd.read_csv(f"./data_original/{q_dir}/sub.txt", sep="\t")
    # Filter 10-K from 2016 to 2023
    df_sub_10k = df_sub.loc[(df_sub["form"] == "10-K") & (df_sub["fy"] > 2015) & (df_sub["fy"] < 2023)]
    print(f"========== {q_dir}: Total {len(df_sub_10k)} ==========")

    count = 0
    for index, row in df_sub_10k.iterrows():
        count += 1
        try:
            cik = row["cik"]
            adsh = row["adsh"]
            accession = row["adsh"].replace("-", "")
            url = f"http://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{adsh}-index.html"

            response = requests.get(url, headers=headers, timeout=20)
            if response.status_code != 200:
                with open("./error_log.txt", "a") as fp:
                    fp.write(f"Error in {q_dir}, document of year {int(row['fy'])}.\n")
                    fp.write(f"http://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{adsh}-index.html\n")
                    fp.write(f"Fetch index: Status Code {response.status_code}\n")
                    fp.write(f"=======================================================\n")
                continue

            element = etree.HTML(response.content)
            tr_list = element.xpath("/html/body/div[4]/div[3]/div/table[position()=1]/tr")

            th_list = tr_list[0].xpath(".//th")
            type_idx = -1
            doc_idx = -1
            for i, th in enumerate(th_list):
                if th.text == "Type":
                    type_idx = i
                elif th.text == "Document":
                    doc_idx = i

            doc_name = None
            for tr in tr_list[1:]:
                td_list = tr.xpath(".//td")
                if td_list[type_idx].text == "10-K":
                    doc_name = td_list[doc_idx].xpath(".//a")[0].text
                    break

            url_doc = f"http://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc_name}"
            print(f"{count}/{len(df_sub_10k)}: {url_doc}")
            response = requests.get(url_doc, headers=headers, timeout=20)
            if response.status_code != 200:
                with open("./error_log.txt", "a") as fp:
                    fp.write(f"Error in {q_dir}, document of year {int(row['fy'])}.\n")
                    fp.write(f"http://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc_name}\n")
                    fp.write(f"Fetch document: Status Code {response.status_code}\n")
                    fp.write(f"=======================================================\n")
                continue

            # Write into final data as html
            with open(f"./data_html/{int(row['fy'])}/{cik}_{accession}_{doc_name}", "w", encoding="utf-8") as fp:
                fp.write(response.text)

        except Exception as e:
            with open("./error_log.txt", "a") as fp:
                fp.write(f"Error in {q_dir}, document of year {int(row['fy'])}.\n")
                fp.write(f"http://www.sec.gov/Archives/edgar/data/{row['cik']}/{row['adsh'].replace('-', '')}/\n")
                fp.write(f"{e}\n")
                fp.write(f"=======================================================\n")
