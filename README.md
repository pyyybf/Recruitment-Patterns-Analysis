# Recruitment-Patterns-Analysis

Text Mining for Analyzing Recruitment Patterns in Tech Industries Using a Corpus of SEC Filings.

## Getting Started

Download data (2016q1–2023q2) from [Financial statement dataset from the SEC](https://www.sec.gov/dera/data/financial-statement-data-sets), and unzip them to the `data_original` folder. 

Run following command to fetch html data to `data_html` folder.
```shell
python fetch_html.py
```

Run following command to generate the url list of html pages in the `html_url` folder.
```shell
python generate_html_url_list.py
```

Run following command to generate the txt files of html pages in the `data_txt` folder.
```shell
python generate_txt_by_html.py
```
