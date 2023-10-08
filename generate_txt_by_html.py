import os
import re

for year in os.listdir("./data_html"):
    if year == ".DS_Store":
        continue
    if not os.path.exists(f"./data_txt/{year}"):
        os.makedirs(f"./data_txt/{year}")
    print(year)

    length = len(os.listdir(f"./data_html/{year}"))
    for i, html_file in enumerate(os.listdir(f"./data_html/{year}")):
        if i % 100 == 0:
            print(f"{i}/{length}")
        if html_file == ".DS_Store":
            continue
        with open(f"./data_html/{year}/{html_file}", "r") as source:
            html_text = source.read()
            text = re.sub(r"<[^>]+>", "", html_text)
            text = re.sub(r"\n{3,}", "\n\n", text)

        with open(f"./data_txt/{year}/{html_file}.txt", "w") as target:
            target.write(text)
