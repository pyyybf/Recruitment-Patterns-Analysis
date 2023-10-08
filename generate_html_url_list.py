import os

for year in os.listdir("./data_html"):
    if year == ".DS_Store":
        continue
    urls = []
    for html_file in os.listdir(f"./data_html/{year}"):
        if html_file == ".DS_Store":
            continue
        names = html_file.split("_")
        urls.append(f"http://www.sec.gov/Archives/edgar/data/{names[0]}/{names[1]}/{'_'.join(names[2:])}")
    with open(f"./html_url/{year}.txt", "w") as fp:
        fp.write("\n".join(urls))
