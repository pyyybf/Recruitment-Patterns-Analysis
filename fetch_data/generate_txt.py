import os
import re
from tqdm import tqdm

from utils import paths


def generate_txt_by_html(txt_dir, html_dir):
    for year in os.listdir(html_dir):
        if year == ".DS_Store":
            continue
        if not os.path.exists(f"{txt_dir}/{year}"):
            os.makedirs(f"{txt_dir}/{year}")

        html_list = [html_file for html_file in os.listdir(f"{html_dir}/{year}") if html_file.endswith(".htm")]
        with tqdm(total=len(html_list), unit="file", desc=f"Generate txt from html file of {year}") as pbar_gen:
            for html_file in html_list:
                with open(f"{html_dir}/{year}/{html_file}", "r") as source:
                    html_text = source.read()
                    # Remove html tags
                    text = re.sub(r"<[^>]+>", "\n", html_text)
                    # Remove redundant line breaks
                    text = re.sub(r"\n{3,}", "\n\n", text)

                with open(f"{txt_dir}/{year}/{html_file}.txt", "w") as target:
                    target.write(text)

                pbar_gen.update(1)


if __name__ == "__main__":
    generate_txt_by_html(txt_dir=paths.txt_dir, html_dir=paths.html_dir)
