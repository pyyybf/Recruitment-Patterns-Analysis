import re
from num2words import num2words
from word2number import w2n

english_numbers = [num2words(i) for i in range(30)]
verbs = ["have", "has", "had", "employed", "employ", "employs", "consisted of", "consists of", "made up of",
         "makes of", "makes up of", "totaled"]
nouns = ["employee", "employees", "full-time", "full time", "headcount", "teammates", "people", "persons",
         "team members", "officers", "officer", "personnel", "staff"]
date_prefix = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
               "november", "december", "since", "year", "during", "fiscal"]

pattern = fr"({'|'.join(verbs)})\s+([a-zA-Z]+\s)*?(([\d, ()]+|no|an|{'|'.join(english_numbers)})( hundreds| thousands)*)\s+([a-zA-Z-]+\s)*?(?:{'|'.join(nouns)})"


def convert_number(num_str):
    if "(" in num_str:
        numbers = re.findall(r"([a-zA-Z0-9-,\s]*)\(([a-zA-Z0-9-,\s]+)", num_str)
        for number in numbers:
            for num in number:
                num = convert_number(num.strip())
                if num >= 0:
                    return num
        return -1
    elif num_str == "no":
        return 0
    elif "hundreds" in num_str:
        return convert_number(num_str.split()[0]) * 100
    elif "thousands" in num_str:
        return convert_number(num_str.split()[0]) * 1000
    # remove spaces
    num_str = re.sub(r"\s+", "", num_str)
    try:
        return w2n.word_to_num(num_str)
    except Exception as e:
        pass
    # remove numbers longer than 3
    num_str = num_str.split(",")
    num_str = [num[:3] for num in num_str]
    num_str = "".join(num_str)
    if num_str.isdigit():
        return int(num_str)
    return -1


def useful_line(line, ban_words):
    if "employ" not in line:
        return False
    for word in ban_words:
        if len(re.findall(fr"\b{word}\b", line)) > 0:
            return False
    return True


def match_line(line, ban_words):
    if not useful_line(line, ban_words):
        return -1
    res = re.findall(pattern, line)
    for match in res:
        if match[1].strip().lower() in date_prefix:
            continue
        employee_num = match[2].strip()
        if employee_num.isdigit() and 2016 <= int(employee_num) <= 2022:
            continue
        employee_num = re.sub(r"\s+", " ", employee_num)
        employee_num = convert_number(employee_num)
        if 0 <= employee_num < 2000000:
            return employee_num
    return -1


def match_employee_num(lines, year, record_file, ban_words={}):
    employee_nums = []
    employee_infos = []
    for line in lines:
        employee_num = match_line(line, ban_words)
        if employee_num >= 0 and employee_num != year:
            employee_nums.append(employee_num)
            employee_infos.append(f"{employee_num} <= {line}")

    if len(employee_nums) > 0:
        with open(record_file, "a") as fp:
            fp.write("\n".join(employee_infos))
            fp.write("\n\n")
        return max(employee_nums)
    return -1
