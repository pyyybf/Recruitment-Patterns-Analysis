import re
from num2words import num2words
from word2number import w2n

numbers = r"[\d, ()]+"
english_numbers = [num2words(i) for i in range(21)]
verbs = ["have", "has", "had", "employed", "employ", "employs", "consisted of", "consists of", "made up of",
         "makes of"]
nouns = ["employee", "employees", "headcount", "teammates", "people", "persons", "team members"]

pattern = fr"({'|'.join(verbs)})\s+([a-z]+\s)*?({numbers}|no|{'|'.join(english_numbers)})\s+([a-z-,]+\s)*?(?:{'|'.join(nouns)})"


def convert_number(num_str):
    if num_str.isdigit():
        return int(num_str)
    elif num_str == "no":
        return 0
    elif "(" in num_str:
        numbers = num_str.split("(")
        num1 = numbers[0].strip()
        num2 = numbers[1].strip(")").strip()
        if num1.isdigit():
            return int(num1)
        elif num2.isdigit():
            return int(num2)
        num1 = convert_number(num1)
        num2 = convert_number(num2)
        if num1:
            return num1
        else:
            return num2
    try:
        return w2n.word_to_num(num_str)
    except Exception as e:
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
    if len(res) > 0:
        employee_num = res[0][2].strip().replace(",", "")
        employee_num = re.sub(r"\s+", "", employee_num)
        employee_num = convert_number(employee_num)
        if employee_num >= 0:
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
