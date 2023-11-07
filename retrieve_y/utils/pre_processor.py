import re


def split_paragraph(lines):
    split_lines = []
    # 姑且试试按英语句号split一下吧
    for line in lines:
        sentences = re.sub(r"&.{4};", " ", line)
        sentences = sentences.split(". ")
        for sentence in sentences:
            if len(sentence.strip()) > 0:
                split_lines.append(sentence.strip())
    return split_lines
