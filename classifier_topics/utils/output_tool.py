def output(*values, output_path=None):
    values = [str(val) for val in values]
    text = " ".join(values)
    if output_path is None:
        print(text)
        return
    with open(output_path, "a") as fp:
        fp.write(f"{text}\n")
