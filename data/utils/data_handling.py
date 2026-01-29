import os

def jsonl_to_array(jsonl_path, json_path, pretty=True):
    with open(jsonl_path, "r", encoding="utf-8") as src, \
         open(json_path, "w", encoding="utf-8") as dst:
        if pretty:
            indent, sep = "  ", ",\n"
            dst.write("[\n")
        else:
            indent, sep = "", ","
            dst.write("[")
        first = True
        for line in src:
            line = line.rstrip("\n")
            if not line:
                continue
            if first:
                first = False
            else:
                dst.write(sep)
            dst.write(f"{indent}{line}")
        dst.write("\n]\n" if pretty else "]")

def write_hf_to_json(ds, output_path):
    jsonl_path = output_path + "l"
    ds.to_json(
        jsonl_path,
        batch_size=10_000,
        num_proc=8,
        lines=True,   
        orient="records",
        force_ascii=False,
    )
    jsonl_to_array(jsonl_path, output_path)
    os.remove(jsonl_path)

