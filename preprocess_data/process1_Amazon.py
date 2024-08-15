
import gzip
import simplejson
import numpy as np

def parse(filename):
    f = gzip.open(filename, 'rt')  # 'rt' means read in text mode
    entry = {}
    times = 0
    for l in f:
        l = l.strip()
        colonPos = l.find(':')
        if colonPos == -1:
            yield entry
            entry = {}
            continue
        eName = l[:colonPos]
        rest = l[colonPos+2:]
        entry[eName] = rest
        times = times + 1
        if times % 10000 == 0:
            print(f'json{times}')
    yield entry

def extract_fields(entry):
    product_id = entry.get("product/productId", "")
    user_id = entry.get("review/userId", "")
    score = entry.get("review/score", "")
    review_time = entry.get("review/time", "")
    return (product_id, user_id, score, review_time)

def process_file(filename):
    product_id_map = {}
    user_id_map = {}
    product_counter = 0
    user_counter = 0
    processed_data = []
    times = 0

    for e in parse(filename):
        if e:
            product_id, user_id, score, review_time = extract_fields(e)

            if product_id not in product_id_map:
                product_id_map[product_id] = product_counter
                product_counter += 1

            if user_id not in user_id_map:
                user_id_map[user_id] = user_counter
                user_counter += 1

            mapped_product_id = product_id_map[product_id]
            mapped_user_id = user_id_map[user_id]
            processed_data.append((mapped_user_id, mapped_product_id, float(score), int(review_time)))

        times = times + 1
        if times % 10000 == 0:
            print(f'np {times}')

    return np.array(processed_data)

# # 示例用法
filename = "all.txt.gz"
data_array = process_file(filename)

# 将 NumPy 数组保存到 CSV 文件中
csv_filename = "Amazon_data.csv"
np.savetxt(csv_filename, data_array, delimiter=" ", fmt='%d %d %.1f %d', header="user_id product_id score time", comments='%')

print(f"Data successfully saved to {csv_filename}")


