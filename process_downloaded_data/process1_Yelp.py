import pandas as pd
import time
import os

# 加载数据
file_path = 'Labelled Yelp Dataset.csv'
df = pd.read_csv(file_path)
#df = df.drop(columns=['Review'])#*

# 重新命名User_id
unique_user_ids = df['User_id'].unique()
user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_user_ids, start=1)}
df['User_id'] = df['User_id'].map(user_id_map)

# 重新命名Product_id
unique_product_ids = df['Product_id'].unique()
product_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_product_ids, start=1)}
df['Product_id'] = df['Product_id'].map(product_id_map)

# 日期格式转换为时间戳
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y').apply(lambda x: int(time.mktime(x.timetuple())))

# 按时间戳排序
df = df.sort_values(by='Date')

# 保存处理后的数据
output_folder = 'yelp_hotel_labelled'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_file_path = os.path.join(output_folder, 'Labelled_Yelp_Dataset_processed.csv')
df.to_csv(output_file_path, index=False)
print(f"Processed data saved to '{output_file_path}'")
