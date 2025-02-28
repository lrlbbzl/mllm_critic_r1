import json
import os
import random
import re

import pandas as pd
from PIL import Image as I
from datasets import Dataset, Features, Value, Image
from tqdm import tqdm

random.seed(42)


def format_math_explanation(text: str):
    answer_match = re.search(r'Answer:\s*([A-Z])', text)
    if answer_match:
        answer = answer_match.group(1)
    else:
        raise ValueError("No answer found in the text")

    text = re.sub(r'Answer:\s*[A-Z]', '', text).strip()
    formatted_text = f"<think>{text}</think> <answer>{answer}</answer>"
    return formatted_text

def trans_size(image_path, min_size=28):
    try:
        image = I.open(image_path)
        width, height = image.size

        if width < min_size or height < min_size:
            new_size = (max(width, min_size), max(height, min_size))
            image = image.resize(new_size, I.Resampling.LANCZOS)
            new_image_path = image_path.replace('.jpg', '_resized.jpg')  # 修改文件名以区分
            image.save(new_image_path)
            return new_image_path  # 返回调整后的图像路径
        return image_path  # 如果图像尺寸已经满足要求，直接返回原路径
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None



def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


input_file = '/opt/tiger/R1-Multimodal-Journey/local_scripts/qa_tuning_20k.json'

data = {
    'image': [],
    'image_path': [],
    'problem': [],
    'solution': [],
}

with open(input_file, 'r') as f:
    data_all = json.load(f)

data_all_filtered = []
for data_tmp in tqdm(data_all, desc="Processing images", unit="image"):
    image_path = os.path.join('/opt/tiger/geo_data/images', data_tmp['image'])
    is_valid = trans_size(image_path)
    if is_valid is not None:
        data_tmp['image'] = is_valid
        data_all_filtered.append(data_tmp)

print('len(data_all): ', len(data_all))
print('len(data_all_filtered): ', len(data_all_filtered))

random.shuffle(data_all_filtered)
for item in data_all_filtered:
    image_path = os.path.join('/opt/tiger/geo_data/images', item.get('image'))
    problem = item['question']
    solution = format_math_explanation(item['answer'])

    data['image'].append(image_path)
    data['image_path'].append(image_path)
    data['problem'].append(problem)
    data['solution'].append(solution)

df = pd.DataFrame(data)

features = Features({
    'image': Image(),
    'image_path': Value('string'),
    'problem': Value('string'),
    'solution': Value('string')
})

train_dataset = Dataset.from_pandas(df, features=features)

train_save_path = "geo_data/train-00000-of-00001.parquet"

train_dataset.to_parquet(train_save_path)

print(f"Saved to {train_save_path}")
