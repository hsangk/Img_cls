from transformers import pipeline
import os
from PIL import Image
import requests
import csv
import time

start = time.time()
# checkpoint = "openai/clip-vit-large-patch14"
checkpoint = "openai/clip-vit-base-patch16"
detector = pipeline(model=checkpoint, task="zero-shot-image-classification")
print("load : ", time.time() - start)



path = './images/train/'
test_path = './images/test/0/'
classes = {'dog':0, 'elephant':1, 'giraffe':2,'guitar':3,'horse':4,'house':5,'person':6}
cls_list = os.listdir(path)
data_list = []
for i in cls_list:
    tmp_path = path + i+'/'

    tmp_images_path = sorted([os.path.join(tmp_path, f)
                               for f in os.listdir(tmp_path)])
    for j in tmp_images_path:
        data_list.append(j)
test_list = os.listdir(test_path)
test_images_filepaths = []
for i in test_list:
    test_images_filepaths.append(test_path+i)
predictions_ = []
a = 1
for url in test_images_filepaths:
    image = Image.open(url)

    start = time.time()
    predictions = detector(image, candidate_labels=cls_list)
    print("predict : ", time.time() - start)
    print(a)
    a += 1
    result = classes[predictions[0]['label']]
    predictions_.append(result)



csv_file_path = 'predictions_clip.csv'

# Write the predictions to the CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['', 'answer value'])  # Write the header
    for i, prediction in enumerate(predictions_):
        writer.writerow([i, prediction])