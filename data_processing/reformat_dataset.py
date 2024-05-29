import os
import shutil
import csv

# Replace this with the path to your CSV file
csv_file_path = 'labels.csv'

# Replace this with the path to your images folder
images_folder_path = 'images'

# Create a dictionary to store the label to folder mapping
label_folder_mapping = {}

# Read the CSV file
with open('labels.csv' , mode = 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        # Get the label and image file name
        label = row['label']
        file_name = row['image_name']

        # Create the label folder if it doesn't exist
        if label not in label_folder_mapping:
            label_folder_mapping[label] = os.path.join(images_folder_path, label)
            os.makedirs(label_folder_mapping[label], exist_ok=True)

        # Move the image file to the label folder
        image_file_path = os.path.join(images_folder_path, file_name + '.jpg')
        shutil.move(image_file_path, label_folder_mapping[label])