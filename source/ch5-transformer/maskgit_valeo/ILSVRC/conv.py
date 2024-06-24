import json

def convert_to_json(file_path):
    """
    Converts a file with annotations to a JSON dictionary.
    
    :param file_path: Path to the annotation file.
    :return: JSON dictionary with keys as file names and values as class labels.
    """
    annotations_dict = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            file_name = parts[0]
            class_label = parts[1]
            annotations_dict[file_name] = class_label
    
    return annotations_dict

# Convert and save to JSON
file_path = 'val_annotations.txt'
annotations_dict = convert_to_json(file_path)

with open('annotations.json', 'w') as json_file:
    json.dump(annotations_dict, json_file, indent=4)

print("Conversion complete. JSON saved to 'annotations.json'.")



