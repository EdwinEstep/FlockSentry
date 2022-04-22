import os

CUSTOM_DATA_PATH = "training/all_image_data_simplified/"

# OLD:
# - 0: chicken
# - 1: fox
# - 2: housepet
# - 3: bird
# - 4: person
# - 5: cat
# - 6: dog
# - 7: horse
# - 8: sheep
# - 9: cow
# - 10: bear
# - 11: bobcat
# - 12: mountain_lion
# - 13: raccoon
# - 14: coyote
# NEW:
# - 0: chicken
# - 1: fox
# - 2: bird
# - 3: person
# - 4: cat
# - 5: dog

keep = ["0", "1", "3", "4", "5", "6", "14"]
replace_dict = {"0": "0", "1": "1", "3": "2", "4": "3", "5": "4", "6": "5", "14": "1"}

for filename in os.listdir(CUSTOM_DATA_PATH):
    # lines we want to keep
    useful_lines = []
    rIdx = 0
    wIdx = 0
    if not filename.endswith(".txt"):
        continue
    # read all the lines in the file
    with open(CUSTOM_DATA_PATH + filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            items = line.split(" ")  # items[0] = class
            if items[0] in keep:
                useful_lines.append(rIdx)  # note that line (all other lines will be removed)
            rIdx = rIdx + 1
    # copy only the lines we want to keep back to the file
    with open(CUSTOM_DATA_PATH + filename, "w") as f:
        for line in lines:
            if wIdx in useful_lines:
                items = line.split(" ")  # items[0] = class
                items[0] = replace_dict[items[0]]
                new_line = " ".join(items)
                f.write(new_line)
            wIdx = wIdx + 1

# remove empty text files and the images that are associated with them
for filename in os.listdir(CUSTOM_DATA_PATH):
    label_path = CUSTOM_DATA_PATH + filename
    if not label_path.endswith(".txt"):
        continue
    if os.path.getsize(label_path) == 0:
        image_path = CUSTOM_DATA_PATH + filename.replace(".txt", ".jpg")
        if not os.path.exists(image_path):
            image_path = image_path.replace(".jpg", ".png")
        if not os.path.exists(image_path):
            image_path = image_path.replace(".png", ".jpeg")
        if not os.path.exists(image_path):
            print(f"not found: {image_path}")
            continue
        os.remove(label_path)
        os.remove(image_path)
