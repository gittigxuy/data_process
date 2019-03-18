import json


className = {
    1: 'bus',
    2: 'traffic light',
    3: 'traffic sign',
    4: 'person',
    5: 'bike',
    6: 'truck',
    7: 'motor',
    8: 'car',
    9: 'train',
    10: 'rider',
}

classNum = [1, 2, 3, 4, 5, 6, 7, 9, 10]


def writeNum(Num):
    with open("outputval.json", "a+", encoding='utf-8') as f:
        f.write(str(Num))


inputfile = []
inner = {}
##向test.json文件写入内容
PATH_TO_JSON = '/home/xuy/桌面/code/python/challenageAI/bdd_dataset/labels/labels/bdd100k_labels_images_val.json'
with open(PATH_TO_JSON, "r+") as f:
    allData = json.load(f)
    # data = allData["annotations"]
    print("read ready")


    for data in allData:
        for item in data['labels']:
            # print(item.keys())
            if 'box2d' in item:
                inner = {
                    "filename": str(data["name"]).zfill(6),
                    "name": item['category'],
                    "box2d": item['box2d'],
                    "id": item['id'],
                    "truncated": item['attributes']['truncated'],
                    "occluded": item['attributes']['occluded']
                }
                inputfile.append(inner)

    inputfile = json.dumps(inputfile,indent=4)
    writeNum(inputfile)
