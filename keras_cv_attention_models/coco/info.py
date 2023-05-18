COCO_LABELS = """person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign,
    parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie,
    suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket,
    bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut,
    cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven,
    toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush"""
COCO_80_LABEL_DICT = {id: ii.strip() for id, ii in enumerate(COCO_LABELS.split(","))}
INVALID_ID_90 = [11, 25, 28, 29, 44, 65, 67, 68, 70, 82]
COCO_90_LABEL_DICT = {id: ii for id, ii in zip(set(range(90)) - set(INVALID_ID_90), COCO_80_LABEL_DICT.values())}
COCO_90_LABEL_DICT.update({ii: "Unknown" for ii in INVALID_ID_90})
COCO_80_to_90_LABEL_DICT = {id_80: id_90 for id_80, id_90 in enumerate(set(range(90)) - set(INVALID_ID_90))}
