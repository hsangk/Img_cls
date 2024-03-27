from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
import cv2

# Clone 필요 : https://github.com/IDEA-Research/GroundingDINO
CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "./ckpt/groundingdino_swint_ogc.pth"
DEVICE = "cpu"
IMAGE_PATH = "./data/hsk/human.jpg"
TEXT_PROMPT = "giraffe. house. human."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

import time
start = time.time()
model = load_model(CONFIG_PATH, CHECKPOINT_PATH)
print("load_model: ", time.time() - start)



start = time.time()
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD,
    device=DEVICE,
)
print("predict: ", time.time() - start)

print(phrases)

idxs = []

for i in range(len(phrases)):
    if phrases[i] == 'human' or phrases[i] == 'person':
        idxs.append(i)

# boxes_t = [boxes[idx] for idx in idxs][0].unsqueeze(0)
# logits_t = [logits[idx] for idx in idxs][0].unsqueeze(0)
# phrases_t = [[phrases[idx] for idx in idxs][0]]


start = time.time()
annotated_frame, xyxy = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
print("annotate: ", time.time() - start)
# xyxy = list(map(int, xyxy[0]))
# a = cv2.rectangle(image_source, tuple(xyxy[:2]), tuple(xyxy[-2:]), color=(0, 0, 0))
cv2.imwrite("./data/hsk/human_pred.jpg", annotated_frame)