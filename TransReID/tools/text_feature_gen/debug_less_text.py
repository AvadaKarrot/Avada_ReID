import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("/home/test/LIVA/ZWQ/pretrained/ViT-L-14.pt.1", device=device)
model = model.float()  # 确保模型为float32
# image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)

# domain_text = {'day': "A photo of a human taken during the day"}
domain_text = []
with open('prompts.txt','r') as f:
    for ind,l in enumerate(f):
        domain_text.append(l.strip())

text = clip.tokenize(domain_text).to(device)

with torch.no_grad():
    # image_features = model.encode_image(image)
    text_features = model.encode_text(text).float()
    
    # logits_per_image, logits_per_text = model(image, text)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

torch.save(text_features, '/home/test/LIVA/ZWQ/pretrained/text_features_4.pt')
# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
print('text_features:{}'.format(text_features.shape))