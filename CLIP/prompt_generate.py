import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("/home/test/LIVA/ZWQ/pretrained/ViT-L-14.pt.1", device=device)
model = model.float()  # 确保模型为float32
# image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)

domain_text = {'day': "A photo of a human taken during the day"}
with open('prompts.txt','r') as f:
    for ind,l in enumerate(f):
        domain_text.update({str(ind):l.strip()})
import clip
domain_token = dict([(k,clip.tokenize(t)) for k,t in domain_text.items()])

text_features_list = []
for i, val in enumerate(domain_token.items()):
  name, dtk = val
  if name == 'day':
    continue
  with torch.no_grad():
      # image_features = model.encode_image(image)
      text_features = model.encode_text(dtk.cuda()).float()
      text_features_list.append(text_features)
      # logits_per_image, logits_per_text = model(image, text)
      # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
all_text_features = torch.cat(text_features_list, dim=0)
torch.save(all_text_features, '/home/test/LIVA/ZWQ/pretrained/text_features_prompt.pt')
# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
print('text_features:{}'.format(all_text_features.shape))