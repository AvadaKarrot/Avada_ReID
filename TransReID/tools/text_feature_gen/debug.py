import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("/home/test/LIVA/ZWQ/pretrained/ViT-L-14.pt.1", device=device)
model = model.float()  # 确保模型为float32
# image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize([
                    #   "a photo of a human body's eyebrows", 
                      "a photo of a human body's nose", 
                      "a photo of a human body's head",
                    #   "a photo of a human body's left elbows",
                    #   "a photo of a human body's right elbows",
                      "a photo of a human body's arms",
                    #   "a photo of a human body's left arms",
                    #   "a photo of a human body's right arms",
                      "a photo of a human body's eyes",
                    #   "a photo of a human body's left eye",
                    #   "a photo of a human body's right eye",
                      "a photo of a human body's ears",
                    #   "a photo of a human body's left ear",
                    #   "a photo of a human body's right ear",
                      "a photo of a human body's neck",
                    #   "a photo of a human body's left ankles",
                    #   "a photo of a human body's right ankles",
                      "a photo of a human body's mouth",
                      "a photo of a human body's shoulder",
                    #   "a photo of a human body's left wrist",
                    #   "a photo of a human body's right wrist",
                      "a photo of a human body's hands",                    
                      "a photo of a human body's left hand",
                    #   "a photo of a human body's right hand",                      
                    #   "a photo of a human body's left hip",
                    #   "a photo of a human body's right hip",
                      "a photo of a human body's feet",
                    #   "a photo of a human body's left foot",
                    #   "a photo of a human body's right foot",   
                      "a photo of a human body's abdomen",                                  
                      ]).to(device)

with torch.no_grad():
    # image_features = model.encode_image(image)
    text_features = model.encode_text(text).float()
    
    # logits_per_image, logits_per_text = model(image, text)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

torch.save(text_features, '/home/test/LIVA/ZWQ/pretrained/text_features_12.pt')
# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
print('text_features:{}'.format(text_features.shape))