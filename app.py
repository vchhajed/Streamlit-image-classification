import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import json
import pandas as pd

# Load the pre-trained MobileNetV2 model
model = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)
model.eval()

# Define transformations for preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ImageNet class index to label mapping
with open("imagenet_classes.json") as f:
    imagenet_class_labels = json.load(f)

# Streamlit app
st.title("Image Classification App")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    
    # Preprocess the image for inference
    img_tensor = preprocess(image)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(img_tensor)
        probabilities = F.softmax(predictions[0], dim=0)
    
    # Display top 5 predicted classes in a table
    st.write("Top 5 Predictions:")
    # Display top 5 predicted classes in a DataFrame
    top5_probs, top5_classes = torch.topk(probabilities, 5)
    class_name_list = []
    prob_list = []
    for prob, class_idx in zip(top5_probs, top5_classes):
        class_name_list.append(imagenet_class_labels[class_idx])
        prob_list.append(f"{prob.item():.4f}")
    
    df = pd.DataFrame({"Class Name": class_name_list, "Probability": prob_list})
    
    st.dataframe(df)  # Display the DataFrame in Streamlit
