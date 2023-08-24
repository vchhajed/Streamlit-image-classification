# Streamlit-image-classification

This is a simple image classification app built using Streamlit, Streamlit Cloud, PyTorch, Docker, and Kubernetes.

## Running the Streamlit App Locally

1. Install the required dependencies using the following command:
   ```
   python3 -m pip install -r requirements.txt
   python3 -m streamlit run app.py
   ```
   - Once the app is build an running you can use Streamlit Cloud to deploy your Streamlit applicaiton through Github Repo.


(Optional)
## Dockerizing the App
- Build 
```
docker build -t steamlit-image-classification-app .
```
- Run 
```
docker run -p 8501:8501 streamlit-image-classification
```

## Deploying on Kubernetes
- Apply the Kubernetes deployment YAML to create the deployment
```
kubectl apply -f deployment.yaml
```