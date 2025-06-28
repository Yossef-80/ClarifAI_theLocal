import os
import torch
from PIL import Image
import numpy as np
from facenet_pytorch import InceptionResnetV1
import pickle  # For saving embeddings

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️ Using device: {device}")

# Load FaceNet model
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
torch.save(facenet.state_dict(), 'face_embedding_vggface2.pth')


def build_face_db(folder='Embeddings/organized_faces-2'):
    db = {}
    for name in os.listdir(folder):
        person_dir = os.path.join(folder, name)
        if not os.path.isdir(person_dir):
            continue
        embeddings = []
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB').resize((160, 160))
                img_tensor = torch.tensor(np.array(img) / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    emb = facenet(img_tensor)
                    embeddings.append(emb.squeeze().cpu().numpy())
            except Exception as e:
                print(f"❌ Failed on {img_path}: {e}")
        if embeddings:
            db[name] = np.mean(embeddings, axis=0)
    return db

# Build face DB
face_db = build_face_db('Embeddings/organized_faces-2')

# Save to file using pickle
with open('face_db.pkl', 'wb') as f:
    pickle.dump(face_db, f)

print(f"✅ Face DB saved with {len(face_db)} people: {list(face_db.keys())}")

