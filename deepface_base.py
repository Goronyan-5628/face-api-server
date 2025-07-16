# -*- coding: utf-8 -*-
import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from deepface import DeepFace
from scipy.spatial.distance import cosine, euclidean
from concurrent.futures import ThreadPoolExecutor

# ✅ Firestore 認証追加
import firebase_admin
from firebase_admin import credentials, firestore

# 🔹 Firebase 認証（適切なパスに変更）
cred = credentials.Certificate("./firebase-admin-key.json")  # ← ここを正しいパスに
firebase_admin.initialize_app(cred)
db = firestore.client()

# ✅ TensorFlow の oneDNN 最適化を無効化
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 🔹 顔検出 + 切り抜き + パディング処理
def process_face(image_path, output_path, target_size=(160, 160)):
    try:
        faces = DeepFace.extract_faces(image_path, detector_backend="ssd", enforce_detection=False)
        if not faces:
            return {"error": f"No face detected in {image_path}"}
        face = (faces[0]["face"] * 255).astype(np.uint8)
        h, w, _ = face.shape
        max_dim = max(h, w)
        padded = np.ones((max_dim, max_dim, 3), dtype=np.uint8) * 255
        sx = (max_dim - w) // 2
        sy = (max_dim - h) // 2
        padded[sy:sy + h, sx:sx + w] = face
        resized = cv2.resize(padded, target_size)
        Image.fromarray(resized).save(output_path)
        return output_path
    except Exception as e:
        return {"error": f"Face processing failed: {str(e)}"}

# 🔹 Firestoreから追加プロフィール情報を取得
def fetch_member_info(image_name):
    docs = db.collection("members").where("imageNames", "array_contains", image_name).stream()
    for doc in docs:
        data = doc.to_dict()
        return {
            "name": data.get("name"),
            "group": data.get("group"),
            "age": data.get("age"),
            "imageUrl": data.get("imageUrl")
        }
    return {
        "name": None,
        "group": None,
        "age": None,
        "imageUrl": None
    }

# 🔹 入力画像の処理
image_paths = sys.argv[1:]
if not image_paths:
    print(json.dumps({"error": "No images provided"}))
    sys.exit(1)

processed_images = []
for idx, image_path in enumerate(image_paths):
    output_path = f"cropped_user_{idx}.jpg"
    result = process_face(image_path, output_path)
    if isinstance(result, dict) and "error" in result:
        print(json.dumps(result))
        sys.exit(1)
    processed_images.append(output_path)

# 🔹 特徴量抽出（4096次元）
def extract_features(image_path):
    return DeepFace.represent(
        img_path=image_path,
        model_name="VGG-Face",
        detector_backend="skip",
        enforce_detection=False
    )[0]['embedding']

with ThreadPoolExecutor(max_workers=1) as executor:
    features_list = list(executor.map(extract_features, processed_images))

if not features_list:
    print(json.dumps({"error": "No valid features extracted"}))
    sys.exit(1)

user_features = np.median(np.vstack(features_list), axis=0)

# 🔹 メンバー特徴量の読み込み
FEATURES_CSV = "member_features_vggface_direct.csv"
df = pd.read_csv(FEATURES_CSV)
feature_cols = [f"v{i+1}" for i in range(4096)]
member_matrix = df[feature_cols].values

if df.empty:
    print(json.dumps({"error": "No member data found in CSV"}))
    sys.exit(1)

# 🔹 類似度計算
df["cosine_similarity"] = 1 - np.apply_along_axis(lambda x: cosine(user_features, x), 1, member_matrix)
df["euclidean_distance"] = np.linalg.norm(member_matrix - user_features, axis=1)
df["similarity_score"] = (1.2 * df["cosine_similarity"]) - (0.8 * df["euclidean_distance"])
df_sorted = df.sort_values(by="similarity_score", ascending=False).head(10)

# 🔹 Firestoreからプロフィールを補完
output = []
for _, row in df_sorted.iterrows():
    info = fetch_member_info(row["image_name"])
    result = {
        "image_name": row["image_name"],
        "cosine_similarity": row["cosine_similarity"],
        "euclidean_distance": row["euclidean_distance"],
        "similarity_score": row["similarity_score"],
        **info
    }
    output.append(result)

# 🔹 結果をJSONで出力
print(json.dumps(output, ensure_ascii=False))
