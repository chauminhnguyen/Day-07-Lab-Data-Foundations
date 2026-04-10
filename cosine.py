from src.chunking import compute_similarity
from src.embeddings import LOCAL_EMBEDDING_MODEL, LocalEmbedder
import os

pairs = [
    ["Bệnh thoái hóa thần kinh gây mất trí nhớ.", "Alzheimer là bệnh thoái hóa thần kinh."],
    ["Con người có thể bị lây sán dây từ cá.", "Ăn cá sống có thể bị sán dây."],
    ["Vi khuẩn lao truyền qua đường hô hấp.", "Lao phổi là bệnh truyền nhiễm."],
    ["Vệ sinh cá nhân giảm lây bệnh.", "Rửa tay hay sạch phòng bệnh."],
    ["Các mạch máu giãn ở hậu môn gây trĩ.", "Trĩ là tình trạng viêm ở hậu môn."],
]
embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))

# Calculate cosine similarity for all pairs
def calculate_pair_similarities():
    similarities = []
    for a, b in pairs:
        # Embed the sentences
        emb_a = embedder([a])[0]
        emb_b = embedder([b])[0]
        # Compute cosine similarity
        sim = compute_similarity(emb_a, emb_b)
        similarities.append((a, b, sim))
        print(f"Similarity between '{a}' and '{b}': {sim:.4f}")
    return similarities

if __name__ == "__main__":
    calculate_pair_similarities() 