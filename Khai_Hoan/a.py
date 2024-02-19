import torch
import torch.nn as nn

# Định nghĩa bộ từ vựng và câu
vocab_size = 100
sentence = ["Tôi", "yêu", "Việt", "Nam"]

# Chuyển đổi từ "Tôi" thành chỉ số trong bộ từ vựng
word_to_index = {"Tôi": 0, "yêu": 1, "Việt": 2, "Nam": 3}
index_of_Toi = word_to_index["Tôi"]

# Khởi tạo lớp nhúng
embedding_dim = 4
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# Lấy vectơ nhúng của từ "Tôi"
embedded_Toi = embedding_layer(torch.LongTensor([index_of_Toi]))

print("Vectơ nhúng của từ 'Tôi':")
print(embedded_Toi)
