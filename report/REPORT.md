# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Minh Châu
**Nhóm:** X2  
**Ngày:** 10/4/2026  

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**  
> Hai câu có độ tương đồng cao về ngữ nghĩa, vector embedding của chúng nằm gần nhau trong không gian vector.

**Ví dụ HIGH similarity:**
- Sentence A: Bệnh Alzheimer là một bệnh thoái hóa thần kinh.
- Sentence B: Alzheimer gây suy giảm trí nhớ và nhận thức.
- Tại sao tương đồng: Cả hai đều mô tả cùng một bệnh và cùng ý nghĩa.

**Ví dụ LOW similarity:**
- Sentence A: Hôm nay trời rất đẹp.
- Sentence B: Bệnh lao phổi là bệnh truyền nhiễm nguy hiểm.
- Tại sao khác: Hai câu thuộc hai chủ đề hoàn toàn khác nhau.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**  
> Cosine similarity đo góc giữa các vector nên phản ánh tốt sự tương đồng về hướng (ngữ nghĩa), không bị ảnh hưởng bởi độ lớn vector như Euclidean distance.

---

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size = 500, overlap = 50. Bao nhiêu chunks?**

- Effective step = chunk_size - overlap = 500 - 50 = 450  
- Số chunks ≈ ceil((10000 - 500) / 450) + 1  
= ceil(9500 / 450) + 1  
= ceil(21.11) + 1 = 22 + 1 = **23 chunks**

**Đáp án:** ~23 chunks

**Nếu overlap tăng lên 100 thì sao?**  
- Step = 500 - 100 = 400 → số chunks tăng lên  
- Overlap lớn giúp giữ context tốt hơn giữa các chunk nhưng làm tăng số lượng chunk và chi phí xử lý.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Bệnh  

**Tại sao nhóm chọn domain này?**  
> Nhóm muốn xây dựng hệ thống hỏi đáp y khoa sử dụng RAG, nên cần dữ liệu về bệnh để hỗ trợ truy vấn và trả lời chính xác.

---

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata |
|---|--------------|-------|----------|----------|
| 1 | alzheimer.md | https://tamanhhospital.vn/alzheimer/ | 27966 | source, category |
| 2 | benh-san-day.md | https://tamanhhospital.vn/benh-san-day/ | 12700 | source, category |
| 3 | benh-tri.md | https://tamanhhospital.vn/benh-tri/ | 12569 | source, category |
| 4 | benh-dai.md | https://tamanhhospital.vn/benh-dai/ | 12700 | source, category |
| 5 | benh-lao-phoi.md | https://tamanhhospital.vn/benh-lao-phoi/ | 12704 | source, category |

---

### Metadata Schema

| Trường | Kiểu | Ví dụ | Vai trò |
|-------|------|------|--------|
| source | string | URL | Dùng để trích dẫn |
| category | string | bệnh truyền nhiễm | Lọc theo loại bệnh |

---

## 3. Chunking Strategy (15 điểm)

### Baseline Analysis

| Tài liệu | Strategy | Chunk Count | Avg Length | Context |
|----------|----------|-------------|------------|--------|
| alzheimer | fixed_size | 183 | 199.4 | Partial |
| alzheimer | by_sentences | 74 | 368.9 | Yes |
| alzheimer | recursive | 1254 | 20.7 | Yes |

### Strategy Của Tôi

**Loại:** SentenceChunker

**Mô tả:**

    Sentence Chunking phân tích ngữ nghĩa và cú pháp để chia tài liệu thành các đơn vị câu hoàn chỉnh:

    Nhận diện ranh giới câu (Sentence Boundary Detection): Thuật toán sử dụng các dấu câu cơ bản (dấu chấm, dấu hỏi, dấu chấm than) làm điểm neo.

    Xử lý ngoại lệ: Các Sentence Chunker chất lượng (thường kết hợp với các mô hình NLP như spaCy, NLTK hoặc Stanford CoreNLP) có khả năng phân biệt dấu chấm kết thúc câu với dấu chấm trong các từ viết tắt (ví dụ: Dr., ThS., mg., U.S.).

    Gom nhóm (Optional): Các câu sau khi được tách ra có thể được gom lại thành các chunk lớn hơn (chứa 2-3 câu liền kề) kèm theo một khoảng trượt (overlap) nhỏ để đảm bảo mạch văn bản được liên tục khi đưa vào các mô hình tìm kiếm.

**Tại sao chọn:**

    Độ dài ngữ nghĩa lý tưởng: Mức độ dài trung bình của sentence chunking là đủ lớn để chứa một ý tưởng trọn vẹn, giúp các mô hình ngôn ngữ lớn (LLM) sau này thực hiện tóm tắt hoặc trả lời câu hỏi chính xác hơn.
    Khắc phục sự phân mảnh của Recursive: Phương pháp recursive trong bảng tạo ra quá nhiều chunk với độ dài trung bình quá ngắn. Việc băm nát tài liệu y khoa (như báo cáo Alzheimer) thành các mảnh vài chữ sẽ phá vỡ mối liên hệ giữa các thực thể y tế (ví dụ: mất liên kết giữa triệu chứng và chẩn đoán).
    Phương pháp fixed_size (cắt theo kích thước cố định) chỉ giữ được ngữ cảnh một phần ("Partial"). Trong y khoa, việc cắt ngang một câu vì hết "quota" độ dài có thể làm thay đổi hoàn toàn ý nghĩa.

---

### So Sánh

| Strategy                       | Ưu điểm                                                                                           | Nhược điểm                                                                                    |
| ------------------------------ | ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Fixed-size chunking    | Triển khai đơn giản, tốc độ cao, dễ kiểm soát kích thước chunk                                    | Cắt ngẫu nhiên theo token/ký tự → dễ phá vỡ câu và ngữ nghĩa, làm giảm chất lượng retrieval   |
| Recursive chunking | Linh hoạt, ưu tiên giữ cấu trúc lớn (paragraph → sentence → word), phù hợp với nhiều loại văn bản | Không ổn định (chunk size không đồng đều), có thể tạo chunk quá nhỏ hoặc quá phân mảnh        |
| Sentence chunking           | Giữ nguyên đơn vị ngữ nghĩa tự nhiên (câu), giúp tăng độ chính xác retrieval; dễ kiểm soát logic  | Số lượng chunk tăng (overhead indexing cao hơn), đôi khi mất context dài nếu không có overlap |

---

**Kết luận:**

> SentenceChunker là lựa chọn phù hợp nhất cho domain y tế vì đảm bảo tính toàn vẹn ngữ nghĩa ở mức câu – đơn vị chứa thông tin quan trọng như triệu chứng, chẩn đoán và hướng dẫn điều trị. Mặc dù tạo ra nhiều chunk hơn, việc kết hợp với overlap giúp duy trì context liên tục và cải thiện đáng kể chất lượng retrieval trong hệ thống RAG.

### So Sánh: Strategy của tôi vs Baseline

| Strategy | Chunk Count | Avg Length | Precision@5 | Retrieval Quality |
|----------|-------------|------------|-------------|-------------------|
| by_sentences    |         207 |      376.2 |      0.6400 | 64.0%      |
| recursive       |         599 |      128.9 |      0.6400 | 64.0%      |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | SentenceChunker | 6/10 | Giữ trọn vẹn cấu trúc ngữ pháp của từng câu, không bao giờ cắt ngang từ. Dễ dàng triển khai và áp dụng cho mọi loại văn bản thô. | Mất bối cảnh tổng thể. AI đọc 1 câu độc lập có thể không biết nó thuộc bệnh nào. Dễ làm đứt gãy các đoạn thông tin liên kết. |
| Minh | LateChunker |  6/10 |Giữ được tính mạch lạc và mối liên kết thông tin chặt chẽ nhờ việc duy trì ngữ cảnh lớn (long context) ở bước biểu diễn ban đầu (indexing). | Có thể gây ra hiện tượng trích xuất thừa thông tin không cần thiết nếu bước "late split" không được cấu hình độ dài cửa sổ (window size) phù hợp.|
| Yến | MarkdownHeadChunker | 8/10 | Từng vector là nội dung ngắn, thống nhất. Kèm với header 1,2 trong metadata giúp hệ thống retrieval có đủ thông tin từ đề mục | Phụ thuộc hoàn toàn vào định dạng gốc. Sẽ vô tác dụng nếu tài liệu là văn bản thô (plain text) không có sẵn các ký tự đánh dấu # hoặc các đoạn không có đánh dấu # như tiêu đề hoặc mở đầu|

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Chiến lược MarkdownHeadChunker tối ưu nhất. Bảo toàn ngữ cảnh (Context-Aware): Giữ trọn vẹn cấu trúc phân cấp tài liệu . Khi truy xuất, AI luôn biết chính xác đoạn text đang đọc thuộc bệnh lý nào.

---

## 4. My Approach (10 điểm)

### Chunking
- Dùng regex để split câu  
- Group theo số câu  
- Không split câu dài (giữ nguyên semantics)

---

### EmbeddingStore

**add_documents**
- Embed từng document  
- Lưu (content, embedding, metadata)

**search**
- Embed query  
- Tính cosine similarity  
- Sort và lấy top-k

---

### KnowledgeBaseAgent

**answer**
- Retrieve top-k chunks  
- Build prompt:
Context: ...
Question:       
Answer:

- Gọi LLM và trả về kết quả

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Bệnh thoái hóa thần kinh gây mất trí nhớ. | Alzheimer là bệnh thoái hóa thần kinh. | high | 0.7137 | Yes |
| 2 | Con người có thể bị lây sán dây từ cá. | Ăn cá sống có thể bị sán dây. | high | 0.7015 | Yes |
| 3 | Vi khuẩn lao truyền qua đường hô hấp. | Lao phổi là bệnh truyền nhiễm. | high | 0.7129 | Yes |
| 4 | Vệ sinh cá nhân giảm lây bệnh. | Rửa tay hay sạch phòng bệnh. | low | 0.4877 | No |
| 5 | Các mạch máu giãn ở hậu môn gây trĩ. | Trĩ là tình trạng viêm ở hậu môn. | low | 0.5058 | Yes |

**Nhận xét:**
> Bảng kết quả cho thấy mô hình hoạt động khá tốt trong việc nhận diện các cặp câu có mức độ tương đồng ngữ nghĩa cao, thể hiện qua việc dự đoán đúng 3/5 trường hợp. Đặc biệt, các cặp liên quan đến kiến thức y khoa rõ ràng và trực tiếp (như bệnh Alzheimer, sán dây, hay lao phổi) đều được mô hình gán mức “high” chính xác với điểm similarity trên 0.7. Tuy nhiên, mô hình vẫn gặp khó khăn ở những trường hợp diễn đạt khác nhau nhưng cùng ý nghĩa, điển hình là cặp 4 khi hai câu đều nói về vai trò của vệ sinh cá nhân trong phòng bệnh nhưng lại bị dự đoán “low”. Ngoài ra, ở cặp 5, dù dự đoán “low” nhưng vẫn được đánh giá là đúng, cho thấy có thể tồn tại sự không nhất quán giữa ngưỡng phân loại và cách gán nhãn. Tổng thể, mô hình có xu hướng nắm bắt tốt các quan hệ ngữ nghĩa trực tiếp nhưng cần cải thiện khả năng hiểu các cách diễn đạt linh hoạt và gián tiếp hơn.

---

## 6. Results — Cá nhân (10 điểm)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Bệnh trĩ có ảnh hưởng khả năng sinh sản không? | Không |
| 2 | Ăn cá có bị sán không? | Có. Có loại sán dây ở bên trong cá. Có khả năng lây bệnh cho người |
| 3 | Làm sao biết mình bị Alzheimer? | Sa sút trí nhớ và khả năng nhận thức; Khó khăn diễn đạt bằng ngôn ngữ; Thay đổi hành vi, tâm trạng, tính cách; Nhầm lẫn thời gian hoặc địa điểm; Đặt đồ vật sai vị trí và không thể nhớ lại mình đã từng làm gì |
| 4 | Những đối tượng nào có nguy cơ cao chuyển từ tình trạng lao tiềm ẩn sang bệnh lao phổi? | Người nhiễm HIV; Người sử dụng ma túy dạng chích; Người bị sụt cân (~10%); Bệnh nhân mắc bụi phổi silic, suy thận, đái tháo đường; Người từng phẫu thuật cắt dạ dày hoặc ruột non; Người ghép tạng hoặc dùng corticoid/thuốc ức chế miễn dịch kéo dài; Bệnh nhân ung thư đầu cổ |
| 5 | Trong trường hợp bị động vật cắn hoặc cào xước, quy trình sơ cứu và các biện pháp y tế cần thực hiện ngay để ngăn chặn virus dại là gì? | **Sơ cứu:** Rửa vết thương bằng nước sạch + xà phòng/povidone iodine ≥15 phút; sát trùng bằng cồn 70% hoặc povidone-iodine; băng bó và đưa đến cơ sở y tế. **Y tế:** Tiêm vắc xin phòng dại càng sớm càng tốt; có thể tiêm thêm huyết thanh kháng dại. **Theo dõi động vật:** Quan sát biểu hiện bất thường của vật nuôi (cắn vô cớ, tiết nước bọt, chết sau vài ngày) |

### Kết Quả Của Tôi

Dựa trên các log được cung cấp, dưới đây là bảng đã được cập nhật lại chính xác các thông tin về Query, nội dung Chunk Top-1, Điểm số (Score) và mức độ liên quan (Relevant):

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Bệnh trĩ có ảnh hưởng khả năng sinh sản không? | Các phương pháp trị bệnh trĩ thường dùng tại bệnh viện và tại nhà | 0.805 | Yes | Không |
| 2 | Con người có thể bị lây sán dây từ cá. | Tùy theo loại vật chủ trung gian (bò, heo, cá…) mà loại sán dây ký sinh vào người | 0.691 | Yes | Có, từ cá chưa được nấu chín |
| 3 | Làm sao biết mình bị Alzheimer? | Nghiên cứu về hội chứng Down và bản sao bổ sung của nhiễm sắc thể | 0.652 | Yes | Mất trí nhớ, khó tập trung, thay đổi hành vi |
| 4 | Những đối tượng nguy cơ cao lao phổi? | Điều hướng website (Trang chủ > CHUYÊN MỤC BỆNH HỌC > Hô hấp > Bệnh lao phổi) | 0.744 | No | Người miễn dịch yếu, lao động, suy dinh dưỡng |
| 5 | Quy trình sơ cứu động vật cắn? | Triệu chứng của bệnh dại (thể cuồng và thể liệt) | 0.729 | No | Rửa 15p water, antiseptic, vaccine |

**Tổng kết:**  
Bao nhiêu queries trả về chunk relevant trong top-3? 3 / 5

---

## 7. What I Learned (5 điểm)

**Từ nhóm:**
- Trade-off giữa Chunk size và Context: Hiểu rõ rằng việc chia nhỏ văn bản giúp tăng độ phân giải cho vector search nhưng dễ làm mất ngữ cảnh liền mạch. Ngược lại, chunk lớn giữ được ý chính nhưng dễ làm loãng "tín hiệu" khiến điểm tương đồng (score) không phản ánh đúng nội dung trọng tâm.

- Sự chênh lệch giữa Score và Relevance: Nhận thấy điểm số cao (0.7 - 0.8) không đồng nghĩa với việc kết quả đó là đúng (như trường hợp query về "Alzheimer" nhưng trả về "Bệnh trĩ"). Điều này cho thấy sự cần thiết của việc tinh chỉnh hàm embedding hoặc ngưỡng (threshold) lọc.

**Từ nhóm khác:**
- Metadata Filtering: Đây là kỹ thuật cực kỳ hiệu quả để thu hẹp không gian tìm kiếm. Việc gắn tag (loại bệnh, đối tượng) giúp loại bỏ các kết quả "nhiễu" có vector gần nhau nhưng khác biệt về ngữ nghĩa chuyên môn.

- Hybrid Search: Kết hợp giữa tìm kiếm ngữ nghĩa (Dense Retrieval) và tìm kiếm từ khóa truyền thống (BM25) giúp xử lý tốt các thuật ngữ chuyên ngành hoặc tên riêng viết tắt mà vector thường bỏ sót.

**Nếu làm lại:**
- Thêm metadata nâng cao  
- Xây dựng evaluation chuẩn

---

## Tự Đánh Giá

| Tiêu chí | Điểm |
|----------|------|
| Warm-up | 5/5 |
| Document selection | 9/10 |
| Chunking strategy | 13/15 |
| My approach | 9/10 |
| Similarity predictions | 4/5 |
| Results | 8/10 |
| Core implementation (tests) | 28/30 |
| Demo | 4/5 |
| **Tổng** | **80 / 100** |

---

**Nhận xét cuối:**  
> Hiểu pipeline tốt, nhưng cần cải thiện phần evaluation và sử dụng embeddings thực để đạt hiệu quả tốt hơn.

======================================== test session starts =========================================
platform win32 -- Python 3.14.3, pytest-9.0.3, pluggy-1.6.0 -- D:\vinuni\Day-07-Lab-Data-Foundations\venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\vinuni\Day-07-Lab-Data-Foundations
plugins: anyio-4.13.0
collected 42 items                                                                                    

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED           [  2%] 
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                    [  4%] 
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED             [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED              [  9%] 
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                   [ 11%] 
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED   [ 14%] 
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED         [ 16%] 
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED          [ 19%] 
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED        [ 21%] 
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                          [ 23%] 
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED          [ 26%] 
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                     [ 28%] 
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                 [ 30%] 
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                           [ 33%] 
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED  [ 35%] 
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED      [ 38%] 
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED      [ 42%] 
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                          [ 45%] 
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED            [ 47%] 
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED              [ 50%] 
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                    [ 52%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED         [ 54%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED           [ 57%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED            [ 61%] 
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                     [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                    [ 66%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED               [ 69%] 
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED           [ 71%] 
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED      [ 73%] 
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED          [ 76%] 
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                [ 78%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED          [ 80%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED     [ 85%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED    [ 88%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED   [ 92%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc
 PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

========================================= 42 passed in 0.50s ========================================= 
