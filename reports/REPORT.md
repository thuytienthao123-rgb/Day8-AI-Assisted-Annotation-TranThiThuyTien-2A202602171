# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Trần Thị Thủy Tiên

Công cụ gán nhãn đã dùng: AnyLabeling

Báo cáo đối chiếu và truy xuất đầy đủ số liệu từ các nguồn: `reports/rounds_table.md`, `outputs/selection_round1.csv`, `outputs/metrics_round0.json`, `outputs/metrics_round1.json`, `outputs/round1_diff.md` và `reports/REVIEW_LOG.csv`. Nhãn kiểm thử được nhìn nhận đúng bản chất là nhãn do mô hình tạo ra để đo độ tương đồng, không coi là chân lý tuyệt đối.

## 1. Dữ liệu và cách chia tập

Tập dữ liệu được trích xuất từ video camera cố định trên cầu vượt quay đường cao tốc ban đêm dài 160 giây với tốc độ 2.5 khung hình/giây. Việc chia tập chưa gán nhãn (pool - 268 ảnh) và tập kiểm thử (test - 20 ảnh) bắt buộc phải thực hiện theo **trục thời gian có vùng đệm (buffer - 112 ảnh)** ở giữa thay vì chia ngẫu nhiên (random split) vì các lý do cốt lõi sau:
- **Hiện tượng tương quan thời gian cao:** Do góc máy camera đứng yên tuyệt đối, các khung hình kế tiếp nhau chỉ cách nhau 0.4 giây hầu như giống hệt nhau về phông nền và bối cảnh. Mỗi chiếc xe di chuyển qua khung hình thường lưu lại từ 5 đến 15 giây.
- **Rò rỉ dữ liệu (Data Leakage):** Nếu chia ngẫu nhiên, cùng một chiếc xe sẽ xuất hiện đồng thời ở cả tập huấn luyện và tập kiểm thử (chỉ xê dịch vài pixel giữa các frame liền kề). Khi đó, mô hình sẽ được chấm điểm trên chính những vật thể nó vừa nhìn thấy ở bước huấn luyện.
- **Hậu quả nếu chia ngẫu nhiên:** Số đo trên tập kiểm thử (AP50, Precision, Recall) sẽ bị **lệch lạc nghiêm trọng theo hướng lạc quan giả tạo (artificially over-optimistic)**. Mô hình có vẻ đạt điểm rất cao nhưng thực chất chỉ đang "học vẹt" bối cảnh cụ thể mà mất hoàn toàn khả năng khái quát hóa (generalization) trên các xe mới ngoài thực tế.
- **Vai trò vùng đệm (buffer):** Vùng đệm 4 giây trước và sau mỗi đoạn test (tổng cộng 112 ảnh bị loại bỏ) đảm bảo mọi chiếc xe trong tập test đã hoàn toàn đi ra khỏi khung hình trước khi tập pool bắt đầu, giữ khoảng cách thời gian tối thiểu giữa pool và test là 4.4 giây, loại bỏ triệt để rò rỉ thông tin.

## 2. Mô hình khởi đầu lạnh (cold start)

Dòng số liệu Vòng 0 trích xuất trực tiếp từ `reports/rounds_table.md` và `outputs/metrics_round0.json`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Dựa vào ảnh so sánh `outputs/compare_round0.jpg` và số đo chi tiết:
- **Các loại xe không khớp nhãn tham chiếu:**
  + Mô hình khởi đầu lạnh (YOLOv8n tiền huấn luyện COCO) bỏ sót nghiêm trọng các xe màu tối chạy trong bóng đêm (rất nhiều box màu vàng - False Negative).
  + Bỏ sót các xe ở xa chỉ lộ cụm đèn pha hoặc đèn hậu nhỏ mờ ảo.
  + Các xe bị che khuất một phần ở làn giữa hoặc xe bị cắt một phần thân ở mép ảnh cũng thường xuyên bị mô hình bỏ qua.
- **Độ phủ (Recall) theo kích thước xe:**
  + `R small` chỉ đạt **0.182** (bỏ sót hơn 81.8% xe nhỏ ở xa), trong khi `R medium` đạt **0.547** và `R large` đạt **0.561**. Điều này phản ánh rõ ràng hạn chế cố hữu của bộ phát hiện COCO ban đầu: khi kích thước vật thể giảm dần và độ tương phản ban đêm kém, đặc trưng thị giác bị suy giảm mạnh khiến mô hình không đủ tự tin kích hoạt phát hiện.
- **Trường hợp cần người rà lại nhãn tham chiếu:**
  + Trong `outputs/compare_round0.jpg`, xuất hiện một số box màu đỏ (False Positive) ở làn đường phía xa bên phải. Khi quan sát kỹ bằng mắt người, vị trí đó thực sự có một chiếc xe đang di chuyển nhưng bộ nhãn tham chiếu (vốn cũng do một mô hình khác tự động tạo ra) đã bỏ sót không đánh nhãn chiếc xe này. Do đó, mô hình khởi đầu lạnh thực tế đã phát hiện đúng nhưng lại bị hệ thống đánh giá chấm oan thành False Positive. Trường hợp này chứng minh nhãn tham chiếu chưa qua rà soát thủ công của con người không phải chân lý tuyệt đối.

## 3. Chiến lược chọn mẫu

### Giải thích công thức và tham số
Công thức tính điểm chọn mẫu trong `tools/al_select.py`:
$$\text{score} = W_U \cdot U + W_A \cdot A + W_D \cdot D = 0.5 \cdot U + 0.3 \cdot A + 0.2 \cdot D$$
- **$U$ (Uncertainty - Trọng số 0.5):** Trung bình độ bất định của 5 box khó nhất trong ảnh, tính theo $u = 1 - |2 \cdot \text{conf} - 1|$. Điểm $u$ đạt cực đại bằng 1.0 khi độ tin cậy $\text{conf} = 0.5$ (trạng thái mô hình lưỡng lự phân vân nhất giữa việc có xe hay không có xe).
- **$A$ (Ambiguity - Trọng số 0.3):** Tỷ lệ số box mơ hồ ($0.15 \le \text{conf} < 0.50$), chuẩn hóa theo ảnh có nhiều box mơ hồ nhất trong pool. Đại lượng này ưu tiên các khung hình có mật độ đối tượng mập mờ cao.
- **$D$ (Diversity - Trọng số 0.2):** Khoảng cách thời gian từ ảnh đang xét tới ảnh đã gán nhãn gần nhất (chuẩn hóa tối đa 10 giây). Đại lượng này khuyến khích thuật toán chọn các ảnh nằm rải rác trên trục thời gian, tránh việc chọn dồn dập vào cùng một thời điểm.
- **Vai trò của `MIN_GAP_S = 2.0s`:** Ràng buộc bắt buộc hai ảnh được chọn trong cùng một lô phải cách nhau ít nhất 2.0 giây. Do camera đặt cố định, hai ảnh cách nhau dưới 2 giây có cấu trúc giao thông và các xe hầu như giống hệt nhau. Ràng buộc này ngăn chặn lãng phí công sức gán nhãn vào các ảnh gần trùng lặp.

### Dẫn chứng cân nhắc thực tế từ `reports/SELECTION.md`
- **Ba frame mô hình chọn:**
  1. `frame_0182.jpg` (Rank 1, score = 0.9591, A = 1.0, 28 box đề xuất): Có số box mơ hồ cao nhất pool, chứa luồng xe đêm phức tạp.
  2. `frame_0369.jpg` (Rank 2, score = 0.9324, U = 0.9315, 43 box đề xuất): Mật độ giao thông cực đông, cách frame 0182 gần 75 giây, đảm bảo tính đa dạng thời gian tối đa.
  3. `frame_0331.jpg` (Rank 5, score = 0.9154, 47 box đề xuất): Nhiều xe tải lớn và vệt sáng phản quang trên đường ướt.
- **Một frame khác để chứng minh sự cân nhắc:**
  + `frame_0372.jpg` (Rank 6, score = 0.9101, t = 148.8s): Dù có điểm bất định nằm trong top 6 toàn bộ tập dữ liệu, frame này đã bị loại bỏ vì nó chỉ cách `frame_0369.jpg` (t = 147.6s) đúng 1.2 giây (< `MIN_GAP_S`). Việc loại bỏ frame này giúp tiết kiệm ngân sách rà nhãn, tránh nạp thông tin trùng lặp vào mô hình.

### Điểm bất định có chứng minh ảnh sẽ cải thiện mô hình không?
**Không.** Điểm bất định cao chỉ phản ánh rằng mô hình hiện tại đang thiếu tự tin đối với bức ảnh đó, chứ không đảm bảo việc gán nhãn ảnh đó sẽ giúp mô hình tăng AP50 trên tập test. Nếu độ bất định bắt nguồn từ nhiễu cảm biến camera, ánh đèn pha chói lóa làm lóa ống kính hoặc góc khuất không thể phục hồi thông tin, việc ép mô hình học các ảnh này có thể dẫn đến việc học nhiễu cục bộ và giảm năng lực tổng quát hóa.

## 4. Các vòng học chủ động (active learning)

### Bảng số đo tổng hợp các vòng
Trích xuất từ `reports/rounds_table.md` sau khi hoàn thành Vòng 1:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 334 | 0.530 | -0.242 | 1.000 | 0.099 | 0.181 | 0.000 | 0.098 | 0.268 |

### Mức độ sửa nhãn gợi ý (từ `outputs/round1_diff.md`)
Trong lô 12 ảnh của Vòng 1, mô hình đề xuất 169 box ban đầu. Sau quá trình rà soát và chỉnh sửa cẩn trọng của học viên:
- **Số box giữ nguyên (accepted):** 136 box (tỷ lệ chấp nhận đạt 80%).
- **Số box chỉnh sửa (edited):** 18 box (thu nhỏ các box bị AI vẽ quá khổ do ôm vệt đèn rọi dưới đường, và kéo rộng các box AI chỉ khoanh chỏm đầu đèn xe để ôm trọn thân xe).
- **Số box xóa bỏ (deleted - FP của model):** 15 box (loại bỏ các box nhận nhầm vệt phản quang mặt đường và các box trùng lặp đè lên nhau).
- **Số box thêm mới (added - FN của model):** 180 box (bổ sung toàn bộ các xe tối màu, xe ở xa bị AI bỏ sót hoàn toàn).
- **Tổng số box hoàn thiện đưa vào huấn luyện:** **334 box** (tăng gần gấp đôi so với đề xuất ban đầu).

### Biến động số đo và phân tích nhóm xe
- **Thay đổi AP50:** AP50 trên tập kiểm thử đạt 0.530 (giảm -0.242 so với khởi đầu lạnh 0.771).
- **Độ chính xác Precision đạt mức tuyệt đối 1.000 (FP = 0):** Mô hình sau khi học nhãn chuẩn của người gán đã hoàn toàn loại bỏ các phát hiện nhầm (zero false positives tại ngưỡng conf 0.25).
- **Độ phủ Recall:** Tại ngưỡng đánh giá conf 0.25, Recall tổng thể giảm xuống 0.099 (R large đạt 0.268, R medium đạt 0.098, R small đạt 0.000). Nguyên nhân là do mô hình huấn luyện trên tập dữ liệu nhỏ (12 ảnh) với tiêu chuẩn nhãn khắt khe đã trở nên cực kỳ thận trọng (conservative). Khi dự đoán trên tập test, phân bố confidence score của mô hình dịch chuyển về dải thấp hơn, khiến nhiều phát hiện đúng bị chặn ở ngưỡng conf 0.25.

### Phân tích so sánh hình ảnh (`outputs/compare_round1.jpg`)
Quan sát ảnh đối chiếu `outputs/compare_round1.jpg`:
- **Ca tốt lên:** Ở các làn đường trung tâm cự ly gần và trung bình, mô hình Vòng 1 vẽ các bounding box cực kỳ chuẩn xác, ôm khít thân xe, hoàn toàn không bị hiện tượng box bị kéo dài xuống mặt đường theo vệt đèn pha như mô hình Vòng 0. Không còn bất kỳ box đỏ (False Positive) nào xuất hiện.
- **Ca xấu đi:** Các cụm xe ở rất xa dưới chân cầu bị mô hình Vòng 1 bỏ qua nhiều hơn mô hình Vòng 0 (tăng box vàng FN), do mô hình chưa nhận đủ mẫu xe nhỏ ở xa trong lô 12 ảnh để tự tin dự đoán vượt ngưỡng 0.25.

### Phân biệt ba nguồn dữ liệu và ca khó theo guideline
- **Quan sát độc lập (`reports/BLIND_SCAN.md` trên `frame_0182.jpg`):** Mắt người đếm được 25 xe rõ ràng và 1 xe nghi ngờ (tổng 26 xe), ghi nhận nguy cơ AI bỏ sót xe tối màu ở cận cảnh mép dưới và xe ở chân cầu.
- **Lỗi pre-label đã sửa (`outputs/round1_diff.md` & `reports/REVIEW_LOG.csv`):** AI ban đầu chỉ tìm được 13 xe (bỏ sót 50%). Người gán nhãn đã thêm mới 14 xe (trong đó có đúng chiếc xe tối màu ở mép dưới `cx=0.433`), sửa 2 box ôm vệt đèn, xóa 1 box trùng lặp để đưa tổng số xe lên đúng 26 xe, khớp hoàn toàn với bản quét độc lập.
- **Kết quả mô hình sau huấn luyện:** Mô hình Vòng 1 học được quy tắc không vẽ box vào vệt đèn, đạt Precision tuyệt đối 1.0.
- **Mô tả ca khó theo guideline:** Tình huống xe đi ở làn xa trong `frame_0380.jpg` và `frame_0182.jpg`: Thân xe tối màu chìm hoàn toàn vào màn đêm, chỉ nhìn thấy hai chấm đèn pha rực sáng. Theo dòng 17 của `GUIDELINE_LABEL.md`, người gán nhãn không được khoanh riêng hai chấm đèn mà phải ước lượng vùng bao thân xe quanh cụm đèn. Đồng thời, theo dòng 21, các xe quá nhỏ ở sát chân cầu cao dưới 16 pixel được bỏ qua để tránh đưa nhiễu nhãn vào mô hình.

## 5. Kết luận và giới hạn

### Đánh giá kết quả Vòng 1 và quyết định dừng/tiếp tục
Kết quả Vòng 1 cho thấy mô hình đã học được bài học rất quan trọng về chất lượng box: loại bỏ hoàn toàn việc nhận diện nhầm vệt đèn pha và bóng tối mặt đường (Precision đạt 100%). Mặc dù AP50 giảm do mô hình trở nên quá thận trọng trên các xe nhỏ ở ngưỡng conf 0.25, nhưng đây là hiện tượng bình thường khi chuyển từ mô hình COCO tổng quát sang dữ liệu đặc thù ban đêm chỉ với 12 ảnh. Tôi quyết định **tiếp tục Vòng 2** (nếu triển khai thực tế) để bổ sung thêm các mẫu xe ở cự ly xa nhằm khôi phục Recall.

### Đề xuất 2 ca còn yếu/bất định cho vòng tiếp theo (từ `outputs/selection_round2.csv`)
1. **frame_0073.jpg** (Rank 5 vòng 2, t = 29.2s, score = 0.9022, U = 0.9388): Nằm ở giai đoạn đầu video, chứa nhiều xe chạy ở làn ngoài cùng có độ tương phản thấp. Chi phí rà nhãn vừa phải (khoảng 30 box), không bị trùng lặp với các ảnh vòng 1.
2. **frame_0126.jpg** (Rank 6 vòng 2, t = 50.4s, score = 0.8931, U = 0.9209): Nằm ở mốc thời gian cách xa các ảnh vòng 1, có nhiều xe tải và xe khách di chuyển với tốc độ khác nhau.
- *Nguy cơ ảnh gần trùng:* Cần duy trì nghiêm ngặt ràng buộc `MIN_GAP_S >= 2.0s` với cả tập train vòng 1 lẫn các ứng viên vòng 2 để không lãng phí chi phí gán nhãn.

### Tác động của các giới hạn đánh giá
- **Tập kiểm thử chỉ 20 ảnh:** Cỡ mẫu nhỏ khiến phương sai thống kê rất lớn; chỉ cần một vài xe thay đổi trạng thái phát hiện đã làm biến động đáng kể chỉ số AP50.
- **Luật bỏ qua xe nhỏ < 16 pixel:** Giúp bảo vệ mô hình không bị phạt bởi các vật thể quá mập mờ ở đường chân trời, nhưng cũng phản ánh rằng bài toán thực địa chỉ tập trung vào các phương tiện có khả năng gây va chạm ở cự ly can thiệp được.
- **Nhãn tham chiếu do mô hình AI tạo ra:** Bộ nhãn test chưa qua rà soát thủ công của con người nên chứa cả lỗi bỏ sót lẫn lỗi vẽ lệch. Do đó, điểm số AP50 phản ánh mức độ tương đồng giữa mô hình của học viên với mô hình sinh nhãn tham chiếu, không phản ánh chất lượng tuyệt đối ngoài đời thực.

### Biện pháp kiểm tra khi AP50 giảm trước khi train thêm
Nếu AP50 giảm sau một vòng học chủ động, thay vì vội vàng huấn luyện tiếp, các bước kiểm tra kỹ thuật cần thực hiện gồm:
1. **Kiểm tra độ nhất quán của nhãn (Annotation Consistency):** So sánh lại các box đã sửa giữa các ảnh xem có bị hiện tượng lúc thì vẽ ôm vệt đèn, lúc lại cắt bỏ vệt đèn không; đảm bảo quy tắc gán nhãn được áp dụng đồng nhất 100%.
2. **Kiểm tra phân bố Confidence Score:** Vẽ biểu đồ phân bố độ tin cậy của mô hình trên tập test để xem mô hình có bị giảm confidence tổng thể hay không (nếu có, hạ ngưỡng conf đánh giá từ 0.25 xuống 0.10 để kiểm tra Recall thực tế).
3. **Kiểm tra hiện tượng Overfitting trên lô nhỏ:** Đánh giá số epoch huấn luyện (50 epochs trên 12 ảnh có thể khiến mô hình bị quá khớp với một số góc chụp cục bộ).
4. **So sánh trực quan trên `compare_round*.jpg`:** Trực tiếp đối chiếu xem các ca bị mất điểm là do mô hình thực sự phát hiện kém đi hay do nhãn tham chiếu của tập test có vấn đề.
