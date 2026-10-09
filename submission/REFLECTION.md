# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Quốc Cường
**Khoá:** A20-K4
**Tier đã chạy:** T4
**Ngày:** 2026-10-09

> **Tình trạng bài làm:** Đã chạy NB0–NB3, bonus NB3b và phần sinh câu trả lời/bảng so sánh của NB4. Chưa hoàn tất chấm tự động NB4 do hết quota GPU và mất file runtime trước khi sao lưu. Vì vậy bài này không có win rate hay khoảng tin cậy của giám khảo.
>
> **Nguồn số liệu:** [Notebook có output](recovered-colab/Lab22_DPO_T4.executed.ipynb), [JSON số liệu DPO trích từ output](recovered-colab/dpo_metrics.json), [bảng bonus](recovered-colab/variants-table.txt) và [8 ví dụ đã bị rút gọn khi in](recovered-colab/fixed-responses-truncated.txt). Các ảnh trong bài được trích từ output notebook. Không có bản sao đầy đủ của side_by_side.jsonl, trọng số mô hình hoặc judge_summary.json.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Tesla T4; log Unsloth báo dung lượng GPU 14.563 GB (không phải VRAM sử dụng cao nhất) |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Cách huấn luyện | Mô hình 4-bit + LoRA r=16, alpha=32; 33.030.144 tham số được huấn luyện; max_length=768, seed=42 |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned; 1.000 mẫu, 1 epoch, 125 bước |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy, tiếng Việt; 800 cặp train / 100 cặp held-out, chia theo prompt |
| Chosen dài hơn rejected (NB2) | 65,9%; trung vị chosen 94 token, rejected 86 token |
| DPO: β / tốc độ học / số epoch | 0,1 / 5e-6 / 1; loss sigmoid, 100 bước |
| Giám khảo | Cấu hình mặc định: Skywork/Skywork-Reward-V2-Qwen3-4B và Skywork/Skywork-Reward-V2-Llama-3.2-3B; chưa có kết quả chấm hoặc sanity accuracy |
| Chi phí | Sử dụng Colab T4; chưa có số liệu chi phí để báo cáo |

NB0 đã khớp loss tham chiếu 0,6981 trên bộ số kiểm tra. Trường hợp policy trùng reference cho loss 0,6931, phù hợp với log(2). NB1 in loss SFT cuối là 1,3604 và đã thực hiện lưu/gộp mô hình SFT trong phiên cũ; các trọng số này không còn trong bản notebook tải về.

Ảnh bằng chứng: [loss SFT](recovered-colab/screenshots/02-sft-loss.png), [độ dài preference](recovered-colab/screenshots/02b-pref-length.png).

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | Thanh tiến trình hiển thị 29 phút 21 giây cho 100 bước; không coi đây là tổng thời gian NB3 gồm nạp mô hình và precompute reference |
| VRAM cao nhất | Chưa có số đo peak VRAM trong output được lưu |
| Loss được log đầu tiên | 0,693640 |
| Training loss trung bình do trainer trả về | 0,653700 |
| Loss ở lần log train cuối / validation loss cuối | 0,615305 / 0,623540 |
| Reward chosen / rejected cuối trên train | 0,919913 / 0,737830 |
| Reward gap cuối trên train | 0,182083 |
| Reward chosen / rejected trên held-out | 0,939881 / 0,773611 |
| Độ chính xác reward trên held-out | 75% |
| Margin trên held-out | 0,166270 |
| Chẩn đoán tự động | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 576 → 570 ký tự, theo số đã làm tròn trong output |

Trường final_train_loss trong JSON lấy từ result.training_loss, nên tôi phân biệt nó với loss ở lần log cuối. Accuracy reward 75% đo việc chosen có reward cao hơn rejected trên tập preference held-out; đây **không phải** tỉ lệ câu trả lời DPO thắng SFT trong NB4.

---

## 3. Đọc đường reward (≥ 100 từ)

Ảnh: [03-dpo-reward-curves.png](recovered-colab/screenshots/03-dpo-reward-curves.png).

Cả rewards/chosen và rewards/rejected trên train đều có xu hướng tăng, dù dao động ở các bước cuối. Reward chosen cuối đạt 0,919913, cao hơn rejected 0,737830, tạo margin 0,182083. Trên held-out, chosen tăng từ 0,243570 ở bước 25 lên 0,939881 ở bước 100; rejected cũng tăng từ 0,205110 lên 0,773611. Margin held-out tăng lần lượt khoảng 0,038460 → 0,119564 → 0,159435 → 0,166270 ở các bước 25, 50, 75 và 100. Như vậy margin tăng vì chosen tăng nhiều hơn rejected, không phải vì rejected giảm.

Train và held-out cùng tăng margin; giá trị cuối của held-out thấp hơn train khoảng 0,0158. Tôi chưa thấy dấu hiệu chỉ train cải thiện còn held-out đứng yên trong lượt chạy này. Tuy nhiên, một lần chạy với 100 cặp held-out chưa đủ để khẳng định không có overfitting hoặc mô hình tổng quát tốt cho mọi loại câu hỏi.

Chương trình gắn nhãn INTENDED. Nhãn này phù hợp với việc margin dương và tăng, nhưng không hoàn toàn khớp mô tả lý tưởng trong README là chosen tăng còn rejected giảm: thực tế cả hai cùng tăng. Tôi cần đọc từng đường reward thay vì dùng nhãn tự động làm kết luận duy nhất. Lượt DPO chính không thể hiện xu hướng likelihood displacement trên reward trung bình, còn một số biến thể ở NB3b có hiện tượng này.

Về câu hỏi NB0, reward margin phụ thuộc vào **chênh lệch** thay đổi log-prob của chosen và rejected so với reference. Ví dụ chosen giảm 3 nat nhưng rejected giảm 5 nat thì chênh lệch vẫn tăng 2 nat, nên loss DPO có thể giảm dù chosen trở nên ít có khả năng xuất hiện hơn. Ngoài ra, tổng log-prob cộng theo token khiến độ dài ảnh hưởng tới thang điểm; dữ liệu có 65,9% chosen dài hơn cần được kiểm tra thiên vị độ dài. Điều đó không có nghĩa DPO luôn ưu tiên câu dài trong mọi trường hợp.

---

## 4. So sánh SFT vs SFT+DPO

Ảnh: [04-side-by-side-table.png](recovered-colab/screenshots/04-side-by-side-table.png).

Notebook xác nhận đã sinh câu trả lời cho **8 câu cố định + 50 câu held-out**. Tuy nhiên, file chứa câu trả lời đầy đủ không được sao lưu trước khi mất runtime. Chưa có judge_summary.json; bảng dưới đây ghi trạng thái thực tế, không thay kết quả chưa đo bằng số 0.

| Nhóm | Số prompt đã sinh | n đã chấm hợp lệ | DPO thắng | SFT thắng | Hoà | Win rate / CI 95% | Win rate cặp dài gần bằng | Câu dài hơn thắng |
|---|---:|---|---|---|---|---|---|---|
| held-out | 50 | Chưa có | Chưa có | Chưa có | Chưa có | Chưa có | Chưa có | Chưa có |
| hữu ích — helpfulness | 4 | Chưa có | Chưa có | Chưa có | Chưa có | Chưa có | Chưa có | Chưa có |
| an toàn — safety | 4 | Chưa có | Chưa có | Chưa có | Chưa có | Chưa có | Chưa có | Chưa có |

**Giám khảo:** dự kiến hội đồng hai RM theo §1, nhưng chưa hoàn tất. **Sanity accuracy, score_length_spearman, per_judge và judge_agreement:** chưa có. **Position consistency:** chưa có vì chưa chạy giám khảo API.

Tôi chưa thể kết luận khoảng tin cậy có chứa 0,5 hay không, giám khảo có đọc tốt tiếng Việt hay không, hoặc DPO có thắng SFT. Độ dài trung bình đã in giảm từ khoảng 576 xuống 570 ký tự, nên không thấy tăng độ dài trung bình trong lần sinh này; riêng con số đó không loại trừ thiên vị độ dài trong từng cặp hoặc trong cách giám khảo chấm. Cũng chưa thể đo mức bất đồng giữa hai RM hay rò rỉ sở thích. Theo mô tả của lab, hai RM khác mô hình nền nhưng cùng nhóm Skywork, nên hạn chế về tính độc lập của giám khảo vẫn cần nêu khi có kết quả.

**Ví dụ hữu ích — h2, gợi ý món ăn từ 5 kg gạo và 12 quả trứng:** Phần output còn lại cho thấy SFT gợi ý gà nướng, khoai tây, bánh mì và cá, chưa bám sát nguyên liệu người dùng cung cấp. DPO cũng gợi ý gà nướng, rồi liệt kê nhiều nguyên liệu bổ sung và lặp lại nước cốt dừa/nước cốt chanh. Trong đoạn hiển thị, DPO chưa cho thấy cải thiện về tính ngắn gọn và khả năng làm theo yêu cầu; câu trả lời còn có dấu hiệu lặp. Đây là một phản ví dụ cho suy luận rằng reward accuracy cao tự động kéo theo câu trả lời hữu ích hơn.

**Ví dụ an toàn — s2, yêu cầu viết tin nhắn đe doạ bạn cùng lớp:** Cả SFT lẫn DPO đều từ chối và hướng tới giải quyết mâu thuẫn theo cách khác. Bản DPO bổ sung lời nhắc về quy định trong trường học, nhưng phần hiển thị chưa cho thấy khác biệt đủ rõ để kết luận DPO an toàn hơn. Hai câu trả lời cũng còn chuỗi thẻ tool_call không phù hợp trong văn bản hiển thị, ảnh hưởng chất lượng trình bày. Tôi ghi nhận hiện tượng này nhưng chưa xác định nguyên nhân từ output hiện có.

Hai phân tích trên chỉ dựa trên đoạn văn đã được notebook rút gọn khi in; chúng không thay thế đánh giá trên câu trả lời đầy đủ hoặc 50 câu held-out.

---

## 5. Đánh đổi theo β (bonus make beta-sweep)

**Chưa chạy β-sweep.** Dòng β=0,1 dưới đây là lượt DPO chính, không phải một thí nghiệm sweep đã hoàn tất.

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0,05 | — | — | Chưa chạy | Không có số liệu |
| 0,1 | 0,166270 | 75% | INTENDED | Lượt NB3 chính |
| 0,5 | — | — | Chưa chạy | Không có số liệu |

**Ba câu giả thuyết:** Tôi dự đoán thay đổi β sẽ ảnh hưởng cả gradient trong quá trình học lẫn thang reward, nên cần giữ nguyên dữ liệu và lịch huấn luyện để so sánh. Với β=0,05 hoặc 0,5, tôi chưa thể dự đoán chắc accuracy sẽ cao hơn β=0,1 vì kết quả còn phụ thuộc learning rate và số bước. Tôi sẽ ưu tiên so accuracy held-out, chất lượng câu trả lời và độ dài thay vì xếp hạng bằng margin thô, vì margin đã được nhân với β.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Quyết định tôi muốn đánh giá lại là chạy toàn bộ NB3b trước khi hoàn tất chấm điểm NB4. Đây là thứ tự thể hiện trong notebook đã lưu: sau DPO chính, tôi chạy DPO, RPO, DPO-norm, LD-DPO và ORPO trên cấu hình bonus, rồi mới sinh câu trả lời NB4. Giá trị của lựa chọn này là có thể quan sát trực tiếp sự khác nhau giữa các loss trên cùng 300 mẫu train và 100 mẫu eval, thay vì chỉ đọc công thức. Kết quả giúp tôi thấy DPO-norm và LD-DPO có likelihood displacement, trong khi RPO có reward chosen dương lớn hơn nhưng accuracy chưa vượt DPO.

Phương án thay thế là ưu tiên hoàn thành NB4 ngay sau NB3, sao lưu kết quả bắt buộc, rồi mới chạy bonus nếu tài nguyên còn cho phép. Khi nhìn lại, đây là phương án phù hợp hơn với giới hạn GPU và hạn nộp. Việc đã chạy xong huấn luyện và sinh câu trả lời không bảo đảm bài làm hoàn chỉnh: phiên Colab sau đó không còn thư mục làm việc, và tôi chỉ giữ được notebook có output. Vì chưa sao lưu side_by_side.jsonl, tôi không thể chuyển ngay sang giám khảo API để hoàn thành phần đánh giá dù bước sinh đã xong.

Nếu làm lại, tôi sẽ thay đổi thứ tự ưu tiên, lưu mô hình và adapter sau từng giai đoạn, đồng thời sao lưu câu trả lời ngay sau NB4 §1. Tôi cũng sẽ bổ sung checkpoint định kỳ vì cấu hình hiện tại không lưu checkpoint giữa quá trình train. Với tôi, bài học chính là thiết kế cách lưu và khôi phục kết quả ngay từ đầu; nhiều thí nghiệm bonus chưa bù được việc thiếu bằng chứng đánh giá cuối cùng của phần bắt buộc.

---

## 7. Bộ đo chuẩn (bonus NB6)

**Chưa thực hiện.** Các cell NB6 trong notebook chưa có output.

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---|---|---|---|
| IFEval | Chưa chạy | Chưa có | Chưa có | Chưa có |
| GSM8K | Chưa chạy | Chưa có | Chưa có | Chưa có |
| Global-MMLU-vi | Chưa chạy | Chưa có | Chưa có | Chưa có |

Chưa có cơ sở để nhận xét alignment tax, chênh lệch so với sai số chuẩn hoặc sự nhất quán với đánh giá NB4.

---

## 8. Biến thể loss (bonus NB3b)

Ảnh: [03b-variants.png](recovered-colab/screenshots/03b-variants.png).

Thí nghiệm này dùng **300 cặp train / 100 cặp eval**, đo độ dài trên **20 probe prompt** với max_new_tokens=256. DPO trong bảng là lượt bonus riêng, không phải lượt DPO chính dùng 800 mẫu ở §2. Margin được tính bằng reward chosen trừ reward rejected từ số in trong notebook và làm tròn tới 6 chữ số thập phân.

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình (ký tự) | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 64% | 0,025620 | 391,90 | Chosen và rejected đều dương; nhãn INTENDED |
| RPO | 61% | 0,034783 | 392,35 | Reward chosen 0,553003 và rejected 0,518220; margin lớn hơn DPO nhưng accuracy thấp hơn |
| DPO-norm | 62% | 0,008711 | 393,85 | Hai reward đều âm, rejected âm hơn; LIKELIHOOD DISPLACEMENT |
| LD-DPO | 55% | 0,024804 | 394,15 | Hai reward đều âm, rejected âm hơn; LIKELIHOOD DISPLACEMENT |
| ORPO | Khoảng 64% | Không được lưu theo cùng thước đo | 390,50 | Khởi tạo từ SFT merged; eval_log_odds_ratio = -0,624976 |

Trong 5 biến thể, LD-DPO cho câu trả lời dài nhất, hơn DPO bonus 2,25 ký tự trung bình; ORPO ngắn nhất, thấp hơn DPO 1,40 ký tự. Khoảng cách lớn nhất giữa hai biến thể chỉ là 3,65 ký tự. Không có số đo độ dài của SFT trên cùng 20 probe prompt trong output này, nên tôi chỉ so giữa các biến thể, không khẳng định biến thể nào làm thay đổi độ dài nhiều nhất so với SFT. Tôi cũng không so trực tiếp các độ dài khoảng 390 ký tự này với 576/570 ký tự ở NB4, vì tập prompt và giới hạn sinh khác nhau.

Về cơ chế, RPO bổ sung loss SFT trên chosen, tạo động lực giữ khả năng sinh câu trả lời được chọn. Chuẩn hoá theo độ dài trong DPO-norm làm thay đổi cách tổng log-prob đóng góp vào loss; LD-DPO ở đây dùng ld_alpha=0,5 để điều chỉnh đóng góp của phần độ dài dư. ORPO kết hợp NLL của chosen với thành phần ưu tiên theo log-odds. Những khác biệt đó là cơ sở để đặt giả thuyết về độ dài và likelihood displacement, nhưng dữ liệu 20 prompt cùng chênh lệch độ dài nhỏ chưa đủ để chứng minh một quan hệ nhân quả rõ ràng. Kết quả RPO cũng cho thấy tăng margin không đồng nghĩa accuracy tăng, còn margin của ORPO không thể thay bằng eval_log_odds_ratio rồi so trực tiếp với DPO.

---

## 9. GRPO (bonus NB7)

**Chưa thực hiện.** Các cell NB7 chưa có output.

| Chỉ số | Giá trị |
|---|---|
| Độ chính xác trước / sau, số câu kiểm tra | Chưa có |
| Sai số chuẩn | Chưa tính vì chưa có kết quả thực nghiệm |

Chưa thể kết luận reward định dạng hay reward đáp án tăng trước, hoặc chênh lệch có vượt mức nhiễu hay không.

---

## Danh sách bonus

- [x] NB3b — đã chạy 5 biến thể, còn output và ảnh; chưa giữ được adapter/JSON runtime gốc của bonus
- [ ] NB5 — GGUF SFT+DPO
- [ ] NB6 — benchmark
- [ ] NB7 — GRPO
- [ ] β-sweep
- [ ] Chấm chéo bằng hai họ mô hình
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình
- [ ] BONUS-CHALLENGE.md (không chấm điểm)

---

## Điều bất ngờ nhất

Reward accuracy held-out đạt 75% nhưng đoạn trả lời DPO cho câu hỏi về món ăn vẫn lặp nguyên liệu và chưa bám sát yêu cầu. Điều này khiến tôi phân biệt rõ hơn giữa học thứ tự ưu tiên trên dữ liệu preference và cải thiện chất lượng câu trả lời thực tế; phần đánh giá NB4 còn thiếu là một giới hạn quan trọng của bài làm này.
