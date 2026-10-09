# Kết quả khôi phục từ notebook Colab

Nguồn: C:/Users/windows/Downloads/Lab22_DPO_T4.ipynb.
Các chỉ số và ảnh dưới đây được trích trực tiếp từ output đã lưu, không chạy lại mô hình và không tạo số liệu mới.

## Đã giữ lại

- Lab22_DPO_T4.executed.ipynb: bản sao nguyên vẹn notebook có output, bao gồm lỗi cuối phiên.
- screenshots/: 4 biểu đồ/bảng phần bắt buộc và 1 biểu đồ NB3b. Đây là PNG hiển thị nhúng trong notebook; kích thước có thể khác file fig.savefig gốc.
- dpo_metrics.json: nội dung JSON đã in ở NB3 (cell index 71, tính từ 0).
- variants-table.txt: bảng kết quả bonus NB3b đã in, số liệu có thể bị làm tròn khi hiển thị.
- fixed-responses-truncated.txt: câu trả lời của 8 prompt cố định được in rút gọn. KHÔNG dùng file này thay side_by_side.jsonl để chấm.

## Kết quả quan sát được

- NB0 khớp loss tham chiếu 0.6981; loss khởi đầu 0.6931.
- SFT final loss: 1.3604.
- Dữ liệu preference: 800 train / 100 eval; chosen dài hơn trong 65.9% cặp.
- DPO train loss: 0.6537003755569458; held-out reward accuracy: 0.75; held-out margin: 0.16626992575824262.
- Nhãn chẩn đoán được chương trình in: INTENDED. Cả chosen và rejected reward cuối đều dương; khi viết phản tư cần phân tích biểu đồ, không diễn giải nhãn này thành rejected chắc chắn giảm.
- NB3b có output cho DPO, RPO, DPO-norm, LD-DPO và ORPO.
- NB4 đã sinh 8 fixed + 50 held-out; độ dài trung bình được in làm tròn: SFT 576, DPO 570 ký tự.
- Sau đó cell kiểm tra báo /content/lab22 không tồn tại; cell kế tiếp tạo thư mục rỗng rồi lỗi import config. Cell tổng hợp chưa chạy.

## Không thể khôi phục từ notebook này

Không có toàn bộ 58 cặp câu trả lời trong output, không có trọng số SFT/DPO, và chưa có kết quả giám khảo. Do đó không tạo side_by_side.jsonl, judge_summary.json hay cấu hình adapter giả.

Nếu không có bản sao runtime riêng, cần chạy lại NB1–NB4 trên GPU. Bỏ qua toàn bộ NB3b nếu ưu tiên hoàn thành phần bắt buộc. Sao lưu mô hình SFT và adapter DPO sau từng giai đoạn nếu muốn tránh huấn luyện lại; sao lưu side_by_side.jsonl ngay sau NB4 §1 để có thể chấm riêng qua API trên CPU hoặc RM trên GPU. Chỉ lưu notebook không lưu các file runtime.

Giữ các kết quả lần chạy mới cùng nhau; không ghép JSON/adapter/câu trả lời của hai lần chạy rồi coi là một thí nghiệm.
