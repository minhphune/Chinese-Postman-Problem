# Chinese Postman Problem cho robot kiểm tra

Đây là hiện thực Python cho bài toán **Chinese Postman Problem (CPP)** trong bài viết `main.pdf`: xây dựng một tour đóng đi qua mọi hành lang/cung bắt buộc ít nhất một lần, xuất phát và trở về depot, với chi phí nhỏ nhất. Dự án có cả phiên bản vô hướng (UCPP), có hướng (DCPP), cơ chế tái lập kế hoạch khi trọng số thay đổi, kiểm thử tự động và benchmark tái lập được.

## Nội dung chính

- **UCPP chính xác:** phát hiện các đỉnh bậc lẻ, tạo metric closure bằng Dijkstra, giải minimum-weight perfect matching bằng Blossom, khai triển các shortest path rồi dựng Euler circuit.
- **DCPP chính xác:** kiểm tra liên thông mạnh, tính mất cân bằng `in-degree - out-degree`, giải bài toán min-cost transportation trên metric closure có hướng bằng successive shortest augmenting paths, rồi dựng directed Euler circuit.
- **Định danh an toàn cho đa đồ thị:** `base_id` biểu diễn hành lang/cung vật lý; `instance_id` biểu diễn một lần sử dụng cụ thể trong đa đồ thị Euler hóa. Thiết kế này bảo toàn cạnh song song và các bản sao.
- **Dự phòng động:** giữ một tour neo khi chỉ trọng số thay đổi, tính cận regret có chứng chỉ và chỉ tái lập khi cận vượt ngưỡng.
- **So sánh tour Euler:** Hierholzer được dùng cho tour thực tế; Fleury được giữ để minh họa và benchmark.

## Cấu trúc

| Tệp | Vai trò |
| --- | --- |
| `main.pdf` | Bài viết 71 trang: nền tảng đồ thị, chứng minh, mô hình robot, mã nguồn và kết quả. |
| `adaptive_cpp_robot.py` | UCPP, Dijkstra, metric closure, Blossom, bitmask oracle, Fleury, Hierholzer và regret certificate. |
| `directed_cpp.py` | DCPP, Dijkstra có hướng, min-cost transportation và directed Hierholzer. |
| `benchmark_cpp.py` | Sinh đồ thị có seed cố định; benchmark tĩnh và mô phỏng cập nhật trọng số động. |
| `test_cpp_robot.py` | 10 kiểm thử cho ví dụ chuẩn, đồ thị Euler sẵn, cạnh song song, trọng số 0, DCPP, transport và certificate. |
| `results/static.csv` | Trung vị/IQR benchmark tĩnh sau 15 lần đo mỗi cấu hình. |
| `results/dynamic.csv` | Kết quả bốn chính sách tái lập trong 50 bước cập nhật. |

## Cài đặt và chạy

Yêu cầu Python 3.10+ (đã dùng type-hint hiện đại), cùng các gói trong `requirements.txt`.

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

Chạy hai ví dụ chuẩn:

```powershell
python adaptive_cpp_robot.py
python directed_cpp.py
```

Kết quả mong đợi:

- UCPP ví dụ: base cost `36`, matching cost `7`, optimal cost `43`.
- DCPP ví dụ: base cost `7`, augmentation cost `2`, optimal cost `9`.

Để tái tạo benchmark:

```powershell
python benchmark_cpp.py
```

Lệnh benchmark sẽ **ghi đè** `results/static.csv` và `results/dynamic.csv` bằng phép đo của máy đang chạy; thời gian tĩnh vì thế có thể khác, còn cấu hình và seed được giữ cố định.

## Kết quả benchmark hiện có

Benchmark tĩnh dùng 15 lần đo sau warm-up. Khi số edge instance sau Euler hóa tăng từ khoảng 104 lên 824, thời gian Hierholzer tăng từ `0.075 ms` lên `0.596 ms`, trong khi Fleury tăng từ `0.974 ms` lên `55.272 ms`. Điều này phù hợp với vai trò của Hierholzer là lựa chọn mặc định để dựng tour.

Trong kịch bản động 50 bước:

| Chính sách | Số lần tái lập | Regret lớn nhất | Cận certificate lớn nhất |
| --- | ---: | ---: | ---: |
| `always` | 50 | 0% | 0% |
| `local_0.15` | 11 | 0% | - |
| `certificate_0.05` | 17 | 0% | 4.997% |
| `certificate_0.10` | 0 | 0.244% | 5.523% |

## Điều kiện và phạm vi mô hình

- UCPP yêu cầu đồ thị liên thông (sau khi bỏ các đỉnh cô lập), cạnh không khuyên, `base_id` duy nhất và trọng số hữu hạn không âm.
- DCPP yêu cầu digraph liên thông mạnh trên các đỉnh có cung, cung không khuyên, `base_id` duy nhất và trọng số hữu hạn không âm.
- Mỗi bài toán tạo một tour đóng cho một robot và một depot.
- Regret certificate chỉ hợp lệ khi topology, hướng, tập cạnh/cung bắt buộc, depot và tập tour khả thi không đổi; đóng/mở hành lang là thay đổi cấu trúc và phải giải lại.
- API transportation của DCPP hiện dùng `__source__` và `__sink__` làm tên nút nội bộ. Không dùng hai chuỗi này làm nhãn đỉnh cho đến khi phần cài đặt được thay bằng sentinel không thể va chạm.

## Kiểm tra tính đúng

Các module kiểm tra các bất biến quan trọng: liên thông, miền trọng số, cân bằng parity/bán bậc, tour đóng tại depot, mỗi instance được dùng đúng một lần và đẳng thức chi phí. Bộ test còn đối chiếu Blossom với bitmask oracle trên bài toán nhỏ, kiểm tra transport bằng brute force 2x2, và kiểm tra cận regret trên các cập nhật ngẫu nhiên có seed cố định.

## Tài liệu

Xem `main.pdf` để có định nghĩa, chứng minh, phân tích độ phức tạp, mô hình trọng số cho robot kiểm tra và diễn giải đầy đủ các bảng kết quả.
