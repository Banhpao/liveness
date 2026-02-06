Folder backend nơi chứa :
-  pycatche - éo quan tâm
-  dataset nơi chứa dataset của người dùng, dùng để detect 
- ChamCongFaceID/backend/build_dataset.py - dùng để build bộ data của người đó ( nhớ nhập tên chihs xác vì tên người được detect cũng là tên của dataset)
- ChamCongFaceID/backend/face_db.pkl  - file dataset được mã hóa 512 chiều dùng thằng arcface để làm ( có 3 thằng bự nhất face_recognition của open cv, facenet của nhân còn này là arcface)
- ChamCongFaceID/backend/faceid_stream.py - backend của cả hệ thống nơi xử lý mọi tác vụ 
- ChamCongFaceID/backend/phat.pt - model chống giả mạo ( đã gửi riêng)




Folder frontend nơi chứa :

- ChamCongFaceID/frontend/static - sau này định làm avata cho người dùng nhma lười 
- ChamCongFaceID/frontend/templates - chứa html của chương trình ( mẹo để vọc file html có 2 phần riêng biệt phía trên script là giao diện nếu có chỉnh giao diện thì chỉnh trên script còn nếu muốn chỉnh cách script hiển thị thì chỉnh trong script)
- ChamCongFaceID/frontend/routers.py - nơi chứa 2 hàm liên quan đến video stream 
- ChamCongFaceID/main.py - quan trọng nhất front end nơi chứa api, vận hành giao diện




Peperline ( luồng chạy chương trình)

Camera frame
  ↓
ROI + Anti-spoof
  ↓
ANTI state (lọc fake)
  ↓
FACEID state (ArcFace + voting)
  ↓
SUCCESS state
  ↓
Frontend overlay hiện
  ↓
[3s countdown]
   ├─ Commit → Firebase
   └─ Re-scan → Reset
  ↓
IDLE


chi tiết hơn về detect thì:
- khi mới bật chương tình sẽ ở chế độ (state) ILDE và model chống giả mạo được thiết kế để chạy liên tục, khi nhận diện được có mặt người ( cả 2 class real/fake) thì bắt đầu nhảy qua state anti ( chống giả mạo) state anti này sẽ bắt người dùng phải đứng trong khung roi của giao diện trong x giây ( t là 2s) nếu trong 2s đó hơn 80% các class xuất hiện trong khung roi đó là real thì mới bắt đầu qua state faceid
- khi đã qua được stae faceid ( tức là đồng nghĩa đã khẳng định là mặt thật) thì không quan tâm class real nữa thay vào đó sẽ dùng từng frame TRONG KHUNG ROI để trích xuất đặc trưng và nhận diện faceid bằng arcface, người dùng phải ở yên trong đó trong x giây (t là 3s) sau 3s nếu 80% số frame trả ra là tên người đó thì khẳng định là người đó. sau khi thành công hiển thị giao diện phụ SUCCESS nếu người dùng nhận thấy không phải mình thì nhấn re-scan sẽ rết và quay về state ILDE để làm lại từ đầu còn nếu đúng thì để yên 3s sẽ tự động lưu thông tin chấm công vào firebase database. và thực hiện lại quy trình cho người kế 



Chống giả mạo: 
- file phat.pt chỉ kẻ 2 class   - class fake được kẻ toàn bộ điện thoại, giấy, màn hình chứa ảnh giả mạo 
                                - class real được kẻ chỉ khuôn mặt của người và không kẻ trong class fake 
                                => có nghĩa là class fake ở đây được định nghĩa là phải có xuất hiện các vật thể thứ 3 như điện thoại, tờ giấy hay màn hình còn class real thì là mặt người 
                                => giới hạn data nằm ở chỗ nếu hình được nhận diện mất hoàn toàn viền điện thoại hay những vật thể thứ 3 thì auto nhận là real còn nếu để cực sát nhưng để lộ 1 phần viền thì vẫn nhận là fake
                                => model chỉ bắt đầu sai khi vật thể đặt cực kì gần còn lại ở mức hoàn hảo  


- lý do yêu cầu người dùng đứng trong khung roi là vì;
 - dataset arcface có tận 512 chiều nếu lấy full frame sẽ có những chi tiết thừa nhưng arcface lại quá chi tiết khiến dataset lỗi và không chính xác thế nên khi build dataset thâm ý của ta là bắt người dùng phải đứng trong khung roi để build ngụ ý là để loại bỏ các chi tiết dư thừa xung quanh 
 - nếu dataset đã có roi và phân tích chiều trên roi đó rồi thì khi nhận diện cũng phải ép người dùng đứng trong khung roi đó để trùng chi tiết với dataset tăng độ chi tiết và là để khắc phục giới hạn của dataset là yếu khi ở quá gần thì ta bắt người dùng phải ở xa 




 cấu trúc database realtime: ví dụ 

attendance/
 └── 2026-02-01/
     └── huykhoi/
         ├── time: "23:13:16"
         └── terminal: "K-04"



chạy file main.py thôi là được 