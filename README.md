# AI_Nhom27
Thuật toán Random Forest là một thuật toán học có giám sát (supervised learning) được sử dụng cho cả phân lớp và hồi quy. Thuật toán này được xây dựng dựa trên việc kết hợp nhiều cây quyết định (decision trees) để tạo ra một mô hình dự đoán chính xác và ổn định hơn. 

Các bước của thuật toán Random Forest như sau:
	-Chọn ngẫu nhiên một tập con của dữ liệu huấn luyện.
	-Xây dựng một cây quyết định trên tập con này.
	-Lặp lại các bước 1 và 2 k lần (k là số lượng cây quyết định được chọn) để xây dựng nhiều cây quyết định khác nhau.
	-Đưa ra dự đoán cho một điểm dữ liệu mới bằng cách đưa nó qua tất cả các cây quyết định và tính toán số lượng phiếu bầu cho mỗi lớp.
	-Chọn lớp có số phiếu bầu cao nhất là kết quả cuối cùng.

Ưu điểm của thuật toán Random Forest:Random forests được coi là một phương pháp chính xác và mạnh mẽ vì số cây quyết định tham gia vào quá trình này
	Khả năng xử lý các tập dữ liệu lớn.
	Khả năng xử lý các tập dữ liệu có nhiều biến.
	Khả năng xử lý các tập dữ liệu có giá trị thiếu.
	Khả năng xác định mức độ quan trọng của các biến trong mô hình.
	Khả năng xác định mức độ chính xác của mô hình.
	Có thể được sử dụng trong cả hai vấn đề phân loại và hồi quy.

Nhược điểm của thuật toán Random Forest: Random forests chậm tạo dự đoán bởi vì nó có nhiều cây quyết định.
	Có thể bị overfitting nếu số cây quá lớn.
	Không thể giải thích kết quả dự đoán một cách chi tiết như các thuật toán khác như cây quyết định.

Thuật toán Random Forest được sử dụng trong nhiều lĩnh vực khác nhau như:
-Phân loại hình ảnh
-Dự đoán giá cổ phiếu.
-Dự đoán giá nhà đất.
-Phân loại bệnh tim mạch
-Phân tích tín dụng
-Dự đoán giá nhà
-Dự đoán khả năng khách hàng sẽ mua sản phẩm.
-Dự đoán khả năng khách hàng sẽ hủy đơn hàng.
-Dự đoán khả năng khách hàng sẽ trả nợ.
-Dự đoán khả năng khách hàng sẽ chuyển đổi.
-Nhiều lĩnh vực khác...
Nó được sử dụng trong các bài toán phân loại và hồi quy với tập dữ liệu lớn và có nhiều biến. Nó cũng được sử dụng khi cần xác định mức độ quan trọng của các biến trong mô hình.

Ví dụ về việc áp dụng thuật toán này cho bài toán Demand Forecasting
https://insights.magestore.com/posts/cai-dat-random-forest-cho-bai-toan-demand-forecasting
Để viết chương trình dự đoán hủy phòng sử dụng thuật toán rừng ngẫu nhiên trong Python, bạn có thể tham khảo các bước sau:

1.Nhập các thư viện cần thiết, chẳng hạn như pandas, numpy, sklearn, matplotlib, seaborn.

import pandas as pd import numpy as np import sklearn import matplotlib.pyplot as plt import seaborn as sns
Có một số cách để cài đặt các thư viện này trong Python. Một cách phổ biến là sử dụng pip, một trình quản lý gói cho Python. Bạn có thể sử dụng lệnh pip install trong cửa sổ dòng lệnh để cài đặt các thư viện. Ví dụ:

pip install pandas pip install numpy pip install sklearn pip install matplotlib pip install seaborn

Bạn cũng có thể cài đặt nhiều thư viện cùng một lúc bằng cách liệt kê chúng sau lệnh pip install. Ví dụ:

pip install pandas numpy sklearn matplotlib seaborn

Một cách khác là sử dụng Anaconda, một nền tảng khoa học dữ liệu cho Python. Bạn có thể tải xuống và cài đặt Anaconda từ trang web chính thức: https://www.anaconda.com/products/individual. Sau khi cài đặt Anaconda, bạn có thể sử dụng lệnh conda install trong cửa sổ dòng lệnh để cài đặt các thư viện. Ví dụ:

conda install pandas conda install numpy conda install scikit-learn conda install matplotlib conda install seaborn

Bạn cũng có thể cài đặt nhiều thư viện cùng một lúc bằng cách liệt kê chúng sau lệnh conda install. Ví dụ:

conda install pandas numpy scikit-learn matplotlib seaborn


2.Tải tập dữ liệu từ Kaggle và đọc nó vào một khung dữ liệu pandas.

Để tải tập dữ liệu từ Kaggle và đọc nó vào một khung dữ liệu pandas, bạn có thể làm theo các bước sau:

Tạo một API token của Kaggle bằng cách truy cập trang web https://www.kaggle.com/<username>/account và nhấn vào nút Create New API Token. Điều này sẽ tải xuống một tệp kaggle.json chứa thông tin xác thực của bạn.
Đặt tệp kaggle.json vào thư mục ~/.kaggle/ trên máy của bạn và đặt quyền truy cập cho nó bằng lệnh chmod 600 ~/.kaggle/kaggle.json.
Cài đặt gói kaggle cho Python bằng lệnh pip install kaggle hoặc pip install --user kaggle.
Nhập gói kaggle và xác thực API của bạn bằng các lệnh sau:
import kaggle kaggle.api.authenticate()

Tải xuống tập dữ liệu từ Kaggle bằng lệnh kaggle.api.dataset_download_files hoặc kaggle.api.dataset_download_file. Ví dụ:
kaggle.api.dataset_download_files(‘ahsan81/hotel-reservations-classification-dataset’, path=‘data/’) kaggle.api.dataset_download_file(‘ahsan81/hotel-reservations-classification-dataset’, file_name=‘hotel_bookings.csv’, path=‘data/’)

Đọc tập dữ liệu vào một khung dữ liệu pandas bằng lệnh pd.read_csv. Ví dụ:
import pandas as pd df = pd.read_csv(‘data/hotel_bookings.csv.zip’)
3.Khám phá và tiền xử lý dữ liệu, bao gồm kiểm tra các giá trị bị thiếu, loại bỏ hoặc thay thế chúng, mã hóa các biến phân loại, chia tập dữ liệu thành tập huấn luyện và kiểm tra.
Để khám phá và tiền xử lý dữ liệu, bạn có thể làm theo các bước sau:

- Kiểm tra các giá trị bị thiếu trong dữ liệu bằng lệnh df.isnull().sum() hoặc df.info(). Điều này sẽ cho bạn biết số lượng giá trị bị thiếu trong mỗi cột của khung dữ liệu.
- Loại bỏ hoặc thay thế các giá trị bị thiếu theo nhu cầu. Bạn có thể sử dụng lệnh df.dropna() để loại bỏ các hàng hoặc cột có chứa giá trị bị thiếu. Bạn cũng có thể sử dụng lệnh df.fillna() để thay thế các giá trị bị thiếu bằng một giá trị cố định hoặc một giá trị thống kê như mean, median, mode, v.v.
- Mã hóa các biến phân loại để chuyển đổi chúng thành dạng số. Bạn có thể sử dụng lệnh pd.get_dummies() để tạo các biến giả cho các biến phân loại có tính chất danh nghĩa. Bạn cũng có thể sử dụng lệnh sklearn.preprocessing.LabelEncoder() để gán các nhãn số cho các biến phân loại có tính chất thứ tự.
- Chia tập dữ liệu thành tập huấn luyện và kiểm tra để đánh giá hiệu suất của mô hình. Bạn có thể sử dụng lệnh sklearn.model_selection.train_test_split() để ngẫu nhiên chia tập dữ liệu thành hai phần với tỷ lệ mong muốn. Ví dụ:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Trong đó X là ma trận đặc trưng và y là vectơ nhãn. Tham số test_size xác định tỷ lệ phần trăm của tập kiểm tra trong tập dữ liệu ban đầu. Tham số random_state xác định hạt giống cho quá trình ngẫu nhiên.
4.Nhập lớp RandomForestClassifier từ sklearn.ensemble và khởi tạo một đối tượng rừng ngẫu nhiên với các tham số mong muốn, chẳng hạn như số lượng cây quyết định (n_estimators), số lượng tính năng được chọn ngẫu nhiên cho mỗi cây (max_features), chiều sâu tối đa của mỗi cây (max_depth), v.v.
Để nhập lớp RandomForestClassifier từ sklearn.ensemble và khởi tạo một đối tượng rừng ngẫu nhiên với các tham số mong muốn, bạn có thể làm theo các bước sau:

Nhập lớp RandomForestClassifier bằng lệnh sau:
from sklearn.ensemble import RandomForestClassifier

Khởi tạo một đối tượng rừng ngẫu nhiên bằng cách gọi hàm khởi tạo của lớp và truyền vào các tham số mong muốn. Ví dụ:
rf = RandomForestClassifier(n_estimators=100, max_features=‘sqrt’, max_depth=10, random_state=42)

Trong đó n_estimators là số lượng cây quyết định trong rừng, max_features là số lượng tính năng được chọn ngẫu nhiên cho mỗi cây, max_depth là chiều sâu tối đa của mỗi cây, và random_state là hạt giống cho quá trình ngẫu nhiên. Bạn có thể xem danh sách đầy đủ các tham số có thể thiết lập ở đây: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
5.Gọi phương thức fit trên đối tượng rừng ngẫu nhiên và truyền vào tập huấn luyện X_train và y_train để huấn luyện mô hình.
Để gọi phương thức fit trên đối tượng rừng ngẫu nhiên và truyền vào tập huấn luyện X_train và y_train để huấn luyện mô hình, bạn có thể sử dụng lệnh sau:

rf.fit(X_train, y_train)

Điều này sẽ huấn luyện mô hình rừng ngẫu nhiên bằng cách sử dụng các đặc trưng trong X_train và các nhãn trong y_train. Bạn có thể xem chi tiết về phương thức fit ở đây: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.fit
6.Gọi phương thức predict trên đối tượng rừng ngẫu nhiên và truyền vào tập kiểm tra X_test để sinh ra các dự đoán cho biến mục tiêu (hủy phòng).
Để gọi phương thức predict trên đối tượng rừng ngẫu nhiên và truyền vào tập kiểm tra X_test để sinh ra các dự đoán cho biến mục tiêu (hủy phòng), bạn có thể sử dụng lệnh sau:

y_pred = rf.predict(X_test)

Điều này sẽ trả về một mảng numpy chứa các dự đoán của mô hình rừng ngẫu nhiên cho các đặc trưng trong X_test. Bạn có thể xem chi tiết về phương thức predict ở đây: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.predict
7.Đánh giá hiệu suất của mô hình bằng cách sử dụng các phép đo như độ chính xác, ma trận nhầm lẫn, báo cáo phân loại, v.v.
Để đánh giá hiệu suất của mô hình bằng cách sử dụng các phép đo như độ chính xác, ma trận nhầm lẫn, báo cáo phân loại, v.v., bạn có thể làm theo các bước sau:

Nhập các phép đo từ gói sklearn.metrics. Ví dụ:
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

Sử dụng các phép đo trên các dự đoán và nhãn thực tế của tập kiểm tra. Ví dụ:
acc = accuracy_score(y_test, y_pred) cm = confusion_matrix(y_test, y_pred) cr = classification_report(y_test, y_pred)

In kết quả của các phép đo để xem hiệu suất của mô hình. Ví dụ:
print(‘Accuracy:’, acc) print(‘Confusion matrix:\n’, cm) print(‘Classification report:\n’, cr)

Bạn có thể xem chi tiết về các phép đo khác ở đây: https://scikit-learn.org/stable/modules/model_evaluation.html
8.Trực quan hóa kết quả bằng cách sử dụng các biểu đồ như biểu đồ cột, biểu đồ hộp, biểu đồ ROC, v.v.
Để trực quan hóa kết quả bằng cách sử dụng các biểu đồ như biểu đồ cột, biểu đồ hộp, biểu đồ ROC, v.v., bạn có thể làm theo các bước sau:

Nhập các thư viện cần thiết để vẽ các biểu đồ, chẳng hạn như matplotlib, seaborn, sklearn.metrics. Ví dụ:
import matplotlib.pyplot as plt import seaborn as sns from sklearn.metrics import roc_curve, roc_auc_score

Chọn loại biểu đồ phù hợp với mục đích của bạn. Ví dụ:

Biểu đồ cột: thích hợp để so sánh các giá trị của các nhóm hoặc danh mục khác nhau1.

Biểu đồ hộp: thích hợp để thể hiện phân phối của một biến liên tục và xác định các ngoại lệ hoặc giá trị bất thường2.

Biểu đồ ROC: thích hợp để đánh giá hiệu suất của một mô hình phân loại nhị phân bằng cách so sánh tỷ lệ dương tính thực (TPR) và tỷ lệ dương tính giả (FPR) ở các ngưỡng khác nhau3.

Sử dụng các hàm có sẵn trong các thư viện để vẽ các biểu đồ và tùy chỉnh các tham số như tiêu đề, nhãn trục, màu sắc, kích thước, v.v. Ví dụ:

Biểu đồ cột: có thể sử dụng hàm plt.bar() hoặc sns.barplot() để vẽ biểu đồ cột1. Ví dụ:

plt.bar(x=[‘A’, ‘B’, ‘C’], height=[10, 20, 15], color=[‘red’, ‘green’, ‘blue’]) plt.title(‘Bar chart example’) plt.xlabel(‘Categories’) plt.ylabel(‘Values’) plt.show()

Biểu đồ hộp: có thể sử dụng hàm plt.boxplot() hoặc sns.boxplot() để vẽ biểu đồ hộp2. Ví dụ:
plt.boxplot(x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) plt.title(‘Box plot example’) plt.xlabel(‘Variable’) plt.ylabel(‘Values’) plt.show()

Biểu đồ ROC: có thể sử dụng hàm roc_curve() và roc_auc_score() để tính toán các giá trị TPR và FPR và diện tích dưới đường cong (AUC), sau đó sử dụng hàm plt.plot() để vẽ biểu đồ ROC3. Ví dụ:
fpr, tpr, thresholds = roc_curve(y_test, y_pred) auc = roc_auc_score(y_test, y_pred) plt.plot(fpr, tpr, label=‘ROC curve (AUC = %0.2f)’ % auc) plt.plot([0, 1], [0, 1], ‘k–’) plt.title(‘ROC curve example’) plt.xlabel(‘False Positive Rate’) plt.ylabel(‘True Positive Rate’) plt.legend(loc=‘lower right’) plt.show()
