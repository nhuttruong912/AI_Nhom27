

# Khai báo thư viện
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

# Tải xuống tập dữ liệu từ Kaggle và đọc nó vào pandas dataframe
# Đảm bảo rằng file kaggle.json đã nằm trong thư mục ~/.kaggle/
import kaggle
kaggle.api.authenticate()
#kaggle.api.dataset_download_files('ahsan81/hotel-reservations-classification-dataset', path='C:/Users/nhutt/.kaggle')
df = pd.read_csv('C:/Users/nhutt/.kaggle/hotel-reservations-classification-dataset.zip')
"""
Khám phá và tiền xử lý dữ liệu, bao gồm kiểm tra các giá trị bị thiếu, loại bỏ hoặc thay thế chúng,
mã hóa các biến phân loại, chia dữ liệu thành các tập huấn luyện và kiểm tra
"""
# Kiểm tra các giá trị còn thiếu
#print(df.isnull().sum())

#df = df.dropna(subset=['booking_status'])
#df = df.fillna(0)


# Mã hóa các biến phân loại (categorical variables) bằng bộ mã hóa nhãn (label encoder)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Booking_ID'] = le.fit_transform(df['Booking_ID'])
df['no_of_adults'] = le.fit_transform(df['no_of_adults'])
df['no_of_children'] = le.fit_transform(df['no_of_children'])
df['no_of_weekend_nights'] = le.fit_transform(df['no_of_weekend_nights'])
df['no_of_week_nights'] = le.fit_transform(df['no_of_week_nights'])
df['type_of_meal_plan'] = le.fit_transform(df['type_of_meal_plan'])
df['required_car_parking_space'] = le.fit_transform(df['required_car_parking_space'])
df['room_type_reserved'] = le.fit_transform(df['room_type_reserved'])
df['arrival_year'] = le.fit_transform(df['arrival_year'])
df['arrival_month'] = le.fit_transform(df['arrival_month'])
df['arrival_date'] = le.fit_transform(df['arrival_date'])
df['market_segment_type'] = le.fit_transform(df['market_segment_type'])
df['repeated_guest'] = le.fit_transform(df['repeated_guest'])
df['no_of_previous_cancellations'] = le.fit_transform(df['no_of_previous_cancellations'])
df['no_of_previous_bookings_not_canceled'] = le.fit_transform(df['no_of_previous_bookings_not_canceled'])
df['avg_price_per_room'] = le.fit_transform(df['avg_price_per_room'])
df['no_of_special_requests'] = le.fit_transform(df['no_of_special_requests'])
df['booking_status'] = le.fit_transform(df['booking_status'])

# Tách dữ liệu thành tập huấn luyện và tập kiểm tra
from sklearn.model_selection import train_test_split
X = df.drop('booking_status', axis=1)  #1 tương đương trục cột
y = df['booking_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""
Trong đó X là ma trận đặc trưng và y là vectơ nhãn.
Tham số test_size xác định tỷ lệ phần trăm của tập kiểm tra trong tập dữ liệu ban đầu.
Tham số random_state xác định hạt giống cho quá trình ngẫu nhiên.
"""

"""
Khởi tạo một đối tượng rừng ngẫu nhiên với các tham số mong muốn, 
chẳng hạn như số cây quyết định (n_estimators), 
số tính năng được chọn ngẫu nhiên cho mỗi cây (max_features), 
độ sâu tối đa của mỗi cây (max_depth), vân vân
"""
rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', max_depth=10, random_state=42)

# Gọi phương thức fit trên đối tượng rừng ngẫu nhiên và chuyển vào tập huấn luyện X_train và y_train để huấn luyện mô hình.
rf.fit(X_train, y_train)
# fit sẽ huấn luyện mô hình rừng ngẫu nhiên bằng cách sử dụng các đặc trưng trong X_train và các nhãn trong y_train

# Gọi phương thức dự đoán (predict) trên đối tượng rừng ngẫu nhiên và chuyển vào tập kiểm tra X_test để tạo dự đoán cho biến mục tiêu
y_pred = rf.predict(X_test)
#predict sẽ trả về một mảng numpy chứa các dự đoán của mô hình rừng ngẫu nhiên cho các đặc trưng trong X_test
"""
Đánh giá hiệu suất của mô hình bằng cách sử dụng các số liệu như
độ chính xác (accuracy score), ma trận nhầm lẫn (confusion matrix), báo cáo phân loại (classification report) , v.v.
"""
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
print('Accuracy:', acc)
print('Confusion matrix:\n', cm)
print('Classification report:\n', cr)


# Trực quan hóa kết quả bằng các biểu đồ như biểu đồ cột (bar chart), biểu đồ hộp (box plot), đường cong ROC, v.v.

# Vẽ biểu đồ cột để so sánh số lần hủy thực tế (Actual) và dự đoán (Predicted)

plt.bar(x=['Actual', 'Predicted'], height=[y_test.sum(), y_pred.sum()], color=['blue', 'orange'])
plt.title('Bar chart of actual vs predicted cancellation counts')
plt.xlabel('Cancellation')
plt.ylabel('Count')
plt.show()

# Vẽ biểu đồ hộp để hiển thị phân phối lead time cho từng nhóm hủy bỏ

sns.boxplot(x='booking_status', y='lead_time', data=df)
plt.title('Box plot of lead time by cancellation group')
plt.xlabel('Cancellation')
plt.ylabel('Lead time')
plt.show()

