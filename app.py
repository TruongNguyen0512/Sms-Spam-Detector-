# import thư viện cần thiết
import tkinter as tk
import nltk 
from tkinter import messagebox
from nltk.corpus import stopwords  
import pickle 
import string 
import sklearn
nltk.download('stopwords')


# Hàm xử lý text
def process_text(text):
    # 1
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    # 2
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    # 3
    return clean_words
# Hàm xử lý sự kiện khi nhấn nút "Check Spam"
def check_spam():
    # Nhận giá trị từ ô nhập text
    text = text_entry.get()
    
    # TODO: Thực hiện kiểm tra xem text có phải là spam hay không dựa vào model và vectorizer đã train
    # Hàm xử lý text 
    process_text(text)

    # Tiền xử lý tin nhắn 
    processed_text = process_text(text)
    # load the vectorizer
    loaded_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    # load the model
    loaded_model = pickle.load(open('classification.pkl', 'rb'))
    # Vectorize the input text
    text_vectorized = loaded_vectorizer.transform(processed_text)

    # Make prediction using the trained model
    prediction = loaded_model.predict(text_vectorized)  
    if prediction[0] == 1:
        # Hiển thị thông báo kết quả
        messagebox.showinfo("Spam Detection", "The message is a spam sms.")
    else:
        # Hiển thị thông báo kết quả
        messagebox.showinfo("Spam Detection", "The message is not a spam sms.")
    
    

# Tạo giao diện
root = tk.Tk()
root.configure(bg='gray')
root.geometry("700x200")
# Tiêu đề
title_label = tk.Label(root, text="Spam Detection", font=("Arial", 16), fg="red")  # Đặt chữ màu đỏ cho tiêu đề
title_label.pack(pady=20)

# Ô nhập text
text_entry = tk.Entry(root, width=70)  # Đặt chiều rộng của ô nhập text lớn hơn
text_entry.pack(pady=10)

# Nút "Check Spam"
check_button = tk.Button(root, text="Check Spam", command=check_spam,fg='red')
check_button.pack(pady=10)

# Chạy giao diện
root.mainloop()