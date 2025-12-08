import torch
import time
import math
# Import class từ file tham chiếu mà bạn đã đặt trong thư mục engine/
from engine.soda_qd1102_ref_qkm_gpu import SODA_QD1102 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_qd1102_example(N_size=2048, steps=100):
    """
    Thực thi SODA-QD1102 Engine cơ bản và đo lường sự đồng bộ lượng tử (R_q).
    """
    if not torch.cuda.is_available():
        print("Cảnh báo: Không tìm thấy CUDA. Chạy trên CPU sẽ rất chậm.")
    
    # 1. Khởi tạo Engine SODA-QD1102
    print(f"Khởi tạo SODA-QD1102: N={N_size} trên {device}...")
    
    # Khởi tạo instance của Class SODA_QD1102
    engine = SODA_QD1102(N=N_size, quantum_k=1.0)
    
    R_list = []
    start_time = time.time()
    
    # 2. Vòng lặp Tiến hóa
    # (Lưu ý: Phiên bản này chỉ chạy QKM, không có Local Learning đầy đủ)
    for step in range(steps):
        # Bước này thực thi QKM Step (tính toán O(N^2))
        Rq = engine.qkm_step()
        R_list.append(Rq)
        
        if step % (steps // 10) == 0 and step > 0:
            print(f"Step {step}/{steps} | Current R_q: {Rq:.4f}")

    elapsed_time = time.time() - start_time
    
    # 3. Kết quả
    print("\n-------------------------------------------")
    print(f"Hoàn thành {steps} bước trong {elapsed_time:.2f} giây.")
    print(f"Quantum Order ban đầu: {R_list[0]:.4f}")
    print(f"Quantum Order cuối cùng: {R_list[-1]:.4f}")
    print("-------------------------------------------")
    
    return R_list

if __name__ == "__main__":
    # Tham số chạy mẫu
    Rq_history = run_qd1102_example(N_size=2048, steps=100)
