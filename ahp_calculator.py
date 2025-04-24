"""
AHP Calculator Module - Sử dụng thuật toán phân tích thứ bậc (AHP) để tính toán lựa chọn ô tô tối ưu
"""
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Sử dụng Agg backend để tránh lỗi khi không có GUI

# Đăng ký các font chữ tiếng Việt
# try:
#     pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))
# except:
#     pass

class AHPCalculator:
    """
    Class triển khai phương pháp phân tích thứ bậc (AHP) 
    để tính ra các trọng số của các tiêu chí và điểm số của các phương án.
    """
    
    def __init__(self):
        # Giá trị chỉ số ngẫu nhiên RI
        self.RANDOM_CONSISTENCY_INDEX = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]
    
    def calculate_weights(self, matrix):
        """
        Tính trọng số cho ma trận so sánh cặp.
        
        Args:
            matrix: Ma trận vuông chứa các giá trị so sánh cặp
            
        Returns:
            Mảng một chiều chứa trọng số cho mỗi tiêu chí/phương án
        """
        # Chuyển đổi thành mảng numpy nếu cần
        matrix = np.array(matrix, dtype=float)
        n = len(matrix)
        
        # Tính tổng mỗi cột
        col_sums = matrix.sum(axis=0)
        
        # Chuẩn hóa ma trận bằng cách chia mỗi phần tử cho tổng cột
        normalized_matrix = matrix / col_sums
        
        # Tính trọng số bằng cách lấy trung bình mỗi hàng
        weights = normalized_matrix.mean(axis=1)
        
        return weights
    
    def calculate_consistency(self, matrix, weights):
        """
        Tính chỉ số nhất quán CR để kiểm tra tính nhất quán của ma trận so sánh cặp.
        
        Args:
            matrix: Ma trận vuông chứa các giá trị so sánh cặp
            weights: Mảng một chiều chứa trọng số đã tính toán
            
        Returns:
            Dictionary chứa lambdaMax, CI, CR, và isConsistent
        """
        matrix = np.array(matrix, dtype=float)
        n = len(matrix)
        
        # Tính Ax (nhân ma trận với vector trọng số)
        weighted_sum = np.dot(matrix, weights)
        
        # Tính λmax
        consistency_vector = weighted_sum / weights
        lambda_max = np.mean(consistency_vector)
        
        # Tính CI (Consistency Index)
        ci = (lambda_max - n) / (n - 1) if n > 1 else 0
        
        # Tính CR (Consistency Ratio)
        ri = self.RANDOM_CONSISTENCY_INDEX[n - 1] if n <= len(self.RANDOM_CONSISTENCY_INDEX) else 1.49
        cr = ci / ri if ri > 0 else 0
        
        # Kiểm tra tính nhất quán (CR < 0.1 được coi là nhất quán)
        is_consistent = cr < 0.1
        
        return {
            "lambdaMax": float(lambda_max),
            "ci": float(ci),
            "cr": float(cr),
            "isConsistent": bool(is_consistent)
        }
    
    def calculate_final_scores(self, criteria_weights, alternative_scores):
        """
        Tính điểm số tổng hợp cho các phương án bằng cách nhân trọng số tiêu chí
        với điểm số của từng phương án và cộng lại.
        
        Args:
            criteria_weights: Mảng chứa trọng số của các tiêu chí
            alternative_scores: Dictionary chứa điểm số của các phương án theo từng tiêu chí
            
        Returns:
            Mảng chứa điểm số tổng hợp của các phương án
        """
        n_alternatives = len(next(iter(alternative_scores.values())))
        final_scores = np.zeros(n_alternatives)
        
        for i, (criterion, weight) in enumerate(zip(alternative_scores.keys(), criteria_weights)):
            scores = alternative_scores[criterion]
            final_scores += weight * np.array(scores)
            
        return final_scores.tolist()
    
    def calculate_ahp(self, criteria_matrix, alternative_matrices):
        """
        Thực hiện phân tích AHP đầy đủ.
        
        Args:
            criteria_matrix: Ma trận so sánh cặp các tiêu chí
            alternative_matrices: Dictionary chứa ma trận so sánh cặp cho mỗi tiêu chí
            
        Returns:
            Kết quả AHP đầy đủ
        """
        # Tính trọng số tiêu chí
        criteria_weights = self.calculate_weights(criteria_matrix)
        
        # Kiểm tra tính nhất quán của ma trận tiêu chí
        consistency = self.calculate_consistency(criteria_matrix, criteria_weights)
        
        # Tính điểm số cho từng phương án theo từng tiêu chí
        alternative_weights = {}
        for criterion, matrix in alternative_matrices.items():
            alternative_weights[criterion] = self.calculate_weights(matrix)
        
        # Tính điểm số tổng hợp
        final_scores = self.calculate_final_scores(criteria_weights, alternative_weights)
        
        # Chuẩn bị kết quả trả về
        criteria_names = list(alternative_matrices.keys())
        
        return {
            "criteria": {
                "names": criteria_names,
                "weights": criteria_weights.tolist(),
                "consistencyRatio": float(consistency["cr"]),
                "lambdaMax": float(consistency["lambdaMax"]),
                "consistencyIndex": float(consistency["ci"])
            },
            "alternatives": {
                "names": list(range(len(final_scores))),  # Thay bằng tên thực tế nếu có
                "scores": final_scores,
                "weightedScores": {
                    criterion: weights.tolist() 
                    for criterion, weights in alternative_weights.items()
                }
            }
        }
    
    def generate_bar_chart(self, car_names, scores, title="Đánh giá tổng hợp các xe"):
        """Tạo biểu đồ cột hiển thị điểm số các xe"""
        plt.figure(figsize=(10, 6))
        bars = plt.bar(car_names, scores, color='skyblue')
        
        # Thêm giá trị lên đỉnh mỗi cột
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.3f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Xe', fontsize=12)
        plt.ylabel('Điểm số', fontsize=12)
        plt.ylim(0, max(scores) * 1.2)  # Để có khoảng trống cho labels
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Lưu vào buffer thay vì file
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300)
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
    
    def generate_radar_chart(self, car_names, criteria_names, criteria_scores):
        """Tạo biểu đồ radar so sánh các xe theo nhiều tiêu chí"""
        # Số tiêu chí
        N = len(criteria_names)
        
        # Góc cho mỗi trục
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Khép kín biểu đồ
        
        # Tạo hình
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Màu sắc
        colors = plt.cm.get_cmap('tab10', len(car_names))
        
        # Vẽ biểu đồ cho mỗi xe
        for i, car_name in enumerate(car_names):
            values = [criteria_scores[criterion][i] for criterion in criteria_names]
            values += values[:1]  # Khép kín biểu đồ
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=car_name, color=colors(i))
            ax.fill(angles, values, color=colors(i), alpha=0.1)
        
        # Thiết lập các thông số
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(criteria_names, fontsize=12)
        
        # Thêm legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title('So sánh các xe theo từng tiêu chí', fontsize=15, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Lưu vào buffer thay vì file
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300)
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
    
    def generate_report(self, ahp_results, cars):
        """
        Tạo báo cáo PDF chi tiết về kết quả phân tích AHP.
        
        Args:
            ahp_results: Kết quả phân tích AHP
            cars: Danh sách thông tin xe
            
        Returns:
            BytesIO: Buffer chứa file PDF
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
        
        styles = getSampleStyleSheet()
        # Thêm style riêng cho tiếng Việt nếu cần
        styles.add(ParagraphStyle(name='TitleVI', 
                                 parent=styles['Title'],
                                 fontName='Helvetica-Bold'))
        styles.add(ParagraphStyle(name='NormalVI', 
                                 parent=styles['Normal'],
                                 fontName='Helvetica'))
        
        # Chuẩn bị các elements
        elements = []
        
        # Tiêu đề báo cáo
        elements.append(Paragraph('BÁO CÁO PHÂN TÍCH AHP', styles['TitleVI']))
        elements.append(Spacer(1, 0.25*inch))
        elements.append(Paragraph('SO SÁNH VÀ LỰA CHỌN Ô TÔ TỐI ƯU', styles['TitleVI']))
        elements.append(Spacer(1, 0.5*inch))
        
        # Danh sách các xe được so sánh
        elements.append(Paragraph('Các xe được so sánh:', styles['Heading2']))
        elements.append(Spacer(1, 0.1*inch))
        
        car_names = [f"{car['brand']} {car['name']} {car['model']}" for car in cars]
        car_data = [['STT', 'Tên xe', 'Giá (VNĐ)', 'Công suất (HP)', 'Tiêu thụ (L/100km)']]
        
        for i, car in enumerate(cars):
            car_data.append([
                str(i+1),
                f"{car['brand']} {car['name']} {car['model']}",
                f"{int(car['price']):,}",
                str(car['horsepower']),
                car['fuelConsumption']
            ])
        
        # Tạo bảng thông tin xe
        car_table = Table(car_data, colWidths=[0.5*inch, 2.5*inch, 1.25*inch, 1.25*inch, 1.25*inch])
        car_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        elements.append(car_table)
        elements.append(Spacer(1, 0.5*inch))
        
        # Trọng số các tiêu chí
        elements.append(Paragraph('Trọng số các tiêu chí:', styles['Heading2']))
        elements.append(Spacer(1, 0.1*inch))
        
        criteria_data = [['Tiêu chí', 'Trọng số', 'Tỷ lệ (%)']]
        for name, weight in zip(ahp_results['criteria']['names'], ahp_results['criteria']['weights']):
            criteria_data.append([name, f"{weight:.4f}", f"{weight*100:.2f}%"])
        
        # Tạo bảng trọng số
        criteria_table = Table(criteria_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        criteria_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        elements.append(criteria_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # Thông tin về tính nhất quán
        cr_color = "green" if ahp_results['criteria']['consistencyRatio'] < 0.1 else "red"
        cr_text = f"Tỷ số nhất quán (CR): {ahp_results['criteria']['consistencyRatio']:.4f} "
        cr_text += f"({'Chấp nhận được' if ahp_results['criteria']['consistencyRatio'] < 0.1 else 'Cần xem xét lại'})"
        
        elements.append(Paragraph(cr_text, styles['Normal']))
        elements.append(Paragraph(f"Chỉ số nhất quán (CI): {ahp_results['criteria']['consistencyIndex']:.4f}", styles['Normal']))
        elements.append(Paragraph(f"Lambda Max: {ahp_results['criteria']['lambdaMax']:.4f}", styles['Normal']))
        elements.append(Spacer(1, 0.5*inch))
        
        # Kết quả phân tích AHP
        elements.append(Paragraph('Kết quả phân tích AHP:', styles['Heading2']))
        elements.append(Spacer(1, 0.1*inch))
        
        result_data = [['Hạng', 'Tên xe', 'Điểm số', 'Đánh giá']]
        
        # Lấy điểm và sắp xếp theo thứ tự giảm dần
        scores = ahp_results['alternatives']['scores']
        rankings = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        for rank, idx in enumerate(rankings):
            score = scores[idx]
            evaluation = ""
            if rank == 0:
                evaluation = "Lựa chọn tốt nhất"
            elif score > 0.8 * scores[rankings[0]]:
                evaluation = "Rất tốt"
            elif score > 0.6 * scores[rankings[0]]:
                evaluation = "Tốt"
            else:
                evaluation = "Cân nhắc"
                
            result_data.append([
                str(rank + 1),
                car_names[idx],
                f"{score:.4f}",
                evaluation
            ])
        
        # Tạo bảng kết quả
        result_table = Table(result_data, colWidths=[0.5*inch, 2.5*inch, 1*inch, 2*inch])
        result_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('BACKGROUND', (0, 1), (0, 1), colors.lightyellow),
            ('BACKGROUND', (3, 1), (3, 1), colors.lightyellow),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (2, 0), (2, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        elements.append(result_table)
        elements.append(Spacer(1, 0.5*inch))
        
        # Tạo biểu đồ cột
        bar_chart_buffer = self.generate_bar_chart(car_names, scores, "Điểm số tổng hợp của các xe")
        bar_chart_img = Image(bar_chart_buffer, width=450, height=300)
        elements.append(bar_chart_img)
        elements.append(Spacer(1, 0.25*inch))
        
        # Tạo biểu đồ radar cho các tiêu chí
        criteria_scores = ahp_results['alternatives']['weightedScores']
        radar_chart_buffer = self.generate_radar_chart(car_names, 
                                                     ahp_results['criteria']['names'], 
                                                     criteria_scores)
        radar_chart_img = Image(radar_chart_buffer, width=400, height=400)
        elements.append(radar_chart_img)
        
        # Kết luận
        elements.append(Spacer(1, 0.5*inch))
        elements.append(Paragraph('Kết luận và đánh giá:', styles['Heading2']))
        elements.append(Spacer(1, 0.1*inch))
        
        best_car_index = rankings[0]
        best_car = cars[best_car_index]
        best_car_name = f"{best_car['brand']} {best_car['name']} {best_car['model']}"
        
        conclusion = f"""
        Dựa trên phân tích AHP với các tiêu chí đã cho, xe <b>{best_car_name}</b> là lựa chọn tối ưu với điểm 
        số cao nhất {scores[best_car_index]:.4f}. Xe này đặc biệt nổi bật ở các tiêu chí:
        """
        
        elements.append(Paragraph(conclusion, styles['NormalVI']))
        
        # Tìm tiêu chí mạnh nhất của xe tốt nhất
        best_criteria = []
        for criterion in ahp_results['criteria']['names']:
            criterion_scores = criteria_scores[criterion]
            if criterion_scores[best_car_index] == max(criterion_scores):
                best_criteria.append(criterion)
        
        for criterion in best_criteria:
            elements.append(Paragraph(f"• {criterion}", styles['Normal']))
        
        # Xây dựng PDF
        doc.build(elements)
        buffer.seek(0)
        return buffer

# Tạo instance để sử dụng
ahp_calculator = AHPCalculator()