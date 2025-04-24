"""
Flask API cho ứng dụng Hệ thống hỗ trợ quyết định lựa chọn ô tô sử dụng phương pháp AHP
"""
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import json
import logging
import sys
import os
# Thêm thư mục hiện tại vào sys.path để import được module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import ahp_calculator từ cùng thư mục
from ahp_calculator import ahp_calculator

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ahp_flask_api')

app = Flask(__name__)
CORS(app)  # Cho phép CORS để frontend có thể gọi API

@app.route('/')
def home():
    """
    Trang chủ cho ứng dụng Flask
    """
    return "AHP Calculator Flask API is running!"

@app.route('/api/health', methods=['GET'])
def health_check():
    """API kiểm tra trạng thái hoạt động của server Flask"""
    return jsonify({
        'status': 'healthy',
        'service': 'AHP Calculator API',
        'version': '1.0.0'
    })

@app.route('/api/calculate', methods=['POST'])
def calculate_ahp():
    """
    API tính toán AHP từ dữ liệu đầu vào
    
    Cần cung cấp:
    - criteriaMatrix: Ma trận so sánh tiêu chí
    - alternativeMatrices: Từ điển chứa ma trận so sánh các phương án theo từng tiêu chí
    
    Trả về:
    - Kết quả AHP với trọng số tiêu chí, điểm số phương án, và các thông số khác
    """
    try:
        data = request.json
        logger.info("Received AHP calculation request")
        
        if not data or 'criteriaMatrix' not in data or 'alternativeMatrices' not in data:
            logger.error("Invalid request data structure")
            return jsonify({
                'error': 'Invalid request data. Required fields: criteriaMatrix, alternativeMatrices'
            }), 400
        
        criteria_matrix = data['criteriaMatrix']
        alternative_matrices = data['alternativeMatrices']
        
        # Xác thực cấu trúc dữ liệu
        if not isinstance(criteria_matrix, list) or not all(isinstance(row, list) for row in criteria_matrix):
            logger.error("Invalid criteria matrix format")
            return jsonify({'error': 'Invalid criteria matrix format'}), 400
        
        if not isinstance(alternative_matrices, dict):
            logger.error("Alternative matrices must be a dictionary")
            return jsonify({'error': 'Alternative matrices must be a dictionary'}), 400
        
        # Thực hiện tính toán AHP
        result = ahp_calculator.calculate_ahp(criteria_matrix, alternative_matrices)
        
        logger.info("AHP calculation completed successfully")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in AHP calculation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    """
    API tạo báo cáo PDF dựa trên kết quả AHP
    
    Cần cung cấp:
    - ahpResults: Kết quả tính toán AHP
    - cars: Danh sách thông tin các xe được so sánh
    
    Trả về:
    - File PDF báo cáo
    """
    try:
        data = request.json
        logger.info("Received report generation request")
        
        if not data or 'ahpResults' not in data or 'cars' not in data:
            logger.error("Invalid request data for report generation")
            return jsonify({
                'error': 'Invalid request data. Required fields: ahpResults, cars'
            }), 400
        
        ahp_results = data['ahpResults']
        cars = data['cars']
        
        # Tạo báo cáo PDF
        pdf_buffer = ahp_calculator.generate_report(ahp_results, cars)
        
        logger.info("Report generated successfully")
        
        # Trả về file PDF
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='ahp_analysis_report.pdf'
        )
    
    except Exception as e:
        logger.error(f"Error in report generation: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting AHP Calculator Flask API")
    app.run(host='0.0.0.0', port=5001, debug=True)
