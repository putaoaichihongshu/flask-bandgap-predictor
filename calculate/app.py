from flask import Flask, request, jsonify, render_template
import torch
from model import FullNet
import numpy as np

app = Flask(__name__)

# 加载模型和权重
model = FullNet()
checkpoint = torch.load('results/best_overall_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def predict_gap(data):
    try:
        # 提取输入数据
        site_a = torch.tensor([data['MA'], data['FA'], data['Cs']], dtype=torch.float32).view(1, -1)
        site_x = torch.tensor([data['Br'], data['Cl'], data['I']], dtype=torch.float32).view(1, -1)

        # 模型预测
        with torch.no_grad():
            prediction = model(site_a, site_x).item()
        return {'gap': prediction}
    except Exception as e:
        return {'error': str(e)}

# API 路由
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("收到的数据:", data)  # 调试信息
        
        # 获取输入值 - 修改 CS 为 Cs
        ma = float(data['MA'])
        fa = float(data['FA'])
        cs = float(data['CS'])  # 匹配前端的 'CS'
        br = float(data['Br'])
        cl = float(data['Cl'])
        i = float(data['I'])
        
        # 验证输入数据
        if not (0 <= ma <= 1 and 0 <= fa <= 1 and 0 <= cs <= 1):
            return jsonify({'error': 'A位离子比例必须在0到1之间'}), 400
        
        if not (0 <= br <= 1 and 0 <= cl <= 1 and 0 <= i <= 1):
            return jsonify({'error': 'X位离子比例必须在0到1之间'}), 400
            
        if not (abs(ma + fa + cs - 1.0) < 0.01):
            return jsonify({'error': 'A位离子比例之和必须为1'}), 400
            
        if not (abs(br + cl + i - 1.0) < 0.01):
            return jsonify({'error': 'X位离子比例之和必须为1'}), 400
        
        # 准备输入张量
        site_a = torch.tensor([[ma, fa, cs]], dtype=torch.float32)
        site_x = torch.tensor([[br, cl, i]], dtype=torch.float32)
        
        # 进行预测
        with torch.no_grad():
            prediction = model(site_a, site_x)
            print("预测结果:", float(prediction[0][0]))  # 调试信息
            
        return jsonify({'bandgap': float(prediction[0][0])})
    
    except Exception as e:
        print("预测错误:", str(e))  # 调试信息
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)