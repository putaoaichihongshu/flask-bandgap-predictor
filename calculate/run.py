from app import app
import os
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render 会通过环境变量指定端口
    app.run(host='0.0.0.0', port=port)