import os
from src.web.app import app

if __name__ == '__main__':
    # Check if model exists
    if not os.path.exists('model.pth'):
        print("Warning: model.pth not found. Please run train.py first.")
        print("Running with untrained model...")
    
    print("Starting web server...")
    print("Visit http://localhost:8080 to play!")
    app.run(host='0.0.0.0', port=8080) 