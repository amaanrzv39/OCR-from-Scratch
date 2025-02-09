
# OCR from scratch

This project implements an Optical Character Recognition (OCR) system from scratch using the Convolutional Recurrent Neural Network (CRNN) architecture. The CRNN model combines the strengths of Convolutional Neural Networks (CNNs) for feature extraction and Recurrent Neural Networks (RNNs) for sequence modeling, making it particularly effective for recognizing text in images.

The goal of this project is to provide a simple, yet powerful, implementation of OCR that can be trained on custom datasets and used to recognize text in various real-world scenarios.

```
OCR-from-Scratch/
├── Image Samples/         # Few Image samples for testing
├── notebook/              # Jupyter notebook contains implementation details
├── models/                # Contains best model weights
├── app.py                 # Streamlit application
├── model.py               # Define model architecture for prediction
└── requirements.txt       # Required packages
```

## Usage
1. Clone repo
```
git clone https://github.com/amaanrzv39/OCR-from-Scratch.git
cd OCR-from-Scratch
```
2. Setup virtual env
```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3. Install required packages
```
pip install -r requirements.txt
```
4. Run
```
streamlit run app.py --server.port=8000 
```

Contributions are welcome! Feel free to open issues or submit pull requests.

📜 License

This project is licensed under the MIT License.

