# Machine Learning Lip Reading - Tensor Flow

This repository contains an application for lip reading based on LipNet, an advanced machine learning model originally designed for lip movement recognition. The project was inspired by Nicholas Renotte, an AI expert at IBM, and adapted to create a user-friendly full-stack app using Streamlit.

<img width="640" alt="ML_Streamlit_LiptNet" src="https://github.com/TobiasMaissen/LipNet_Tensorflow/assets/38552872/e1c5c08a-5ebc-4d36-835f-c16ae3ac1c3c">



## What is LipNet?

LipNet is an advanced machine learning model developed to analyze lip movements and transcribe words. It utilizes neural networks and deep learning to accomplish this task. LipNet has the potential to be valuable in various applications, including improving accessibility for the hearing-impaired, enhancing speech recognition in noisy environments, and applications in forensics.

## Project Description

This project goes beyond implementing the LipNet model and presents a comprehensive solution that brings all the necessary components together. The key components of this project include:

### Functions and Code in `utils.py`

`utils.py` contains essential functions for data processing and manipulation. Here's an overview of the functions:

- `load_video(path:str)`: This function loads video frames, processes them, and normalizes the data for further analysis.

- `load_alignments(path:str)`: It reads alignment data from a file and converts it into a format suitable for the lip reading model.

- `load_data(path: str)`: This function combines the data loading process, which includes video frames and alignments, and prepares it for model input.

### Functions and Code in `modelutil.py`

`modelutil.py` contains the code for defining the LipNet-based model. Here's what it does:

- `load_model()`: This function constructs the LipNet model by defining a sequence of layers, including 3D convolution, LSTM, dense layers, and more. It also loads the model weights from a checkpoint.


Here's a user guide section for your README file based on the provided instructions:

---

# User Guide

Before getting started with the LipNet-based lip reading app, please follow these steps:

1. **Environment Setup:**
   - Make sure you have Python 3.9 installed.
   - Check the specific versions of the dependencies in the `LipNet_github.ipynb` notebook. It's crucial to use these versions to avoid any errors.

2. **Data and Checkpoints:**
   - In the `LipNet_github.ipynb` notebook, you'll find links to download the necessary data and model checkpoints.
   - Create a folder named 'models' to store the downloaded checkpoint files. This will save you from the need to train the model from scratch.

3. **App Setup:**
   - Create a folder named 'app' in your project directory.
   - Inside the 'app' folder, place the following Python files: `utils.py`, `streamlitapp.py`, and `modelutil.py`.

4. **Running the App:**
   - The rest of the setup details and usage instructions are embedded in the code and can be found in the provided Python files.
   - Execute the `streamlitapp.py` script to launch the user-friendly LipNet-based lip reading application. For run the app use the command `streamlit run streamlitapp.py`

Now, you're all set to use the lip reading app for transcribing spoken words from lip movements. Enjoy the app and its advanced machine learning capabilities!

## Notes

This project is an excellent demonstration of how advanced machine learning models like LipNet can be integrated into real-world applications. It showcases the synergy between artificial intelligence and full-stack development to create a powerful and user-friendly product.
