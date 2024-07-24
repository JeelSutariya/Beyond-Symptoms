# Beyond Symptoms: Multi-Disease Prediction System

The proposed progressive Disease Prediction System symbolizes an advanced fusion of technological innovation and medical expertise aimed at transforming healthcare by enabling the early identification of some of the more chronic diseases. Utilizing specialized machine learning models, this research aims to predict disease onset based on user-provided health data. The system features a user-friendly web interface, ensuring seamless interaction for data input and result visualization. Each disease prediction model, having explored various machine learning models for optimal accuracy, employs specific algorithms to ensure precision and reliability. Emphasis is placed on data privacy, ethical considerations, and compliance with healthcare regulations. The research not only contributes to efficient resource allocation and enhanced population health but also empowers individuals to proactively manage their well-being. Challenges encountered during development, such as data variability, model complexity, and regulatory compliance, were addressed with innovative solutions. The research’s future scope envisions continuous growth, contributing to advancements in healthcare prognosis and fostering a proactive approach to health management.

## Features

- Predict the likelihood of Diabetes, Heart Disease, and Parkinson's Disease
- User-friendly interface for inputting health metrics
- Visualizations of model performance (ROC curve and confusion matrix)
- Responsive design for various screen sizes

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/JeelSutariya/Beyond-Symptoms.git
   cd beyond-symptoms
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Download the datasets and place them in the `C:\Users\Jeel.Sutariya JEEL-SUTARIYA-N (*YOUR-USERNAME*)\Documents\Beyond_Symptoms_Datasets` directory.

## Usage

1. Train the models:
   ```
   python models/train_diabetes.py
   python models/train_heart.py
   python models/train_parkinsons.py
   ```

2. Run the Streamlit app:
   ```
   streamlit run src/app.py
   ```

3. Open your web browser and go to `http://localhost:8501` to use the application.

## Project Structure

```
beyond_symptoms/
│
├── data/
│   ├── diabetes.csv
│   ├── heart.csv
│   └── parkinsons.csv
│
├── models/
│   ├── train_diabetes.py
│   ├── train_heart.py
│   ├── train_parkinsons.py
│   └── model_utils.py
│
├── saved_models/
│   ├── diabetes_model.pkl
│   ├── heart_disease_model.pkl
│   └── parkinsons_model.pkl
│
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── pages/
│   │   ├── __init__.py
│   │   ├── home.py
│   │   ├── diabetes.py
│   │   ├── heart_disease.py
│   │   └── parkinsons.py
│   └── utils/
│       ├── __init__.py
│       └── preprocessing.py
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── images/
│       ├── logo.png
│       ├── diabetes.png
│       ├── heart.png
│       └── parkinsons.png
│
├── tests/
│   ├── test_models.py
│   └── test_preprocessing.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.