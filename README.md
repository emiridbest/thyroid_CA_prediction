# Tradi

A machine learning-based web app for predicting the recurrence of thyroid cancers.


## Features



## Installation

1. Clone the repository:
```bash
git clone https://github.com/emiridbest/thyroid_CA_prediction.git
cd tradi
```

2. Create a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:
```properties
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run code/main.py
```

2. Enter patient details
3. Select your preferred timeframe
4. Click "Predict" to generate insights

## Project Structure

```
thyroid_pred/
│
├── code/
│   ├── main.py                    # Main application
│   ├── api_test.py                # api test for open_ai
│   └── sk_model                   # Random Forest classification algorithA
│
├── dataset/                      # Training and testing data
├── venv/                         # Virtual environment
├── .env***                       # Environment variables
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```



## Dependencies

- Python 3.12+
- pandas
- streamlit
- matplotlib
- openai
- python-dotenv
- seaborn
- scikit-learn
- imbalanced learn

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

MIT License - See LICENSE file for details

## Disclaimer

- This tool is for educational purposes only.
- Primarily designed for use by clinicians
- Results might reassuring but should be subject to clinical revalidation
