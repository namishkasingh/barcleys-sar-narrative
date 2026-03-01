# Barclays SAR Narrative Project 
 
## ?? Overview 
 
Machine Learning project for Suspicious Activity Report (SAR) analysis and narrative generation. This system helps financial institutions detect and analyze potential fraudulent activities using advanced ML techniques. 
 
## ? Features 
 
- **Frontend Dashboard**: Interactive visualization of SAR data and model outputs 
- **Backend API**: RESTful APIs for model inference and data processing 
- **Multiple ML Models**: XGBoost, Random Forest, and Neural Networks for fraud detection 
- **CTGAN Integration**: Synthetic data generation for imbalanced dataset handling 
- **Audit Trail**: Comprehensive logging of all system activities 
- **Vector Database**: Similarity search for historical SAR cases 
- **Automated Graphing**: Generation of performance metrics and visualizations 
 
## ?? Generated Graphs 
 
The system automatically generates several analysis graphs: 
 
- `graph1_ctgan_balancing.png` - CTGAN synthetic data balancing effects 
- `graph2_model_comparison.png` - Performance comparison across models 
- `graph3_risk_distribution.png` - Risk score distribution analysis 
- `graph4_feature_importance.png` - Key features for fraud detection 
- `graph5_roc_curve.png` - ROC curves for different models 
- `graph6_audit_growth.png` - Audit trail growth over time 
- `graph7_threshold_analysis.png` - Threshold optimization analysis 
 
## ??? Technology Stack 
 
- **Frontend**: Python, Dash/Streamlit, Plotly 
- **Backend**: FastAPI/Flask, REST APIs 
- **Machine Learning**: XGBoost, Scikit-learn, TensorFlow 
- **Data Processing**: Pandas, NumPy, PyArrow 
- **Synthetic Data**: CTGAN (Conditional Tabular GAN) 
- **Visualization**: Matplotlib, Seaborn, Plotly 
- **Database**: Vector DB (FAISS/ChromaDB) for embeddings 
- **Version Control**: Git, GitHub 
 
## ?? Installation 
 
````bash 
Clone the repository 
git clone https://github.com/namishkasingh/barcleys-sar-narrative.git 
cd barcleys-sar-narrative 
 
Create virtual environment 
python -m venv venv 
source venv/bin/activate  # On Windows: venv\Scripts\activate 
 
Install dependencies 
pip install -r requirements.txt 
```` 
 
## ?? Usage 
 
### Run the Frontend Dashboard 
````bash 
python frontend/sar_dashboard.py 
```` 
 
### Start the Backend API 
````bash 
cd backend 
python api/app.py 
```` 
 
### Generate Graphs 
````bash 
python generate_graphs.py 
```` 
 
## ?? Project Structure 
 
```` 
ÃÄÄ backend/ 
³   ÃÄÄ api/              # API endpoints 
³   ³   ÃÄÄ app.py 
³   ³   ÀÄÄ app_with_db.py 
³   ÀÄÄ data/ 
³       ÀÄÄ synthetic/     # Generated synthetic data 
ÃÄÄ frontend/ 
³   ÀÄÄ sar_dashboard.py   # Main dashboard 
ÃÄÄ notebooks/ 
³   ÀÄÄ ctgan_training/    # CTGAN training notebooks 
ÃÄÄ audit_trail/           # System audit logs 
ÃÄÄ vector_db/             # Vector embeddings storage 
ÃÄÄ generate_graphs.py      # Graph generation script 
ÃÄÄ graph*.png             # Generated visualizations 
ÀÄÄ README.md              # This file 
```` 
 
## ?? Notes on Large Files 
 
The following files are excluded from this repository due to size constraints: 
 
- **creditcard.csv** (~143 MB) - Available on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) 
- **archive.zip** (~66 MB) - Historical data archive 
 
To use these files, download them separately and place them in the appropriate directories. 
 
## ?? Model Performance 
 
 
## ?? Author 
 
**Namishka Singh** 
- GitHub: [@namishkasingh](https://github.com/namishkasingh) 
 
## ?? License 
 
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
 
## ?? Acknowledgments 
 
- Barclays for the project opportunity 
- Kaggle for the creditcard dataset 
- Open source community for amazing tools 
