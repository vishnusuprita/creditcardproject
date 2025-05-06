import streamlit as st

st.title("📈 Page 2: Metrics & Visualizations")
st.write("Graphs, charts, evaluation scores, etc.")
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report

st.title("📈 Model Evaluation Metrics and Visualizations")
st.markdown("---")

# Load your trained models (make sure you have these models saved in the same directory)
model_names = ["Logistic Regression", "Decision Tree", "Random Forest", "AdaBoost"]
models = {
    "Logistic Regression": pickle.load(open(r"C:\Users\Admin\Desktop\python_project\models\logistic_model.pkl", 'rb'))
,
    "Decision Tree": pickle.load(open(r"C:\Users\Admin\Desktop\python_project\models\Descisiontree_model.pkl", 'rb')),
    "Random Forest": pickle.load(open(r"C:\Users\Admin\Desktop\python_project\models\randomtree.pkl", 'rb')),
    "AdaBoost": pickle.load(open(r"C:\Users\Admin\Desktop\python_project\models\ada_model.pkl", 'rb')),
}

X_test = pickle.load(open(r"C:\Users\Admin\Desktop\python_project\X_test.pkl", 'rb'))
y_test = pickle.load(open(r"C:\Users\Admin\Desktop\python_project\y_test.pkl", 'rb'))

# Load your test data (assumes you have X_test and y_test saved separately)
# X_test = pd.read_csv('X_test.csv')
# y_test = pd.read_csv('y_test.csv')

# Flatten y_test if needed
y_test = y_test.values.flatten()

# Evaluation Function
def evaluate_model(name, model, X_test, y_test):
    st.subheader(f"🔵 {name}")

    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:,1]
    
    # Metrics
    st.write("**Classification Report:**")
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report)

    # Confusion Matrix
    st.write("**Confusion Matrix:**")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    # ROC Curve
    st.write("**ROC Curve:**")
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = roc_auc_score(y_test, y_probs)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax2.plot([0,1], [0,1], 'k--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend(loc="lower right")
    st.pyplot(fig2)

    st.markdown("---")

# Loop through all models
for name in model_names:
    evaluate_model(name, models[name], X_test, y_test)
