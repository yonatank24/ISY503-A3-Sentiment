# ISY503 Assessment 3 - Sentiment Analysis

**Author:** Jhonatan Mosquera Caro

**Subject:** ISY503 Intelligent Systems - Torrens University Australia  
**Dataset:** Multi-Domain Sentiment Dataset (Blitzer et al., 2007)  
**Dataset URL:** http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html  

---

## Project Overview

Sentiment analysis system that classifies Amazon product reviews as 
Positive or Negative using a Neural Network (MLP) trained on real 
customer data.

**MLP Test Accuracy:** 77.11%  
**Logistic Regression Test Accuracy:** 76.76%  
**Best model:** MLP Neural Network  
**Total reviews used:** 1,889 (after outlier removal)

---

## Files

| File | Description |
|------|-------------|
| `ISY503_A3_FINAL_v4.ipynb` | Full ML pipeline - run in Google Colab |
| `sentiment_analyser.html` | Web interface - open in any browser |

---

## How to Run

### Jupyter Notebook
1. Upload `Multi_Domain_Sentiment_Dataset.zip` to your Google Drive
2. Open `ISY503_A3_FINAL_v4.ipynb` in Google Colab
3. Run all cells in order from top to bottom

### Web Interface
1. Download `sentiment_analyser.html`
2. Double-click the file to open in any browser
3. Type a review and click Analyse

---

## ML Pipeline

| Step | Description |
|------|-------------|
| Load | BeautifulSoup parses XML review files |
| Shuffle | Randomise to remove ordering bias |
| Clean | Lowercase, punctuation removal, lemmatisation, stopwords |
| Outlier removal | Remove reviews < 3 or > 500 words |
| Encode | CountVectorizer converts text to word count vectors |
| Normalise | MaxAbsScaler standardises feature vectors |
| Split | 70% training / 15% validation / 15% test |
| Batches | Python generator function, batch size 32 |
| Train | MLP Neural Network - 128 → 64 neurons, ReLU, Adam |
| Compare | Logistic Regression trained on same data |
| Evaluate | Accuracy, classification report, confusion matrix |

---

## Ethical Considerations

- **Labelling bias** - labels assigned by dataset creators may not reflect true sentiment for all reviewers
- **Class balance** - equal positive/negative examples avoid model bias
- **Deployment risk** - model should support human decisions, not replace them

---

## References

Blitzer, J., Dredze, M., & Pereira, F. (2007). Biographies, Bollywood, 
boom-boxes and blenders. *Proceedings of the 45th Annual Meeting of the ACL*, 187-205.
