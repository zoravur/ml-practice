# Jax Algorithms
## Common machine learning algorithms implemented in Jax

All notebooks are located in the `notebooks/` directory.

| Model                | Dataset                               | Filename                  |
|----------------------|---------------------------------------|---------------------------|
| Linear Regression    | California housing dataset            | linear-regression.ipynb   |
| Logistic Regression  | Wisconsin Diagnostic Breast Cancer    | logistic-regression.ipynb |
| MLP                  | MNIST                                 | mlp.ipynb                 |
| CNN                  | CIFAR-10	                           | cnn.ipynb                 |
| CNN                  | CIFAR-10	                           | cifar-redux.ipynb         |
| VAE                  | MNIST                                 | TODO                      |
| BoW + TFIDF + LogReg | IMDB Sentiment                        | nlp-intro.ipynb           |

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter:
   ```bash
   jupyter lab
   ```

4. Open any of the notebook files to explore the implementations.