**Molecular Machine Learning Competition**

**Molecular Machine Learning Project**

**Overview**

This repository contains code and data for predicting the photostability (T80) of organic molecules, especially under small-data constraints. The project incorporates:

1.  A primary training set (42 molecules) and test set (9 molecules).
2.  An additional dataset (OligomerFeatures.csv) with ~2200 molecules, mostly unlabeled, used for semi-supervised (pseudo-labeling) techniques.
3.  Various scripts for data exploration, feature engineering, and model training.

**Directory Structure**

data/

-   train.csv
-   test.csv
-   sample_submission.csv
-   OligomerFeatures.csv

images/

-   correlation_basic_descriptors.png
-   correlation_dos_features.png
-   correlation_top_features.png

scripts/

-   advanced.py (older advanced pipeline or feature generation script)
-   eda.py (exploratory data analysis)
-   prefeature.py (preliminary feature engineering or pipeline)
-   pseudo.py (semi-supervised pseudo-labeling approach)
-   rforest.py (random forest experiments)
-   submission.py (generates final submission CSV)
-   archive/

-   explore.py (archived EDA or older code)
-   models.py (archived model code)

submission.csv (example final submission file at the project root)

**Key Scripts**

1.  eda.py

-   Basic exploratory data analysis on the training and test sets, plus potential generation of correlation plots (saved in images/).

3.  advanced.py

-   Early advanced pipeline experiments or additional feature transformations.

5.  prefeature.py

-   Preliminary feature engineering script that may include dimensionality reduction, DOS transformations, or 3D descriptors.

7.  pseudo.py

-   Implements a semi-supervised approach, combining the labeled set with unlabeled OligomerFeatures data.
-   Trains an ensemble for confidence estimates, filters pseudo-labeled data by confidence, and retrains the final model.

9.  rforest.py

-   Focused experiments with random forest. Could be an older script or partial approach.

11. submission.py

-   Loads the best model/pipeline and outputs a final submission.csv for the competition.

**Data Description**

-   train.csv and test.csv are the primary competition data (42 training molecules, 9 test molecules).
-   OligomerFeatures.csv is a larger dataset (~2200 molecules) with only ~40 T80-labeled samples; used for pseudo-labeling.
-   sample_submission.csv is an example format for the competition.

**Usage**

1.  Place train.csv, test.csv, and OligomerFeatures.csv in the data/ folder.
2.  From the project root, run one of the scripts in scripts/, for example:\
    python scripts/pseudo.py\
    This might:

-   Load and preprocess the data.
-   Perform pseudo-labeling with an ensemble model.
-   Evaluate on the original labeled set.
-   Generate a final submission.csv.

4.  If you want to do a straightforward approach, you can run:\
    python scripts/submission.py\
    which loads a previously trained model and outputs a submission.csv.
5.  For exploratory analysis, run:\
    python scripts/eda.py\
    to generate correlation plots or initial data insights (images saved to images/).

**Project Notes and Next Steps**

-   Consider applying PCA or other dimensionality reduction to DOS/SDOS features (prefeature.py or advanced.py).
-   Refine pseudo-labeling by adjusting confidence thresholds or iterative labeling in pseudo.py.
-   Experiment with different ensembles (e.g., XGBoost, CatBoost) and hyperparameter tuning to lower MSLE.
-   Evaluate results via cross-validation on the original 42-labeled data.

**License and Credits**

No specific license provided here. If you want to open-source your code, consider adding an MIT or Apache-2.0 LICENSE file.\
Credit to the original dataset providers and any references in the competition or OligomerFeatures dataset.

Portions of the code were inspired by or generated with the help of ChatGPT.

**Contact**

Author: Sean Ryan\
Feel free to open issues or pull requests if you have questions or improvements.# molecular-ml
