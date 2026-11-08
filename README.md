DefenseAI

=========



Aerospace-Inspired Equipment Intelligence Platform



DefenseAI is a research prototype that combines remaining useful life prediction, explainable machine learning, sensor-data inference, and document retrieval in an interactive web application.



The current implementation focuses on simulated turbofan-engine degradation using NASA C-MAPSS FD001. Users can explore validation results, inspect sensor trends, understand model predictions through SHAP, generate predictions from uploaded CSV records, and search PDF documents for relevant passages.



Technology Stack

\---------------



Python, Pandas, NumPy, scikit-learn, XGBoost, SHAP, Streamlit, Plotly, Matplotlib, and pypdf.



Implemented Capabilities

\-----------------------



\- Dataset download, structural validation, and remaining useful life label generation.

\- Engine-level training and validation splits to keep individual engines separate.

\- Comparative evaluation of a mean baseline, Random Forest, and XGBoost.

\- Validation error analysis by engine and remaining-life range.

\- Global and local SHAP explanations.

\- CSV-based inference with required-column and numeric-value validation.

\- Checks for sensor values outside observed training ranges.

\- PDF text extraction and passage retrieval with source names and page references.

\- A unified Streamlit interface for analysis, inference, and document search.



Model Evaluation

\----------------



The current experiment uses 80 engines for training and 20 engines for validation. Validation metrics cover all 4,070 records from the held-out engines.



| Model | Validation MAE | Validation RMSE |

|---|---:|---:|

| XGBoost | 25.68 cycles | 35.03 cycles |

| Random Forest | 25.67 cycles | 35.11 cycles |

| Mean baseline | 55.36 cycles | 65.72 cycles |



XGBoost was selected based on validation RMSE, although its performance was similar to Random Forest on this split.



The evaluation also includes per-engine errors, signed prediction errors, and performance across actual remaining-life ranges. SHAP explanations show how individual features contribute to the model’s predictions.



Validation data was used for model selection and XGBoost early stopping. These figures are development-stage validation results, not official test-set results. The official test set has not yet been evaluated.



Local Installation

\------------------



Run the following commands in Anaconda Prompt:



```bat

conda create -n defenseai python=3.11 -y

conda activate defenseai

python -m pip install -r requirements.txt

```



Reproducing the Pipeline

\-----------------------



From the repository root, run:



```bat

python setup\_project.py

python prepare\_fd001.py

python analyze\_and\_split.py

python train\_models.py

python analyze\_errors.py

python explain\_model.py

```



The download step requires an internet connection and access to the configured third-party dataset mirror. The preparation script records the source URLs and downloaded-file hashes in a validation report.



Datasets and trained model binaries are excluded from the repository. Run the pipeline before launching the application to generate the required local files.



Launching the Application

\-------------------------



From the repository root, run:



```bat

python -m streamlit run dashboard\\main.py --server.port 8501

```



The application provides three main sections:



Validation Dashboard  

Displays model comparisons, historical engine trajectories, validation errors, and saved SHAP explanations.



Sensor Inference  

Accepts compatible sensor records, validates model inputs, generates remaining useful life predictions, and exports results.



Document Search  

Extracts text from uploaded PDFs and retrieves relevant passages with document names and PDF page positions.



Document Retrieval

\------------------



The current retrieval implementation uses TF-IDF and cosine similarity to rank passages by keyword relevance. It returns extracted text rather than generated answers.



This module does not yet include an LLM, semantic embeddings, or generative question answering. It requires text-based PDFs; OCR for scanned documents is not implemented.



Repository Components

\---------------------



| File | Purpose |

|---|---|

| `setup\_project.py` | Creates the project structure and checks package imports |

| `prepare\_fd001.py` | Downloads and validates FD001 data and creates RUL labels |

| `analyze\_and\_split.py` | Creates engine-level splits and exploratory charts |

| `train\_models.py` | Trains and compares the baseline, Random Forest, and XGBoost |

| `analyze\_errors.py` | Generates validation error diagnostics |

| `explain\_model.py` | Produces global and local SHAP explanations |

| `dashboard/main.py` | Defines application navigation |

| `dashboard/app.py` | Displays validation results and historical analysis |

| `dashboard/predict.py` | Handles CSV input validation and model inference |

| `dashboard/documents.py` | Provides PDF passage retrieval |



Scope and Limitations

\---------------------



DefenseAI is a \*\*research\*\* and portfolio project, not an operational maintenance system.



\- C-MAPSS contains simulated turbofan-engine degradation data, not classified military equipment records.

\- Remaining useful life predictions are expressed in operating cycles, not hours or calibrated failure probabilities.

\- SHAP explains model behavior; it does not establish physical fault causes.

\- Passing feature-range checks does not guarantee that an input is representative of training data or that its prediction is reliable.

\- Retrieved passages are not verified maintenance instructions.

\- Current metrics come from one engine-level validation split.

\- The prototype is not certified for real-world maintenance or operational readiness decisions.



Planned Development

\-------------------



Future work includes semantic document retrieval, source-grounded LLM responses, evaluated fine-tuning experiments, structured agent workflows, and a readiness and cost simulator with explicitly documented assumptions.



These capabilities are planned extensions and are not part of the current implementation.


