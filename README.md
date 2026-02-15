# Fake Account Detection System

## Description
This project was made for the **Final Year Project** of **BSc.CSIT (7th semester) 2078 batch**. It was made by **Raunak Tuladhar (me)**, **Sargaa Manandhar** and **Shishir Timalsina**. The project we made was supervised by our supervisor **Mr. Hiranya Prasad Bastakoti**.

Fake Account Detection System is a web-based application designed to identify and flah fraulent or fake accounts. It is **Twitter** (now known as **X**) based system. It likely targets the profile details such as followers counts, following counts and tweet counts. It utilizes machine learning, specifically deep learning models, in its backend to analyze user data and detect suspicious patterns. The system likely processes various forms of information, potentially including structured user data and unstructured textual content, to classify accounts as either legitimate or fake. A frontend provides the user interface for interaction.

## Algorithms Used

### 1. BERT (Bidirectional Encoder Representations from Transformers)

- **Description:**  
BERT is a cutting-edge pre-trained transformer-based neural network model for Natural Language Processing (NLP). It processes text bidirectionally, enabling it to grasp the full context and nuance of words within a sentence. BERT generates high-dimensional vector representations (embeddings) that encapsulate the semantic and syntactic meaning of textual input.

- **Role in this project:**  
BERT (bert-base-uncased) is utilized as a powerful feature extractor for the textual description of user profiles. The system computes the mean of BERT's last hidden state to obtain a fixed-size (768-dimensional) embedding. These embeddings provide rich, contextualized textual features that help the model understand the content and style of profile descriptions, which can be crucial indicators of fake accounts. BERT's parameters are frozen, meaning it's used directly for feature generation without further training during the classification phase.


### 2. FNN (Feedforward Neural Network)

- **Description:**  
A Feedforward Neural Network is a fundamental type of artificial neural network where data flows in one direction, from the input layer through one or more hidden layers, to the output layer. The ImprovedFakeProfileClassifier implemented in the project is a multi-layered FNN that incorporates nn. Linear layers for transformations, nn. BatchNorm1d for stabilizing training, nn. ReLU as an activation function, and nn. Dropout for regularization to prevent overfitting.

- **Role in Project:**  
This FNN serves as the main classifier responsible for making the final "real" or "fake" prediction. It takes a concatenated feature vector as its input, comprising:
    * The 768-dimensional BERT embeddings extracted from the profile description.
    * Three standardized numerical features: followers_counts, following_counts, and tweet_counts.
    * The FNN learns complex, non-linear patterns within these combined features to classify profiles.

### 3. StandardScaler

- **Description:**  
StandardScaler is a preprocessing technique from the scikit-learn library. It transforms numerical features so that they have a mean of 0 and a standard deviation of 1. This process is also known as Z-score normalization.

- **Role in Project:**  
Applied to the numerical features (followers_count, friends_count, statuses_count). Standardization ensures that these features, which often have different scales, do not disproportionately influence the FNN during training and inference. This is a common and important step to improve the performance and stability of many machine learning models.

## Technologies Used

### Frontend

* **HTML:** For structuring web pages.

* **CSS:** For styling the web pages.

* **JavaScript:** For interactive functionality.

* **npm / Node.js:** Used for managing frontend packages and dependencies.

### Backend

* **Python:** The primary programming language for the server-side logic.

* **Flask:** A lightweight web framework for building the API and serving the frontend.

* **FLask-CORS:** For handling Cross-Origin Resource Sharing.

* **SQLite:** A file-based relational database for storing user information

* **bcrypt:** For securely hashing and verifying user passwords.

* **Google OAuth 2.0 Client Library:** For implementing Google Login functionality.

### Machine Learning / Data Science

* **PyTorch:** The deep learning framework used for defining, training, and running the neural network models.

* **Hugging Face Transformers:** For easily loading and utilizing pre-trained BERT models and tokenizers.

* **Numpy:** Essential for numerical operations and array manipulation.

* **Pandas:** Used for data loading, manipulation, and preprocessing, especially in the training script. 

* **Scikit-learn:** Provides the StandardScaler for preprocessing numerical features and various metrics for model evaluation.

* **Matplotlib and Seaborn:** Used for data visualization and plotting training results (e.g., loss curves, confusion matrix).

* **TQDM:** For displaying progress bars during computationally intensive tasks like BERT embedding extraction and model training.

* **Pickle:** For serializing and deserializing Python objects, specifically used to save and load the trained data.

### Development & Other

* **Git:** Version control system.

* **Visual Studio Code:** Integrated Development Environment (IDE) settings.

* **Python `venv`:** Python virtual environment for managing project dependencies.

## How to run

Before you start, ensure you have the following installed:
* Python 3.x: (e.g., Python 3.8+)
* pip: Python's package installer (usually comes with Python).
* Node.js & npm: For managing frontend dependencies and building CSS.

### 1. Backend Setup and Execution

   1. Navigate to the Backend Directory: `cd backend`

   2. Create and Activate a Python Virtual Environment: (It's good practice to isolate project dependencies)
   
        * python -m venv venv
        * **On Windows:**
            `.\venv\Scripts\activate`
        * **On macOS/Linux:**
         `source venv/bin/activate`

   3. Install Python Dependencies:
        `pip install -r requirements.txt`


   4. Obtain/Train Machine Learning Models:  
      The **app.py** expects model files (final_model.pt or bert_lstm_model.pt) and the BERT model directory (bert_model/). The train_model.py script will generate final_model.pt (and best_model.pt) and download the necessary BERT components (bert-base-uncased) if they don't exist locally.

       * Ensure you have fusers.csv and users.csv in the backend directory, as **train_model.py** relies on them.
       * Run the training script to generate the models:
            `python train_model.py`  
            
        This might take a while as it downloads BERT and trains the model. It will save final_model.pt (and best_model.pt) and a bert_model directory.


   5. Start the Flask Backend Server:  `python app.py`  

      The server will typically run on http://localhost:5000. You should see output something like this in your terminal:

      ```bash
      * Running on http://127.0.0.1:5000/ 
      (Press CTRL+C to quit)
      ```

### 2. Frontend Setup (for Development)

  While the Flask backend serves the static frontend files directly, you might want to set up the frontend separately for development, especially to watch for
  Tailwind CSS changes.

   1. Open a New Terminal and Navigate to the Frontend Directory:
        `cd frontend`


   2. Install Node.js Dependencies:
        `npm install`

   3. Build/Watch Tailwind CSS:  
      To compile input.css into style.css and watch for changes during development:


        `npx tailwindcss -i ./input.css -o ./style.css --watch`  

      Keep this terminal window open and running while you develop the frontend.

### 3. Access the Application

  Once the Flask backend server is running, open your web browser and navigate to:

  `http://localhost:5000` 

  You should see the application's homepage.
