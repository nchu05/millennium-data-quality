# Millennium Data Quality Project Setup and Usage

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/jerrychen04/millennium-data-quality.git
    cd millennium-data-quality
    ```

2. **Create a virtual environment:**
    ```sh
    python -m venv venv
    ```

3. **Activate the virtual environment:**
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```

4. **Install the dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Running the Main Script

To run the main script, execute the following command:
```sh
python backtester/main.py
```

## Running Tests

To run the unit tests, use the following command:
```sh
python -m unittest discover -s unit_tests
```

## To Create A Strategy

Go to `order_generator.py` and create a new instance of `OrderGenerator`. See example code in main.py to run strategy. Write corresponding unit tests