# Guide to Packaging Project

1. Create setup python file
    Create a `setup.py` file in the project directory

   ```sh
   .
   ├── MANIFEST.in
   ├── package
   │   ├──__init__.py
   │   └──__main__.py
   ├── poetry.lock
   ├── pyproject.toml
   ├── README.md
   ├── requirements.txt
   ├── setup.py

   ```

2. Clean previous distributions

   ```sh
   rm -rf build dist *.egg-info
   ```

3. Build using poetry

   ```sh
   poetry build
   ```

4. Install package

   ```sh
   pip install -e path/to/package
   ```
