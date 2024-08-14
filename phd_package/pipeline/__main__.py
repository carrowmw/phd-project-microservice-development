# phd_package/pipeline/__main__.py

from .pipeline_generator import Pipeline

if __name__ == "__main__":
    pipeline = Pipeline()
    print(f"Running pipeline... {pipeline}\n")
    pipeline.run_pipeline()
