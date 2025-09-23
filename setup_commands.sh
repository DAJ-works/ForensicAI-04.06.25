mkdir -p videoproof
cd videoproof

python -m venv venv

# Activate virtual environment (Windows)
# venv\Scripts\activate

# Activate virtual environment (Linux/Mac)
source venv/bin/activate

# Create project structure
mkdir -p {backend/{preprocessing,models,event_detection,api,database},frontend/{components,pages,services},docs,scripts,data/{raw,processed}}

# Create main requirements file
touch requirements.txt