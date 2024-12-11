conda create -n CompreCap python=3.9
conda activate CompreCap
pip install -r requirements.txt
pip install spacy
python -m spacy download en_core_web_lg