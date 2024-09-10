import spacy
from spacy.util import get_package_path

model_name = "en_core_web_sm"
model_path = get_package_path(model_name)
print(model_path)
