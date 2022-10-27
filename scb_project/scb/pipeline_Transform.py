import fitz
from sentence_transformers import SentenceTransformer,util
from .pipeline_Extract import Extraction

def Transformation(Extract):
        text1 = Extract
        text1 = text1.split('\n')
        print("text1", text1)
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        doc = fitz.open(r'D:\\scb_bazaar\\scb_bazaar\\scb_dynamic_dropdown\\scb_project\\financepdf.pdf')
        text2 = doc[0].get_text()
        embeddings2 = model.encode(text2, convert_to_tensor=True)
        value_dict = {}
        for i in text1:
            embeddings1 = model.encode(i, convert_to_tensor=True)
            cosine_scores = util.cos_sim(embeddings1, embeddings2)
            if cosine_scores.item() > 0.40:
                value_dict[i] = cosine_scores.item()
        value_str = ""
        for item in value_dict:
            value_str += item + ':' + str(value_dict[item]) + ' '
        print(type(value_str))
        print(f"Converted string- {value_str}")
        return value_str