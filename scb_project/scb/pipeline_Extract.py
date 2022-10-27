
import io
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
import json


def extract_text_by_page(file_path):
    with open(file_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh, 
                                      caching=True,
                                      check_extractable=True):
            resource_manager = PDFResourceManager()
            fake_file_handle = io.StringIO()
            converter = TextConverter(resource_manager, fake_file_handle)
            page_interpreter = PDFPageInterpreter(resource_manager, converter)
            page_interpreter.process_page(page)
            
            text = fake_file_handle.getvalue()
            yield text
    
            # close open handles
            converter.close()
            fake_file_handle.close()
    
def Extraction(pdf_path):
    print(pdf_path)
    allpages =""
    for page in extract_text_by_page(pdf_path):
        allpages = allpages +page
    return allpages


# class DataClassification:
#     def __init__(self, file_path):
#         self.file_path = file_path
#
#     def Classification(self):
#         class_df = class_predict(self.file_path)
#         print(class_df)#         return class_df


# pipe2 = Pipeline(steps=[('DataExtraction', DataExtraction()),])
