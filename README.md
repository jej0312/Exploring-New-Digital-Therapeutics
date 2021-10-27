python version: 3.8.6  

```plain
C:.  
└─DigitalTherapeutics  
    ├─README.md  
    ├─data  
    │   ├─disease  
    │   │   └─psychiatric_disorder_CUIs.pickle (crawled data (Mental Illness) from NCBI)  
    │   ├─patent (all should not be shared)  
    │   └─mapping  
    │       ├─umls-2021AA-mrconso.zip (from UMLS. can download for free after sign up)  
    │       ├─MRCONSO.RRF (from UMLS. can download for free after sign up)  
    │       ├─MRCONSO_reduced.RRF (from UMLS. can download for free after sign up)  
    │       └─disease_label_all_210727.csv (patent and its target information)  
    └─py  
        ├─__init__.py  
        ├─pymongo_handler.py (contains id, pwd)  
        ├─DTx_companies.py (crawling lists of DTx_companies)  
        ├─patent_description.py (crawling patents' description from WIPSON; contains id, pwd)  
        ├─bern.py (using API for BioBERT NER)  
        ├─filtering_psychiatric_disorders.py (filtering the CUIs of psychiatric disorders; from NCBI)  
        ├─mapping_MeSH_CUI.py (mapping the IDs -- MeSH-CUI)  
        ├─BERTopic.py (nonDTx technology detection by topic modeling)  
        └─recommending_nondtx.py (nonDTx tech detection by sentence similarity and target recommendation)   