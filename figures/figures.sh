python3 parse_cusparse.py ../hypre_test/data/CuSparse.out > cusparse.csv
python3 parse_amgt.py ../hypre_test/data/No_Mixed.out > nomix.csv
python3 parse_amgt.py ../hypre_test/data/Mixed.out > mix.csv
python3 merge_pro.py
python3 Fig7.py

# python3 Fig8.py


# python3 preprocess_parse.py > preprocess_test.csv
# python3 Fig9.py