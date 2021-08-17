rm -rf /from_local/results /from_local/pp_data
mkdir /from_local/results /from_local/pp_data
python test_main.py /data/Challenge.fa
./seal/test_main_cpp /from_local/pp_data /from_local/trained_model /from_local/results 2000
