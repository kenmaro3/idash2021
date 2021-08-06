rm -rf results
mkdir results
python test_main.py Challenge/Challenge.fa
./seal/test_main_cpp /from_local/pp_csv /from_local/lr_csv /from_local/results

