#include <iostream>
#include <algorithm>
#include <cassert>
#include <string>
#include <fstream>
#include <chrono>

#include "seal/seal.h"

using namespace std;
using namespace seal;
using namespace std::chrono;

inline double get_time_msec(void){
    return static_cast<double>(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count())/1000000;
}

std::vector<std::string> split2(std::string str, char del) {
    int first = 0;
    int last = str.find_first_of(del);
    std::vector<std::string> result;
    while (first < str.size()) {
        std::string subStr(str, first, last - first);
        result.push_back(subStr);
        first = last + 1;
        last = str.find_first_of(del, first);
        if (last == std::string::npos) {
            last = str.size();
        }
    } return result;
}

std::vector<std::vector<std::string> >
csv2vector(std::string filename, int ignore_line_num = 0){
    std::ifstream reading_file;
    reading_file.open(filename, std::ios::in);
    if(!reading_file){
        std::vector<std::vector<std::string> > data;
        return data;
    }
    std::string reading_line_buffer;
    for(int line = 0; line < ignore_line_num; line++){
        getline(reading_file, reading_line_buffer);
        if(reading_file.eof()) break;
    }

    std::vector<std::vector<std::string> > data;
    while(std::getline(reading_file, reading_line_buffer)){
        if(reading_line_buffer.size() == 0) break;
        std::vector<std::string> temp_data;
        temp_data = split2(reading_line_buffer, ',');
        data.push_back(temp_data);
    }
    return data;
}

vector<vector<double>> vector_2d_to_double(vector<vector<string>> x){
  int size1 = x.size();
  int size2 = x[0].size();
  vector<vector<double>> res(size1, vector<double>(size2));
  for(int i=0; i<size1; i++){
    for(int j=0; j<size2; j++){
      res[i][j] = stod(x[i][j]);
    }
  }
  return res;
}

vector<vector<int>> vector_2d_to_int(vector<vector<string>> x){
  int size1 = x.size();
  int size2 = x[0].size();
  vector<vector<int>> res(size1, vector<int>(size2));
  for(int i=0; i<size1; i++){
    for(int j=0; j<size2; j++){
      res[i][j] = stoi(x[i][j]);
    }
  }
  return res;
}


vector<vector<double>> pp_xs(vector<vector<double>> xs, int l, int ls, int n, int dim, int bs){
  //assert(xs.size() == bs);
  assert(xs[0].size() == dim);

  vector<vector<double>> res;
  for(int i=0; i<n-1; i++){
    vector<double> tmp;
    for(int j=0; j<l; j++){
      for(int k=0; k<dim; k++){
        tmp.push_back(xs[i*l+j][k]);
      }
    }
    res.push_back(tmp);
  }

  vector<double> tmp;
  for(int j=0; j<ls; j++){
    for(int k=0; k<dim; k++){
      tmp.push_back(xs[(n-1)*l+j][k]);
    }
  }
  res.push_back(tmp);
  assert(res.size()==n);
  return res;
}


vector<double> pp_w(vector<double> w, int dim){
  assert(w.size() == dim);
  std::reverse(w.begin(), w.end());
  return w;
}


vector<double> pp_b(double b, int pmd){
  vector<double> res(pmd);
  for(int i=0; i<pmd; i++){
    res[i] = b;
  }
  return res;
}

vector<Ciphertext> encode_encrypt_input(vector<vector<double>> xs, CKKSEncoder &encoder, Encryptor &encryptor, double scale, int n){
  vector<Plaintext> tmp(n);
  for(int i=0; i<n; i++){
    //Plaintext tmp1;
    //encoder.encode_as_coeff(xs[i], scale, tmp1);
    encoder.encode_as_coeff(xs[i], scale, tmp[i]);
    //tmp.push_back(tmp1);
  }

  vector<Ciphertext> res(n);
  for(int i=0; i<n; i++){
    encryptor.encrypt(tmp[i], res[i]);
    //res.push_back(tmp1);
  }
  return res;
}

Plaintext encode_w(vector<double> w, CKKSEncoder &encoder, double scale){
  Plaintext res;
  encoder.encode_as_coeff(w, scale, res);
  return res;
}

Plaintext encode_b(vector<double> b, CKKSEncoder &encoder, double scale){
  Plaintext res;
  encoder.encode_as_coeff(b, scale, res);
  return res;
}

vector<Ciphertext> mult_xs_w(vector<Ciphertext> xs, Plaintext w, Evaluator &evaluator, RelinKeys relin_keys, int n){
  vector<Ciphertext> res(n);
  for(int i=0; i<n; i++){
    Ciphertext tmp;
    evaluator.multiply_plain(xs[i], w, res[i]);
    evaluator.relinearize_inplace(res[i], relin_keys);

    //res.push_back(tmp);
  }
  return res;
}

vector<Ciphertext> add_xs_b(vector<Ciphertext> xs, Plaintext b, Evaluator &evaluator, RelinKeys relin_keys, int n){
  vector<Ciphertext> res(n);
  for(int i=0; i<n; i++){
    Ciphertext tmp;

    int scale_n = int(round(log2(xs[i].scale())));
    b.scale() = pow(2.0, scale_n);

    evaluator.add_plain(xs[i], b, res[i]);
    //res.push_back(tmp);
  }
  return res;
}


vector<vector<double>> decrypt_decode_res(vector<Ciphertext> xs, CKKSEncoder &encoder, Decryptor &decryptor, int n){
  vector<Plaintext> tmp(n);
  vector<vector<double>> res;

  for(int i=0; i<n; i++){
    //Plaintext tmp1;
    decryptor.decrypt(xs[i], tmp[i]);
    //tmp.push_back(tmp1);
  }

  for(int i=0; i<n; i++){
    vector<double> tmp1;
    encoder.decode_as_coeff(tmp[i], tmp1);
    res.push_back(tmp1);
  }
  return res;
}

vector<double> psp_res(vector<vector<double>> xs, int l, int ls, int n, int dim, int bs){
  assert(xs.size() == n);
  assert(n > 0);

  vector<double> res(l*(n-1)+ls);
  for(int i=0; i<n-1; i++){
    for(int j=0; j<l; j++){
      //res.push_back(xs[i][dim-1+dim*j]);
      res[i*l+j] = xs[i][dim-1+dim*j];
    }
  }
  for(int j=0; j<ls; j++){
    //res.push_back(xs[n-1][dim-1+dim*j]);
    res[l*(n-1)+j] = xs[n-1][dim-1+dim*j];
  }
  return res;
}


vector<double> debug(vector<vector<double>> xs, vector<double> w, double b, int dim, int bs){
  //assert(xs.size() == bs);
  assert(xs[0].size() == dim);
  assert(w.size() == dim);

  vector<vector<double>> tmp1;
  for(int i=0; i<bs; i++){
    vector<double> tmp2;
    for(int j=0; j<dim; j++){
      tmp2.push_back(xs[i][j]*w[j]);
    }
    tmp1.push_back(tmp2);
  }

  vector<double> res;
  for(int i=0; i<bs; i++){
    double tmp3 = 0;
    for(int j=0; j<dim; j++){
      tmp3 += tmp1[i][j];
    }
    res.push_back(tmp3+b);
  }

  return res;
}


vector<vector<double>> test_slice(vector<vector<double>> x, int size){
    vector<vector<double>> tmp;
    for(int i=0; i<size; i++){
      tmp.push_back(x[i]);
    }
    return tmp;
}


double sigmoid(double x, double gain)
{
  return 1.0 / (1.0 + exp(-gain * x));
}

vector<vector<double>> apply_softmax(vector<vector<double>> x){
  int size1 = x.size();
  int size2 = x[0].size();
  vector<vector<double>> res(size1, vector<double>(size2));
  vector<double> tmp(size2);
  for(int i=0; i<size2; i++){
    double tmp1 = exp(x[0][i]);
    for(int j=1; j<size1; j++){
      tmp1 = tmp1 + exp(x[j][i]);
    }
    tmp[i] = tmp1;
  }

  for(int i=0; i<size2; i++){
    for(int j=0; j<size1; j++){
      res[j][i] = exp(x[j][i])/tmp[i];
    }
  }

  return res;

}

vector<vector<double>> apply_sigmoid(vector<vector<double>> x){
  int size1 = x.size();
  int size2 = x[0].size();
  vector<vector<double>> res(size1, vector<double>(size2));
  for(int i=0; i<size1; i++){
    for(int j=0; j<size2; j++){
      res[i][j] = sigmoid(x[i][j], 1);
    }
  }
  return res;
}


vector<int> parse_result(vector<vector<double>> res){
  vector<int> parsed_res(res[0].size());
  for(int i=0; i<res[0].size(); i++){
    int tmp_index = -1;
    double tmp_value = -100;
    for(int j=0; j<res.size(); j++){
      if(res[j][i] > tmp_value){
        tmp_index = j;
        tmp_value = res[j][i];
      }
    }
    parsed_res[i] = tmp_index;
  }
  return parsed_res;
}


void write_result_to_file(vector<int> x, string filename){
  ofstream myfile(filename);
  int vsize = x.size();
  for (int n=0; n<vsize; n++)
  {
      myfile << x[n] << endl;
  }
}

void write_result_to_file(vector<double> x, string filename){
  ofstream myfile(filename);
  int vsize = x.size();
  for (int n=0; n<vsize; n++)
  {
      myfile << x[n] << endl;
  }
}

void write_label_to_file(vector<vector<int>> x, string filename){
  ofstream myfile(filename);
  int vsize = x.size();
  for (int n=0; n<vsize; n++)
  {
      myfile << x[n][0] << endl;
  }

}

void write_probability_to_file(vector<vector<double>> x, string filename){
  ofstream myfile(filename);
  int size1 = x.size();
  int size2 = x[0].size();
  for(int j=0; j<size2; j++){
    for(int i=0; i<size1; i++){
      myfile << x[i][j] << ",";
    }
    myfile << endl;
  }
}

void write_roundtrip_to_file(double x, string filename){
  ofstream myfile(filename);
  myfile << x << " msec for roundtrip";
}

void write_encryption_to_file(double x, string filename){
  ofstream myfile(filename);
  myfile << x << " msec for encryption";
}

void write_decryption_to_file(double x, string filename){
  ofstream myfile(filename);
  myfile << x << " msec for decryption";
}

void write_computation_to_file(double x, string filename){
  ofstream myfile(filename);
  myfile << x << " msec for computation";
}


int main(int argc, char *argv[]){
    assert(argc==5); 
    // "usage ./test_main input_data_folder input_lr_folder output_folder batch_size"
    string input_data_folder = argv[1];
    string input_lr_folder = argv[2];
    string output_folder = argv[3];
    int bs = atoi(argv[4]);

    printf("hello, world\n");
    printf("\n====================================================\n");
    printf("input_data_folder: %s\n", input_data_folder.c_str());
    printf("input_lr_folder: %s\n", input_lr_folder.c_str());
    printf("output_folder: %s\n", output_folder.c_str());
    printf("batch_size: %d\n", bs);

    EncryptionParameters parms(scheme_type::ckks);


    ///* 8192
    size_t poly_modulus_degree = 8192;
    int pmd = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 60 }));
    double scale = pow(2.0, 40);
    //*/

    SEALContext context(parms);

    // key gen
    printf("\n====================================================\n");
    cout << "keygen and context...." << endl;
    KeyGenerator keygen(context);
    auto secret_key = keygen.secret_key();
    PublicKey public_key;
    keygen.create_public_key(public_key);
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    GaloisKeys gal_keys;
    keygen.create_galois_keys(gal_keys);
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);

    CKKSEncoder encoder(context);

    printf("\n====================================================\n");
    printf("setting for lr...\n");
    int dim = 200;
    //int bs = 8000;
    //vector<double> w = give_me_w(dim);
    //vector<double> b;

    // not using npy loading due to the unstableness...
    //LoadNpy("/from_local/lr_npy/weight.npy", w);
    //LoadNpy("/from_local/lr_npy/bias.npy", b);
    //LoadNpy("/from_local/pp_npy/x_test.npy", xs);

    vector<vector<double>> w = vector_2d_to_double(csv2vector(input_lr_folder + "/weight.csv"));
    vector<vector<double>> b = vector_2d_to_double(csv2vector(input_lr_folder + "/bias.csv"));
    vector<vector<double>> xs = vector_2d_to_double(csv2vector(input_data_folder + "/x_test.csv"));
    vector<vector<int>> ys = vector_2d_to_int(csv2vector(input_data_folder + "/y_test.csv"));


    printf("w size: %d, %d\n", w.size(), w[0].size());
    printf("b size: %d\n", b.size());
    printf("xs size: %d, %d\n", xs.size(), xs[0].size());
    printf("ys size: %d, %d\n", ys.size(), ys[0].size());



    int l = min(pmd/2/2/dim, bs); // 1 ctxt can have at most l of x
    int n = bs/l;
    int nt = bs%l;
    if(nt!=0) n += 1; // n ctxts in total
    int ls = l;
    if(nt!=0) ls = nt; // last ctxt has ls of x
    assert(bs==l*(n-1)+ls);
    printf("dim=%d, bs=%d, l=%d, n=%d, t=%d, ls=%d\n", dim, bs, l, n, nt, ls);


    double start_roundtrip, end_roundtrip;
    double start_encryption, end_encryption;
    double start_decryption, end_decryption;
    double start_computation, end_computation;

    start_roundtrip = get_time_msec();
    printf("\n====================================================\n");
    printf("encryption...\n");
    start_encryption = get_time_msec();
    //xs = test_slice(xs, bs);
    vector<vector<double>> ppd_xs = pp_xs(xs, l, ls, n, dim, bs);

    vector<Ciphertext> enc_xs = encode_encrypt_input(ppd_xs, encoder, encryptor, scale, n);
    end_encryption = get_time_msec();


    printf("\n====================================================\n");
    printf("calculation...\n");
    start_computation = get_time_msec();
    vector<vector<Ciphertext>> res_ctxt(4, vector<Ciphertext>(n));
    for(int i=0; i<4; i++){
      printf("label loop: %d\n", i);
      vector<double> ppd_w = pp_w(w[i], dim);
      vector<double> ppd_b = pp_b(b[i][0], pmd);
      Plaintext plain_w = encode_w(ppd_w, encoder, scale);
      Plaintext plain_b = encode_b(ppd_b, encoder, scale);
      vector<Ciphertext> xs_w = mult_xs_w(enc_xs, plain_w, evaluator, relin_keys, n);
      xs_w = add_xs_b(xs_w, plain_b, evaluator, relin_keys, n);
      //res_ctxt.push_back(xs_w);
      res_ctxt[i] = xs_w;
    }
    end_computation = get_time_msec();


    printf("\n====================================================\n");
    printf("decryption...\n");
    start_decryption = get_time_msec();
    vector<vector<double>> res(4, vector<double>(l*(n-1)+ls));
    for(int i=0; i<4; i++){
      vector<vector<double>> dec_res = decrypt_decode_res(res_ctxt[i], encoder, decryptor, n);
      vector<double> psp_x_w = psp_res(dec_res, l, ls, n, dim, bs);
      res[i] = psp_x_w;
    }
    end_decryption = get_time_msec();

    //printf("after calc: %d, %d\n", res.size(), res[0].size());

    printf("\n====================================================\n");
    printf("parsing result...\n");
    vector<vector<double>> res_probability = apply_softmax(res);
    vector<int> parsed_res = parse_result(res);
    printf("\n====================================================\n");
    printf("done...\n");
    end_roundtrip = get_time_msec();
    printf("time for prediction %f\n", end_roundtrip - start_roundtrip);

    printf("\n====================================================\n");
    printf("writing all the results to file...\n");
    write_encryption_to_file(end_encryption - start_encryption, output_folder + "/encryption.csv");
    write_decryption_to_file(end_decryption - start_decryption, output_folder + "/decryption.csv");
    write_computation_to_file(end_computation - start_computation, output_folder + "/computation.csv");
    write_roundtrip_to_file(end_roundtrip - start_roundtrip, output_folder + "/roundtrip.csv");
    write_probability_to_file(res_probability, output_folder + "/probability.csv");
    //write_probability_to_file(res, output_folder + "/res.csv");
    write_result_to_file(parsed_res, output_folder + "/result_cipher_label.txt");


    //printf("\n====================================================\n");
    //printf("debug purpose......\n");
    //vector<vector<double>> res_raw;
    //for(int i=0; i<4; i++){
    //  vector<double> tmp = debug(xs, w[i], b[i][0], dim, bs);
    //  res_raw.push_back(tmp);
    //}
    //vector<int> parsed_raw_res = parse_result(res_raw);

    //write_result_to_file(parsed_raw_res, output_folder + "/result_raw.txt");
    //write_label_to_file(ys, output_folder + "/result_label.txt");

    //for(int i=0; i<parsed_res.size(); i++){
    //  printf("%d, ", parsed_res[i]);
    //}
    //printf("\n");


    //for(int i=0; i<ys.size(); i++){
    //  printf("%d, ", ys[i][1]);
    //}
    //printf("\n");

    //
    //int tot = 0;
    //int right = 0;
    //for(int i=0; i<ys.size(); i++){
    //  if(int(parsed_res[i]) == int(ys[i][1])){
    //    right+=1;
    //  }
    //  tot+=1;
    //}

    //printf("\n====================================================\n");
    //printf("tot: %d, right: %d, acc: %f\n", tot, right, double(right)/double(tot));


    return 0;

}
