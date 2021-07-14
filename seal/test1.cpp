#include <iostream>
#include <algorithm>
#include <cassert>


#include "seal/seal.h"

using namespace std;
using namespace seal;

#include <chrono>
using namespace std::chrono;
inline double get_time_msec(void){
    return static_cast<double>(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count())/1000000;
}

//vector<vector<double>> give_me_ys(int bs, int dim){
//  vector<vector<double>> res;
//  for(int i=0; i<bs; i++){
//    vector<double> tmp;
//    for(int j=0; j<dim; j++){
//      tmp.push_back(double(i)*0.1 + double(j));
//    }
//    res.push_back();
//  }
//  return res;
//}

vector<double> give_me_w(int dim){
  vector<double> res;
  for(int i=0; i<dim; i++){
    res.push_back(double(i)*0.1 + double(i));
  }
  return res;
}

vector<vector<double>> give_me_xs(int batch_size, int dim){
  vector<vector<double>> xs(batch_size, vector<double>(dim));
  for(int i=0; i<batch_size; i++){
    for(int j=0; j<dim; j++){
      xs[i][j] = double(i)*0.1 + double(j);
    }
  }
  return xs;
}


void print_vec_1d(vector<double> x, int size){
  for(int i=0; i<size; i++){
    printf("%f, ", x[i]);
  }
  printf("\n");
}

void print_vec_2d(vector<vector<double>> xs, int size){
  for(int i=0; i<xs.size(); i++){
    print_vec_1d(xs[i], size);
  }
}

vector<vector<double>> pp_xs(vector<vector<double>> xs, int bs, int dim){
  assert(xs.size() == bs);
  assert(xs[0].size() == dim);
  return xs;
}

vector<vector<double>> pp_ys(vector<vector<double>> ys){

}

vector<double> pp_w(vector<double> w, int dim){
  assert(w.size() == dim);
  std::reverse(w.begin(), w.end());
  return w;
}

vector<Ciphertext> encode_encrypt_input(vector<vector<double>> xs, CKKSEncoder &encoder, Encryptor &encryptor, double scale, int n){
  vector<Plaintext> tmp;
  for(int i=0; i<n; i++){
    Plaintext tmp1;
    encoder.encode_as_coeff(xs[i], scale, tmp1);
    tmp.push_back(tmp1);
  }

  vector<Ciphertext> res;
  for(int i=0; i<n; i++){
    Ciphertext tmp1;
    encryptor.encrypt(tmp[i], tmp1);
    res.push_back(tmp1);
  }
  return res;
}

Plaintext encode_w(vector<double> w, CKKSEncoder &encoder, double scale){
  Plaintext res;
  encoder.encode_as_coeff(w, scale, res);
  return res;
}

vector<Ciphertext> mult_xs_w(vector<Ciphertext> xs, Plaintext w, Evaluator &evaluator, RelinKeys relin_keys, int n){
  vector<Ciphertext> res;
  for(int i=0; i<n; i++){
    Ciphertext tmp;
    evaluator.multiply_plain(xs[i], w, tmp);
    res.push_back(tmp);
  }
  return res;
}


vector<vector<double>> decrypt_decode_res(vector<Ciphertext> xs, CKKSEncoder &encoder, Decryptor &decryptor, int n){
  vector<Plaintext> tmp;
  vector<vector<double>> res;

  for(int i=0; i<n; i++){
    Plaintext tmp1;
    decryptor.decrypt(xs[i], tmp1);
    tmp.push_back(tmp1);
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

  vector<double> res;
  for(int i=0; i<n-1; i++){
    for(int j=0; j<l; j++){
      res.push_back(xs[i][dim-1+dim*j]);
    }
  }
  for(int j=0; j<ls; j++){
    res.push_back(xs[n-1][dim-1+dim*j]);
  }
  return res;
}




int main(){
    cout << "hello, world" << endl;
    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = 8192;
    int pmd = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 60 }));
    double scale = pow(2.0, 40);
    SEALContext context(parms);

    // key gen
    cout << "ititialization of ckks keys etc" << endl;
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

    int dim = 200;
    int bs = 2000;
    vector<double> w = give_me_w(dim);
    vector<vector<double>> xs = give_me_xs(bs, dim);

    //print_vec_1d(w, dim);
    //cout << endl;
    //print_vec_2d(xs, dim);

    int l = min(pmd/2/2/dim, bs); // 1 ctxt can have at most l of x
    int n = bs/l;
    int nt = bs%l;
    if(nt!=0) n += 1; // n ctxts in total
    int ls = l;
    if(nt!=0) ls = nt; // last ctxt has ls of x
    printf("dim=%d\nbs=%d\nl=%d\nn=%d\nnt=%d\nls=%d\n", dim, bs, l, n, nt, ls);

    double start, end;
    start = get_time_msec();
    vector<vector<double>> ppd_xs = pp_xs(xs, bs, dim);

    //print_vec_2d(ppd_xs, dim*(l+1));
    vector<double> ppd_w = pp_w(w, dim);
    cout << "\npp done" << endl;

    vector<Ciphertext> enc_xs = encode_encrypt_input(ppd_xs, encoder, encryptor, scale, n);
    Plaintext plain_w = encode_w(ppd_w, encoder, scale);
    cout << "\nenc done" << endl;
    vector<Ciphertext> xs_w = mult_xs_w(enc_xs, plain_w, evaluator, relin_keys, n);
    cout << "\nmult done" << endl;
    vector<vector<double>> dec_res = decrypt_decode_res(xs_w, encoder, decryptor, n);
    cout << "\ndec done" << endl;
    vector<double> psp_x_w = psp_res(dec_res, l, ls, n, dim, bs);
    cout << "\npsp done" << endl;
    end = get_time_msec();
    cout << "\npsp_x_w" << endl;
    print_vec_1d(psp_x_w, bs);
    printf("time: %f\n", end-start);

    return 0;

}
