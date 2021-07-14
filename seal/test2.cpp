#include <iostream>
#include <algorithm>
#include <cassert>
#include <numeric> 
#include <stdio.h>


#include "seal/seal.h"

using namespace std;
using namespace seal;


#include <chrono>
using namespace std::chrono;
inline double get_time_msec(void){
    return static_cast<double>(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count())/1000000;
}

vector<vector<double>> give_me_ys(int bs, int dim){
  vector<vector<double>> res;
  for(int i=0; i<bs; i++){
    vector<double> tmp;
    for(int j=0; j<dim; j++){
      tmp.push_back(double(i)*0.1 + double(j));
    }
    res.push_back(tmp);
  }
  return res;
}


vector<vector<double>> give_me_xs(int batch_size, int dim){
  vector<vector<double>> xs(batch_size, vector<double>(dim));
  for(int i=0; i<batch_size; i++){
    for(int j=0; j<dim; j++){
      xs[i][j] = double(i+1)*0.1 + double(j);
    }
  }
  return xs;
}

void print_vec_1d(vector<int> x, int size){
  for(int i=0; i<size; i++){
    printf("%d, ", x[i]);
  }
  printf("\n");
}

void print_vec_1d(vector<double> x, int size){
  for(int i=0; i<size; i++){
    printf("%f, ", x[i]);
  }
  printf("\n");
}

void print_vec_2d(vector<vector<int>> xs, int size){
  for(int i=0; i<xs.size(); i++){
    print_vec_1d(xs[i], size);
  }
}
void print_vec_2d(vector<vector<double>> xs, int size){
  for(int i=0; i<xs.size(); i++){
    print_vec_1d(xs[i], size);
  }
}

vector<vector<double>> pp_xs(vector<vector<double>> xs, int dim, int bs, int hyc){
  assert(xs.size() == bs);
  assert(xs[0].size() == dim);

  vector<vector<double>> res;
  for(int i=0; i<bs; i++){
      vector<double> tmp;
      for(int j=0; j<hyc; j++){
          for(int k=0; k<dim; k++){
              tmp.push_back(xs[i][k]);
          }
      }
      res.push_back(tmp);
  }

  assert(res.size()==bs);
  assert(res[0].size()==hyc*dim);

  return res;

  //int tmp_num = pmd/2/dim;

  //vector<vector<double>> res;
  //for(int i=0; i<n; i++){
  //  vector<double> tmp(pmd/2);
  //    for(int j=0; j<tmp_num; j++){
  //        for(int k=0; k<dim; k++){
  //            tmp[j*dim+k] = xs[i][k];
  //        }
  //    }
  //    res.push_back(tmp);
  //}
  //assert(res.size() == n);
  //assert(res[0].size() == pmd/2);
  
  //return res;
}

vector<vector<double>> pp_ys(vector<vector<double>> ys, int pmd, int dim, int target_num, int hcn, int hyc){
    assert(ys.size()==target_num);
    assert(ys[0].size()==dim);

    assert(hyc==2000);
    assert(pmd/2/dim > hyc);
        vector<vector<double>> res;
        for(int i=0; i<hcn; i++){
            vector<double> tmp2;
            for(int j=0; j<hyc; j++){
                for(int k=0; k<dim; k++){
                    tmp2.push_back(ys[i*hyc+j][k]);
                }

            }
            res.push_back(tmp2);
        }

    assert(res.size()==hcn);
    assert(res[0].size()==hyc*dim);

    return res;


    //for(int i=0; i<target_num; i++){
    //    for(int j=0; j<dim; j++){
    //        res.push_back(ys[i][j]);
    //    }
    //}
    //assert(res.size() == dim*target_num);

    //return res;
}


vector<Ciphertext> encode_encrypt_input(vector<vector<double>> xs, CKKSEncoder &encoder, Encryptor &encryptor, double scale, int bs, int hyc, int dim){
  assert(xs.size()==bs);
  assert(xs[0].size()==hyc*dim);
  vector<Plaintext> tmp;
  for(int i=0; i<bs; i++){
    Plaintext tmp1;
    encoder.encode(xs[i], scale, tmp1);
    tmp.push_back(tmp1);
  }

  vector<Ciphertext> res;
  for(int i=0; i<bs; i++){
    Ciphertext tmp1;
    encryptor.encrypt(tmp[i], tmp1);
    res.push_back(tmp1);
  }
  assert(res.size()==bs);
  return res;
}

vector<Plaintext> encode_ys(vector<vector<double>> ys, CKKSEncoder &encoder, double scale, int hcn){
  vector<Plaintext> res;
  for(int i=0; i<hcn; i++){
      Plaintext tmp;
    encoder.encode(ys[i], scale, tmp);
    res.push_back(tmp);
  }
  assert(res.size()==hcn);
  return res;
}

vector<vector<Ciphertext>> calc_dist(vector<Ciphertext> xs, vector<Plaintext> ys, Evaluator &evaluator, RelinKeys relin_keys, GaloisKeys galois_keys, int hcn, int dim, int bs){
  assert(xs.size()==bs);
  assert(ys.size()==hcn);




  vector<vector<Ciphertext>> xs_tmp;
  for(int i=0; i<bs; i++){
      vector<Ciphertext> xs_tmp2;
      for(int j=0; j<hcn; j++){
          xs_tmp2.push_back(xs[i]);
      }
      xs_tmp.push_back(xs_tmp2);
  }
  assert(xs_tmp.size()==bs);
  assert(xs_tmp[0].size()==hcn);

  vector<vector<Ciphertext>> calc_tmp;
  for(int i=0; i<bs; i++){
      vector<Ciphertext> calc_tmp2;
      for(int j=0; j<hcn; j++){
        Ciphertext tmp1;
        evaluator.sub_plain(xs_tmp[i][j], ys[j], tmp1);
        evaluator.square_inplace(tmp1);
        evaluator.relinearize_inplace(tmp1, relin_keys);
        calc_tmp2.push_back(tmp1);
      }
      calc_tmp.push_back(calc_tmp2);
  }

  vector<vector<Ciphertext>> rot_tmp;
  for(int i=0; i<bs; i++){
      vector<Ciphertext> rot_tmp2;
      for(int j=0; j<hcn; j++){
          rot_tmp2.push_back(calc_tmp[i][j]);
      }
      rot_tmp.push_back(rot_tmp2);
  }

  for(int i=0; i<bs; i++){
      vector<Ciphertext> rot_tmp2;
      for(int j=0; j<hcn; j++){
          for(int k=1; k<dim; k++){
            Ciphertext tmp;
            evaluator.rotate_vector(rot_tmp[i][j], k, galois_keys, tmp);
            evaluator.add_inplace(calc_tmp[i][j], tmp);
          }
      }
  }

  assert(calc_tmp.size()==bs);
  assert(calc_tmp[0].size()==hcn);

  return calc_tmp;
  

  //vector<Ciphertext> tmp_sub;
  //for(int i=0; i<n; i++){
  //  Ciphertext tmp1;
  //  evaluator.sub_plain(xs[i], ys, tmp1);
  //  tmp_sub.push_back(tmp1);
  //}
  //for(int i=0; i<n; i++){
  //  evaluator.square_inplace(tmp_sub[i]);
  //  evaluator.relinearize_inplace(tmp_sub[i], relin_keys);
  //}

  //vector<Ciphertext> res(n);
  //for(int i=0; i<n; i++){
  //    res[i] = tmp_sub[i];
  //}
  //for(int i=0; i<n; i++){
  //    for(int j=1; j<dim; j++){
  //    }
  //}
  //return res;
}


vector<vector<vector<double>>> decrypt_decode_res(vector<vector<Ciphertext>> xs, CKKSEncoder &encoder, Decryptor &decryptor, int bs, int hcn){
    assert(xs.size()==bs);
    assert(xs[0].size()==hcn);
    
  vector<vector<Plaintext>> tmp;
  for(int i=0; i<bs; i++){
      vector<Plaintext> tmp1;
      for(int j=0; j<hcn; j++){
          Plaintext tmp2;
        decryptor.decrypt(xs[i][j], tmp2);
        tmp1.push_back(tmp2);
      }
      tmp.push_back(tmp1);
  }

  vector<vector<vector<double>>> res;

  for(int i=0; i<bs; i++){
      vector<vector<double>> res1;
      for(int j=0; j<hcn; j++){
          vector<double> res2;
          encoder.decode(tmp[i][j], res2);
          res1.push_back(res2);
      }
      res.push_back(res1);
  }
  assert(res.size()==bs);
  assert(res[0].size()==hcn);
  return res;
}

int find_index(vector<int> x, int value)
{  
    size_t len = x.size();
    for( size_t n = 0 ; n < len ; ++n ){
        if( x[ n ] == value ){
            return n;
        }
    }
    return -1;
}
int find_index(vector<double> x, int value)
{  
    size_t len = x.size();
    for( size_t n = 0 ; n < len ; ++n ){
        if( x[ n ] == value ){
            return n;
        }
    }
    return -1;
}

vector<int> find_knn(vector<double> x, int size, int target_num){
    assert(x.size()==target_num);
    vector<int> test2;
    for(int i=0; i<x.size(); i++){
        test2.push_back(static_cast<int>(x[i]*10000));
    }

    std::sort(x.begin(), x.end());

    vector<int> indices(size);
    for(int i=0; i<size; i++){
        indices[i] = find_index(test2, x[i]);
    }

    return indices;
}


vector<vector<int>> psp_res(vector<vector<vector<double>>> xs, int target_num, int hcn, int hyc, int dim, int bs, int knn_size){
  assert(xs.size() == bs);
  assert(xs[0].size()==hcn);
  assert(target_num == hyc*hcn);



  vector<vector<double>> res;
  for(int i=0; i<bs; i++){
    vector<double> tmp(target_num);
    for(int j=0; j<hcn; j++){
        for(int k=0; k<hyc; k++){
            tmp[j*hyc+k] = xs[i][j][k];
        }
    }
    res.push_back(tmp);
  }

  vector<vector<int>> res2;
  for(int i=0; i<bs; i++){
      res2.push_back(find_knn(res[i], knn_size, target_num));
  }

  assert(res2.size()==bs);
  assert(res2[0].size()==knn_size);
  return res2;
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

    int dim = 2;
    int bs = 2000;
    int target_num = 20000;
    int knn_size = 3;
    vector<vector<double>> ys = give_me_ys(target_num, dim);
    vector<vector<double>> xs = give_me_xs(bs, dim);

    //print_vec_2d(ys, dim);
    cout << endl;
    print_vec_2d(xs, dim);

    int hyc = 2000; // hm_ys_in_c == 8000/2/dim = 2000
    int hcn = target_num/hyc; //hm_c_need
    assert(hcn=10);



    int l = 1; // d=8192, l=2 if pmd=8192*2
    int n = bs/l;
    int nt = bs % l;
    int ls;
    if(nt != 0){
        n += 1;
        ls = l;
    }else{
        ls = nt;
    }
    printf("dim=%d\nbs=%d\ntarget_num=%d\nl=%d\nn=%d\nnt=%d\nls=%d\n", dim, bs, target_num, l, n, nt, ls);

    double start, end;

    start = get_time_msec();
    vector<vector<double>> ppd_xs = pp_xs(xs, dim, bs, hyc);

    //print_vec_2d(ppd_xs, dim*target_num+4);
    vector<vector<double>> ppd_ys = pp_ys(ys, pmd, dim, target_num, hcn, hyc);
    //print_vec_1d(ppd_ys, dim*target_num+4);
    cout << "\npp done" << endl;

    vector<Ciphertext> enc_xs = encode_encrypt_input(ppd_xs, encoder, encryptor, scale, bs, hyc, dim);
    vector<Plaintext> plain_ys = encode_ys(ppd_ys, encoder, scale, hcn);
    cout << "\nenc done" << endl;
    vector<vector<Ciphertext>> c_res = calc_dist(enc_xs, plain_ys, evaluator, relin_keys, gal_keys, hcn, dim, bs);
    cout << "\ncalc done" << endl;
    vector<vector<vector<double>>> dec_res = decrypt_decode_res(c_res, encoder, decryptor, bs, hcn);
    cout << "\ndec done" << endl;
    //vector<vector<int>> pspd = psp_res(dec_res, target_num, l, ls, n, dim, bs);
    vector<vector<int>> pspd = psp_res(dec_res, target_num, hcn, hyc, dim, bs, knn_size);
    cout << "\npsp done" << endl;
    cout << "\npspd" << endl;
    print_vec_2d(pspd, knn_size);
    end = get_time_msec();

    printf("time: %f\n", end - start);

    return 0;

}
