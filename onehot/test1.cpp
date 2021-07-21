#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <istream>
#include <sstream>
#include <cassert>
#include <map>

#include <chrono>
using namespace std::chrono;
inline double get_time_millisec(void){
    return static_cast<double>(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count())/1000000;
}

using namespace std;


void print_vec_1d(vector<string> x){
  for(int i=0; i<x.size(); i++){
    printf("%s, ", x[i].c_str());
  }
  printf("\n");
}

std::multimap<string, vector<string>> load_dictionary(string filename){
  // load file from python to string
  string test1;
  vector<string> test3;

  std::ifstream ifs("cols_from_python.txt");
  std::string str;
  if(ifs.fail()){
    std::cerr << "file read failed" << std::endl;
  }
  while(getline(ifs,str)){
    test1 = str;
    //cout << test1 << endl;
  }


  // separate with ,
  std::stringstream test2(test1);
  string buf;

  while(getline(test2,buf, ',')){
    test3.push_back(buf);
  }

  // make dict(multimap)
  // dict[i] is vector<string>, which is showing found alpabet at column i from training dataset
  std::multimap<string, vector<string>> dic;
  string tmp_index;
  vector<string> tmp_vec;
  for(int i=0; i<test3.size(); i++){
    string separator = "_";
    int separator_length = separator.length();
    string string = test3[i];

    auto offset = std::string::size_type(0);
    auto pos = string.find(separator, offset);
    auto index = string.substr(offset, pos - offset);
    if(tmp_index != index){
      dic.insert(std::make_pair(tmp_index, tmp_vec));
      tmp_vec.clear();
    }
    offset = pos + separator_length;
    pos = string.find(separator, offset);
    // assert just in case, if this fails, string is not the shape like X_Y
    assert(pos == std::string::npos);
    auto alphabet = string.substr(offset);
    tmp_vec.push_back(alphabet);
    tmp_index = index;
  }

  return dic;

}


void show_dictionary(std::multimap<string, vector<string>> dic){
  for(int i=0; i<10; i++){
    auto itr = dic.find(to_string(i));
    if( itr == dic.end() ) {
        cout << "not found.\n";
    }else{
        cout << itr->first << ": ";
        print_vec_1d(itr->second);
    }
  }

}



int main(){
  printf("hello, world\n");

  double start, end;

  string filename = "cols_from_python.txt";


  printf("\nlaod_map======================================================\n");
  start = get_time_millisec();
  std::multimap<string, vector<string>> dic = load_dictionary(filename);
  end = get_time_millisec();
  printf("time to load dic [millisec]: %f", end-start);
  printf("\n\nshow_map (only head) ========================================\n");
  show_dictionary(dic);
  printf("dictionary length: %d\n", dic.size());



}
